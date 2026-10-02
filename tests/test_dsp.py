"""Cut-point search and edge fades; ducking, loudness, true peak and limiting."""

import numpy as np
import pytest

from speech2song.audio.dsp import (
    fade_edges,
    frame_levels_db,
    limit,
    loudness_lufs,
    master,
    oversampled_peaks,
    quietest_point,
    quote_duck,
    true_peak_db,
)

from .fixtures.synth import SR, noise_burst, sine


def test_quietest_point_finds_the_gap() -> None:
    signal = np.concatenate([noise_burst(0.3), np.zeros(int(0.05 * SR)), noise_burst(0.3)])
    gap_start, gap_end = int(0.3 * SR), int(0.35 * SR)
    index, level = quietest_point(signal, 0, len(signal) - 1, int(0.2 * SR), frame=441)
    assert gap_start <= index <= gap_end
    assert level < -100


def test_ties_go_to_the_point_nearest_the_boundary() -> None:
    silence = np.zeros(SR, dtype=np.float32)
    index, _ = quietest_point(silence, 100, 40_000, 12_345, frame=441)
    assert index == 12_345
    index, _ = quietest_point(silence, 100, 40_000, 99_999, frame=441)
    assert index == 40_000  # clamped to the search range


def test_start_and_end_cuts_leave_room_for_the_fade() -> None:
    """Speech from 0.5 s to 1.0 s; the transcript thinks it starts 40 ms late."""
    signal = np.zeros(int(1.5 * SR), dtype=np.float32)
    onset, offset = int(0.5 * SR), int(1.0 * SR)
    signal[onset:offset] = noise_burst(0.5)[: offset - onset]
    pad = int(0.035 * SR)
    start, _ = quietest_point(signal, onset - 6000, onset + 4000, onset + 1764, frame=441,
                              side="start", pad=pad)  # fmt: skip
    assert onset - pad - 300 <= start <= onset - pad
    end, _ = quietest_point(signal, offset - 4000, offset + 6000, offset - 1764, frame=441,
                            side="end", pad=pad)  # fmt: skip
    assert offset + pad <= end <= offset + pad + 300


def test_short_pauses_cut_where_they_can() -> None:
    """A 20 ms pause is shorter than the pad: cut at the far edge of the pause."""
    signal = noise_burst(0.4)
    gap_lo, gap_hi = int(0.2 * SR), int(0.22 * SR)
    signal[gap_lo:gap_hi] = 0
    index, _ = quietest_point(signal, gap_lo - 2000, gap_hi + 2000, gap_hi, frame=441,
                              side="start", pad=int(0.035 * SR))  # fmt: skip
    assert gap_lo <= index <= gap_hi


def test_search_range_is_respected() -> None:
    signal = np.concatenate([np.zeros(1000), noise_burst(0.2)]).astype(np.float32)
    index, _ = quietest_point(signal, 2000, 5000, 3000, frame=441)
    assert 2000 <= index <= 5000


def test_frame_levels_of_a_known_sine() -> None:
    tone = sine(1000, 0.5, amp=0.5)
    levels = frame_levels_db(tone, np.array([SR // 4]), frame=4410)
    assert levels[0] == pytest.approx(20 * np.log10(0.5 / np.sqrt(2)), abs=0.1)


def test_fades_touch_only_the_edges() -> None:
    block = np.stack([noise_burst(0.5, seed=1), noise_burst(0.5, seed=2)], axis=1)
    faded = fade_edges(block, 441)
    np.testing.assert_array_equal(faded[441:-441], block[441:-441])
    assert abs(faded[0]).max() < abs(block[0]).max() * 0.01 + 1e-9
    ramp = faded[:441, 0] / block[:441, 0]
    assert np.all(np.diff(ramp) > 0) and 0 < ramp[0] < 0.01 and 0.99 < ramp[-1] < 1


def test_zero_fade_and_short_blocks() -> None:
    block = noise_burst(0.01)[:, None]
    np.testing.assert_array_equal(fade_edges(block, 0), block)
    short = fade_edges(block[:10], 441)  # fade shrinks to half the block
    assert short.shape == (10, 1)


# --- Ducking -----------------------------------------------------------------------------


def test_a_quote_ducks_once_easing_down_before_and_up_after() -> None:
    sr = 1000
    gain = quote_duck([(3000, 5000)], 10000, sr, depth_db=-9, lead_s=0.6, release_s=1.5)
    db = 20 * np.log10(gain)
    assert db[2399] > -0.01  # nothing yet, 0.6 s before the first word
    assert db[2700] == pytest.approx(-4.5, abs=0.3)  # half-way down at the middle of the ramp
    assert db[3000] < -8.9 and db[4999] == pytest.approx(-9, abs=0.01)  # down for the quote
    assert db[5750] == pytest.approx(-4.5, abs=0.3)  # half-way back up
    assert db[6500] > -0.05 and db[-1] == 0  # recovered 1.5 s after the last word
    assert np.all(np.diff(db[2400:3000]) <= 1e-6) and np.all(np.diff(db[5000:6500]) >= -1e-6)


def test_overlapping_quotes_take_the_deeper_duck_and_edges_are_safe() -> None:
    sr = 1000
    gain = quote_duck([(0, 1000), (1500, 2500)], 3000, sr, depth_db=-9, lead_s=0.6,
                      release_s=1.5)  # fmt: skip
    db = 20 * np.log10(gain)
    assert db[500] == pytest.approx(-9, abs=0.01)  # a quote at the very start
    assert db[1250] < -6  # 500 ms of pause between the quotes: not back up
    assert db.min() >= -9 - 1e-6  # never deeper than the depth
    assert quote_duck([], 100, sr, depth_db=-9, lead_s=0.6, release_s=1.5).min() == 1.0
    assert quote_duck([(50, 50)], 100, sr, depth_db=-9, lead_s=0.6, release_s=1.5).min() == 1.0
    flat = quote_duck([(10, 90)], 100, sr, depth_db=-9, lead_s=0.0, release_s=0.0)
    assert flat[9] == 1.0 and flat[50] == pytest.approx(10 ** (-9 / 20), rel=1e-4)


# --- Loudness, true peak, limiting --------------------------------------------------------


def test_true_peak_finds_inter_sample_overs() -> None:
    t = np.arange(SR) / SR
    quarter = np.sin(2 * np.pi * SR / 4 * t + np.pi / 4).astype(np.float32)
    assert 20 * np.log10(np.abs(quarter).max()) == pytest.approx(-3.01, abs=0.01)
    assert true_peak_db(quarter) == pytest.approx(0.0, abs=0.2)  # samples miss the crest
    assert len(oversampled_peaks(np.stack([quarter, quarter], axis=1))) == SR


def test_loudness_matches_pyloudnorm() -> None:
    import pyloudnorm

    rng = np.random.default_rng(3)
    music = rng.standard_normal((SR * 7, 2)).astype(np.float32) * 0.1
    music[SR * 2 : SR * 4] *= 0.01  # a quiet stretch the relative gate drops
    music[SR * 5 :] = 0  # silence the absolute gate drops
    expected = pyloudnorm.Meter(SR).integrated_loudness(music.astype(np.float64))
    assert loudness_lufs(music, SR) == pytest.approx(expected, abs=0.01)
    assert loudness_lufs(music[:, 0], SR) == pytest.approx(
        pyloudnorm.Meter(SR).integrated_loudness(music[:, 0].astype(np.float64)), abs=0.01
    )


def test_loudness_of_a_known_sine() -> None:
    # BS.1770: a 997 Hz full-scale sine reads -3.01 LUFS in one channel.
    tone = sine(997, 3.0, amp=1.0)
    assert loudness_lufs(tone, SR) == pytest.approx(-3.01, abs=0.1)
    assert loudness_lufs(np.zeros(SR * 2, dtype=np.float32), SR) is None
    assert loudness_lufs(tone[:1000], SR) is None  # shorter than one block


def test_limiter_holds_the_ceiling_and_leaves_quiet_parts_alone() -> None:
    quiet = noise_burst(2.0, amp=0.05)
    loud = np.concatenate([quiet, noise_burst(0.2, amp=0.9, seed=1), quiet])
    out = limit(np.stack([loud, loud], axis=1), SR, -3.0)[:, 0]
    assert true_peak_db(out) <= -3.0 + 0.05
    np.testing.assert_allclose(out[:SR], loud[:SR], rtol=1e-6)  # far from the peak


def test_master_hits_the_loudness_target_under_the_ceiling() -> None:
    music = np.stack([noise_burst(20.0, amp=0.05), noise_burst(20.0, amp=0.05, seed=2)], axis=1)
    music[SR * 5 : SR * 5 + 400] *= 15  # transients the limiter has to catch
    result = master(music, SR, target_lufs=-14.0, ceiling_db=-1.0)
    assert result.lufs == pytest.approx(-14.0, abs=0.1)
    assert result.true_peak_db <= -1.0
    assert true_peak_db(result.audio) <= -1.0


# --- The speech guard ---------------------------------------------------------------------


def test_the_guard_keeps_every_word_clear_of_the_music() -> None:
    from speech2song.audio.dsp import (
        GUARD_ATTACK_DB_S,
        band_filter,
        channel_levels_db,
        speech_guard,
        word_level_db,
    )

    sr = 16000
    rng = np.random.default_rng(4)
    music = (0.05 * rng.standard_normal((6 * sr, 2))).astype(np.float32)
    speech = np.zeros_like(music)
    loud, soft = (sr, 2 * sr), (4 * sr, 5 * sr)  # a word 20 dB over the music, one at par
    speech[loud[0] : loud[1]] = 0.5 * rng.standard_normal((sr, 2))
    speech[soft[0] : soft[1]] = 0.05 * rng.standard_normal((sr, 2))
    gain, margins = speech_guard(speech, music, sr, [loud, soft], margin_db=10)
    assert margins[0] == pytest.approx(20, abs=1) and margins[1] == pytest.approx(0, abs=1)
    db = 20 * np.log10(gain)
    assert db[loud[0] : loud[1]].min() > -0.01  # clear already: untouched
    inside = db[soft[0] : soft[1] - sr // 50]  # the last frames start to recover
    assert np.allclose(inside, margins[1] - 10, atol=0.05)  # down by exactly the shortfall
    lead = round(5 / GUARD_ATTACK_DB_S * sr)  # the dip is half-way down this far ahead
    assert db[soft[0] - lead] < -2 and db[soft[0] - sr // 2] > -0.01
    guarded = music * gain[:, None]
    mid = np.array([(soft[0] + soft[1]) // 2])
    under = channel_levels_db(band_filter(guarded, sr), mid, soft[1] - soft[0])[0]
    after = word_level_db(band_filter(speech, sr), *soft, sr) - under
    assert after == pytest.approx(10, abs=0.3)
    assert speech_guard(speech, music, sr, [], margin_db=10)[0].min() == 1.0


def test_words_close_together_are_guarded_without_recovering_between_them() -> None:
    from speech2song.audio.dsp import GUARD_GROUP_S, dips, speech_guard

    sr = 16000
    rng = np.random.default_rng(5)
    music = (0.05 * rng.standard_normal((9 * sr, 2))).astype(np.float32)
    speech = np.zeros_like(music)
    words = [(sr, 3 * sr // 2), (7 * sr // 4, 9 * sr // 4),  # a phrase: 250 ms apart
             (6 * sr, 13 * sr // 2)]  # a word on its own, far from it  # fmt: skip
    for (a, b), amp in zip(words, (0.05, 0.063, 0.05), strict=True):  # 0, +2, 0 dB over the music
        speech[a:b] = amp * rng.standard_normal((b - a, 2))
    assert GUARD_GROUP_S > 0.25
    gain, margins = speech_guard(speech, music, sr, words, margin_db=10)
    assert margins[0] < margins[1] < 3
    db = 20 * np.log10(gain)
    pause = db[3 * sr // 2 + sr // 100 : 7 * sr // 4 - sr // 100]
    assert pause.max() < margins[1] - 10 + 0.3  # held at the shallower dip, not recovering
    found = dips(gain, sr)
    assert len(found) == 2  # one dip for the phrase, one for the lone word
    assert found[0][0] < 1.0 and found[0][1] > 2.25  # eased in ahead and out after the phrase


def test_word_spans_follow_the_clips() -> None:
    from speech2song.audio.mixing import WORD_TAIL_S, word_spans
    from speech2song.models import Clip, PlacedClip, Word

    clip = Clip.model_construct(id="c1", start_s=10.0, end_s=12.0)
    placed = [PlacedClip(clip_id="c1", section_id="s2", file="x.wav", start_sample=1000,
                         end_sample=1000 + 2 * 100, start_s=10.0)]  # fmt: skip
    words = [Word(w="before", start=9.0, end=9.5), Word(w="early", start=9.9, end=10.3),
             Word(w="hello", start=10.5, end=11.0), Word(w="end", start=11.9, end=12.0),
             Word(w="If", start=11.97, end=12.4)]  # fmt: skip
    spans = word_spans(placed, {"c1": clip}, words, 100)
    assert {s.section_id for s in spans} == {"s2"}
    assert [(s.word, s.start, s.end) for s in spans] == [
        ("early", 1000, 1000 + round((0.3 + WORD_TAIL_S) * 100)),  # starts inside the clip
        ("hello", 1050, 1000 + round((1.0 + WORD_TAIL_S) * 100)),
        ("end", 1190, 1200),  # kept inside the clip
    ]  # "If", the next sentence's first word, only grazes the clip's end cut  # fmt: skip


def test_quote_regions_run_from_the_first_word_to_the_last() -> None:
    from speech2song.audio.mixing import WordSpan, quote_regions
    from speech2song.models import PlacedClip

    placed = [PlacedClip(clip_id="c1", section_id="s2", file="x.wav", start_sample=1000,
                         end_sample=2000, start_s=0.0),
              PlacedClip(clip_id="c2", section_id="s5", file="y.wav", start_sample=5000,
                         end_sample=6000, start_s=0.0)]  # fmt: skip
    words = [WordSpan("c1", "s2", "a", 1100, 1300), WordSpan("c1", "s2", "b", 1500, 1800)]
    assert quote_regions(placed, words) == [(1100, 1800), (5000, 6000)]  # c2: no words
