"""Cut-point search and edge fades."""

import numpy as np
import pytest

from speech2song.audio.dsp import fade_edges, frame_levels_db, quietest_point

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
