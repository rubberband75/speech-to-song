"""Song lengths (M8): the preset's forms, fitting a song to its window, the anchor's
tempo, key and chords, choosing several lengths' quotes in one call, the per-length
folders, conditioning a length's music on the anchor's take, and runs made before M8
becoming their summary length."""

import json
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml
from typer.testing import Result

from speech2song.arrangement import (
    anchor_progressions,
    arc_schema,
    bar_options,
    build_arc_prompt,
    build_arrangement,
    chord_cycle,
    default_parts,
    ending_seconds,
    fit_parts,
    song_seconds,
)
from speech2song.backends import music_elevenlabs, transcribe_base
from speech2song.config import ConfigError, load_preset
from speech2song.costs import CostLog
from speech2song.manifest import Run, migrate_v1
from speech2song.models import (
    Arrangement,
    AsrWord,
    ClipSet,
    Melody,
    SelectionResult,
    TakeChoice,
    TakeMeta,
    Transcript,
    VersionsAnswer,
)
from speech2song.music_plan import MAX_REFERENCE_MS, anchor_range
from speech2song.selection import (
    EarlierQuote,
    align_ids,
    build_prompt,
    build_versions_prompt,
    clip_targets,
    prompt_digest,
    renamed,
    validate_versions,
)

from .conftest import PRESETS_DIR, SHORT_BARS, Talk
from .fixtures.fake_claude import response
from .fixtures.fake_elevenlabs import FakeElevenLabs
from .fixtures.synth import synthetic_talk, write_wav
from .test_arrangement import _clip, _melody
from .test_end_to_end import TalkTranscriber

Cli = Callable[..., Result]
PRESET = load_preset("cinematic_future_bass", PRESETS_DIR)
# The summary's selection prompt (select_clips.md): editing it would ask Claude again for
# the summary of every run made so far.
SUMMARY_PROMPT_DIGEST = "b20e27f1463e02da"


# --- The preset's lengths ---------------------------------------------------------------------


def test_lengths_come_from_the_preset() -> None:
    short = PRESET.for_length("short")
    assert short.arc == ["intro", "speech_bed", "build", "gap", "drop", "speech_bed", "outro"]
    assert short.song_seconds == (61, 75)
    assert (short.role_bars("intro"), short.role_bars("drop")) == (4, 8)
    targets = clip_targets(short)
    assert targets.count_range == (2, 4) and targets.total_speech_seconds == 25
    highlights = PRESET.for_length("highlights")
    assert clip_targets(highlights).count_range == (3, 7)
    assert PRESET.for_length("summary") is PRESET  # the preset itself, unchanged
    assert clip_targets(PRESET).count_range == (6, 10)
    assert PRESET.length_description("summary").startswith("four to six minutes")


def test_an_arc_entry_can_give_its_bars() -> None:
    highlights = PRESET.for_length("highlights")
    clips = [_clip("a", 4.0), _clip("b", 4.0), _clip("c", 4.0)]
    parts = default_parts(highlights, clips)
    drops = [p.bars for p in parts if p.role == "drop"]
    builds = [p.bars for p in parts if p.role == "build"]
    assert drops == [8, 16] and builds == [4, 8]  # "build 8", "drop 16": the climax is longer


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"lengths": {"long": {"description": "x"}}}, "unknown length"),
        ({"lengths": {"summary": {"arc": ["speech_bed"]}}}, "may only give a description"),
        ({"lengths": {"short": {"bars": {"speech_bed": 4}}}}, "is not a music role"),
        ({"lengths": {"short": {"clips": {"counts": 3}}}}, "unknown keys"),
        ({"lengths": {"short": {"seconds": [75, 61]}}}, "0 < min < max"),
        ({"lengths": {"short": {"arc": ["intro", "outro"]}}}, "must contain at least one"),
        ({"arc": ["intro", "speech_bed 4", "outro"]}, "must be a music role"),
    ],
)
def test_bad_lengths_are_rejected(tmp_path: Path, change: dict, message: str) -> None:
    data = yaml.safe_load((PRESETS_DIR / "cinematic_future_bass.yaml").read_text())
    data.update(change)
    (tmp_path / "cinematic_future_bass.yaml").write_text(yaml.safe_dump(data))
    with pytest.raises(ConfigError, match=message):
        load_preset("cinematic_future_bass", tmp_path)


def test_a_preset_without_the_length_says_how_to_add_it() -> None:
    bare = PRESET.model_copy(update={"lengths": {}})
    with pytest.raises(ConfigError, match="has no 'short' length"):
        bare.for_length("short")


# --- Fitting a song to its window -------------------------------------------------------------


def test_bar_options_stay_within_half_and_double() -> None:
    assert bar_options(8) == [4, 8, 12, 16]
    assert bar_options(1) == [1, 2]
    assert bar_options(32) == [16, 32]  # parts are at most 32 bars


def test_fitting_lands_the_song_in_its_window() -> None:
    short = PRESET.for_length("short")
    clips = [_clip("a", 11.0), _clip("b", 11.0)]  # long lines: the music has to make room
    parts = default_parts(short, clips)
    before = song_seconds(parts, clips, 120.0, short, "held_chord")
    assert before > 75
    fitted, warning = fit_parts(parts, clips, 120.0, short, (61, 75), "held_chord")
    after = song_seconds(fitted, clips, 120.0, short, "held_chord")
    assert warning is None and 61 <= after <= 75
    for old, new in zip(parts, fitted, strict=True):
        if old.role != "speech_bed" and old.role != "gap":
            assert new.bars in bar_options(old.bars)
        else:
            assert new.bars == old.bars  # speech passages and the gap keep their length
    inside, _ = fit_parts(fitted, clips, 120.0, short, (61, 75), "held_chord")
    assert inside == fitted  # a song that already fits is left as it is


def test_a_song_that_cannot_fit_says_so() -> None:
    short = PRESET.for_length("short")
    clips = [_clip("a", 30.0), _clip("b", 30.0)]
    fitted, warning = fit_parts(default_parts(short, clips), clips, 120.0, short, (61, 75),
                                "held_chord")  # fmt: skip
    assert warning is not None and "outside the 61-75 s" in warning
    assert all(
        p.bars == min(bar_options(8 if p.role == "drop" else 4))
        for p in fitted
        if p.role in ("intro", "build", "drop", "outro")
    )  # as short as it gets


# --- The anchor's chords ----------------------------------------------------------------------


def test_chord_cycles() -> None:
    assert chord_cycle(["Am", "F", "Am", "F"]) == ["Am", "F"]
    assert chord_cycle(["Am", "F", "C"]) == ["Am", "F", "C"]


def test_other_lengths_take_the_anchors_chords() -> None:
    clips = [_clip("a", 6.0), _clip("b", 6.0), _clip("c", 6.0)]
    melody = _melody(clips)
    anchor = build_arrangement(default_parts(PRESET, clips), clips, melody, PRESET)
    chords = anchor_progressions(anchor)
    anchor_drops = [s for s in anchor.sections if s.role == "drop"]
    assert chords["drop"] == [chord_cycle(s.chords) for s in anchor_drops]
    assert "speech_bed" not in chords and "gap" not in chords

    short = PRESET.for_length("short")
    mine = clips[:2]
    song = build_arrangement(default_parts(short, mine), mine, melody, short, anchor_chords=chords)
    (drop,) = [s for s in song.sections if s.role == "drop"]
    cycle = chords["drop"][-1]  # the one drop of a short takes the anchor's last (the climax)
    assert drop.chords == [cycle[i % len(cycle)] for i in range(drop.bars)]
    outro = song.sections[-1]
    assert outro.chords[-1] == "Am"  # still lands home


def test_the_arc_prompt_asks_for_the_lengths_duration() -> None:
    clips = [_clip("a", 4.0), _clip("b", 4.0)]
    _, summary = build_arc_prompt(PRESET, clips, 120.0, None)
    # unchanged for the summary, so Claude's arcs for songs made before M8 stay cached
    assert "- keep the whole song between about 3 and 6 minutes, longer when there is more " \
        "speech\n" in summary  # fmt: skip
    _, short = build_arc_prompt(PRESET.for_length("short"), clips, 120.0, None)
    assert "between about 61 and 75 seconds" in short
    assert "intro, speech_bed, build, gap, drop, speech_bed, outro" in short
    _, highlights = build_arc_prompt(PRESET.for_length("highlights"), clips, 120.0, None)
    assert "between about 2 and 3.5 minutes" in highlights
    assert arc_schema(["intro"], ["a"])["required"] == ["parts", "ending", "notes"]


# --- Choosing several lengths' quotes in one call ------------------------------------------


def _transcript() -> Transcript:
    return synthetic_talk()[1]


def _versions(transcript: Transcript, picks: dict[str, tuple[int, int]],
              versions: dict[str, list[str]]) -> dict[str, Any]:  # fmt: skip
    clips = [{"id": cid, "start_sentence": a, "end_sentence": b,
              "text": " ".join(s.text for s in transcript.sentences[a - 1 : b]),
              "score": round(0.95 - 0.05 * n, 2), "role": "hook", "reason": "it stands alone"}
             for n, (cid, (a, b)) in enumerate(picks.items())]  # fmt: skip
    return {"clips": clips, "versions": [{"length": length, "order": order, "notes": "n"}
                                         for length, order in versions.items()]}  # fmt: skip


def test_the_summary_prompt_is_unchanged() -> None:
    assert prompt_digest() == SUMMARY_PROMPT_DIGEST


def test_versions_prompt_lists_each_length_and_earlier_quotes() -> None:
    transcript = _transcript()
    targets = {"highlights": clip_targets(PRESET.for_length("highlights")),
               "short": clip_targets(PRESET.for_length("short"))}  # fmt: skip
    earlier = [EarlierQuote("c4", 4, 4, "s4w1 s4w2", ("summary",))]
    system, user = build_versions_prompt(transcript, PRESET, targets, earlier)
    assert "versions of different lengths" in system and "15 seconds" in system
    assert "- highlights: two to three and a half minutes with two drops" in user
    assert ". 3 to 7 lines (about 5), each 3 to 15 s (up to 25 s" in user
    assert "- short: about a minute with one drop" in user and "at most 25 s of speech" in user
    assert '- c4, sentences 4-4 (summary): "s4w1 s4w2"' in user
    assert "Keep the strongest of them" in user
    # the summary alone keeps today's prompt, with the earlier quotes in front
    _, alone = build_prompt(transcript, PRESET, clip_targets(PRESET), earlier)
    assert "Keep the strongest of them" in alone and "summarize the whole talk" in alone


def test_versions_are_checked_per_length() -> None:
    transcript = _transcript()
    targets = {"highlights": clip_targets(PRESET.for_length("highlights")),
               "short": clip_targets(PRESET.for_length("short"))}  # fmt: skip
    good = VersionsAnswer.model_validate(_versions(
        transcript, {"c1": (1, 1), "c2": (3, 3), "c3": (5, 5), "c4": (8, 8)},
        {"highlights": ["c1", "c2", "c3", "c4"], "short": ["c1", "c4"]}))  # fmt: skip
    assert validate_versions(good, transcript, targets) == []
    assert [c.id for c in good.selection("short").clips] == ["c1", "c4"]

    bad = VersionsAnswer.model_validate(_versions(
        transcript, {"c1": (1, 1), "c2": (2, 8)},
        {"highlights": ["c1", "c9"]}))  # fmt: skip
    messages = [p.message for p in validate_versions(bad, transcript, targets)]
    assert "short: no version was given" in messages
    assert any(m.startswith("highlights: its order names clips that are not in") for m in messages)
    assert any(m.startswith("highlights:") and "clips were returned" in m for m in messages)


def test_kept_quotes_keep_their_ids_across_lengths() -> None:
    transcript = _transcript()
    earlier = [
        EarlierQuote("c7", 5, 5, "x", ("summary",)),
        EarlierQuote("c1", 2, 2, "y", ("summary",)),
    ]
    answer = VersionsAnswer.model_validate(_versions(
        transcript, {"c1": (1, 1), "c2": (5, 5)}, {"short": ["c1", "c2"]}))  # fmt: skip
    renames = align_ids(answer.clips, earlier)
    # c2 is the summary's c7; Claude's new c1 can't keep an id the summary uses elsewhere
    assert renames == {"c2": "c7", "c1": "c2"}
    fixed = renamed(answer, renames)
    assert [c.id for c in fixed.clips] == ["c2", "c7"]
    assert fixed.versions[0].order == ["c2", "c7"]
    assert align_ids(answer.clips, []) == {}


# --- Runs made before M8 ----------------------------------------------------------------------


def test_a_schema_1_run_becomes_its_summary(tmp_path: Path) -> None:
    root = tmp_path / "runs" / "20261001-000000-old"
    (root / "clips").mkdir(parents=True)
    (root / "06_music").mkdir()
    (root / "clips" / "clip_001.wav").write_bytes(b"clip")
    (root / "06_music" / "take_001.mp3").write_bytes(b"take")
    (root / "03_clips.json").write_text("{}")
    (root / "05_arrangement_probe1.json").write_text("{}")
    (root / "02_transcript.json").write_text("{}")
    record = {"status": "complete", "version": 1, "fingerprint": "f",
              "started_at": "2026-10-01T00:00:00+00:00"}  # fmt: skip
    clip = root / "clips" / "clip_001.wav"
    stat = clip.stat()
    data = {
        "schema_version": 1, "run_id": root.name, "created_at": "2026-10-01T00:00:00+00:00",
        "tool_version": "0.1.0", "input": {"path": "/x.mp3", "sha256": "", "size": 0},
        "preset": "cinematic_future_bass",
        "options": {"isolate_voice": False, "clips": 5, "take": 2, "claude_model": "m"},
        "stages": {"ingest": record, "align": record, "select": record, "mix": record},
        "hash_memo": {str(clip.resolve()): {"ino": stat.st_ino, "size": stat.st_size,
                                            "mtime_ns": stat.st_mtime_ns, "sha256": "abc"}},
    }  # fmt: skip
    (root / "manifest.json").write_text(json.dumps(data))
    run = Run.open(tmp_path / "runs", root.name)
    assert run.migrated and run.manifest.schema_version == 2
    assert sorted(run.manifest.stages) == ["align", "ingest"]
    summary = run.song("summary")
    assert sorted(summary.stages) == ["mix", "select"]
    assert (summary.options.clips, summary.options.take) == (5, 2)
    assert run.manifest.options.claude_model == "m" and run.manifest.length == "summary"
    for rel in ("clips/clip_001.wav", "06_music/take_001.mp3", "03_clips.json",
                "05_arrangement_probe1.json"):  # fmt: skip
        assert summary.path(rel).exists() and not run.path(rel).exists()
    assert run.path("02_transcript.json").exists()  # the talk's files stay at the root
    moved = str(summary.path("clips/clip_001.wav").resolve())
    assert run.manifest.hash_memo[moved].sha256 == "abc"  # no re-hashing after the move
    assert Run.open(tmp_path / "runs", root.name).migrated is False  # once


def test_migrating_a_run_without_a_song_moves_nothing(tmp_path: Path) -> None:
    root = tmp_path / "r"
    root.mkdir()
    (root / "02_asr.json").write_text("{}")
    data = migrate_v1(root, {"schema_version": 1, "stages": {}, "options": {}})
    assert "songs" not in data and not (root / "summary").exists()


# --- Several lengths of a run, end to end -----------------------------------------------------


@pytest.fixture
def length_song(cli: Cli, tmp_path: Path) -> Path:
    """The cinematic preset with 1-2 bar sections and small lengths (a short song fitted
    to 40-46 s, highlights without a window), one music take."""
    data = yaml.safe_load((PRESETS_DIR / "cinematic_future_bass.yaml").read_text())
    for role, bars in SHORT_BARS.items():
        data["section_roles"][role]["bars"] = bars
    data["lengths"]["short"].update(
        seconds=[40, 46], bars={"intro": 1, "build": 1, "drop": 2, "outro": 1},
        clips={"count": 2, "count_tolerance": 0, "total_speech_seconds": 10})  # fmt: skip
    data["lengths"]["highlights"].update(
        seconds=None, arc=["intro", "speech_bed", "build", "gap", "drop", "speech_bed", "outro"],
        bars={"intro": 1, "build": 1, "drop": 2, "outro": 1},
        clips={"count": 3, "count_tolerance": 0, "total_speech_seconds": 15})  # fmt: skip
    presets = tmp_path / "length_presets"
    presets.mkdir()
    (presets / "cinematic_future_bass.yaml").write_text(yaml.safe_dump(data))
    (tmp_path / "config.yaml").write_text(
        f"presets_dir: {presets}\nruns_dir: runs\nmusic_takes: 1\n"
    )
    return presets


def _new_run(cli: Cli, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Run, Transcript]:
    """A new run of the synthetic talk, transcribed (by a fake Whisper)."""
    audio, truth, _ = synthetic_talk()
    source = write_wav(tmp_path / "talk.wav", np.stack([audio, 0.8 * audio], axis=1))
    asr = [AsrWord(w=w.w.capitalize(), start=w.start, end=w.end, conf=0.9) for w in truth.words]
    monkeypatch.setattr(transcribe_base, "make_transcriber",
                        lambda config, options: TalkTranscriber(asr))  # fmt: skip
    result = cli("run", str(source), "--stop-after", "transcribe")
    assert result.exit_code == 0, result.output
    run = Run.open(tmp_path / "runs")
    return run, Transcript.model_validate_json(run.path("02_transcript.json").read_text())


HIGHLIGHTS_SHORT = ({"c1": (1, 1), "c2": (3, 3), "c3": (8, 8)},
                    {"highlights": ["c1", "c2", "c3"], "short": ["c1", "c3"]})  # fmt: skip


def test_two_lengths_in_one_command(
    cli: Cli, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, install: Callable,
    length_song: Path,
) -> None:  # fmt: skip
    run, transcript = _new_run(cli, tmp_path, monkeypatch)
    fake = install(response(_versions(transcript, *HIGHLIGHTS_SHORT)))
    result = cli("run", "--run", "latest", "--length", "highlights,short", "--yes")
    assert result.exit_code == 0, result.output
    assert len(fake.requests) == 1  # one call chose both lengths' quotes
    prompt = fake.requests[0]["messages"][0]["content"]
    assert "- highlights: " in prompt and "- short: " in prompt
    assert "chosen in the same call as highlights" in result.output
    assert result.output.index("── highlights ──") < result.output.index("── short ──")

    run = Run.open(tmp_path / "runs")
    assert run.manifest.length == "short" and run.made_lengths() == ["highlights", "short"]
    assert run.manifest.shared is not None and run.manifest.shared.anchor == "highlights"
    highlights, short = run.song("highlights"), run.song("short")
    for song in (highlights, short):
        assert song.path("07_mix/master.wav").exists()
    shared = SelectionResult.model_validate_json(short.path("03_selection.json").read_text())
    assert shared.shared_from == "highlights" and shared.attempts[0].usd == 0.0
    assert [c.id for c in shared.answer().clips] == ["c1", "c3"]
    clips = ClipSet.model_validate_json(short.path("03_clips.json").read_text())
    assert clips.order == ["c1", "c3"]

    # the anchor's tempo, key and octave, and its chords in the music sections
    melodies = [Melody.model_validate_json(s.path("04_melody.json").read_text())
                for s in (highlights, short)]  # fmt: skip
    assert melodies[0].bpm == melodies[1].bpm and melodies[0].key == melodies[1].key
    assert melodies[0].octave_shift == melodies[1].octave_shift
    by_clip = [{c.clip_id: c for c in m.clips} for m in melodies]
    assert by_clip[0]["c1"].notes == by_clip[1]["c1"].notes  # the same quote, the same notes
    anchor = Arrangement.model_validate_json(highlights.path("05_arrangement.json").read_text())
    mine = Arrangement.model_validate_json(short.path("05_arrangement.json").read_text())
    (drop,) = [s for s in mine.sections if s.role == "drop"]
    cycle = anchor_progressions(anchor)["drop"][-1]
    assert drop.chords == [cycle[i % len(cycle)] for i in range(drop.bars)]
    preset = load_preset("cinematic_future_bass", length_song).for_length("short")
    seconds = mine.total_seconds + ending_seconds(preset, mine.ending or preset.ending)
    assert 40 <= seconds <= 46  # fitted to its window
    assert "fitted to 40-46 s" in result.output

    entries = CostLog(run.costs_path).read()
    assert [(e.stage, e.length) for e in entries] == [("select", "highlights")]
    costs = cli("costs")
    assert "highlights" in costs.output
    status = cli("status")
    assert "Lengths: highlights (anchor), short" in status.output
    for length in ("highlights", "short"):
        assert any(line.startswith(f"│ {length}") and "mix" in line and "complete" in line
                   for line in status.output.splitlines())  # fmt: skip

    again = cli("run", "--run", "latest", "--length", "highlights,short", "--yes")
    assert again.exit_code == 0
    assert not [line for line in again.output.splitlines() if line.startswith("▶ ")]

    # A length chosen later sees the run's quotes; its selection is its own call.
    install(response({"clips": [], "suggested_order": [], "notes": ""}))
    dry = cli("select", "--length", "summary", "--dry-run")
    assert dry.exit_code == 0 and "select up to 10 clips" in dry.output


def test_one_length_at_a_time_and_clips_per_length(
    cli: Cli, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, install: Callable,
    length_song: Path,
) -> None:  # fmt: skip
    run, transcript = _new_run(cli, tmp_path, monkeypatch)
    assert cli("mix", "--length", "short,highlights").exit_code == 1  # one length
    bad = cli("select", "--length", "long")
    assert bad.exit_code == 1 and "use short, highlights, summary" in bad.output
    both = cli("select", "--length", "short,highlights", "--clips", "2")
    assert both.exit_code == 1 and "--clips needs a single --length" in both.output
    picks = {"c1": (1, 1), "c2": (8, 8)}
    install(response(_versions(transcript, picks, {"short": ["c1", "c2"]})))
    result = cli("select", "--length", "short", "--clips", "2", "--yes")
    assert result.exit_code == 0, result.output
    run = Run.open(tmp_path / "runs")
    assert run.song("short").options.clips == 2 and run.manifest.length == "short"
    assert not run.song("summary").path("03_selection.json").exists()


# --- Conditioning a length's music on the anchor's take ---------------------------------------


def test_anchor_range_takes_the_matching_chunk() -> None:
    spans = [(0, 10_000), (50_000, 100_000)]
    assert anchor_range(spans, 1, 2) == (0, 10_000)
    start, end = anchor_range(spans, 1, 1)  # a single chunk takes the anchor's last
    assert end - start == MAX_REFERENCE_MS and start == 60_000  # its middle 30 s
    assert anchor_range([(0, 5_000)], 3, 3) == (0, 5_000)


def test_other_lengths_are_conditioned_on_the_anchors_take(
    cli: Cli, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, install: Callable,
    length_song: Path,
) -> None:  # fmt: skip
    run, transcript = _new_run(cli, tmp_path, monkeypatch)
    config = Path("config.yaml")
    config.write_text(config.read_text().replace("music_takes: 1", "music_takes: 2"))
    monkeypatch.setenv("ELEVENLABS_API_KEY", "test-key")
    fake = FakeElevenLabs()
    monkeypatch.setattr(music_elevenlabs, "client_factory", lambda key: fake)
    install(response(_versions(transcript, *HIGHLIGHTS_SHORT)))
    result = cli("run", "--run", "latest", "--length", "highlights,short",
                 "--music-backend", "elevenlabs", "--yes")  # fmt: skip
    assert result.exit_code == 0, result.output
    assert len(fake.compose_calls) == 4 and fake.upload_calls == []  # nothing is uploaded
    for call in fake.compose_calls[:2]:  # the anchor: no conditioning
        assert all("conditioning_ref" not in c for c in call["composition_plan"]["chunks"])
    run = Run.open(tmp_path / "runs")
    highlights, short = run.song("highlights"), run.song("short")
    chosen = TakeChoice.model_validate_json(highlights.path("06_music/analysis.json").read_text())
    meta = TakeMeta.model_validate_json(
        highlights.path(f"06_music/take_{chosen.chosen:03d}.meta.json").read_text())  # fmt: skip
    plan = fake.compose_calls[2]["composition_plan"]["chunks"]
    conditioned = [c for c in plan if "conditioning_ref" in c]
    assert len(conditioned) == len(plan) - 1  # every chunk but the tail past the end
    for chunk in conditioned:
        ref = chunk["conditioning_ref"]
        assert ref["song_id"] == meta.song_id and chunk["condition_strength"] == "low"
        assert 0 < ref["range"]["end_ms"] - ref["range"]["start_ms"] <= MAX_REFERENCE_MS
    assert f"conditioned on take {meta.take} of highlights" in result.output
    assert short.options.anchor_take == meta.take
    mine = TakeMeta.model_validate_json(short.path("06_music/take_001.meta.json").read_text())
    assert mine.params["anchor"] == {"length": "highlights", "take": meta.take,
                                     "song_id": meta.song_id}  # fmt: skip

    other = 3 - chosen.chosen  # choosing another anchor take keeps the short's music
    assert cli("mix", "--length", "highlights", "--take", str(other)).exit_code == 0
    again = cli("generate", "--length", "short", "--yes")
    assert again.exit_code == 0 and "generate: cached" in again.output
    assert len(fake.compose_calls) == 4


# --- A run made before M8 keeps its caches -----------------------------------------------------


def _as_schema_1(root: Path) -> None:
    """Put a run back the way runs were before M8: the summary's files and records at
    the run root."""
    data = json.loads((root / "manifest.json").read_text())
    song = data.pop("songs")["summary"]
    data["stages"].update(song["stages"])
    data["options"].update({k: v for k, v in song["options"].items() if k in ("clips", "take")})
    for key in ("length", "shared"):
        data.pop(key, None)
    data["schema_version"] = 1
    for item in (root / "summary").iterdir():
        shutil.move(str(item), str(root / item.name))
    (root / "summary").rmdir()
    (root / "manifest.json").write_text(json.dumps(data))


def test_a_run_made_before_lengths_keeps_its_caches(
    cli: Cli, talk: Talk, install: Callable, short_song: Path
) -> None:
    from .fixtures.fake_claude import clip_answer

    run = talk[0]
    install(response(clip_answer(talk[1], [(1, 1), (8, 8)])))
    for args in (("select", "--yes"), ("melody",), ("arrange",), ("generate",), ("mix",)):
        assert cli(*args).exit_code == 0
    _as_schema_1(run.root)
    status = cli("status")
    assert status.exit_code == 0, status.output
    assert "its song is now the run's summary length" in status.output
    rows = [line for line in status.output.splitlines() if line.startswith("│ summary")]
    assert len(rows) == 7 and all("complete" in row for row in rows)  # no arc: 7 stages
    reopened = Run.open(run.root.parent, run.id)
    assert reopened.manifest.shared is not None and reopened.manifest.shared.anchor == "summary"
    assert reopened.song("summary").path("07_mix/master.wav").exists()
