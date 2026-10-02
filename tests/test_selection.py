"""Clip selection logic: prompt, validation and clean-up of Claude's answer."""

import json

import pytest

from speech2song.config import load_preset
from speech2song.errors import StageError
from speech2song.models import ClipSelection, ClipTargets, RequiredQuote, RunOptions, SelectedClip
from speech2song.selection import (
    SELECTION_SCHEMA,
    build_prompt,
    clip_targets,
    estimate_input_tokens,
    estimate_input_tokens_from_duration,
    estimate_output_tokens,
    feedback_prompt,
    finalize,
    needs_retry,
    prompt_digest,
    text_matches,
    validate,
)

from .conftest import PRESETS_DIR
from .fixtures.synth import transcript_from

TARGETS = ClipTargets(count=3, min_seconds=3, max_seconds=15, total_speech_seconds=30)


def _transcript():
    """Six sentences of 4 s each, separated by 1 s pauses (IDs 1-6)."""
    sentences = []
    t = 1.0
    for s in range(6):
        words = [(f"s{s + 1}w{k}", t + k, t + k + 0.8) for k in range(4)]
        words[-1] = (words[-1][0] + ".", words[-1][1], words[-1][2])
        sentences.append(words)
        t += 5.0
    return transcript_from(sentences)


TRANSCRIPT = _transcript()


def _clip(cid: str, start: int, end: int, score: float = 0.8, text: str | None = None):
    if text is None:
        text = " ".join(s.text for s in TRANSCRIPT.sentences[start - 1 : end])
    return SelectedClip(id=cid, start_sentence=start, end_sentence=end, text=text, score=score,
                        role="hook", reason="because")  # fmt: skip


def _selection(*clips: SelectedClip, order: list[str] | None = None) -> ClipSelection:
    return ClipSelection(
        clips=list(clips), suggested_order=order or [c.id for c in clips], notes="n"
    )


def test_targets_come_from_the_preset_with_cli_override() -> None:
    preset = load_preset("cinematic_future_bass", PRESETS_DIR)
    default = clip_targets(preset, RunOptions())
    assert (default.count, default.count_range, default.longest) == (8, (6, 10), 40)
    exact = clip_targets(preset, RunOptions(clips=5))
    assert (exact.count, exact.count_range) == (5, (5, 5))  # --clips N means exactly N


def test_prompt_lists_sentences_and_targets() -> None:
    preset = load_preset("cinematic_future_bass", PRESETS_DIR)
    system, user = build_prompt(TRANSCRIPT, preset, TARGETS)
    assert "self-contained" in system
    assert "Choose exactly 3 clips" in user
    assert "between 3 and 15 seconds" in user
    assert "at most 30 seconds" in user
    assert "1 | 1.0-4.8 | 3.8s | s1w0 s1w1 s1w2 s1w3." in user
    assert "Meditative spoken word" in user
    assert "$" not in user
    assert len(prompt_digest()) == 16


def test_schema_is_strict() -> None:
    assert SELECTION_SCHEMA["additionalProperties"] is False
    clip_schema = SELECTION_SCHEMA["properties"]["clips"]["items"]
    assert set(clip_schema["required"]) == set(SelectedClip.model_fields)
    assert clip_schema["properties"]["role"]["enum"] == [
        "hook", "build", "payoff", "breakdown", "outro",
    ]  # fmt: skip


def test_text_matching_tolerates_punctuation_but_not_other_words() -> None:
    assert text_matches("S1W0, s1w1 s1w2 s1w3", "s1w0 s1w1 s1w2 s1w3.")
    assert not text_matches("s2w0 s2w1 s2w2 s2w3", "s1w0 s1w1 s1w2 s1w3.")


def test_valid_selection_has_no_problems() -> None:
    selection = _selection(_clip("a", 1, 1), _clip("b", 3, 4), _clip("c", 6, 6))
    assert validate(selection, TRANSCRIPT, TARGETS) == []


@pytest.mark.parametrize(
    ("clip", "kind", "fragment"),
    [
        (_clip("x", 7, 7, text="nothing"), "invalid", "not a valid range (IDs run 1-6)"),
        (_clip("x", 4, 3, text="backwards"), "invalid", "not a valid range"),
        (_clip("x", 2, 2, text="s5w0 s5w1 s5w2 s5w3."), "invalid", "which read"),
        (_clip("x", 2, 4), "invalid", "last 13.8 s"),  # fine: within 15 s
        (_clip("x", 1, 4), "invalid", "clips must be 3-15 s"),
    ],
)
def test_clip_problems(clip: SelectedClip, kind: str, fragment: str) -> None:
    problems = validate(_selection(clip), TRANSCRIPT, TARGETS)
    messages = [p.message for p in problems if p.kind == kind]
    if fragment == "last 13.8 s":
        assert messages == []
    else:
        assert any(fragment in m for m in messages), problems


def test_overlaps_budget_count_and_order_are_reported() -> None:
    nested = _selection(_clip("a", 1, 3), _clip("b", 2, 2), _clip("c", 3, 3), order=["a", "z"])
    kinds = {p.kind for p in validate(nested, TRANSCRIPT, ClipTargets(
        count=2, min_seconds=3, max_seconds=15, total_speech_seconds=10))}  # fmt: skip
    assert kinds == {"overlap", "budget", "count", "order"}


def test_retry_only_for_problems_claude_must_fix() -> None:
    short = validate(_selection(_clip("a", 1, 1)), TRANSCRIPT, TARGETS)  # 1 clip of 3
    assert {p.kind for p in short} == {"count"}
    assert not needs_retry(short)
    bad = validate(_selection(_clip("a", 1, 1, text="made up")), TRANSCRIPT, TARGETS)
    assert needs_retry(bad)


def test_finalize_cleans_up_and_keeps_claudes_ids() -> None:
    selection = _selection(
        _clip("hook", 5, 5, score=0.9),
        _clip("bad", 2, 2, text="hallucinated words here"),
        _clip("weak", 5, 6, score=0.3),  # overlaps "hook", lower score
        _clip("early", 1, 1, score=0.7),
        _clip("mid", 3, 3, score=1.4),  # score gets clamped
        order=["hook", "early", "bad", "mid", "weak"],
    )
    final, warnings = finalize(selection, TRANSCRIPT, TARGETS)
    assert [(c.id, c.start_sentence) for c in final.clips] == [
        ("early", 1), ("mid", 3), ("hook", 5),
    ]  # fmt: skip
    assert final.suggested_order == ["hook", "early", "mid"]
    assert final.clips[1].score == 1.0
    assert final.clips[0].text == TRANSCRIPT.sentences[0].text
    assert any("bad" in w for w in warnings)
    assert any("dropped weak: it overlaps hook" in w for w in warnings)


def test_malformed_ids_are_renumbered_in_time_order() -> None:
    selection = _selection(_clip("third one!", 5, 5), _clip("c9", 1, 1), order=["c9", "third one!"])
    final, warnings = finalize(selection, TRANSCRIPT, TARGETS)
    assert [c.id for c in final.clips] == ["c1", "c2"]
    assert final.suggested_order == ["c1", "c2"]
    assert any("renamed malformed clip IDs" in w for w in warnings)


def test_finalize_enforces_count_and_budget_by_score() -> None:
    clips = [_clip(f"k{i}", i, i, score=i / 10) for i in range(1, 7)]
    final, warnings = finalize(_selection(*clips), TRANSCRIPT, TARGETS)
    assert [c.start_sentence for c in final.clips] == [4, 5, 6]  # the three best
    tight = ClipTargets(count=6, min_seconds=3, max_seconds=15, total_speech_seconds=8)
    final, warnings = finalize(_selection(*clips), TRANSCRIPT, tight)
    assert [c.start_sentence for c in final.clips] == [5, 6]
    assert any("speech budget" in w for w in warnings)


def test_finalize_with_nothing_usable_fails() -> None:
    with pytest.raises(StageError, match="None of Claude's clips"):
        finalize(_selection(_clip("a", 9, 9, text="x")), TRANSCRIPT, TARGETS)


def test_feedback_prompt_lists_problems_and_previous_answer() -> None:
    selection = _selection(_clip("a", 1, 1, text="made up"))
    problems = validate(selection, TRANSCRIPT, TARGETS)
    prompt = feedback_prompt("ORIGINAL", selection, problems)
    assert prompt.startswith("ORIGINAL")
    assert "Your previous answer had these problems" in prompt
    assert (
        json.loads(prompt.split("Previous answer:\n")[1].split("\n\n")[0])["clips"][0]["id"] == "a"
    )


def test_estimates_are_generous_and_offline() -> None:
    assert estimate_input_tokens("x" * 1100, "y" * 1100) == 1000 + 600
    assert estimate_output_tokens(5) == 3000 + 5 * 250
    assert estimate_input_tokens_from_duration(600) > 600 * 2.6 * 1.6


# --- Count ranges, long quotes and required passages -----------------------------------


def _required(start: int, end: int, role: str | None = None) -> RequiredQuote:
    return RequiredQuote(id="q1", text="x", start_sentence=start, end_sentence=end,
                         similarity=1.0, role=role)  # fmt: skip


def test_long_quotes_and_count_ranges() -> None:
    targets = ClipTargets(count=3, min_count=2, max_count=4, min_seconds=3, max_seconds=5,
                          long_max_seconds=15, total_speech_seconds=40)  # fmt: skip
    ok = _selection(_clip("c1", 1, 1), _clip("c2", 2, 3))  # 4 s and 9 s
    assert validate(ok, TRANSCRIPT, targets) == []
    too_long = _selection(_clip("c1", 1, 4))  # 19 s
    problems = validate(too_long, TRANSCRIPT, targets)
    assert any("clips must be 3-15 s" in p.message for p in problems)
    assert any(p.kind == "count" and "2-4 were asked for" in p.message for p in problems)


def test_required_passages_are_asked_for_and_never_dropped() -> None:
    targets = ClipTargets(count=2, min_count=2, max_count=2, min_seconds=3, max_seconds=5,
                          total_speech_seconds=20, required=[_required(4, 6, "outro")])  # fmt: skip
    _, user = build_prompt(TRANSCRIPT, load_preset("cinematic_future_bass", PRESETS_DIR), targets)
    assert "sentences 4-6, role outro" in user and "Choose exactly 2 clips" in user
    answer = _selection(_clip("c1", 1, 1, 0.9), _clip("c2", 5, 5, 0.95), _clip("c3", 2, 2, 0.5))
    problems = validate(answer, TRANSCRIPT, targets)
    assert any(p.kind == "required" for p in problems) and needs_retry(problems)
    final, warnings = finalize(answer, TRANSCRIPT, targets)
    by_id = {c.id: c for c in final.clips}
    # The 14 s required passage (over the 5 s limit) is added; c2 overlaps it and goes;
    # the lowest score goes for the count.
    assert set(by_id) == {"c1", "q1"}
    assert (by_id["q1"].start_sentence, by_id["q1"].end_sentence, by_id["q1"].role) == (
        4, 6, "outro")  # fmt: skip
    assert any("added the required passage q1" in w for w in warnings)
    assert final.suggested_order == ["c1", "q1"]  # in talk order relative to the others


def test_a_required_passage_claude_chose_is_kept_as_is() -> None:
    targets = ClipTargets(count=1, min_seconds=3, max_seconds=5, total_speech_seconds=30,
                          required=[_required(2, 4)])  # fmt: skip
    answer = _selection(_clip("c7", 2, 4, 0.4))  # 14 s: allowed because it is required
    assert validate(answer, TRANSCRIPT, targets) == []
    final, _ = finalize(answer, TRANSCRIPT, targets)
    assert [c.id for c in final.clips] == ["c7"]
