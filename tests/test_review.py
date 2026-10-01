"""The interactive review state machine and loop."""

import io

import pytest
from rich.console import Console

from speech2song.models import Clip, ClipSet
from speech2song.review import ReviewState, run_review


def _state() -> ReviewState:
    return ReviewState(ids=["c1", "c2", "c3", "c4"], order=["c3", "c1", "c4", "c2"])


def test_drop_keep_move_order() -> None:
    state = _state()
    assert state.apply("drop c1") == ("continue", "dropped c1")
    assert state.order == ["c3", "c4", "c2"] and state.dropped == ["c1"]
    state.apply("move c2 1")
    assert state.order == ["c2", "c3", "c4"]
    state.apply("keep c1")
    assert state.order == ["c2", "c3", "c4", "c1"] and state.dropped == []
    state.apply("order c1 c2 c3 c4")
    assert state.order == ["c1", "c2", "c3", "c4"]
    assert state.apply("save")[0] == "save"
    assert state.apply("quit")[0] == "quit"


@pytest.mark.parametrize(
    ("command", "message"),
    [
        ("drop c9", "no clip 'c9'"),
        ("keep c1", "not dropped"),
        ("move c1 9", "position must be 1-4"),
        ("order c1 c2", "exactly once"),
        ("dance", "unknown command"),
    ],
)
def test_bad_commands_explain_themselves(command: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _state().apply(command)


def test_cannot_save_with_nothing_kept() -> None:
    state = ReviewState(ids=["c1"], order=["c1"])
    state.apply("drop c1")
    with pytest.raises(ValueError, match="at least one"):
        state.apply("save")


def _clip_set() -> ClipSet:
    clips = [
        Clip(id=f"c{i}", file=f"clips/clip_00{i}.wav", start_sentence=i, end_sentence=i,
             text=f"line {i}", score=0.5, role="hook", reason="r", nominal_start_s=i,
             nominal_end_s=i + 3, start_s=i, end_s=i + 3, start_sample=0, end_sample=1,
             duration_s=3.0, cut_level_db=(-80.0, -80.0))
        for i in (1, 2, 3)
    ]  # fmt: skip
    return ClipSet(source="01_clean.wav", isolated=False, sample_rate=44100, channels=1,
                   fade_ms=15, clips=clips, order=["c2", "c1", "c3"])  # fmt: skip


def test_review_loop_saves_after_errors_and_previews() -> None:
    answers = iter(["drop c7", "play c2", "drop c3", "move c1 1", "save"])
    previews: list[str] = []
    console = Console(file=io.StringIO(), width=120)
    state = run_review(
        console, _clip_set(), lambda: next(answers),
        lambda clip_id: previews.append(clip_id) or f"preview {clip_id}",
    )  # fmt: skip
    assert state is not None
    assert state.order == ["c1", "c2"] and state.dropped == ["c3"]
    assert previews == ["c2"]
    output = console.file.getvalue()  # type: ignore[attr-defined]
    assert "no clip 'c7'" in output and "preview c2" in output


def test_review_loop_quit_returns_none() -> None:
    answers = iter(["drop c1", "quit"])
    console = Console(file=io.StringIO())
    assert run_review(console, _clip_set(), lambda: next(answers), lambda c: "") is None
