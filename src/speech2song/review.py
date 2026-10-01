"""Interactive clip review: drop, restore, reorder and audition clips before arranging."""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

from rich.console import Console
from rich.markup import escape
from rich.table import Table

from speech2song.models import ClipSet

HELP = """Commands (clip IDs like c3):
  list                 show the clips and the current order
  play ID              write a preview with a second of context (and play it if possible)
  drop ID              remove a clip from the song
  keep ID              bring a dropped clip back (added at the end)
  move ID POS          move a clip to position POS (1 = first)
  order ID ID ...      set the whole order (dropped clips stay dropped)
  save                 save these choices and finish
  quit                 finish without saving"""

Action = Literal["continue", "save", "quit"]


@dataclass
class ReviewState:
    ids: list[str]  # every clip, in time order
    order: list[str]  # kept clips, in playback order
    dropped: list[str] = field(default_factory=list)

    def _known(self, clip_id: str) -> str:
        if clip_id not in self.ids:
            raise ValueError(f"no clip {clip_id!r}; clips are {', '.join(self.ids)}")
        return clip_id

    def apply(self, line: str) -> tuple[Action, str]:
        """Run one command. Raises ValueError with a message for bad input."""
        words = line.split()
        if not words:
            return "continue", ""
        command, args = words[0].lower(), words[1:]
        if command in ("save", "s"):
            if not self.order:
                raise ValueError("keep at least one clip")
            return "save", "saved"
        if command in ("quit", "q"):
            return "quit", "not saved"
        if command in ("list", "l", "help", "?"):
            return "continue", HELP if command in ("help", "?") else ""
        if command == "drop" and len(args) == 1:
            clip = self._known(args[0])
            if clip in self.dropped:
                raise ValueError(f"{clip} is already dropped")
            self.order.remove(clip)
            self.dropped.append(clip)
            return "continue", f"dropped {clip}"
        if command == "keep" and len(args) == 1:
            clip = self._known(args[0])
            if clip not in self.dropped:
                raise ValueError(f"{clip} is not dropped")
            self.dropped.remove(clip)
            self.order.append(clip)
            return "continue", f"kept {clip} (now last)"
        if command == "move" and len(args) == 2:
            clip = self._known(args[0])
            if clip not in self.order:
                raise ValueError(f"{clip} is dropped; keep it first")
            if not args[1].isdigit() or not 1 <= int(args[1]) <= len(self.order):
                raise ValueError(f"position must be 1-{len(self.order)}")
            self.order.remove(clip)
            self.order.insert(int(args[1]) - 1, clip)
            return "continue", f"moved {clip} to {args[1]}"
        if command == "order" and args:
            wanted = [self._known(a) for a in args]
            if sorted(wanted) != sorted(self.order):
                raise ValueError(f"list each kept clip exactly once: {', '.join(self.order)}")
            self.order = wanted
            return "continue", "order set"
        raise ValueError(f"unknown command {line!r}; type help")


def show(console: Console, clip_set: ClipSet, state: ReviewState) -> None:
    table = Table()
    for column in ("#", "Clip", "Time", "Role", "Score", "Text"):
        table.add_column(column)
    clips = {c.id: c for c in clip_set.clips}
    rows = [(str(n), clips[i]) for n, i in enumerate(state.order, start=1)]
    rows += [("dropped", clips[i]) for i in state.dropped]
    for position, clip in rows:
        text = clip.text if len(clip.text) <= 80 else clip.text[:79] + "..."
        table.add_row(
            position, clip.id, f"{clip.start_s:.1f}-{clip.end_s:.1f} s ({clip.duration_s:.1f})",
            clip.role, f"{clip.score:.2f}", escape(text),
        )  # fmt: skip
    console.print(table)


def run_review(
    console: Console,
    clip_set: ClipSet,
    ask: Callable[[], str],
    preview: Callable[[str], str],
) -> ReviewState | None:
    """Loop until save (returns the state) or quit (returns None)."""
    state = ReviewState(
        ids=[c.id for c in clip_set.clips],
        order=list(clip_set.order),
        dropped=list(clip_set.dropped),
    )
    console.print(HELP)
    show(console, clip_set, state)
    while True:
        line = ask()
        words = line.split()
        if words and words[0].lower() in ("play", "p") and len(words) == 2:
            try:
                console.print(preview(state._known(words[1])))
            except ValueError as exc:
                console.print(f"[red]{escape(str(exc))}[/]")
            continue
        try:
            action, message = state.apply(line)
        except ValueError as exc:
            console.print(f"[red]{escape(str(exc))}[/]")
            continue
        if message:
            console.print(message)
        if action == "save":
            return state
        if action == "quit":
            return None
        if words and words[0].lower() not in ("help", "?"):
            show(console, clip_set, state)
