"""Sentence splitting for official transcripts and timed ASR words."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from speech2song.text.normalize import has_word_chars

_CLOSERS = "\"'\u201d\u2019)]}\u00bb"
_OPENERS = "\"'\u201c\u2018([{\u00ab"
# A period after these never ends a sentence ("Dr. Smith", "St. George", "e.g. this").
_ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "st", "jr", "sr", "prof", "rev", "fr", "gen", "sen", "rep",
    "gov", "lt", "col", "capt", "sgt", "hon", "vs", "vol", "pp", "cf", "approx", "fig",
    "e.g", "i.e", "u.s", "mt", "ft",
}  # fmt: skip
_TERMINAL = (".", "!", "?", ":", ";")


def _first_letter(word: str) -> str:
    return next((ch for ch in word.lstrip(_OPENERS) if ch.isalpha()), "")


def ends_sentence(word: str, next_word: str | None) -> bool:
    """Does a sentence end after `word`? `next_word` is None at a line or text end."""
    core = word.rstrip(_CLOSERS)
    if core.endswith(("?", "!")):
        return True
    if core.endswith(("...", "…")):
        return next_word is None or _first_letter(next_word).isupper()
    if not core.endswith("."):
        return False
    stem = core[:-1].lstrip(_OPENERS)
    if len(stem) == 1 and stem.isalpha():
        return False  # an initial, as in "Russell M. Nelson"
    if stem.casefold() in _ABBREVIATIONS:
        return False
    return next_word is None or not _first_letter(next_word).islower()


@dataclass(frozen=True)
class OfficialWord:
    text: str
    line: int  # 1-based line number in the transcript


@dataclass
class OfficialText:
    words: list[OfficialWord]
    sentences: list[tuple[int, int]]  # [start, end) word ranges
    hard_wrapped: bool


def looks_hard_wrapped(lines: Sequence[str]) -> bool:
    """True for text wrapped at a fixed width (e.g. copied from a PDF): most lines end
    mid-sentence. Then single line breaks are not boundaries, only blank lines are."""
    content = [line.strip() for line in lines if line.strip()]
    if len(content) < 5:
        return False
    open_ended = sum(1 for line in content if not line.rstrip(_CLOSERS).endswith(_TERMINAL))
    return open_ended / len(content) >= 0.6


def parse_official(text: str) -> OfficialText:
    """Split an official transcript into words (with line numbers) and sentences.

    Line breaks end sentences (paragraphs, verse lines, captions) unless the text looks
    hard-wrapped. Punctuation-only tokens ("—") attach to a neighbouring word.
    """
    lines = text.lstrip("﻿").replace("\r\n", "\n").replace("\r", "\n").split("\n")
    wrapped = looks_hard_wrapped(lines)
    words: list[OfficialWord] = []
    sentences: list[tuple[int, int]] = []
    start = 0

    def close() -> None:
        nonlocal start
        if start < len(words):
            sentences.append((start, len(words)))
            start = len(words)

    def next_token(line_index: int, token_index: int) -> str | None:
        tokens = lines[line_index].split()
        if token_index + 1 < len(tokens):
            return tokens[token_index + 1]
        if wrapped and line_index + 1 < len(lines) and lines[line_index + 1].split():
            return lines[line_index + 1].split()[0]
        return None

    for line_index, line in enumerate(lines):
        tokens = line.split()
        if not tokens or not wrapped:
            close()
        pending = ""
        for token_index, token in enumerate(tokens):
            if not has_word_chars(token):
                if start < len(words) and words[-1].line == line_index + 1:
                    words[-1] = OfficialWord(f"{words[-1].text} {token}", line_index + 1)
                else:
                    pending = f"{pending}{token} "
                continue
            words.append(OfficialWord(pending + token, line_index + 1))
            pending = ""
            if ends_sentence(token, next_token(line_index, token_index)):
                close()
    close()
    return OfficialText(words, sentences, wrapped)


class TimedWord(Protocol):
    @property
    def w(self) -> str: ...
    @property
    def start(self) -> float: ...
    @property
    def end(self) -> float: ...


def split_timed(
    words: Sequence[TimedWord], *, pause_s: float, max_s: float
) -> list[tuple[int, int]]:
    """Sentences for ASR words: punctuation, long pauses, and a maximum duration."""
    ranges: list[tuple[int, int]] = []
    start = 0
    for i, word in enumerate(words):
        nxt = words[i + 1] if i + 1 < len(words) else None
        if nxt is None or ends_sentence(word.w, nxt.w) or nxt.start - word.end >= pause_s:
            ranges.append((start, i + 1))
            start = i + 1
    result: list[tuple[int, int]] = []
    for lo, hi in ranges:
        result.extend(_split_long(words, lo, hi, max_s))
    return result


def _split_long(
    words: Sequence[TimedWord], lo: int, hi: int, max_s: float
) -> list[tuple[int, int]]:
    if hi - lo < 2 or words[hi - 1].end - words[lo].start <= max_s:
        return [(lo, hi)]
    _gap, cut = max((words[i].start - words[i - 1].end, i) for i in range(lo + 1, hi))
    return _split_long(words, lo, cut, max_s) + _split_long(words, cut, hi, max_s)
