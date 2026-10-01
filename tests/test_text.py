"""Normalization and sentence splitting."""

import pytest

from speech2song.models import AsrWord
from speech2song.text.normalize import normalize_word, tokenize
from speech2song.text.sentences import (
    ends_sentence,
    looks_hard_wrapped,
    parse_official,
    split_timed,
)

from .fixtures.synth import FIXTURES, fake_asr_words


@pytest.mark.parametrize(
    ("word", "tokens"),
    [
        ("Hello", ["hello"]),
        ("Kearon\u2019s", ["kearons"]),
        ("don't", ["dont"]),
        ("\u201cBecause", ["because"]),
        ("back.\u201d", ["back"]),
        ("happened\u2014over", ["happened", "over"]),
        ("eight-year-old", ["eight", "year", "old"]),
        ("and/or", ["and", "or"]),
        ("[we]", ["we"]),
        ("Savior.12", ["savior"]),
        ("1984,", ["1984"]),
        ("U.S.", ["us"]),
        ("Caf\u00e9", ["cafe"]),
        ("\u2014", []),
        ("...", []),
    ],
)
def test_normalize_word(word: str, tokens: list[str]) -> None:
    assert normalize_word(word) == tokens


def test_tokens_point_back_to_their_word() -> None:
    tokens = tokenize(["An", "eight-year-old", "boy"])
    assert [(t.text, t.word) for t in tokens] == [
        ("an", 0),
        ("eight", 1),
        ("year", 1),
        ("old", 1),
        ("boy", 2),
    ]


@pytest.mark.parametrize(
    ("word", "following", "expected"),
    [
        ("end.", None, True),
        ("end.", "Then", True),
        ("end.", "and", False),
        ("back.\u201d", "Next", True),
        ("why?", "and", True),
        ("wow!\u201d", None, True),
        ("taught:", "Because", False),
        ("home;", "Next", False),
        ("M.", "Nelson", False),
        ("(H.", "Oaks", False),
        ("Dr.", "Vale", False),
        ("St.", "George", False),
        ("e.g.", "This", False),
        ("U.S.", "Army", False),
        ("42.", "Then", True),
        ("wait...", "and", False),
        ("wait...", "Then", True),
        ("wait\u2026", None, True),
        ("calling\u2014", "Calling", False),
    ],
)
def test_ends_sentence(word: str, following: str | None, expected: bool) -> None:
    assert ends_sentence(word, following) is expected


def _sentences(text: str) -> list[str]:
    parsed = parse_official(text)
    return [" ".join(w.text for w in parsed.words[lo:hi]) for lo, hi in parsed.sentences]


def test_parse_official_mini_talk() -> None:
    parsed = parse_official((FIXTURES / "mini_talk.txt").read_text(encoding="utf-8"))
    sentences = _sentences((FIXTURES / "mini_talk.txt").read_text(encoding="utf-8"))
    assert not parsed.hard_wrapped
    assert (
        "We were strengthened by President Thaddeus Q. Brightwater\u2019s first lecture, "
        "given nearly 42 years ago." in sentences
    )
    assert "There is a light beyond the harbor" in sentences  # verse line
    assert "The Long Way Back" in sentences  # heading
    assert "Harbor at dawn, with fishing boats" in sentences  # caption
    assert (
        "As Dr. Imogen Vale taught: \u201cShould [we] wander, we will find [our] way back.\u201d"
        in sentences
    )
    last = ["Come home.", "Come home.", "The light is waiting for you and for me."]
    assert sentences[-3:] == last
    lines = {w.text: w.line for w in parsed.words}
    assert lines["Conclusion"] == 14


def test_punctuation_only_tokens_attach_to_a_word() -> None:
    parsed = parse_official("The bell is ringing \u2014\n\u2014 and calling.")
    assert [w.text for w in parsed.words] == ["The", "bell", "is", "ringing \u2014", "\u2014 and",
                                              "calling."]  # fmt: skip


def test_hard_wrapped_text_ignores_single_line_breaks() -> None:
    text = (
        "After our first daughter was born, we waited for\n"
        "years and then, one bright morning, the news\n"
        "arrived. We were grateful for the many people\n"
        "who prayed for us during those long and quiet\n"
        "years of waiting and hoping for a little one.\n"
        "\n"
        "Then everything changed for our small family\n"
        "in ways we did not expect at the time.\n"
    )
    assert looks_hard_wrapped(text.splitlines())
    assert _sentences(text) == [
        "After our first daughter was born, we waited for years and then, one bright "
        "morning, the news arrived.",
        "We were grateful for the many people who prayed for us during those long and quiet "
        "years of waiting and hoping for a little one.",
        "Then everything changed for our small family in ways we did not expect at the time.",
    ]


def test_split_timed_on_punctuation_and_initials() -> None:
    words = fake_asr_words("As Russell M. Nelson taught. We listened.")
    ranges = split_timed(words, pause_s=1.5, max_s=30)
    assert [" ".join(w.w for w in words[lo:hi]) for lo, hi in ranges] == [
        "As Russell M. Nelson taught.",
        "We listened.",
    ]


def test_split_timed_on_long_pause() -> None:
    words = [
        AsrWord(w="and", start=0.0, end=0.2, conf=1),
        AsrWord(w="then", start=0.25, end=0.5, conf=1),
        AsrWord(w="silence", start=2.5, end=3.0, conf=1),
    ]
    assert split_timed(words, pause_s=1.5, max_s=30) == [(0, 2), (2, 3)]


def test_split_timed_breaks_overlong_sentences_at_the_largest_gap() -> None:
    words = fake_asr_words(" ".join(["word"] * 120), word_s=0.3, gap_s=0.05)
    words[60] = words[60].model_copy(update={"start": words[60].start + 0.5})
    ranges = split_timed(words, pause_s=1.5, max_s=30)
    assert ranges[0] == (0, 60)
    assert all(words[hi - 1].end - words[lo].start <= 30 for lo, hi in ranges)
    assert ranges[-1][1] == len(words)
