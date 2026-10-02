"""Required quotes (--quotes): reading the file and finding each quote in the transcript;
and where long quotes are split into parts."""

from pathlib import Path

import pytest

from speech2song.errors import S2SError
from speech2song.models import Word
from speech2song.text.parts import plan_parts
from speech2song.text.quotes import QuoteSpec, find_quote, match_quotes, parse_quotes

from .fixtures.synth import transcript_from

SENTENCES = [
    "We all struggle at times.",
    "The Savior loves all of us, and is tenderly calling for you and for me to come home.",
    "Come home.",
    "In the end, everyone must make their own choice.",
]


def _transcript():
    rows, t = [], 0.0
    for sentence in SENTENCES:
        words = []
        for word in sentence.split():
            words.append((word, t, t + 0.3))
            t += 0.4
        rows.append(words)
        t += 1.0
    return transcript_from(rows)


TRANSCRIPT = _transcript()


def test_txt_quotes_are_lines_or_paragraphs(tmp_path: Path) -> None:
    lines = tmp_path / "q.txt"
    lines.write_text('Come home.\n- "In the end, everyone must make their own choice."\n\n')
    assert [q.text for q in parse_quotes(lines)] == [
        "Come home.", "In the end, everyone must make their own choice."]  # fmt: skip
    wrapped = tmp_path / "w.txt"
    wrapped.write_text("The Savior loves all of us,\nand is tenderly calling\n\nCome home.\n")
    assert [q.text for q in parse_quotes(wrapped)] == [
        "The Savior loves all of us, and is tenderly calling", "Come home."]  # fmt: skip


def test_yaml_quotes_can_give_a_role(tmp_path: Path) -> None:
    path = tmp_path / "q.yaml"
    path.write_text("quotes:\n  - text: Come home.\n    occurrence: last\n"
                    "  - text: We all struggle at times.\n    role: hook\n"
                    "  - Plain text.\n")  # fmt: skip
    assert parse_quotes(path) == [QuoteSpec("Come home.", None, -1),
                                  QuoteSpec("We all struggle at times.", "hook"),
                                  QuoteSpec("Plain text.")]  # fmt: skip
    path.write_text("- text: x\n  occurrence: 0\n")
    with pytest.raises(S2SError, match="occurrence must be"):
        parse_quotes(path)
    path.write_text("- text: x\n  role: chorus\n")
    with pytest.raises(S2SError, match="role must be one of"):
        parse_quotes(path)
    path.write_text("just: a mapping\n")
    with pytest.raises(S2SError, match="must be a list"):
        parse_quotes(path)
    (tmp_path / "empty.txt").write_text("\n\n")
    with pytest.raises(S2SError, match="has no quotes"):
        parse_quotes(tmp_path / "empty.txt")


def test_quotes_are_found_despite_small_differences() -> None:
    text = "the savior loves all of us and is tenderly calling for you and me to come home"
    first, last, score, occurrences = find_quote(TRANSCRIPT, text)
    words = [w.w for w in TRANSCRIPT.words[first:last]]
    assert words[0] == "The" and words[-1] == "home." and score > 0.9 and occurrences == 1
    quotes = [QuoteSpec("Come home.", "outro"), QuoteSpec("everyone must make their own choice")]
    matched = match_quotes(TRANSCRIPT, quotes)
    # "Come home." is its own sentence (3), not the end of sentence 2; a fragment of
    # sentence 4 takes the whole sentence.
    assert [(q.id, q.start_sentence, q.end_sentence, q.role) for q in matched] == [
        ("q1", 3, 3, "outro"), ("q2", 4, 4, None)]  # fmt: skip


def test_a_repeated_quote_uses_the_first_and_says_so() -> None:
    twice = transcript_from([[("Come", 0.0, 0.3), ("home.", 0.4, 0.7)],
                             [("Stay", 2.0, 2.3), ("here.", 2.4, 2.7)],
                             [("Come", 4.0, 4.3), ("home.", 4.4, 4.7)]])  # fmt: skip
    (quote,) = match_quotes(twice, [QuoteSpec("come home")])
    assert (quote.start_sentence, quote.occurrences, quote.similarity) == (1, 2, 1.0)
    last, second = match_quotes(twice, [QuoteSpec("come home", occurrence=-1)]), match_quotes(
        twice, [QuoteSpec("come home", occurrence=2)])  # fmt: skip
    assert [q.start_sentence for q in (*last, *second)] == [3, 3]
    with pytest.raises(S2SError, match="asks for occurrence 3"):
        match_quotes(twice, [QuoteSpec("come home", occurrence=3)])


def test_quotes_that_are_missing_or_overlap_are_refused() -> None:
    with pytest.raises(S2SError, match="Quote 1 was not found"):
        match_quotes(TRANSCRIPT, [QuoteSpec("Ask not what your country can do for you")])
    with pytest.raises(S2SError, match="share sentences"):
        match_quotes(TRANSCRIPT, [QuoteSpec("The Savior loves all of us"),
                                  QuoteSpec("tenderly calling for you and for me")])  # fmt: skip


def _words(spec: list[tuple[str, float]]) -> list[Word]:
    """Words of 0.4 s; each entry's number is the pause before the next word."""
    words, t = [], 0.0
    for text, pause in spec:
        words.append(Word(w=text, start=t, end=t + 0.4))
        t += 0.4 + pause
    return words


def test_long_quotes_split_at_sentences_then_clauses_then_pauses() -> None:
    short = _words([("a", 0.1)] * 10)
    assert plan_parts(short, set(), 15.0) == [(0, 10)]
    # 30 words (about 15 s): a sentence ends after word 14, a comma after word 9.
    spec = [(f"w{i}", 0.1) for i in range(30)]
    spec[9] = ("w9,", 0.1)
    spec[14] = ("w14.", 0.3)
    words = _words(spec)
    assert plan_parts(words, {14}, 9.0) == [(0, 15), (15, 30)]
    assert plan_parts(words, set(), 9.0) == [(0, 15), (15, 30)]  # the longest pause wins
    three = plan_parts(words, {14}, 6.0)
    assert len(three) == 3 and all(words[b - 1].end - words[a].start <= 6.0 for a, b in three)
    assert (0, 10) in three  # the comma: a clause boundary beats a plain word gap
