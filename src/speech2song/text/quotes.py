"""Quotes the user wants in the song (`--quotes FILE`): reading the file and finding each
quote in the transcript. Pure apart from reading the file.

A .txt file holds one quote per line, or, when it has blank lines, one quote per
paragraph (so wrapped text works). A .yaml file is a list of quotes, each a string or a
mapping with `text`, an optional `role`, and an optional `occurrence` (1, 2, ... or
"last") for a quote the talk says more than once; a mapping with a `quotes` list also
works. Without `occurrence`, the first time the talk says it is used.
"""

import itertools
import re
from dataclasses import dataclass
from pathlib import Path

import yaml
from rapidfuzz.distance import Indel

from speech2song.errors import S2SError
from speech2song.models import RequiredQuote, Transcript
from speech2song.text.normalize import normalize_word, tokenize

ROLES = ("hook", "build", "payoff", "breakdown", "outro")
MATCH_MIN = 0.8  # token similarity of a quote to the transcript words it is matched to
LENGTH_SLACK = 0.2  # matched spans may be this much shorter or longer than the quote
_BULLET = re.compile("^\\s*(?:[-*\u2022]|\\d+[.)])\\s+")
_QUOTE_MARKS = "\"'\u201c\u201d\u2018\u2019"


@dataclass(frozen=True)
class QuoteSpec:
    text: str
    role: str | None = None
    occurrence: int | None = None  # 1-based; -1 is the last; None is the first


def _clean(text: str) -> str:
    return " ".join(_BULLET.sub("", text).split()).strip(_QUOTE_MARKS + " ")


def parse_quotes(path: Path) -> list[QuoteSpec]:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise S2SError(f"Can't read the quotes file {path}: {exc}") from exc
    if path.suffix.lower() in (".yaml", ".yml"):
        quotes = _yaml_quotes(raw, path)
    elif re.search(r"\S[ \t]*\n[ \t]*\n\s*\S", raw):  # paragraphs
        quotes = [QuoteSpec(_clean(block)) for block in re.split(r"\n\s*\n", raw)]
    else:
        quotes = [QuoteSpec(_clean(line)) for line in raw.splitlines()]
    quotes = [q for q in quotes if q.text]
    if not quotes:
        raise S2SError(f"The quotes file {path} has no quotes.")
    return quotes


def _yaml_quotes(raw: str, path: Path) -> list[QuoteSpec]:
    try:
        data = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise S2SError(f"The quotes file {path} is not valid YAML: {exc}") from exc
    if isinstance(data, dict):
        data = data.get("quotes")
    if not isinstance(data, list):
        raise S2SError(f"The quotes file {path} must be a list of quotes (or have a `quotes` "
                       "list).")  # fmt: skip
    quotes = []
    for number, item in enumerate(data, start=1):
        if isinstance(item, str):
            quotes.append(QuoteSpec(_clean(item)))
        elif isinstance(item, dict) and isinstance(item.get("text"), str):
            role = item.get("role")
            if role is not None and role not in ROLES:
                raise S2SError(f"Quote {number} in {path}: role must be one of "
                               f"{', '.join(ROLES)}")  # fmt: skip
            occurrence = item.get("occurrence")
            if occurrence == "last":
                occurrence = -1
            elif occurrence is not None and (not isinstance(occurrence, int) or occurrence < 1):
                raise S2SError(f"Quote {number} in {path}: occurrence must be 1, 2, ... or "
                               "\"last\"")  # fmt: skip
            quotes.append(QuoteSpec(_clean(item["text"]), role, occurrence))
        else:
            raise S2SError(f"Quote {number} in {path} must be text, or a mapping with `text`.")
    return quotes


SENTENCE_EDGE_BONUS = 0.01  # breaks ties toward spans that start or end with a sentence


def _spans(transcript: Transcript, text: str) -> list[tuple[float, int, int]]:
    """Candidate (score, first token, end token) spans for `text`, best first."""
    quote = [t for word in text.split() for t in normalize_word(word)]
    tokens = tokenize([w.w for w in transcript.words])
    words = [t.text for t in tokens]
    if not quote or not words:
        return []
    starts = {s.word_start for s in transcript.sentences}
    ends = {s.word_end for s in transcript.sentences}
    n = len(quote)
    slack = max(1, round(n * LENGTH_SLACK))
    found = []
    for start in range(len(words)):
        if words[start] != quote[0] and Indel.normalized_similarity(
            quote[:3], words[start : start + 3]
        ) < 0.5:  # fmt: skip
            continue  # a quote's span starts close to its first words
        for length in range(max(1, n - slack), n + slack + 1):
            end = start + length
            if end > len(words):
                break
            score = Indel.normalized_similarity(quote, words[start:end])
            first, last = tokens[start].word, tokens[end - 1].word + 1
            bonus = SENTENCE_EDGE_BONUS * ((first in starts) + (last in ends))
            found.append((score + bonus, start, end))
    return sorted(found, key=lambda span: -span[0])


def find_quote(
    transcript: Transcript, text: str, occurrence: int | None = None
) -> tuple[int, int, float, int]:
    """(first word, last word exclusive, similarity, occurrences) of the transcript span
    that best matches `text`, comparing normalized words. Spans that line up with
    sentences win ties; `occurrences` counts separate spans that match as well, and
    `occurrence` picks one of them (1-based, -1 for the last; default the first)."""
    spans = _spans(transcript, text)
    if not spans:
        return 0, 0, 0.0, 0
    tokens = tokenize([w.w for w in transcript.words])
    top = spans[0][0]
    equal = sorted((s for s in spans if s[0] >= top - 1e-9), key=lambda span: span[1])
    distinct: list[tuple[float, int, int]] = []
    for span in equal:
        if not distinct or span[1] >= distinct[-1][2]:
            distinct.append(span)
    pick = 0 if occurrence is None else (occurrence - 1 if occurrence > 0 else occurrence)
    if not -len(distinct) <= pick < len(distinct):
        return 0, 0, 0.0, len(distinct)
    _, start, end = distinct[pick]
    quote = [t for word in text.split() for t in normalize_word(word)]
    similarity = Indel.normalized_similarity(quote, [t.text for t in tokens[start:end]])
    return tokens[start].word, tokens[end - 1].word + 1, similarity, len(distinct)


def match_quotes(transcript: Transcript, quotes: list[QuoteSpec]) -> list[RequiredQuote]:
    """Each quote as whole transcript sentences. Raises when a quote can't be found or two
    quotes cover the same sentences."""
    matched: list[RequiredQuote] = []
    for number, quote in enumerate(quotes, start=1):
        first, last, score, occurrences = find_quote(transcript, quote.text, quote.occurrence)
        if quote.occurrence is not None and quote.occurrence > occurrences:
            raise S2SError(f'Quote {number} asks for occurrence {quote.occurrence}, but the '
                           f'talk says it {occurrences} time(s): "{quote.text}"')  # fmt: skip
        if score < MATCH_MIN:
            raise S2SError(f'Quote {number} was not found in the transcript (best match '
                           f'{score:.0%}): "{quote.text}"')  # fmt: skip
        inside = [s for s in transcript.sentences if s.word_start < last and s.word_end > first]
        if not inside:
            raise S2SError(f'Quote {number} matches words outside any sentence: "{quote.text}"')
        matched.append(RequiredQuote(
            id=f"q{number}", text=quote.text, start_sentence=inside[0].id,
            end_sentence=inside[-1].id, similarity=round(score, 3), role=quote.role,  # type: ignore[arg-type]
            occurrences=1 if quote.occurrence is not None else occurrences,
        ))  # fmt: skip
    ordered = sorted(matched, key=lambda q: q.start_sentence)
    for a, b in itertools.pairwise(ordered):
        if b.start_sentence <= a.end_sentence:
            shared = f"{b.start_sentence}-{a.end_sentence}"
            raise S2SError(f"Quotes {a.id[1:]} and {b.id[1:]} share sentences {shared}; "
                           "make them one quote.")  # fmt: skip
    return matched
