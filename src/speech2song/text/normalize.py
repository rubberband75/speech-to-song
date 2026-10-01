"""Word normalization for matching official transcript words against ASR words.

One display word becomes zero or more match tokens: "Kearon's" -> ["kearons"],
"eight-year-old" -> ["eight", "year", "old"], "[we]" -> ["we"], "Savior.12" -> ["savior"].
"""

import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass

_APOSTROPHES = "\u2018\u2019\u201a\u201b\u2032"  # curly/low/reversed single quotes, prime
_QUOTES = "\u201c\u201d\u201e\u201f\u2033"  # curly/low/reversed double quotes, double prime
_DASHES = "\u2010\u2011\u2012\u2013\u2014\u2015\u2212"  # hyphens, en/em dashes, minus
_PUNCT = str.maketrans(
    {
        **dict.fromkeys(_APOSTROPHES, "'"),
        **dict.fromkeys(_QUOTES, '"'),
        **dict.fromkeys(_DASHES, "-"),
        "\u2026": "...",
    }
)
# Footnote markers pasted as plain digits after a word's punctuation: "Savior.12", "us,3".
_FOOTNOTE = re.compile(r"(?<=[^\W\d_][.,;:!?\"')\]])\d{1,3}$")
_SEPARATORS = re.compile(r"[-/]+")
_NON_WORD = re.compile(r"[\W_]+")


@dataclass(frozen=True)
class Token:
    text: str
    word: int  # index of the word it came from


def normalize_word(word: str) -> list[str]:
    text = unicodedata.normalize("NFKC", word).translate(_PUNCT)
    text = _FOOTNOTE.sub("", text)
    tokens = []
    for part in _SEPARATORS.split(text):
        decomposed = unicodedata.normalize("NFKD", part)
        no_accents = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
        token = _NON_WORD.sub("", no_accents.replace("'", "").casefold())
        if token:
            tokens.append(token)
    return tokens


def tokenize(words: Sequence[str]) -> list[Token]:
    return [Token(text, index) for index, word in enumerate(words) for text in normalize_word(word)]


def has_word_chars(text: str) -> bool:
    return any(ch.isalnum() for ch in text)
