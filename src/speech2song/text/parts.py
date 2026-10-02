"""Where a long quote is split into parts, so it can play with short musical pauses in
between. Pure: word timings in, word index ranges out.

A boundary between two words is better the more it reads like a pause: a sentence end
is best, then clause punctuation (comma, semicolon, colon, dash), then any word gap;
within each kind, a longer silence is better. The fewest parts that keep every part
within the limit are used, as evenly long as the good boundaries allow.
"""

import math
from collections.abc import Collection, Sequence

from speech2song.models import Word

SENTENCE_COST = 0.0
CLAUSE_COST = 1.0
WORD_COST = 3.0
PAUSE_CREDIT_S = 0.6  # a gap this long or longer earns the full credit
BALANCE_WEIGHT = 2.0
CLAUSE_MARKS = (",", ";", ":", "-", "\u2013", "\u2014", "...")


def _length(words: Sequence[Word], start: int, end: int) -> float:
    return words[end - 1].end - words[start].start


def boundary_cost(words: Sequence[Word], index: int, sentence_ends: Collection[int]) -> float:
    """Cost of splitting before words[index] (lower is a better place to pause)."""
    before = words[index - 1]
    if index - 1 in sentence_ends:
        kind = SENTENCE_COST
    elif before.w.rstrip("\"')\u201d\u2019").endswith(CLAUSE_MARKS):
        kind = CLAUSE_COST
    else:
        kind = WORD_COST
    gap = max(0.0, words[index].start - before.end)
    return kind - min(gap, PAUSE_CREDIT_S) / PAUSE_CREDIT_S


def plan_parts(
    words: Sequence[Word], sentence_ends: Collection[int], max_part_s: float
) -> list[tuple[int, int]]:
    """Split words[0:n] into parts (start, end exclusive), each at most `max_part_s` long
    from its first word's start to its last word's end when possible. `sentence_ends`
    holds the indices of words that end a sentence. One part when everything fits."""
    n = len(words)
    if n < 2 or _length(words, 0, n) <= max_part_s:
        return [(0, n)]
    costs = [math.inf] + [boundary_cost(words, i, sentence_ends) for i in range(1, n)]
    total = _length(words, 0, n)
    for parts in range(math.ceil(total / max_part_s), n + 1):
        plan = _best_split(words, costs, parts, max_part_s, total / parts)
        if plan is not None:
            return plan
    return [(i, i + 1) for i in range(n)]  # a single word longer than the limit


def _best_split(
    words: Sequence[Word], costs: list[float], parts: int, max_part_s: float, ideal: float
) -> list[tuple[int, int]] | None:
    """The cheapest split into exactly `parts` parts within the limit (dynamic programming
    over word boundaries), or None if there is none."""
    n = len(words)
    best: dict[tuple[int, int], tuple[float, int]] = {(0, 0): (0.0, -1)}  # (k, end) -> cost
    for k in range(1, parts + 1):
        for end in range(1, n + 1):
            if (k == parts) != (end == n):
                continue
            options = []
            for start in range(k - 1, end):
                prior = best.get((k - 1, start))
                if prior is None or _length(words, start, end) > max_part_s:
                    continue
                balance = ((_length(words, start, end) - ideal) / ideal) ** 2
                cut = costs[start] if start else 0.0
                options.append((prior[0] + cut + BALANCE_WEIGHT * balance, start))
            if options:
                best[(k, end)] = min(options)
    if (parts, n) not in best:
        return None
    bounds, end = [], n
    for k in range(parts, 0, -1):
        start = best[(k, end)][1]
        bounds.append((start, end))
        end = start
    return bounds[::-1]
