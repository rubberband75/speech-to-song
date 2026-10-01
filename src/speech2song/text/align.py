"""Align an official transcript to ASR words; build the 02_transcript.json content.

1. Both sides are tokenized (normalize.py); every token remembers its source word.
2. Unspoken lines: official lines barely covered by "solid" matches (runs of 3+ pairs,
   consecutive on both sides) are set aside. Captions and headings end up here.
3. The remaining official tokens are aligned: exact LCS anchors first, then a small
   dynamic program inside each gap pairs similar tokens (misheard names, split words).
4. Whatever is left between pairs: official-only runs are unspoken (3+ tokens) or
   interpolated; ASR-only runs stay as `asr_only` words; runs on both sides are
   substitutions where the official words take the ASR words' time span
   ("1984" spoken as "nineteen eighty-four").
5. Words are emitted in alignment order. Sentences come from the official text, plus
   ASR-only runs long enough to stand alone; short ones join a neighbouring sentence.
"""

from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import fmean
from typing import Literal

from rapidfuzz.distance import Indel

from speech2song.config import AlignConfig
from speech2song.models import (
    AlignmentReport,
    AsrRef,
    AsrResult,
    AsrWord,
    Sentence,
    Span,
    Transcript,
    Word,
    WordFlag,
)
from speech2song.text.normalize import has_word_chars, tokenize
from speech2song.text.sentences import OfficialText, parse_official, split_timed

SOLID_RUN = 3  # consecutive pairs in runs this long count as solid evidence of speech
MAX_GAP_CELLS = 250_000  # larger gaps skip fuzzy pairing (keeps the DP fast)
_MOVES = ((1, 1), (1, 2), (2, 1))

SentenceRange = tuple[int, int, Literal["official", "asr"]]


@dataclass(frozen=True)
class Link:
    """Official token(s) paired with ASR token(s)."""

    o: tuple[int, ...]
    a: tuple[int, ...]
    exact: bool


def similarity(x: str, y: str) -> float:
    return Indel.normalized_similarity(x, y)


def align_tokens(o: Sequence[str], a: Sequence[str], threshold: float) -> list[Link]:
    """Monotonic pairing: exact LCS blocks, plus fuzzy pairs inside the gaps."""
    vocab: dict[str, int] = {}
    o_ids = [vocab.setdefault(t, len(vocab)) for t in o]
    a_ids = [vocab.setdefault(t, len(vocab)) for t in a]
    links: list[Link] = []
    oi = ai = 0
    for block in Indel.opcodes(o_ids, a_ids):
        if block.tag != "equal":
            continue
        links += gap_pairs(o, a, (oi, block.src_start), (ai, block.dest_start), threshold)
        size = block.src_end - block.src_start
        links += [Link((block.src_start + k,), (block.dest_start + k,), True) for k in range(size)]
        oi, ai = block.src_end, block.dest_end
    links += gap_pairs(o, a, (oi, len(o)), (ai, len(a)), threshold)
    return links


def gap_pairs(
    o: Sequence[str],
    a: Sequence[str],
    o_range: tuple[int, int],
    a_range: tuple[int, int],
    threshold: float,
) -> list[Link]:
    """Best monotonic 1:1 / 1:2 / 2:1 pairing inside a gap, maximizing total similarity."""
    (o0, o1), (a0, a1) = o_range, a_range
    n, m = o1 - o0, a1 - a0
    if n == 0 or m == 0 or n * m > MAX_GAP_CELLS:
        return []
    score = [[0.0] * (m + 1) for _ in range(n + 1)]
    move = [[(0, 0)] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        for j in range(m + 1):
            if i == j == 0:
                continue
            best, how = -1.0, (0, 0)
            if i and score[i - 1][j] > best:
                best, how = score[i - 1][j], (1, 0)
            if j and score[i][j - 1] > best:
                best, how = score[i][j - 1], (0, 1)
            for di, dj in _MOVES:
                if i >= di and j >= dj:
                    sim = similarity(
                        "".join(o[o0 + i - di : o0 + i]), "".join(a[a0 + j - dj : a0 + j])
                    )
                    if sim >= threshold and score[i - di][j - dj] + sim > best:
                        best, how = score[i - di][j - dj] + sim, (di, dj)
            score[i][j], move[i][j] = best, how
    links = []
    i, j = n, m
    while i or j:
        di, dj = move[i][j]
        if di and dj:
            links.append(
                Link(tuple(range(o0 + i - di, o0 + i)), tuple(range(a0 + j - dj, a0 + j)), False)
            )
        i, j = i - di, j - dj
    links.reverse()
    return links


def solid_links(links: Sequence[Link]) -> list[Link]:
    """Links in runs of SOLID_RUN+ pairs that are consecutive on both sides.

    Exact or fuzzy: a misheard name inside a run still counts. Unrelated texts produce
    only scattered matches (common words), so they have almost no solid links.
    """
    solid: list[Link] = []
    run: list[Link] = []
    for link in links:
        if run and link.o[0] == run[-1].o[-1] + 1 and link.a[0] == run[-1].a[-1] + 1:
            run.append(link)
            continue
        if len(run) >= SOLID_RUN:
            solid += run
        run = [link]
    if len(run) >= SOLID_RUN:
        solid += run
    return solid


@dataclass
class _Out:
    """A word being built; times are filled in stages."""

    w: str
    start: float
    end: float
    conf: float | None
    flag: WordFlag
    asr: str | None = None
    timing: tuple[int, ...] = ()  # ASR word indices that time an official word


def _spread(items: Sequence[_Out], lo: float, hi: float) -> None:
    """Share [lo, hi] among items in proportion to their text length."""
    weights = [max(1, len(item.w)) for item in items]
    total = sum(weights)
    t = lo
    for item, weight in zip(items, weights, strict=True):
        item.start = t
        t += (hi - lo) * weight / total
        item.end = t


def _unspoken_lines(
    official: OfficialText, o_texts: Sequence[str], o_lines: Sequence[int], a_texts: Sequence[str],
    cfg: AlignConfig,
) -> set[int]:  # fmt: skip
    links = align_tokens(o_texts, a_texts, cfg.fuzzy_threshold)
    solid = {x for link in solid_links(links) for x in link.o}
    total: Counter[int] = Counter()
    covered: Counter[int] = Counter()
    for k, line in enumerate(o_lines):
        total[line] += 1
        covered[line] += k in solid
    ratio = cfg.unspoken_line_ratio
    return {line for line, count in total.items() if covered[line] / count < ratio}


def align_official(
    official: OfficialText, asr_words: Sequence[AsrWord], cfg: AlignConfig, duration_s: float
) -> tuple[list[Word], list[Sentence], AlignmentReport]:
    o_tokens = tokenize([w.text for w in official.words])
    a_tokens = tokenize([w.w for w in asr_words])
    a_texts = [t.text for t in a_tokens]
    o_lines = [official.words[t.word].line for t in o_tokens]

    skip_lines = _unspoken_lines(official, [t.text for t in o_tokens], o_lines, a_texts, cfg)
    active = [k for k, line in enumerate(o_lines) if line not in skip_lines]
    links = align_tokens([o_tokens[k].text for k in active], a_texts, cfg.fuzzy_threshold)
    solid_a = {x for link in solid_links(links) for x in link.a}

    status: dict[int, str] = {}  # official token -> exact | fuzzy | sub | interpolated | unspoken
    timing: dict[int, tuple[int, ...]] = {}  # official token -> ASR words that time it
    used: set[int] = set()  # ASR tokens tied to official text
    order: list[tuple[str, int]] = []  # emission order: ("o" | "a", word index)

    def asr_words_of(token_indices: Sequence[int]) -> tuple[int, ...]:
        return tuple(sorted({a_tokens[x].word for x in token_indices}))

    def gap(o_lo: int, o_hi: int, a_lo: int, a_hi: int) -> None:
        o_part = [active[x] for x in range(o_lo, o_hi)]
        a_part = list(range(a_lo, a_hi))
        comparable = len(a_part) <= 3 * len(o_part) + 2 and len(o_part) <= 3 * len(a_part) + 2
        if o_part and a_part and comparable:
            span = asr_words_of(a_part)
            for k in o_part:
                status[k], timing[k] = "sub", span
            used.update(a_part)
            order.extend(("o", o_tokens[k].word) for k in o_part)
            return
        kind = "unspoken" if len(o_part) >= cfg.unspoken_min_tokens else "interpolated"
        for k in o_part:
            status[k] = kind
        order.extend(("o", o_tokens[k].word) for k in o_part)
        order.extend(("a", a_tokens[x].word) for x in a_part)

    po = pa = 0
    for link in links:
        gap(po, link.o[0], pa, link.a[0])
        span = asr_words_of(link.a)
        for x in link.o:
            status[active[x]] = "exact" if link.exact else "fuzzy"
            timing[active[x]] = span
        used.update(link.a)
        order.extend(("o", o_tokens[active[x]].word) for x in link.o)
        po, pa = link.o[-1] + 1, link.a[-1] + 1
    gap(po, len(active), pa, len(a_tokens))

    tokens_of: defaultdict[int, list[int]] = defaultdict(list)
    for k, token in enumerate(o_tokens):
        tokens_of[token.word].append(k)
    used_words = {a_tokens[x].word for x in used}

    out: list[_Out] = []
    out_index: dict[int, int] = {}  # official word -> position in `out`
    seen: set[tuple[str, int]] = set()
    for side, index in order:
        if (side, index) in seen:
            continue
        seen.add((side, index))
        if side == "a":
            if index not in used_words:
                word = asr_words[index]
                out.append(_Out(word.w, word.start, word.end, word.conf, "asr_only"))
            continue
        states = [status.get(k, "unspoken") for k in tokens_of[index]]
        if all(state == "unspoken" for state in states):
            continue
        sources = tuple(sorted({w for k in tokens_of[index] for w in timing.get(k, ())}))
        text = official.words[index].text
        if sources:
            heard = [asr_words[w] for w in sources]
            exact = all(state == "exact" for state in states)
            out.append(
                _Out(
                    text,
                    min(w.start for w in heard),
                    max(w.end for w in heard),
                    round(fmean(w.conf for w in heard), 3),
                    "matched" if exact else "fuzzy",
                    None if exact else " ".join(w.w for w in heard),
                    sources,
                )
            )
        else:
            out.append(_Out(text, 0.0, 0.0, None, "interpolated"))
        out_index[index] = len(out) - 1

    _finish_times(out, duration_s)
    words = [
        Word(w=o.w, start=round(o.start, 3), end=round(o.end, 3), conf=o.conf, flag=o.flag,
             asr=o.asr)
        for o in out
    ]  # fmt: skip

    ranges: list[list] = []
    for lo, hi in official.sentences:
        positions = [out_index[w] for w in range(lo, hi) if w in out_index]
        if positions:
            ranges.append([min(positions), max(positions) + 1, "official"])
    sentences = make_sentences(words, _cover_orphans(words, sorted(ranges), cfg))

    total = len(official.words)
    flags = Counter(words[i].flag for i in out_index.values())
    good = flags["matched"] + flags["fuzzy"]
    unspoken = total - len(out_index)
    report = AlignmentReport(
        official_words=total,
        matched=flags["matched"],
        fuzzy=flags["fuzzy"],
        interpolated=flags["interpolated"],
        unspoken=unspoken,
        asr_words=len(asr_words),
        asr_only=sum(1 for w in words if w.flag == "asr_only"),
        quality=round(good / total, 4) if total else 0.0,
        quality_spoken=round(good / (total - unspoken), 4) if total > unspoken else 0.0,
        coverage=round(len(used_words) / len(asr_words), 4) if asr_words else 0.0,
        anchor_coverage=round(len(solid_a) / len(a_tokens), 4) if a_tokens else 0.0,
        unspoken_spans=_unspoken_spans(official, out_index),
        asr_only_spans=_asr_only_spans(words),
    )
    return words, sentences, report


def _finish_times(out: list[_Out], duration_s: float) -> None:
    """Share time among words timed by the same ASR words, fill interpolated words from
    their neighbours, and make starts non-decreasing."""
    n = len(out)
    i = 0
    while i < n:
        j = i
        if out[i].timing:
            while j + 1 < n and out[j + 1].timing == out[i].timing:
                j += 1
            if j > i:
                _spread(out[i : j + 1], out[i].start, out[i].end)
        i = j + 1
    i = 0
    while i < n:
        if out[i].flag != "interpolated":
            i += 1
            continue
        j = i
        while j < n and out[j].flag == "interpolated":
            j += 1
        lo = out[i - 1].end if i > 0 else 0.0
        hi = out[j].start if j < n else duration_s
        _spread(out[i:j], lo, max(lo, hi))
        i = j
    for k in range(1, n):
        out[k].start = max(out[k].start, out[k - 1].start)
        out[k].end = max(out[k].end, out[k].start)


def _cover_orphans(
    words: Sequence[Word], ranges: list[list], cfg: AlignConfig
) -> list[SentenceRange]:
    """Words outside every official sentence (ASR-only runs between sentences) become
    their own sentences when long enough, else join the neighbouring sentence."""
    n = len(words)
    covered = [False] * n
    for lo, hi, _ in ranges:
        covered[lo:hi] = [True] * (hi - lo)
    extra: list[list] = []
    i = 0
    while i < n:
        if covered[i]:
            i += 1
            continue
        j = i
        while j < n and not covered[j]:
            j += 1
        if j - i >= cfg.asr_sentence_min_words or not ranges:
            pieces = split_timed(words[i:j], pause_s=cfg.pause_split_s, max_s=cfg.max_sentence_s)
            extra += [[i + lo, i + hi, "asr"] for lo, hi in pieces]
        else:
            before = next((r for r in reversed(ranges) if r[1] == i), None)
            if before is not None:
                before[1] = j
            else:
                next(r for r in ranges if r[0] == j)[0] = i
        i = j
    return [(lo, hi, source) for lo, hi, source in sorted(ranges + extra)]


def _unspoken_spans(official: OfficialText, out_index: dict[int, int]) -> list[Span]:
    """Runs of official words missing from the output, one span per line at most."""
    spans: list[Span] = []
    run: list[int] = []
    for index in [*range(len(official.words)), None]:
        missing = index is not None and index not in out_index
        if missing and run and official.words[index].line == official.words[run[-1]].line:
            run.append(index)
            continue
        if run:
            text = " ".join(official.words[k].text for k in run)
            spans.append(Span(text=text, line=official.words[run[0]].line))
        run = [index] if missing else []
    return spans


def _asr_only_spans(words: Sequence[Word]) -> list[Span]:
    spans: list[Span] = []
    run: list[Word] = []
    for word in [*words, None]:
        if word is not None and word.flag == "asr_only":
            run.append(word)
            continue
        if run:
            text = " ".join(w.w for w in run)
            spans.append(Span(text=text, start=run[0].start, end=run[-1].end))
            run = []
    return spans


def make_sentences(words: Sequence[Word], ranges: Sequence[SentenceRange]) -> list[Sentence]:
    sentences = []
    for number, (lo, hi, source) in enumerate(sorted(ranges), start=1):
        span = words[lo:hi]
        confs = [w.conf for w in span if w.conf is not None]
        sentences.append(
            Sentence(
                id=number,
                text=" ".join(w.w for w in span),
                start=span[0].start,
                end=max(w.end for w in span),
                word_start=lo,
                word_end=hi,
                source=source,
                avg_conf=round(fmean(confs), 3) if confs else None,
            )
        )
    return sentences


def merge_punctuation(words: Sequence[AsrWord]) -> list[AsrWord]:
    """Attach punctuation-only ASR words (e.g. a lone "-") to a neighbouring word."""
    out: list[AsrWord] = []
    pending: AsrWord | None = None
    for word in words:
        if has_word_chars(word.w):
            if pending is not None:
                word = word.model_copy(update={"w": pending.w + word.w, "start": pending.start})
                pending = None
            out.append(word)
        elif out:
            last = out[-1]
            out[-1] = last.model_copy(update={"w": last.w + word.w, "end": max(last.end, word.end)})
        else:
            pending = (
                word if pending is None else pending.model_copy(update={"w": pending.w + word.w})
            )
    return out


def build_transcript(
    asr: AsrResult, official_text: str | None, cfg: AlignConfig, *, source: str
) -> Transcript:
    """The 02_transcript.json document, aligned to `official_text` when given."""
    asr_words = merge_punctuation(asr.words())
    common = {
        "source": source,
        "duration_s": asr.duration_s,
        "language": asr.language,
        "asr": AsrRef(backend=asr.backend, model=asr.model),
    }
    report = None
    if official_text is not None:
        official = parse_official(official_text)
        words, sentences, report = align_official(official, asr_words, cfg, asr.duration_s)
        if report.anchor_coverage >= cfg.min_coverage:
            return Transcript(
                **common,
                words=words,
                sentences=sentences,
                official_transcript_used=True,
                alignment=report,
            )
        report = report.model_copy(update={"fallback": True})
    words = [Word(w=x.w, start=x.start, end=x.end, conf=x.conf, flag="asr") for x in asr_words]
    pieces = split_timed(words, pause_s=cfg.pause_split_s, max_s=cfg.max_sentence_s)
    return Transcript(
        **common,
        words=words,
        sentences=make_sentences(words, [(lo, hi, "asr") for lo, hi in pieces]),
        official_transcript_used=False,
        alignment=report,
    )
