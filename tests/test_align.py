"""Official-transcript alignment, on synthetic text and fake ASR timings."""

import pytest

from speech2song.config import AlignConfig
from speech2song.models import Transcript, Word
from speech2song.text.align import Link, build_transcript, gap_pairs, solid_links

from .fixtures.synth import FIXTURES, asr_result, fake_asr_words

CFG = AlignConfig()
MINI_TALK = (FIXTURES / "mini_talk.txt").read_text(encoding="utf-8")
# What a recognizer might return for mini_talk.txt: misheard names, spelled-out numbers,
# an ad-lib sentence, an "um", a dropped "the", and none of the captions or headings.
MINI_TALK_SPOKEN = (FIXTURES / "mini_talk_spoken.txt").read_text(encoding="utf-8").strip()


def transcribe(official: str | None, spoken: str) -> Transcript:
    transcript = build_transcript(
        asr_result(fake_asr_words(spoken)), official, CFG, source="talk.mp3"
    )
    check_invariants(transcript)
    return transcript


def check_invariants(transcript: Transcript) -> None:
    words = transcript.words
    assert [w.start for w in words] == sorted(w.start for w in words)
    assert all(w.start <= w.end for w in words)
    position = 0
    for number, sentence in enumerate(transcript.sentences, start=1):
        assert sentence.id == number
        assert sentence.word_start == position < sentence.word_end
        span = words[sentence.word_start : sentence.word_end]
        assert sentence.text == " ".join(w.w for w in span)
        assert sentence.start == span[0].start
        position = sentence.word_end
    assert position == len(words)


def word(transcript: Transcript, text: str) -> Word:
    matches = [w for w in transcript.words if w.w == text]
    assert len(matches) == 1, f"{text!r}: {matches}"
    return matches[0]


def test_identical_text_matches_everything() -> None:
    text = "Choices have consequences. But we can change. That is the good news!"
    asr = fake_asr_words(text)
    transcript = transcribe(text, text)
    assert transcript.official_transcript_used
    assert [w.flag for w in transcript.words] == ["matched"] * len(asr)
    assert [(w.start, w.end) for w in transcript.words] == [(w.start, w.end) for w in asr]
    assert [s.text for s in transcript.sentences] == [
        "Choices have consequences.",
        "But we can change.",
        "That is the good news!",
    ]
    report = transcript.alignment
    assert report is not None
    assert (report.quality, report.quality_spoken, report.coverage) == (1.0, 1.0, 1.0)


def test_without_official_text_uses_asr_sentences() -> None:
    transcript = transcribe(None, "First idea here. Second idea here.")
    assert not transcript.official_transcript_used
    assert transcript.alignment is None
    assert {w.flag for w in transcript.words} == {"asr"}
    assert [s.source for s in transcript.sentences] == ["asr", "asr"]


def test_misheard_names_take_the_asr_timing() -> None:
    official = "Brother Ollivander spoke with Marisol about Teichert paintings."
    spoken = "Brother Oliver Ander spoke with Mary Sol about Tyker paintings."
    transcript = transcribe(official, spoken)
    asr = {w.w: w for w in fake_asr_words(spoken)}
    name = word(transcript, "Ollivander")
    assert (name.flag, name.asr) == ("fuzzy", "Oliver Ander")
    assert (name.start, name.end) == (asr["Oliver"].start, asr["Ander"].end)
    assert word(transcript, "Marisol").asr == "Mary Sol"
    painter = word(transcript, "Teichert")  # too different to pair: a substitution
    assert (painter.flag, painter.start, painter.end) == (
        "fuzzy",
        asr["Tyker"].start,
        asr["Tyker"].end,
    )
    assert [w.w for w in transcript.words] == official.split()


def test_numbers_spoken_as_words() -> None:
    transcript = transcribe("In 1984 he declared it.", "In nineteen eighty-four he declared it.")
    asr = {w.w: w for w in fake_asr_words("In nineteen eighty-four he declared it.")}
    year = word(transcript, "1984")
    assert year.flag == "fuzzy"
    assert (year.start, year.end) == (asr["nineteen"].start, asr["eighty-four"].end)
    assert year.asr == "nineteen eighty-four"


def test_one_asr_word_for_two_official_words_is_shared() -> None:
    transcript = transcribe("We can not stop now.", "We cannot stop now.")
    cannot = {w.w: w for w in fake_asr_words("We cannot stop now.")}["cannot"]
    can, not_ = word(transcript, "can"), word(transcript, "not")
    assert can.start == cannot.start
    assert can.end == pytest.approx(not_.start)
    assert not_.end == cannot.end


def test_captions_and_headings_are_unspoken() -> None:
    transcript = transcribe(MINI_TALK, MINI_TALK_SPOKEN)
    report = transcript.alignment
    assert report is not None and not report.fallback
    assert [(s.line, s.text) for s in report.unspoken_spans] == [
        (9, "The Long Way Back"),
        (10, "Harbor at dawn, with fishing boats"),
        (13, "Vale family at the harbor"),
        (14, "Conclusion"),
    ]
    texts = [s.text for s in transcript.sentences]
    assert "Conclusion" not in texts
    assert not any("fishing boats" in t for t in texts)
    assert report.quality_spoken > 0.9
    assert report.coverage > 0.9


def test_mini_talk_word_flags() -> None:
    transcript = transcribe(MINI_TALK, MINI_TALK_SPOKEN)
    assert word(transcript, "Ollivander\u2019s").flag == "fuzzy"
    assert word(transcript, "Brightwater\u2019s").asr == "Bright Waters"
    assert word(transcript, "42").asr == "forty-two"
    assert word(transcript, "1984").asr == "nineteen eighty-four"
    assert word(transcript, "um,").flag == "asr_only"
    the_news = [w for w in transcript.words if w.w == "the" and w.flag == "interpolated"]
    assert len(the_news) == 1


def test_ad_lib_becomes_its_own_asr_sentence() -> None:
    transcript = transcribe(MINI_TALK, MINI_TALK_SPOKEN)
    asr_sentences = [s for s in transcript.sentences if s.source == "asr"]
    assert [s.text for s in asr_sentences] == ["And I want to tell you a little story."]
    report = transcript.alignment
    assert report is not None
    assert "And I want to tell you a little story." in [s.text for s in report.asr_only_spans]


def test_short_interjection_stays_inside_its_sentence() -> None:
    transcript = transcribe(MINI_TALK, MINI_TALK_SPOKEN)
    sentence = next(s for s in transcript.sentences if "daughter" in s.text)
    assert "um," in sentence.text.split()
    assert sentence.source == "official"


def test_dropped_word_is_interpolated_between_neighbours() -> None:
    transcript = transcribe("We saw the bright light today.", "We saw bright light today.")
    the = word(transcript, "the")
    saw, bright = word(transcript, "saw"), word(transcript, "bright")
    assert the.flag == "interpolated"
    assert saw.end <= the.start <= the.end <= bright.start


def test_skipped_clause_is_unspoken() -> None:
    official = "We went home, after a very long and tiring day, and slept."
    transcript = transcribe(official, "We went home, and slept.")
    report = transcript.alignment
    assert report is not None
    assert [s.text for s in report.unspoken_spans] == ["after a very long and tiring day,"]
    assert transcript.sentences[0].text == "We went home, and slept."


def test_wrong_transcript_falls_back_to_asr() -> None:
    official = "Completely unrelated words about gardening, tomatoes and the weather in spring."
    spoken = "We talked about the stars and planets and moons orbiting far away tonight."
    transcript = transcribe(official, spoken)
    assert not transcript.official_transcript_used
    assert transcript.alignment is not None and transcript.alignment.fallback
    assert {w.flag for w in transcript.words} == {"asr"}


def test_empty_asr() -> None:
    transcript = build_transcript(asr_result([], duration_s=5.0), MINI_TALK, CFG, source="x")
    assert transcript.words == []
    assert transcript.sentences == []
    assert transcript.alignment is not None and transcript.alignment.fallback


def test_gap_pairs_prefers_merges_that_fit_better() -> None:
    links = gap_pairs(["anglesey"], ["angle", "sea"], (0, 1), (0, 2), 0.6)
    assert links == [Link((0,), (0, 1), False)]
    links = gap_pairs(["can", "not"], ["cannot"], (0, 2), (0, 1), 0.6)
    assert links == [Link((0, 1), (0,), False)]
    assert gap_pairs(["teichert"], ["tyker"], (0, 1), (0, 1), 0.6) == []


def test_solid_links_need_three_consecutive_pairs() -> None:
    links = [Link((0,), (0,), True), Link((1,), (1,), True), Link((3,), (2,), True)]
    assert solid_links(links) == []
    links.append(Link((4, 5), (3,), False))  # a fuzzy 2:1 pair continues the run
    links.append(Link((6,), (4,), True))
    assert solid_links(links) == links[2:]
