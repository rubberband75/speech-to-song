"""Talks given as a URL: finding the handler, reading a General Conference page, saving the
files, and `ingest`/`run` with a URL. The page and the audio are synthetic, and the
network is never used."""

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from typer.testing import Result

from speech2song import sources
from speech2song.errors import S2SError
from speech2song.manifest import Run
from speech2song.sources import describe_talk, find_handler, is_url
from speech2song.sources.base import Talk, save_talk
from speech2song.sources.general_conference import (
    API_URL,
    GeneralConference,
    talk_from_page,
    talk_text,
)

from .fixtures.synth import speechlike, write_wav

Cli = Callable[..., Result]

PAGE_URL = ("https://www.churchofjesuschrist.org/study/general-conference/2030/04/"
            "a-made-up-talk?lang=eng")  # fmt: skip
AUDIO_URL = "https://assets.example.org/abc-64k-en.mp3"
STEM = "a-made-up-talk-by-elder-jane-q-example"

BODY = """<header>
<video controls="controls"><source src="talk.mp4" type="video/mp4"/></video>
<h1 id="title1">A Made-Up Talk</h1>
<div class="byline"><p class="author-name">By Elder Jane Q. Example</p>
<p class="author-role">Of a Quorum</p>
<div class="image-cropper"><img alt="Elder Example" src="portrait.jpg"/></div></div>
<p class="kicker">A summary nobody says aloud.</p>
</header>
<div class="body-block">
<p id="p1">The first paragraph &amp; its words.<a class="note-ref" href="#note1"><sup
 class="marker" data-value="1"></sup></a></p>
<section><header><h2>A Heading</h2><img alt="A caption" src="photo.jpg"></header>
<p>\u201cWe read a verse\u201d (<a class="scripture-ref" href="/a">Book 1:2</a>; see also
<a class="scripture-ref" href="/b">verse 3</a>). Then we spoke on.</p>
<p>In <a class="scripture-ref" href="/c">Book 4</a> we read <em>more</em>.</p>
<span class="page-break" data-page="2"></span>
<div class="poetry"><div class="stanza">
<p class="line">A line of verse,</p><p class="line">and another.</p>
</div></div>
<figure><img src="z.jpg"/><figcaption>A photo caption</figcaption></figure>
</section></div>
<footer class="notes"><p class="title">Notes</p><ol><li><p>A footnote.</p></li></ol></footer>
"""

SPOKEN = (
    "The first paragraph & its words.\n\n"
    "\u201cWe read a verse\u201d. Then we spoke on.\n\n"
    "In Book 4 we read more.\n\n"
    "A line of verse,\n\n"
    "and another.\n"
)


def _page(**meta: object) -> dict:
    base = {
        "title": "A Made-Up Talk",
        "audio": [{"mediaUrl": AUDIO_URL, "variant": "audio"}],
        "pageAttributes": {"data-content-type": "general-conference-talk"},
    }
    return {"meta": base | meta, "content": {"body": BODY}}


class FakeHttp:
    def __init__(self, page: dict | None = None, audio: bytes = b"audio bytes") -> None:
        self.page = _page() if page is None else page
        self.audio = audio
        self.requests: list[str] = []

    def get(self, url: str) -> bytes:
        self.requests.append(url)
        return json.dumps(self.page).encode()

    def download(self, url: str, dest: Path) -> None:
        self.requests.append(url)
        dest.write_bytes(self.audio)


def test_handlers_match_host_and_path_prefix() -> None:
    assert isinstance(find_handler(PAGE_URL), GeneralConference)
    assert isinstance(find_handler(PAGE_URL.replace("https://www", "http://WWW")),
                      GeneralConference)  # fmt: skip
    for url in ("https://speeches.byu.edu/talks/dallin-h-oaks/timing/",
                "https://www.churchofjesuschrist.org/study/scriptures/bofm/2-ne/2?lang=eng",
                "https://www.churchofjesuschrist.org.example.com/study/general-conference/x",
                "https://churchofjesuschrist.org/study/general-conference/2016/04/x"):  # fmt: skip
        assert find_handler(url) is None, url
    assert is_url("HTTPS://example.org/a") and not is_url("inputs/talk.mp3")
    with pytest.raises(S2SError, match="No handler for this URL") as error:
        describe_talk("https://speeches.byu.edu/talks/dallin-h-oaks/timing/", FakeHttp())
    assert "www.churchofjesuschrist.org/study/general-conference/" in str(error.value)


def test_talk_text_keeps_only_what_is_spoken() -> None:
    text, speaker = talk_text(BODY)
    assert text == SPOKEN
    assert speaker == "Elder Jane Q. Example"


def test_a_talk_is_read_from_the_content_api() -> None:
    http = FakeHttp()
    talk = GeneralConference().describe(PAGE_URL, http)
    assert http.requests == [
        f"{API_URL}?lang=eng&uri=%2Fgeneral-conference%2F2030%2F04%2Fa-made-up-talk"]  # fmt: skip
    assert (talk.title, talk.speaker, talk.audio_url, talk.text) == (
        "A Made-Up Talk", "Elder Jane Q. Example", AUDIO_URL, SPOKEN)  # fmt: skip
    assert (talk.stem, talk.audio_suffix) == (STEM, ".mp3")


def test_pages_that_are_not_talks_with_audio_are_refused() -> None:
    with pytest.raises(S2SError, match="Not a General Conference talk"):
        talk_from_page(PAGE_URL, _page(pageAttributes={"data-content-type": "table-of-contents"}))
    with pytest.raises(S2SError, match="no audio download"):
        talk_from_page(PAGE_URL, _page(audio=[]))
    other = _page(audio=[{"mediaUrl": "https://a.org/x.mp3", "variant": "other"},
                         {"mediaUrl": AUDIO_URL, "variant": "audio"}])  # fmt: skip
    assert talk_from_page(PAGE_URL, other).audio_url == AUDIO_URL
    http = FakeHttp()
    http.get = lambda url: b"<html>not json</html>"  # type: ignore[method-assign]
    with pytest.raises(S2SError, match="sent something unexpected"):
        GeneralConference().describe(PAGE_URL, http)


def test_a_talk_is_downloaded_once_and_edits_are_kept(tmp_path: Path) -> None:
    talk = Talk(PAGE_URL, "A Made-Up Talk", "Elder Jane Q. Example", AUDIO_URL, SPOKEN)
    http = FakeHttp()
    files = save_talk(talk, tmp_path / "inputs", http)
    assert files.audio == tmp_path / "inputs" / f"{STEM}.mp3"
    assert files.written == (files.audio, files.transcript)
    assert files.audio.read_bytes() == b"audio bytes" and files.transcript.read_text() == SPOKEN
    files.transcript.write_text("Edited by hand.\n")
    again = save_talk(talk, tmp_path / "inputs", http)
    assert again.written == () and again.transcript.read_text() == "Edited by hand.\n"
    assert http.requests == [AUDIO_URL]


def test_a_failed_download_leaves_no_file(tmp_path: Path) -> None:
    talk = Talk(PAGE_URL, "A Made-Up Talk", None, AUDIO_URL, SPOKEN)

    class Broken(FakeHttp):
        def download(self, url: str, dest: Path) -> None:
            dest.write_bytes(b"half")
            raise S2SError("connection reset")

    with pytest.raises(S2SError, match="connection reset"):
        save_talk(talk, tmp_path, Broken())
    with pytest.raises(S2SError, match="download was empty"):
        save_talk(talk, tmp_path, FakeHttp(audio=b""))
    assert list(tmp_path.iterdir()) == []


@pytest.fixture
def web(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> FakeHttp:
    """The CLI's web access, serving the synthetic page and a short speech-like WAV."""
    voice, _ = speechlike([(180, 220, 0.3), (220, 160, 0.25)] * 2)
    wav = write_wav(tmp_path / "audio.wav", np.stack([voice, voice], axis=1), subtype="PCM_16")
    http = FakeHttp(audio=wav.read_bytes())
    wav.unlink()
    monkeypatch.setattr(sources, "web", lambda: http)
    return http


def test_ingest_a_url_saves_the_talk_and_records_the_page(
    cli: Cli, web: FakeHttp, tmp_path: Path
) -> None:
    result = cli("ingest", PAGE_URL)
    assert result.exit_code == 0, result.output
    assert "A Made-Up Talk" in result.output and "(downloaded)" in result.output
    run = Run.open(tmp_path / "runs", "latest")
    inputs = (tmp_path / "inputs").resolve()
    assert Path(run.manifest.input.path) == inputs / f"{STEM}.mp3"
    assert run.manifest.transcript is not None
    assert Path(run.manifest.transcript.path) == inputs / f"{STEM}.txt"
    assert run.manifest.source_url == PAGE_URL
    assert run.path("00_source.wav").is_file()
    assert f"From: {PAGE_URL}" in cli("status").output
    again = cli("ingest", PAGE_URL)  # a second run reuses the files
    assert "already there, kept as is" in again.output
    assert web.requests.count(AUDIO_URL) == 1


def test_a_url_dry_run_reads_the_page_and_downloads_nothing(
    cli: Cli, web: FakeHttp, tmp_path: Path
) -> None:
    result = cli("run", PAGE_URL, "--dry-run")
    assert result.exit_code == 0, result.output
    assert "would save" in result.output and "isn't downloaded yet" in result.output
    assert AUDIO_URL not in web.requests
    assert not (tmp_path / "inputs").exists() and not (tmp_path / "runs").exists()


def test_a_url_without_a_handler_is_a_clean_error(cli: Cli, web: FakeHttp) -> None:
    result = cli("run", "https://speeches.byu.edu/talks/dallin-h-oaks/timing/")
    assert result.exit_code == 1
    assert "No handler for this URL" in result.output
