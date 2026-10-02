"""General Conference talks on churchofjesuschrist.org: the audio and the published text.

The site's content API returns a page as JSON (the same data the page renders):
`meta.audio` lists the audio downloads and `content.body` holds the talk as HTML.
The transcript keeps what is spoken (see `html_text`); the byline, in the body's header,
gives the speaker.
"""

import json
import re
from urllib.parse import parse_qs, urlencode, urlsplit

from speech2song.errors import S2SError
from speech2song.sources.base import Http, Talk
from speech2song.sources.html_text import TalkHtml, classes, join_blocks

API_URL = "https://www.churchofjesuschrist.org/study/api/v3/language-pages/type/content"
TALK_TYPE = "general-conference-talk"


class _TalkHtml(TalkHtml):
    """The talk's blocks, and the speaker from the byline (`p.author-name`)."""

    def __init__(self) -> None:
        super().__init__()
        self.author: str | None = None
        self._author: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "p" and "author-name" in classes(attrs):
            self._author = []
        super().handle_starttag(tag, attrs)

    def handle_endtag(self, tag: str) -> None:
        if tag == "p" and self._author is not None:
            self.author = " ".join("".join(self._author).split())
            self._author = None
        super().handle_endtag(tag)

    def handle_data(self, data: str) -> None:
        if self._author is not None:
            self._author.append(data)
        super().handle_data(data)


def talk_text(body: str) -> tuple[str, str | None]:
    """(transcript, speaker) from a talk's body HTML."""
    parser = _TalkHtml()
    parser.feed(body)
    parser.close()
    speaker = re.sub(r"(?i)^by\s+", "", parser.author) if parser.author else None
    return join_blocks(parser.blocks), speaker or None


def talk_from_page(url: str, page: dict) -> Talk:
    """The talk on an API page. Raises when the page is not a talk or has no audio."""
    meta = page.get("meta") or {}
    content = page.get("content") or {}
    kind = (meta.get("pageAttributes") or {}).get("data-content-type")
    if kind != TALK_TYPE:
        raise S2SError(f"Not a General Conference talk (the page is {kind or 'unknown'}): {url}")
    audio = [a for a in meta.get("audio") or [] if a.get("mediaUrl")]
    if not audio:
        raise S2SError(f"The talk page has no audio download: {url}")
    chosen = next((a for a in audio if a.get("variant") == "audio"), audio[0])
    text, speaker = talk_text(content.get("body") or "")
    if not text:
        raise S2SError(f"The talk page has no text: {url}")
    title = meta.get("title") or urlsplit(url).path.rstrip("/").rsplit("/", 1)[-1]
    return Talk(page_url=url, title=title, speaker=speaker, audio_url=chosen["mediaUrl"],
                text=text)  # fmt: skip


def api_url(url: str) -> str:
    parts = urlsplit(url)
    lang = parse_qs(parts.query).get("lang", ["eng"])[0]
    uri = parts.path.removeprefix("/study").rstrip("/")
    return f"{API_URL}?{urlencode({'lang': lang, 'uri': uri})}"


class GeneralConference:
    name = "General Conference talks (churchofjesuschrist.org)"
    prefixes = ("www.churchofjesuschrist.org/study/general-conference/",)

    def describe(self, url: str, http: Http) -> Talk:
        try:
            page = json.loads(http.get(api_url(url)))
        except ValueError as exc:
            raise S2SError(f"The site's content API sent something unexpected for {url}") from exc
        if not isinstance(page, dict):
            raise S2SError(f"The site's content API sent something unexpected for {url}")
        return talk_from_page(url, page)
