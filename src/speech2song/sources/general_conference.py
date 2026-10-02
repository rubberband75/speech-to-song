"""General Conference talks on churchofjesuschrist.org: the audio and the published text.

The site's content API returns a page as JSON (the same data the page renders):
`meta.audio` lists the audio downloads and `content.body` holds the talk as HTML.
The transcript keeps what is spoken: the body's paragraphs and verse lines, one per
block. Headings, the byline, images and captions, footnote markers, the notes, and
scripture citations in parentheses are left out; the aligner treats anything else the
speaker skips as unspoken.
"""

import json
import re
from html.parser import HTMLParser
from urllib.parse import parse_qs, urlencode, urlsplit

from speech2song.errors import S2SError
from speech2song.sources.base import Http, Talk

API_URL = "https://www.churchofjesuschrist.org/study/api/v3/language-pages/type/content"
TALK_TYPE = "general-conference-talk"

SKIPPED = {"header", "footer", "figure", "figcaption", "img", "picture", "video", "audio",
           "sup", "script", "style", "nav", "aside", "table", "h1", "h2", "h3", "h4", "h5",
           "h6"}  # fmt: skip
BLOCKS = {"p", "li", "div", "section", "blockquote", "dd", "dt"}
VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "source",
        "track", "wbr"}  # fmt: skip
CITATION = "\ue000"  # marks a scripture link while the text is assembled
_CITED = re.compile(f"\\s*\\([^()]*{CITATION}[^()]*\\)")


def _clean(text: str) -> str:
    text = _CITED.sub("", text).replace(CITATION, "")
    text = " ".join(text.split())
    return re.sub(r"\s+([.,;:!?])", r"\1", text)  # where a citation was removed


class _TalkHtml(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.skipping: list[str] = []  # open skipped elements, innermost last
        self.chunks: list[str] = []
        self.blocks: list[str] = []
        self.author: str | None = None
        self._author: list[str] | None = None

    def _flush(self) -> None:
        text = _clean("".join(self.chunks))
        self.chunks = []
        if text:
            self.blocks.append(text)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        classes = (dict(attrs).get("class") or "").split()
        if tag == "p" and "author-name" in classes:
            self._author = []
        if tag in VOID:
            if tag == "br" and not self.skipping:
                self._flush()
            return
        if self.skipping or tag in SKIPPED:
            self.skipping.append(tag)
            return
        if tag in BLOCKS:
            self._flush()
        elif tag == "a" and "scripture-ref" in classes:
            self.chunks.append(CITATION)

    def handle_endtag(self, tag: str) -> None:
        if tag == "p" and self._author is not None:
            self.author = " ".join("".join(self._author).split())
            self._author = None
        if self.skipping:
            if tag in self.skipping:
                del self.skipping[len(self.skipping) - 1 - self.skipping[::-1].index(tag) :]
            return
        if tag in BLOCKS:
            self._flush()

    def handle_data(self, data: str) -> None:
        if self._author is not None:
            self._author.append(data)
        if not self.skipping:
            self.chunks.append(data)


def talk_text(body: str) -> tuple[str, str | None]:
    """(transcript, speaker) from a talk's body HTML. The transcript has one block per
    paragraph or verse line, separated by blank lines, like a hand-copied one."""
    parser = _TalkHtml()
    parser.feed(body)
    parser.close()
    parser._flush()
    speaker = re.sub(r"(?i)^by\s+", "", parser.author) if parser.author else None
    text = "\n\n".join(parser.blocks)
    return (text + "\n" if text else ""), speaker or None


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
