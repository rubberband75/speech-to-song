"""BYU Speeches (speeches.byu.edu): the audio and the published text.

A talk's page has it all. Its schema.org `Article` (JSON-LD) gives the title, the speaker
and the audio file, and `div.individual-speech__content` holds the talk as HTML. The
transcript keeps what is spoken (see `html_text`); the closing copyright line is left out.
"""

import html
import json
from html.parser import HTMLParser
from urllib.parse import urlsplit

from speech2song.errors import S2SError
from speech2song.sources.base import Http, Talk
from speech2song.sources.html_text import TalkHtml, join_blocks

CONTENT = "individual-speech__content"


class _JsonLd(HTMLParser):
    """The JSON-LD scripts on a page, as text."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.scripts: list[str] = []
        self._in = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self._in = tag == "script" and dict(attrs).get("type") == "application/ld+json"
        if self._in:
            self.scripts.append("")

    def handle_endtag(self, tag: str) -> None:
        if tag == "script":
            self._in = False

    def handle_data(self, data: str) -> None:
        if self._in:
            self.scripts[-1] += data


def _nodes(value: object) -> list[dict]:
    """The objects in a JSON-LD value: a node, a list of nodes, or an `@graph`."""
    if isinstance(value, list):
        return [n for item in value for n in _nodes(item)]
    if isinstance(value, dict):
        return [value, *_nodes(value.get("@graph", []))]
    return []


def article(page: str) -> dict | None:
    """The page's schema.org Article, if it has one."""
    parser = _JsonLd()
    parser.feed(page)
    parser.close()
    for script in parser.scripts:
        try:
            nodes = _nodes(json.loads(script))
        except ValueError:
            continue
        for node in nodes:
            kinds = node.get("@type")
            if "Article" in (kinds if isinstance(kinds, list) else [kinds]):
                return node
    return None


def _names(value: object) -> list[str]:
    people = value if isinstance(value, list) else [value]
    names = [p.get("name") if isinstance(p, dict) else p for p in people]
    return [html.unescape(n).strip() for n in names if isinstance(n, str) and n.strip()]


def _audio_url(value: object) -> str | None:
    for item in value if isinstance(value, list) else [value]:
        if isinstance(item, dict) and isinstance(item.get("contentUrl"), str):
            return item["contentUrl"]
    return None


def talk_text(page: str) -> str:
    """The transcript in the page's talk content."""
    parser = TalkHtml(root=CONTENT)
    parser.feed(page)
    parser.close()
    return join_blocks([b for b in parser.blocks if not b.startswith("©")])


def talk_from_page(url: str, page: str) -> Talk:
    """The talk on a page. Raises when the page is not a talk or has no audio or text."""
    node = article(page)
    if node is None:
        raise S2SError(f"Not a BYU Speeches talk (the page has no article data): {url}")
    audio = _audio_url(node.get("audio"))
    if not audio:
        raise S2SError(f"The talk page has no audio download: {url}")
    text = talk_text(page)
    if not text:
        raise S2SError(f"The talk page has no text: {url}")
    title = html.unescape(node.get("headline") or node.get("name") or "").strip()
    speakers = _names(node.get("author"))
    return Talk(
        page_url=url,
        title=title or urlsplit(url).path.rstrip("/").rsplit("/", 1)[-1],
        speaker=" and ".join(speakers) or None,
        audio_url=audio,
        text=text,
    )


class ByuSpeeches:
    name = "BYU Speeches (speeches.byu.edu)"
    prefixes = ("speeches.byu.edu/talks/",)

    def describe(self, url: str, http: Http) -> Talk:
        return talk_from_page(url, http.get(url).decode("utf-8", errors="replace"))
