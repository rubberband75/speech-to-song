"""The spoken text of a talk published as HTML, for the handlers that read talk pages.

The transcript keeps what is spoken: paragraphs, list items and verse lines, one per
block. Headings, images and captions, footnote markers and notes are skipped. Citations
are dropped: a group in parentheses or brackets that holds a link or a number, or starts
with "see", and a bracketed sentence (a note such as "[A photo was shown.]"). Other
brackets are editorial words the speaker said, so their words stay. The aligner treats
anything else the speaker skips as unspoken.
"""

import re
from html.parser import HTMLParser

SKIPPED = {"header", "footer", "figure", "figcaption", "img", "picture", "video", "audio",
           "sup", "script", "style", "nav", "aside", "table", "h1", "h2", "h3", "h4", "h5",
           "h6"}  # fmt: skip
BLOCKS = {"p", "li", "div", "section", "blockquote", "dd", "dt"}
VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "source",
        "track", "wbr"}  # fmt: skip
LINK = "\ue000"  # marks a link while the text is assembled
KEPT = ("\ue001", "\ue002")  # parentheses kept as spoken, while citations are dropped
_GROUP = re.compile(r"\(([^()\[\]]*)\)|\[([^()\[\]]*)\]")  # an innermost group


def classes(attrs: list[tuple[str, str | None]]) -> list[str]:
    return (dict(attrs).get("class") or "").split()


def is_citation(inner: str, bracket: bool) -> bool:
    """Whether a group in parentheses (or brackets) cites a source rather than being said."""
    if LINK in inner or re.search(r"\d", inner) or re.match(r"\s*see\b", inner, re.I):
        return True
    return bracket and re.search(r"[.!?]\s*$", inner) is not None


def drop_citations(text: str) -> str:
    """Text without its citations, innermost groups first ("[Book (City, 1982), 93]")."""

    def replace(match: re.Match[str]) -> str:
        paren, bracket = match.groups()
        inner = paren if paren is not None else bracket
        if is_citation(inner, bracket is not None):
            return ""
        return f"{KEPT[0]}{inner}{KEPT[1]}" if paren is not None else inner

    while (dropped := _GROUP.sub(replace, text)) != text:
        text = dropped
    return text.replace(KEPT[0], "(").replace(KEPT[1], ")")


def clean(text: str) -> str:
    text = drop_citations(text).replace(LINK, "")
    text = " ".join(text.split())
    return re.sub(r"\s+([.,;:!?])", r"\1", text)  # where a citation was removed


class TalkHtml(HTMLParser):
    """Collects the text blocks of a talk. With `root`, only the text inside the first
    element with that class is read (the talk on a whole web page)."""

    def __init__(self, root: str | None = None) -> None:
        super().__init__(convert_charrefs=True)
        self.root = root
        self.reading = root is None
        self._root_tag: str | None = None
        self._root_depth = 0  # open elements inside the root with the root's tag
        self.skipping: list[str] = []  # open skipped elements, innermost last
        self.chunks: list[str] = []
        self.blocks: list[str] = []

    def _flush(self) -> None:
        text = clean("".join(self.chunks))
        self.chunks = []
        if text:
            self.blocks.append(text)

    def _enter_root(self, tag: str, attrs: list[tuple[str, str | None]]) -> bool:
        """Track the root element; False while outside it."""
        if self.root is None:
            return True
        if self._root_tag is None:
            if self.root in classes(attrs) and tag not in VOID:
                self._root_tag, self.reading = tag, True
            return False
        if tag == self._root_tag and self.reading:
            self._root_depth += 1
        return self.reading

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if not self._enter_root(tag, attrs):
            return
        if tag in VOID:
            if tag == "br" and not self.skipping:
                self._flush()
            return
        if self.skipping or tag in SKIPPED:
            self.skipping.append(tag)
            return
        if tag in BLOCKS:
            self._flush()
        elif tag == "a":
            self.chunks.append(LINK)

    def handle_endtag(self, tag: str) -> None:
        if not self.reading:
            return
        if self.root is not None and tag == self._root_tag:
            if self._root_depth == 0:
                self._flush()
                self.reading = False
                return
            self._root_depth -= 1
        if self.skipping:
            if tag in self.skipping:
                del self.skipping[len(self.skipping) - 1 - self.skipping[::-1].index(tag) :]
            return
        if tag in BLOCKS:
            self._flush()

    def handle_data(self, data: str) -> None:
        if self.reading and not self.skipping:
            self.chunks.append(data)

    def close(self) -> None:
        super().close()
        if self.reading:
            self._flush()


def join_blocks(blocks: list[str]) -> str:
    """A transcript file's text: blocks separated by blank lines, like a hand-copied one."""
    return "\n\n".join(blocks) + "\n" if blocks else ""
