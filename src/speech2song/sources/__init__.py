"""Inputs given as a URL: the handler whose prefix (host + path) the URL starts with
finds the talk's audio and transcript, and they are saved as the run's input files."""

import re
from pathlib import Path
from urllib.parse import urlsplit

from speech2song.errors import S2SError
from speech2song.sources.base import (
    Http,
    Talk,
    TalkFiles,
    UrlHandler,
    UrlHttp,
    save_talk,
    talk_paths,
)
from speech2song.sources.general_conference import GeneralConference

HANDLERS: tuple[UrlHandler, ...] = (GeneralConference(),)

__all__ = ["HANDLERS", "Talk", "TalkFiles", "describe_talk", "fetch_talk", "find_handler",
           "is_url", "talk_paths"]  # fmt: skip


def is_url(text: str) -> bool:
    return re.match(r"(?i)https?://", text) is not None


def url_key(url: str) -> str:
    """ "host/path" of a URL, the part handler prefixes are matched against."""
    parts = urlsplit(url)
    return f"{parts.netloc.lower()}{parts.path}"


def find_handler(url: str) -> UrlHandler | None:
    key = url_key(url)
    return next((h for h in HANDLERS if any(key.startswith(p) for p in h.prefixes)), None)


def web() -> Http:
    return UrlHttp()


def describe_talk(url: str, http: Http | None = None) -> tuple[UrlHandler, Talk]:
    """The handler for `url` and the talk it finds (reads the page, downloads nothing)."""
    handler = find_handler(url)
    if handler is None:
        known = "\n".join(f"  {prefix}" for h in HANDLERS for prefix in h.prefixes)
        raise S2SError(f"No handler for this URL: {url}\nURLs that work start with:\n{known}")
    return handler, handler.describe(url, http or web())


def fetch_talk(url: str, folder: Path, http: Http | None = None) -> tuple[Talk, TalkFiles]:
    """Find the talk at `url` and save its audio and transcript in `folder`."""
    http = http or web()
    _, talk = describe_talk(url, http)
    return talk, save_talk(talk, folder, http)
