"""Talks given as web pages: what a URL handler finds on a page, and saving its files.

A handler turns a page URL into a `Talk` (title, speaker, audio URL, published text);
`save_talk` downloads the audio and writes the text next to it. Network access goes
through `Http`, so handlers can be tested with canned pages.
"""

import shutil
import urllib.error
import urllib.request
from dataclasses import dataclass
from http.client import HTTPResponse
from pathlib import Path, PurePosixPath
from typing import Protocol
from urllib.parse import urlsplit

from speech2song import __version__
from speech2song.errors import S2SError
from speech2song.manifest import atomic_write_text, slugify

USER_AGENT = f"speech2song/{__version__}"
STEM_MAX = 80


@dataclass(frozen=True)
class Talk:
    page_url: str
    title: str
    speaker: str | None
    audio_url: str
    text: str  # the published transcript: one paragraph or verse line per block

    @property
    def stem(self) -> str:
        """File name stem, e.g. "come-home-by-elder-clark-g-gilbert"."""
        name = f"{self.title} by {self.speaker}" if self.speaker else self.title
        return slugify(name, max_len=STEM_MAX)

    @property
    def audio_suffix(self) -> str:
        return PurePosixPath(urlsplit(self.audio_url).path).suffix.lower() or ".mp3"


class Http(Protocol):
    def get(self, url: str) -> bytes: ...
    def download(self, url: str, dest: Path) -> None: ...


class UrlHttp:
    """Plain HTTPS through urllib (no extra dependency)."""

    timeout_s = 60.0

    def _open(self, url: str) -> HTTPResponse:
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            return urllib.request.urlopen(request, timeout=self.timeout_s)
        except urllib.error.HTTPError as exc:
            raise S2SError(f"Couldn't fetch {url}: HTTP {exc.code} {exc.reason}") from exc
        except (urllib.error.URLError, OSError) as exc:
            raise S2SError(f"Couldn't fetch {url}: {exc}") from exc

    def get(self, url: str) -> bytes:
        with self._open(url) as response:
            return response.read()

    def download(self, url: str, dest: Path) -> None:
        with self._open(url) as response, dest.open("xb") as fh:
            shutil.copyfileobj(response, fh, length=1 << 20)


class UrlHandler(Protocol):
    name: str
    prefixes: tuple[str, ...]  # host + path, without the scheme

    def describe(self, url: str, http: Http) -> Talk: ...


@dataclass(frozen=True)
class TalkFiles:
    audio: Path
    transcript: Path
    written: tuple[Path, ...]  # files this call wrote; the others were already there


def talk_paths(talk: Talk, folder: Path) -> tuple[Path, Path]:
    return folder / f"{talk.stem}{talk.audio_suffix}", folder / f"{talk.stem}.txt"


def save_talk(talk: Talk, folder: Path, http: Http) -> TalkFiles:
    """Download the audio and write the transcript into `folder`. A file that already
    exists is kept as it is (it may have been edited), so a talk is downloaded once."""
    folder.mkdir(parents=True, exist_ok=True)
    audio, transcript = talk_paths(talk, folder)
    written: list[Path] = []
    if not audio.exists():
        partial = audio.with_name(f".{audio.name}.part")
        partial.unlink(missing_ok=True)
        try:
            http.download(talk.audio_url, partial)
            if partial.stat().st_size == 0:
                raise S2SError(f"The audio download was empty: {talk.audio_url}")
            partial.replace(audio)
        finally:
            partial.unlink(missing_ok=True)
        written.append(audio)
    if not transcript.exists():
        atomic_write_text(transcript, talk.text)
        written.append(transcript)
    return TalkFiles(audio, transcript, tuple(written))
