from __future__ import annotations

import re
from typing import Any
from urllib.parse import parse_qs, quote, urldefrag, urlparse

import requests

from ..base import MangaSource, SourceError
from ..models import Chapter, Manga, Page


class ExampleSource(MangaSource):
    """Reference adapter for a public JSON manifest served over HTTP(S)."""

    name = "example"

    def __init__(
        self,
        timeout: float = 20.0,
        session: requests.Session | None = None,
    ) -> None:
        self.timeout = timeout
        self.session = session or requests.Session()
        self._documents: dict[str, dict[str, Any]] = {}
        self._pages: dict[str, list[Page]] = {}

    def can_handle(self, url: str) -> bool:
        parsed = urlparse(url)
        hostname = parsed.hostname or ""
        return (
            parsed.scheme.lower() in {"http", "https"}
            and hostname != "mangadex.org"
            and not hostname.endswith(".mangadex.org")
        )

    def _load_document(self, url: str) -> dict[str, Any]:
        document_url, _ = urldefrag(url)
        if document_url in self._documents:
            return self._documents[document_url]
        if not self.can_handle(document_url):
            raise SourceError("ExampleSource only accepts HTTP(S) JSON manifests")

        try:
            response = self.session.get(document_url, timeout=self.timeout)
            response.raise_for_status()
            document = response.json()
        except requests.Timeout as exc:
            raise SourceError("Timed out while loading the JSON manifest") from exc
        except requests.RequestException as exc:
            raise SourceError(f"Cannot load the JSON manifest: {exc}") from exc
        except ValueError as exc:
            raise SourceError("ExampleSource expected a JSON document") from exc

        if not isinstance(document, dict):
            raise SourceError("Manifest root must be a JSON object")
        self._documents[document_url] = document
        return document

    @staticmethod
    def _required_text(data: dict[str, Any], key: str, context: str) -> str:
        value = data.get(key)
        if value is None or not str(value).strip():
            raise SourceError(f"Missing '{key}' in {context}")
        return str(value).strip()

    def get_manga(self, url: str) -> Manga:
        document_url, _ = urldefrag(url)
        document = self._load_document(document_url)
        manga_data = document.get("manga", document)
        if not isinstance(manga_data, dict):
            raise SourceError("Manifest 'manga' must be an object")
        return Manga(
            id=self._required_text(manga_data, "id", "manga manifest"),
            title=self._required_text(manga_data, "title", "manga manifest"),
            source_url=document_url,
            source=self.name,
        )

    def get_chapters(self, manga: Manga) -> list[Chapter]:
        document = self._load_document(manga.source_url)
        chapters_data = document.get("chapters")
        if not isinstance(chapters_data, list):
            raise SourceError("Manifest 'chapters' must be a list")

        chapters: list[Chapter] = []
        seen_ids: set[str] = set()
        for position, item in enumerate(chapters_data, start=1):
            if not isinstance(item, dict):
                raise SourceError(f"Chapter {position} must be an object")
            chapter_id = self._required_text(item, "id", f"chapter {position}")
            if chapter_id in seen_ids:
                raise SourceError(f"Duplicate chapter id '{chapter_id}'")
            seen_ids.add(chapter_id)

            chapter_url = str(
                item.get("url") or f"{manga.source_url}#chapter={quote(chapter_id, safe='')}"
            )
            chapter = Chapter(
                id=chapter_id,
                manga_id=manga.id,
                number=self._required_text(item, "number", f"chapter {chapter_id}"),
                title=str(item.get("title") or f"Chapter {item['number']}").strip(),
                url=chapter_url,
            )
            chapters.append(chapter)
            if "pages" in item:
                self._pages[chapter.url] = self._parse_pages(item["pages"], chapter.id)
        return chapters

    def _parse_pages(self, pages_data: Any, chapter_id: str) -> list[Page]:
        if not isinstance(pages_data, list) or not pages_data:
            raise SourceError(f"Chapter '{chapter_id}' must contain at least one page")

        pages: list[Page] = []
        seen_indexes: set[int] = set()
        for position, item in enumerate(pages_data, start=1):
            if isinstance(item, str):
                page = Page(index=position, image_url=item)
            elif isinstance(item, dict):
                try:
                    raw_index = item.get("index", position)
                    if isinstance(raw_index, bool):
                        raise ValueError
                    if isinstance(raw_index, str):
                        if not re.fullmatch(r"[+-]?\d+", raw_index.strip()):
                            raise ValueError
                    elif not isinstance(raw_index, int):
                        raise ValueError
                    page = Page(
                        index=int(raw_index),
                        image_url=self._required_text(item, "image_url", f"page {position}"),
                    )
                except (TypeError, ValueError) as exc:
                    raise SourceError(f"Invalid index for page {position}") from exc
            else:
                raise SourceError(f"Page {position} must be a URL string or object")

            if page.index < 0 or page.index in seen_indexes:
                raise SourceError(f"Invalid or duplicate page index {page.index}")
            if urlparse(page.image_url).scheme.lower() not in {"http", "https"}:
                raise SourceError(f"Page {position} must use HTTP(S)")
            seen_indexes.add(page.index)
            pages.append(page)
        return sorted(pages, key=lambda page: page.index)

    def get_pages(self, chapter: Chapter) -> list[Page]:
        if chapter.url in self._pages:
            return self._pages[chapter.url]

        document_url, fragment = urldefrag(chapter.url)
        document = self._load_document(document_url)
        if "pages" in document:
            pages = self._parse_pages(document["pages"], chapter.id)
            self._pages[chapter.url] = pages
            return pages

        manga = self.get_manga(document_url)
        chapters = self.get_chapters(manga)
        wanted_id = self._chapter_id_from_fragment(fragment) or chapter.id
        for candidate in chapters:
            if candidate.id == wanted_id:
                if candidate.url not in self._pages:
                    raise SourceError(f"Chapter '{wanted_id}' does not define pages")
                return self._pages[candidate.url]
        raise SourceError(f"Chapter '{wanted_id}' was not found in the manifest")

    @staticmethod
    def _chapter_id_from_fragment(fragment: str) -> str | None:
        if not fragment:
            return None
        values = parse_qs(fragment).get("chapter")
        return values[0] if values else None

    def get_chapter(self, url: str) -> tuple[Manga, Chapter]:
        document_url, fragment = urldefrag(url)
        document = self._load_document(document_url)
        if fragment:
            manga = self.get_manga(document_url)
            chapter_id = self._chapter_id_from_fragment(fragment)
            for chapter in self.get_chapters(manga):
                if chapter.id == chapter_id:
                    return manga, chapter
            raise SourceError(f"Chapter '{chapter_id}' was not found in the manifest")

        manga_data = document.get("manga")
        chapter_data = document.get("chapter")
        if not isinstance(manga_data, dict) or not isinstance(chapter_data, dict):
            raise SourceError(
                "Direct chapter documents need 'manga', 'chapter', and 'pages' objects"
            )
        manga = Manga(
            id=self._required_text(manga_data, "id", "chapter manga"),
            title=self._required_text(manga_data, "title", "chapter manga"),
            source_url=str(manga_data.get("source_url") or document_url),
            source=self.name,
        )
        chapter = Chapter(
            id=self._required_text(chapter_data, "id", "chapter document"),
            manga_id=manga.id,
            number=self._required_text(chapter_data, "number", "chapter document"),
            title=str(chapter_data.get("title") or "Chapter").strip(),
            url=document_url,
        )
        self._pages[chapter.url] = self._parse_pages(document.get("pages"), chapter.id)
        return manga, chapter
