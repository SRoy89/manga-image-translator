from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from email.utils import parsedate_to_datetime
from typing import Any, Literal
from urllib.parse import urlparse
from uuid import UUID

import requests

from ..base import MangaSource, SourceError
from ..models import Chapter, Manga, Page

LOGGER = logging.getLogger(__name__)
API_BASE_URL = "https://api.mangadex.org"
MANGADEX_HOSTS = {"mangadex.org", "www.mangadex.org"}
USER_AGENT = "auto-manga/0.1 (+https://github.com/zyddnys/manga-image-translator)"
_UUID = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)
_LANGUAGE = re.compile(r"^[a-z]{2,3}(?:-[a-z]{2})?$")
_CHAPTER_HASH = re.compile(r"^[0-9a-fA-F]+$")
_PAGE_FILENAME = re.compile(r"^[A-Za-z0-9._-]+$")


class MangaDexError(SourceError):
    """Base error for public MangaDex API operations."""


class MangaDexUrlError(MangaDexError):
    """Raised when a MangaDex URL or identifier is invalid."""


class MangaDexNotFoundError(MangaDexError):
    """Raised when MangaDex reports that a resource does not exist."""


class MangaDexRateLimitError(MangaDexError):
    """Raised after the limited retry budget for HTTP 429 is exhausted."""


class MangaDexUnavailableError(MangaDexError):
    """Raised when a chapter has no public MangaDex@Home pages."""


class MangaDexResponseError(MangaDexError):
    """Raised when an official endpoint returns malformed data."""


ResourceType = Literal["manga", "chapter"]


@dataclass(frozen=True)
class MangaDexReference:
    resource_type: ResourceType | None
    id: str


@dataclass(frozen=True)
class _ChapterRelease:
    chapter: Chapter
    volume: str | None
    source_number: str | None
    language: str
    version: int
    published_at: str


def _uuid(value: str, context: str) -> str:
    if not _UUID.fullmatch(value):
        raise MangaDexUrlError(f"Invalid MangaDex {context} UUID '{value}'")
    try:
        return str(UUID(value))
    except ValueError as exc:
        raise MangaDexUrlError(f"Invalid MangaDex {context} UUID '{value}'") from exc


def parse_mangadex_url(value: str) -> MangaDexReference:
    """Parse a canonical MangaDex website URL or an otherwise ambiguous raw UUID."""
    candidate = value.strip()
    if not candidate:
        raise MangaDexUrlError("MangaDex URL or UUID cannot be empty")
    if _UUID.fullmatch(candidate):
        return MangaDexReference(None, _uuid(candidate, "resource"))

    parsed = urlparse(candidate)
    if parsed.scheme.lower() != "https" or parsed.hostname not in MANGADEX_HOSTS:
        raise MangaDexUrlError("Expected an HTTPS URL on mangadex.org or a raw UUID")
    try:
        custom_port = parsed.port is not None
    except ValueError as exc:
        raise MangaDexUrlError("MangaDex URL contains an invalid port") from exc
    if parsed.username or parsed.password or custom_port:
        raise MangaDexUrlError("MangaDex URLs cannot contain credentials or custom ports")
    if parsed.query or parsed.fragment:
        raise MangaDexUrlError("MangaDex URLs cannot contain a query string or fragment")

    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) in {2, 3} and parts[0] == "title":
        if len(parts) == 3 and not parts[2].strip():
            raise MangaDexUrlError("MangaDex manga slug cannot be empty")
        return MangaDexReference("manga", _uuid(parts[1], "manga"))
    if len(parts) == 2 and parts[0] == "chapter":
        return MangaDexReference("chapter", _uuid(parts[1], "chapter"))
    raise MangaDexUrlError(f"Unsupported MangaDex path '{parsed.path or '/'}'")


class MangaDexSource(MangaSource):
    """Adapter for public metadata and MangaDex@Home delivery endpoints."""

    name = "mangadex"

    def __init__(
        self,
        timeout: float = 20.0,
        retries: int = 3,
        request_delay: float = 0.25,
        translated_languages: tuple[str, ...] = ("en", "ja", "ko", "zh", "zh-hk"),
        data_saver: bool = False,
        session: requests.Session | None = None,
        at_home_delay: float = 1.5,
    ) -> None:
        if timeout <= 0 or retries < 0 or request_delay < 0 or at_home_delay < 0:
            raise ValueError("Invalid MangaDex HTTP client settings")
        if not translated_languages:
            raise ValueError("MangaDex translated_languages cannot be empty")
        languages = tuple(language.strip().lower() for language in translated_languages)
        if any(not _LANGUAGE.fullmatch(language) for language in languages):
            raise ValueError("Invalid MangaDex translated language code")

        self.timeout = timeout
        self.retries = retries
        self.request_delay = request_delay
        self.at_home_delay = at_home_delay
        self.translated_languages = tuple(dict.fromkeys(languages))
        self.data_saver = data_saver
        self.session = session or requests.Session()
        headers = getattr(self.session, "headers", None)
        if headers is not None:
            headers.setdefault("Accept", "application/json")
            headers.setdefault("User-Agent", USER_AGENT)
        self._next_request = 0.0
        self._next_at_home_request = 0.0

    def can_handle(self, url: str) -> bool:
        try:
            parse_mangadex_url(url)
            return True
        except MangaDexUrlError:
            parsed = urlparse(url.strip())
            return parsed.scheme.lower() == "https" and parsed.hostname in MANGADEX_HOSTS

    def get_manga(self, url: str) -> Manga:
        reference = parse_mangadex_url(url)
        if reference.resource_type == "chapter":
            raise MangaDexUrlError("A MangaDex chapter URL cannot be used with the manga command")
        return self._get_manga_by_id(reference.id)

    def _get_manga_by_id(self, manga_id: str) -> Manga:
        manga_id = _uuid(manga_id, "manga")
        document = self._request_json(f"/manga/{manga_id}", "manga metadata")
        data = self._entity(document, "manga", manga_id)
        attributes = data.get("attributes")
        if not isinstance(attributes, dict):
            attributes = {}
        return Manga(
            id=manga_id,
            title=self._choose_title(attributes, manga_id),
            source_url=f"https://mangadex.org/title/{manga_id}",
            source=self.name,
        )

    def get_chapters(self, manga: Manga) -> list[Chapter]:
        manga_id = _uuid(manga.id, "manga")
        raw_chapters = self._get_paginated_feed(manga_id)
        releases: list[_ChapterRelease] = []
        seen_ids: set[str] = set()
        for position, item in enumerate(raw_chapters, start=1):
            try:
                release = self._release_from_entity(item, manga_id)
            except MangaDexUnavailableError as exc:
                LOGGER.debug("Skipping unavailable MangaDex release %s: %s", position, exc)
                continue
            if release.chapter.id in seen_ids:
                continue
            seen_ids.add(release.chapter.id)
            releases.append(release)

        selected: dict[tuple[str, str, str], _ChapterRelease] = {}
        specials: list[_ChapterRelease] = []
        for release in releases:
            if release.source_number is None:
                specials.append(release)
                continue
            key = (release.volume or "", release.source_number, release.language)
            current = selected.get(key)
            if current is None or self._release_is_better(release, current):
                selected[key] = release

        chosen = [*selected.values(), *specials]
        chosen.sort(key=self._release_sort_key)
        return [release.chapter for release in chosen]

    def _get_paginated_feed(self, manga_id: str) -> list[dict[str, Any]]:
        limit = 100
        offset = 0
        chapters: list[dict[str, Any]] = []
        while True:
            params: dict[str, Any] = {
                "limit": limit,
                "offset": offset,
                "translatedLanguage[]": list(self.translated_languages),
                "includeExternalUrl": 0,
                "includeEmptyPages": 0,
                "order[volume]": "asc",
                "order[chapter]": "asc",
                "order[readableAt]": "asc",
            }
            document = self._request_json(
                f"/manga/{manga_id}/feed", "manga chapter feed", params=params
            )
            page = document.get("data")
            total = document.get("total")
            if not isinstance(page, list) or isinstance(total, bool) or not isinstance(total, int):
                raise MangaDexResponseError(
                    f"Malformed chapter feed for manga {manga_id}: missing data or total"
                )
            if total < 0 or total > 10_000:
                raise MangaDexResponseError(
                    f"Unsupported chapter feed size {total} for manga {manga_id}"
                )
            if any(not isinstance(item, dict) for item in page):
                raise MangaDexResponseError(
                    f"Malformed chapter entity in feed for manga {manga_id}"
                )
            chapters.extend(page)
            offset += len(page)
            if offset >= total:
                return chapters
            if not page:
                raise MangaDexResponseError(
                    f"Chapter feed for manga {manga_id} ended before its reported total"
                )

    def get_chapter(self, url: str) -> tuple[Manga, Chapter]:
        reference = parse_mangadex_url(url)
        if reference.resource_type == "manga":
            raise MangaDexUrlError("A MangaDex manga URL cannot be used with the chapter command")
        chapter_id = reference.id
        document = self._request_json(
            f"/chapter/{chapter_id}",
            "chapter metadata",
            params={"includes[]": "manga"},
        )
        data = self._entity(document, "chapter", chapter_id)
        manga_id = self._related_manga_id(data, chapter_id)
        release = self._release_from_entity(data, manga_id)
        return self._get_manga_by_id(manga_id), release.chapter

    def get_pages(self, chapter: Chapter) -> list[Page]:
        chapter_id = _uuid(chapter.id, "chapter")
        document = self._request_json(
            f"/at-home/server/{chapter_id}", "MangaDex@Home page metadata"
        )
        base_url = document.get("baseUrl")
        chapter_data = document.get("chapter")
        if not isinstance(base_url, str) or not self._valid_base_url(base_url):
            raise MangaDexResponseError(
                f"Malformed MangaDex@Home base URL for chapter {chapter_id}"
            )
        if not isinstance(chapter_data, dict):
            raise MangaDexResponseError(
                f"Malformed MangaDex@Home chapter data for chapter {chapter_id}"
            )
        chapter_hash = chapter_data.get("hash")
        filenames_key = "dataSaver" if self.data_saver else "data"
        filenames = chapter_data.get(filenames_key)
        if not isinstance(chapter_hash, str) or not _CHAPTER_HASH.fullmatch(chapter_hash):
            raise MangaDexResponseError(f"Missing chapter hash for chapter {chapter_id}")
        if not isinstance(filenames, list) or not filenames:
            raise MangaDexUnavailableError(
                f"Chapter {chapter_id} has no downloadable MangaDex@Home pages"
            )
        if any(not self._valid_page_filename(filename) for filename in filenames):
            raise MangaDexResponseError(
                f"Malformed MangaDex@Home page filename for chapter {chapter_id}"
            )

        quality = "data-saver" if self.data_saver else "data"
        root = base_url.rstrip("/")
        return [
            Page(index, f"{root}/{quality}/{chapter_hash}/{filename}")
            for index, filename in enumerate(filenames, start=1)
        ]

    def _request_json(
        self,
        path: str,
        category: str,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        url = f"{API_BASE_URL}{path}"
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            self._wait_for_request(category)
            response: requests.Response | None = None
            try:
                LOGGER.debug("MangaDex API request: %s", category)
                response = self.session.get(url, params=params, timeout=self.timeout)
                status = response.status_code
                if status == 404:
                    raise MangaDexNotFoundError(f"{category} not found (HTTP 404): {path}")
                if status == 429:
                    if attempt >= self.retries:
                        raise MangaDexRateLimitError(
                            f"MangaDex rate limit persisted for {category} (HTTP 429)"
                        )
                    delay = self._retry_delay(response, attempt)
                    LOGGER.warning(
                        "MangaDex rate limited %s; retrying in %.1f seconds",
                        category,
                        delay,
                    )
                    time.sleep(delay)
                    continue
                if 500 <= status < 600:
                    if attempt >= self.retries:
                        raise MangaDexError(
                            f"MangaDex temporary failure for {category} (HTTP {status})"
                        )
                    delay = self._retry_delay(response, attempt)
                    LOGGER.warning(
                        "MangaDex temporary failure for %s (HTTP %s); retrying in %.1f seconds",
                        category,
                        status,
                        delay,
                    )
                    time.sleep(delay)
                    continue
                if status < 200 or status >= 300:
                    raise MangaDexError(f"MangaDex {category} failed (HTTP {status}): {path}")
                try:
                    document = response.json()
                except ValueError as exc:
                    raise MangaDexResponseError(
                        f"MangaDex returned invalid JSON for {category}"
                    ) from exc
                if not isinstance(document, dict) or document.get("result") != "ok":
                    raise MangaDexResponseError(
                        f"MangaDex returned a malformed response for {category}"
                    )
                return document
            except MangaDexError:
                raise
            except (requests.Timeout, requests.ConnectionError) as exc:
                last_error = exc
                if attempt >= self.retries:
                    break
                delay = max(self.request_delay, 0.5 * (2**attempt))
                LOGGER.warning(
                    "MangaDex connection error for %s; retrying in %.1f seconds",
                    category,
                    delay,
                )
                time.sleep(delay)
            except requests.RequestException as exc:
                raise MangaDexError(f"MangaDex request failed for {category}: {exc}") from exc
            finally:
                if response is not None:
                    response.close()
        error_name = type(last_error).__name__ if last_error else "network error"
        raise MangaDexError(
            f"MangaDex request failed for {category} after retries ({error_name})"
        ) from last_error

    def _wait_for_request(self, category: str) -> None:
        now = time.monotonic()
        next_request = self._next_request
        if category == "MangaDex@Home page metadata":
            next_request = max(next_request, self._next_at_home_request)
        wait_for = max(0.0, next_request - now)
        if wait_for:
            time.sleep(wait_for)
        now = time.monotonic()
        self._next_request = now + self.request_delay
        if category == "MangaDex@Home page metadata":
            self._next_at_home_request = now + self.at_home_delay

    def _retry_delay(self, response: requests.Response, attempt: int) -> float:
        retry_after = response.headers.get("Retry-After")
        parsed = self._parse_retry_after(retry_after)
        if parsed is None:
            reset = response.headers.get("X-RateLimit-Retry-After")
            try:
                parsed = max(0.0, float(reset) - time.time()) if reset is not None else None
            except ValueError:
                parsed = None
        return parsed if parsed is not None else max(self.request_delay, 0.5 * (2**attempt))

    @staticmethod
    def _parse_retry_after(value: str | None) -> float | None:
        if value is None:
            return None
        try:
            return max(0.0, float(value))
        except ValueError:
            try:
                moment = parsedate_to_datetime(value)
                if moment.tzinfo is None:
                    moment = moment.replace(tzinfo=timezone.utc)
                return max(0.0, (moment - datetime.now(timezone.utc)).total_seconds())
            except (TypeError, ValueError, OverflowError):
                return None

    @staticmethod
    def _entity(document: dict[str, Any], expected_type: str, expected_id: str) -> dict[str, Any]:
        data = document.get("data")
        if not isinstance(data, dict):
            raise MangaDexResponseError(f"Missing {expected_type} entity {expected_id}")
        entity_id = data.get("id")
        try:
            normalized_id = _uuid(str(entity_id), expected_type)
        except MangaDexUrlError as exc:
            raise MangaDexResponseError(
                f"Malformed {expected_type} ID in response for {expected_id}"
            ) from exc
        if normalized_id != expected_id or data.get("type") not in {None, expected_type}:
            raise MangaDexResponseError(f"Unexpected {expected_type} entity for {expected_id}")
        return data

    @staticmethod
    def _choose_title(attributes: dict[str, Any], fallback: str) -> str:
        candidates: list[dict[str, Any]] = []
        title = attributes.get("title")
        if isinstance(title, dict):
            candidates.append(title)
        alt_titles = attributes.get("altTitles")
        if isinstance(alt_titles, list):
            candidates.extend(item for item in alt_titles if isinstance(item, dict))

        for language in ("en", "vi", "ja-ro"):
            for candidate in candidates:
                value = candidate.get(language)
                if isinstance(value, str) and value.strip():
                    return value.strip()
        for candidate in candidates:
            for value in candidate.values():
                if isinstance(value, str) and value.strip():
                    return value.strip()
        return fallback

    def _release_from_entity(
        self, data: dict[str, Any], manga_id: str
    ) -> _ChapterRelease:
        if data.get("type") not in {None, "chapter"}:
            raise MangaDexResponseError("MangaDex feed contained a non-chapter entity")
        chapter_id_value = data.get("id")
        try:
            chapter_id = _uuid(str(chapter_id_value), "chapter")
        except MangaDexUrlError as exc:
            raise MangaDexResponseError("Malformed chapter ID in MangaDex feed") from exc
        attributes = data.get("attributes")
        if not isinstance(attributes, dict):
            raise MangaDexResponseError(f"Malformed chapter attributes for {chapter_id}")

        external_url = attributes.get("externalUrl")
        pages = attributes.get("pages")
        if isinstance(external_url, str) and external_url.strip():
            raise MangaDexUnavailableError(
                f"Chapter {chapter_id} is hosted on an external website"
            )
        if isinstance(pages, bool) or not isinstance(pages, int) or pages < 1:
            raise MangaDexUnavailableError(f"Chapter {chapter_id} has no public pages")

        language_value = attributes.get("translatedLanguage")
        language = language_value.strip().lower() if isinstance(language_value, str) else "und"
        if language != "und" and not _LANGUAGE.fullmatch(language):
            raise MangaDexResponseError(f"Malformed language for chapter {chapter_id}")
        source_number = self._optional_text(attributes.get("chapter"))
        volume = self._optional_text(attributes.get("volume"))
        display_number = source_number or f"special-{chapter_id[:8]}"
        chapter_title = self._chapter_title(
            volume, source_number, self._optional_text(attributes.get("title"))
        )
        storage_key = self._storage_key(chapter_id, volume, source_number, language)
        version_value = attributes.get("version", 0)
        version = (
            version_value
            if isinstance(version_value, int) and not isinstance(version_value, bool)
            else 0
        )
        published_at = self._optional_text(
            attributes.get("readableAt")
            or attributes.get("publishAt")
            or attributes.get("createdAt")
        ) or ""
        chapter = Chapter(
            id=chapter_id,
            manga_id=manga_id,
            number=display_number,
            title=chapter_title,
            url=f"https://mangadex.org/chapter/{chapter_id}",
            storage_key=storage_key,
        )
        return _ChapterRelease(
            chapter=chapter,
            volume=volume,
            source_number=source_number,
            language=language,
            version=version,
            published_at=published_at,
        )

    @staticmethod
    def _related_manga_id(data: dict[str, Any], chapter_id: str) -> str:
        relationships = data.get("relationships")
        if not isinstance(relationships, list):
            raise MangaDexResponseError(
                f"Chapter {chapter_id} response has no manga relationship"
            )
        for relationship in relationships:
            if isinstance(relationship, dict) and relationship.get("type") == "manga":
                try:
                    return _uuid(str(relationship.get("id")), "manga")
                except MangaDexUrlError as exc:
                    raise MangaDexResponseError(
                        f"Chapter {chapter_id} has a malformed manga relationship"
                    ) from exc
        raise MangaDexResponseError(f"Chapter {chapter_id} has no manga relationship")

    @staticmethod
    def _release_is_better(candidate: _ChapterRelease, current: _ChapterRelease) -> bool:
        if candidate.version != current.version:
            return candidate.version > current.version
        if candidate.published_at != current.published_at:
            return candidate.published_at > current.published_at
        return candidate.chapter.id < current.chapter.id

    def _release_sort_key(self, release: _ChapterRelease) -> tuple[object, ...]:
        try:
            language_index = self.translated_languages.index(release.language)
        except ValueError:
            language_index = len(self.translated_languages)
        return (
            self._number_sort_key(release.volume),
            self._number_sort_key(release.source_number),
            language_index,
            release.published_at,
            release.chapter.id,
        )

    @staticmethod
    def _number_sort_key(value: str | None) -> tuple[int, object]:
        if value is None:
            return (2, "")
        try:
            number = Decimal(value)
            if number.is_finite():
                return (0, number)
        except InvalidOperation:
            pass
        return (1, value.casefold())

    @staticmethod
    def _optional_text(value: Any) -> str | None:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, (str, int, float)):
            return None
        text = str(value).strip()
        return text or None

    @staticmethod
    def _chapter_title(
        volume: str | None, number: str | None, title: str | None
    ) -> str:
        parts: list[str] = []
        if volume:
            parts.append(f"Vol. {volume}")
        parts.append(f"Ch. {number}" if number else "Special")
        if title:
            parts.append(title)
        return " — ".join(parts)

    @staticmethod
    def _storage_key(
        chapter_id: str,
        volume: str | None,
        number: str | None,
        language: str,
    ) -> str:
        parts = [f"v{volume}"] if volume else []
        parts.append(f"c{number}" if number else f"special-{chapter_id[:8]}")
        parts.append(language)
        return "-".join(parts)

    @staticmethod
    def _valid_base_url(value: str) -> bool:
        parsed = urlparse(value)
        try:
            parsed.port
        except ValueError:
            return False
        return (
            parsed.scheme == "https"
            and bool(parsed.hostname)
            and parsed.username is None
            and parsed.password is None
            and not parsed.query
            and not parsed.fragment
        )

    @staticmethod
    def _valid_page_filename(value: Any) -> bool:
        return (
            isinstance(value, str)
            and value not in {".", ".."}
            and bool(_PAGE_FILENAME.fullmatch(value))
        )
