from __future__ import annotations

import base64
import shutil
import tempfile
import unittest
from pathlib import Path
from uuid import UUID

import requests

from auto_manga.config import (
    AppConfig,
    ConfigError,
    DatabaseConfig,
    DownloadConfig,
    MangaDexConfig,
    SourcesConfig,
    StorageConfig,
    TranslationConfig,
    load_config,
)
from auto_manga.crawler.base import MangaSource, SourceError
from auto_manga.crawler.downloader import ChapterDownloader
from auto_manga.crawler.models import Chapter, Manga
from auto_manga.crawler.sources.example_source import ExampleSource
from auto_manga.crawler.sources.mangadex_source import (
    MangaDexNotFoundError,
    MangaDexRateLimitError,
    MangaDexResponseError,
    MangaDexSource,
    MangaDexUnavailableError,
    MangaDexUrlError,
    parse_mangadex_url,
)
from auto_manga.crawler.sources.registry import SourceRegistry
from auto_manga.pipeline.orchestrator import PipelineOrchestrator
from auto_manga.storage.database import Database

MANGA_ID = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
CHAPTER_ID = "11111111-2222-3333-4444-555555555555"
JPEG = base64.b64decode(
    "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAP//////////////////////////////////////////////////////////////////////////////////////"
    "2wBDAf//////////////////////////////////////////////////////////////////////////////////////"
    "wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAX/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/"
    "9oADAMBAAIQAxAAAAEf/8QAFBABAAAAAAAAAAAAAAAAAAAAAP/aAAgBAQABBQJ//8QAFBEBAAAAAAAAAAAAAAAA"
    "AAAAAP/aAAgBAwEBPwF//8QAFBEBAAAAAAAAAAAAAAAAAAAAAP/aAAgBAgEBPwF//8QAFBABAAAAAAAAAAAAAAAA"
    "AAAAAP/aAAgBAQAGPwJ//8QAFBABAAAAAAAAAAAAAAAAAAAAAP/aAAgBAQABPyF//9oADAMBAAIAAwAAABCf/8QA"
    "FBEBAAAAAAAAAAAAAAAAAAAAAP/aAAgBAwEBPxB//8QAFBEBAAAAAAAAAAAAAAAAAAAAAP/aAAgBAgEBPxB//8QA"
    "FhABAQEAAAAAAAAAAAAAAAAAABEh/9oACAEBAAE/EEf/2Q=="
)


class FakeResponse:
    def __init__(
        self,
        status: int,
        payload: object,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.status_code = status
        self.payload = payload
        self.headers = headers or {}
        self.closed = False

    def json(self) -> object:
        if isinstance(self.payload, ValueError):
            raise self.payload
        return self.payload

    def close(self) -> None:
        self.closed = True


class ScriptedSession:
    def __init__(self, *outcomes: FakeResponse | Exception) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[tuple[str, object, float]] = []
        self.headers: dict[str, str] = {}

    def get(self, url: str, params: object = None, timeout: float = 0) -> FakeResponse:
        self.calls.append((url, params, timeout))
        if not self.outcomes:
            raise AssertionError(f"Unexpected HTTP request: {url}")
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def manga_document(
    title: object = None,
    alt_titles: object = None,
    attributes: object = None,
) -> dict[str, object]:
    if attributes is None:
        attributes = {
            "title": title if title is not None else {"en": "English title"},
            "altTitles": alt_titles if alt_titles is not None else [],
        }
    return {
        "result": "ok",
        "data": {
            "id": MANGA_ID,
            "type": "manga",
            "attributes": attributes,
        },
    }


def chapter_entity(
    chapter_id: str,
    *,
    number: str | None = "1",
    volume: str | None = "1",
    language: str = "en",
    pages: int = 2,
    title: str | None = None,
    version: int = 1,
    readable_at: str = "2025-01-01T00:00:00+00:00",
    external_url: str | None = None,
    relationships: bool = False,
) -> dict[str, object]:
    entity: dict[str, object] = {
        "id": chapter_id,
        "type": "chapter",
        "attributes": {
            "chapter": number,
            "volume": volume,
            "translatedLanguage": language,
            "pages": pages,
            "title": title,
            "version": version,
            "readableAt": readable_at,
            "externalUrl": external_url,
        },
    }
    if relationships:
        entity["relationships"] = [{"id": MANGA_ID, "type": "manga"}]
    return entity


def feed_document(data: list[dict[str, object]], total: int | None = None) -> dict[str, object]:
    return {"result": "ok", "data": data, "total": len(data) if total is None else total}


def at_home_document(*, data: object = None, data_saver: object = None, hash_value: object = "abc123") -> dict[str, object]:
    return {
        "result": "ok",
        "baseUrl": "https://node.example.mangadex.network/token",
        "chapter": {
            "hash": hash_value,
            "data": ["1.png", "2.jpg"] if data is None else data,
            "dataSaver": ["1-small.jpg", "2-small.jpg"] if data_saver is None else data_saver,
        },
    }


def source_with(*responses: FakeResponse | Exception, **kwargs: object) -> MangaDexSource:
    return MangaDexSource(
        timeout=1,
        retries=int(kwargs.pop("retries", 0)),
        request_delay=0,
        at_home_delay=0,
        session=ScriptedSession(*responses),  # type: ignore[arg-type]
        **kwargs,
    )


class MangaDexUrlTest(unittest.TestCase):
    def test_manga_url_without_slug(self) -> None:
        result = parse_mangadex_url(f"https://mangadex.org/title/{MANGA_ID}")
        self.assertEqual((result.resource_type, result.id), ("manga", MANGA_ID))

    def test_manga_url_with_slug(self) -> None:
        result = parse_mangadex_url(
            f"https://mangadex.org/title/{MANGA_ID}/example-title"
        )
        self.assertEqual((result.resource_type, result.id), ("manga", MANGA_ID))

    def test_chapter_url(self) -> None:
        result = parse_mangadex_url(f"https://mangadex.org/chapter/{CHAPTER_ID}")
        self.assertEqual((result.resource_type, result.id), ("chapter", CHAPTER_ID))

    def test_raw_uuid_is_valid_but_ambiguous(self) -> None:
        result = parse_mangadex_url(CHAPTER_ID.upper())
        self.assertIsNone(result.resource_type)
        self.assertEqual(result.id, CHAPTER_ID)

    def test_rejects_malformed_uuid(self) -> None:
        with self.assertRaisesRegex(MangaDexUrlError, "Invalid MangaDex manga UUID"):
            parse_mangadex_url("https://mangadex.org/title/not-a-uuid")

    def test_rejects_wrong_domain(self) -> None:
        with self.assertRaisesRegex(MangaDexUrlError, "mangadex.org"):
            parse_mangadex_url(f"https://example.org/title/{MANGA_ID}")

    def test_rejects_unsupported_path(self) -> None:
        with self.assertRaisesRegex(MangaDexUrlError, "Unsupported MangaDex path"):
            parse_mangadex_url(f"https://mangadex.org/user/{MANGA_ID}")

    def test_rejects_invalid_port(self) -> None:
        with self.assertRaisesRegex(MangaDexUrlError, "invalid port"):
            parse_mangadex_url(f"https://mangadex.org:bad/title/{MANGA_ID}")


class MangaDexMangaTest(unittest.TestCase):
    def test_english_title_is_preferred(self) -> None:
        source = source_with(
            FakeResponse(200, manga_document({"vi": "Tiếng Việt", "en": "English"}))
        )
        manga = source.get_manga(MANGA_ID)
        self.assertEqual(manga.title, "English")
        self.assertEqual(manga.source, "mangadex")

    def test_vietnamese_title_is_used_when_english_is_missing(self) -> None:
        source = source_with(
            FakeResponse(
                200,
                manga_document({"ja": "日本語"}, [{"ja-ro": "Romaji"}, {"vi": "Việt"}]),
            )
        )
        self.assertEqual(source.get_manga(MANGA_ID).title, "Việt")

    def test_first_available_title_is_used_after_preferences(self) -> None:
        source = source_with(FakeResponse(200, manga_document({"fr": "Français"})))
        self.assertEqual(source.get_manga(MANGA_ID).title, "Français")

    def test_malformed_attributes_fall_back_to_uuid(self) -> None:
        source = source_with(FakeResponse(200, manga_document(attributes="bad")))
        self.assertEqual(source.get_manga(MANGA_ID).title, MANGA_ID)

    def test_http_404_is_specific(self) -> None:
        source = source_with(FakeResponse(404, {"result": "error"}))
        with self.assertRaisesRegex(MangaDexNotFoundError, "HTTP 404"):
            source.get_manga(MANGA_ID)


class MangaDexChapterListingTest(unittest.TestCase):
    def test_one_page_of_chapters(self) -> None:
        source = source_with(
            FakeResponse(200, feed_document([chapter_entity(CHAPTER_ID)]))
        )
        chapters = source.get_chapters(Manga(MANGA_ID, "Title", "url", "mangadex"))
        self.assertEqual([chapter.id for chapter in chapters], [CHAPTER_ID])
        self.assertEqual(chapters[0].number, "1")

    def test_multiple_api_pages_are_loaded(self) -> None:
        first = [chapter_entity(str(UUID(int=index + 1)), number=str(index + 1)) for index in range(100)]
        last = chapter_entity(str(UUID(int=101)), number="101")
        session = ScriptedSession(
            FakeResponse(200, feed_document(first, total=101)),
            FakeResponse(200, feed_document([last], total=101)),
        )
        source = MangaDexSource(
            request_delay=0, at_home_delay=0, session=session  # type: ignore[arg-type]
        )
        chapters = source.get_chapters(Manga(MANGA_ID, "Title", "url", "mangadex"))
        self.assertEqual(len(chapters), 101)
        self.assertEqual([call[1]["offset"] for call in session.calls], [0, 100])  # type: ignore[index]

    def test_decimal_and_null_chapter_numbers(self) -> None:
        special_id = "22222222-3333-4444-5555-666666666666"
        source = source_with(
            FakeResponse(
                200,
                feed_document(
                    [
                        chapter_entity(CHAPTER_ID, number="10.5"),
                        chapter_entity(special_id, number=None, title="Bonus"),
                    ]
                ),
            )
        )
        chapters = source.get_chapters(Manga(MANGA_ID, "Title", "url", "mangadex"))
        self.assertEqual(chapters[0].number, "10.5")
        self.assertEqual(chapters[1].number, "special-22222222")
        self.assertIn("Bonus", chapters[1].title)

    def test_duplicate_policy_prefers_highest_version_then_latest_release(self) -> None:
        old_id = "22222222-3333-4444-5555-666666666666"
        new_id = "33333333-4444-5555-6666-777777777777"
        source = source_with(
            FakeResponse(
                200,
                feed_document(
                    [
                        chapter_entity(old_id, version=1, readable_at="2025-12-01"),
                        chapter_entity(CHAPTER_ID, version=2, readable_at="2025-01-01"),
                        chapter_entity(new_id, version=2, readable_at="2025-02-01"),
                    ]
                ),
            )
        )
        chapters = source.get_chapters(Manga(MANGA_ID, "Title", "url", "mangadex"))
        self.assertEqual([chapter.id for chapter in chapters], [new_id])

    def test_languages_get_distinct_storage_keys(self) -> None:
        japanese_id = "22222222-3333-4444-5555-666666666666"
        source = source_with(
            FakeResponse(
                200,
                feed_document(
                    [
                        chapter_entity(CHAPTER_ID, language="en"),
                        chapter_entity(japanese_id, language="ja"),
                    ]
                ),
            )
        )
        chapters = source.get_chapters(Manga(MANGA_ID, "Title", "url", "mangadex"))
        self.assertEqual(len(chapters), 2)
        self.assertNotEqual(chapters[0].storage_key, chapters[1].storage_key)

    def test_no_chapters_returns_empty_list(self) -> None:
        source = source_with(FakeResponse(200, feed_document([])))
        self.assertEqual(
            source.get_chapters(Manga(MANGA_ID, "Title", "url", "mangadex")), []
        )

    def test_temporary_api_error_is_retried(self) -> None:
        source = source_with(
            FakeResponse(503, {"result": "error"}),
            FakeResponse(200, feed_document([])),
            retries=1,
        )
        self.assertEqual(
            source.get_chapters(Manga(MANGA_ID, "Title", "url", "mangadex")), []
        )

    def test_timeout_is_retried(self) -> None:
        source = source_with(
            requests.Timeout("temporary timeout"),
            FakeResponse(200, feed_document([])),
            retries=1,
        )
        self.assertEqual(
            source.get_chapters(Manga(MANGA_ID, "Title", "url", "mangadex")), []
        )

    def test_rate_limit_uses_retry_after_and_then_succeeds(self) -> None:
        source = source_with(
            FakeResponse(429, {"result": "error"}, {"Retry-After": "0"}),
            FakeResponse(200, feed_document([])),
            retries=1,
        )
        self.assertEqual(
            source.get_chapters(Manga(MANGA_ID, "Title", "url", "mangadex")), []
        )

    def test_persistent_rate_limit_is_specific(self) -> None:
        source = source_with(FakeResponse(429, {"result": "error"}))
        with self.assertRaises(MangaDexRateLimitError):
            source.get_chapters(Manga(MANGA_ID, "Title", "url", "mangadex"))


class MangaDexDirectChapterTest(unittest.TestCase):
    def test_direct_chapter_uses_real_manga_relationship(self) -> None:
        chapter_doc = {
            "result": "ok",
            "data": chapter_entity(CHAPTER_ID, number="10.5", relationships=True),
        }
        source = source_with(
            FakeResponse(200, chapter_doc),
            FakeResponse(200, manga_document({"en": "Related Manga"})),
        )
        manga, chapter = source.get_chapter(CHAPTER_ID)
        self.assertEqual(manga.id, MANGA_ID)
        self.assertEqual(chapter.manga_id, MANGA_ID)
        self.assertEqual(chapter.number, "10.5")

    def test_manga_url_is_rejected_by_chapter_command(self) -> None:
        source = source_with()
        with self.assertRaisesRegex(MangaDexUrlError, "manga URL"):
            source.get_chapter(f"https://mangadex.org/title/{MANGA_ID}")

    def test_external_chapter_is_not_downloadable(self) -> None:
        chapter_doc = {
            "result": "ok",
            "data": chapter_entity(
                CHAPTER_ID,
                external_url="https://publisher.example/chapter",
                relationships=True,
            ),
        }
        source = source_with(FakeResponse(200, chapter_doc))
        with self.assertRaisesRegex(MangaDexUnavailableError, "external website"):
            source.get_chapter(CHAPTER_ID)


class MangaDexPageTest(unittest.TestCase):
    def setUp(self) -> None:
        self.chapter = Chapter(CHAPTER_ID, MANGA_ID, "1", "One", "url")

    def test_full_resolution_pages_are_ordered(self) -> None:
        source = source_with(FakeResponse(200, at_home_document()))
        pages = source.get_pages(self.chapter)
        self.assertEqual([page.index for page in pages], [1, 2])
        self.assertTrue(pages[0].image_url.endswith("/data/abc123/1.png"))
        self.assertTrue(pages[1].image_url.endswith("/data/abc123/2.jpg"))

    def test_data_saver_mode(self) -> None:
        source = source_with(FakeResponse(200, at_home_document()), data_saver=True)
        pages = source.get_pages(self.chapter)
        self.assertTrue(pages[0].image_url.endswith("/data-saver/abc123/1-small.jpg"))

    def test_empty_page_list_is_rejected(self) -> None:
        source = source_with(FakeResponse(200, at_home_document(data=[])))
        with self.assertRaisesRegex(MangaDexUnavailableError, "no downloadable"):
            source.get_pages(self.chapter)

    def test_malformed_at_home_response_is_rejected(self) -> None:
        source = source_with(
            FakeResponse(200, {"result": "ok", "baseUrl": "not-a-url", "chapter": {}})
        )
        with self.assertRaisesRegex(MangaDexResponseError, "base URL"):
            source.get_pages(self.chapter)

    def test_missing_chapter_hash_is_rejected(self) -> None:
        source = source_with(FakeResponse(200, at_home_document(hash_value=None)))
        with self.assertRaisesRegex(MangaDexResponseError, "Missing chapter hash"):
            source.get_pages(self.chapter)


class MangaDexRegistryTest(unittest.TestCase):
    def test_manga_and_chapter_urls_select_mangadex(self) -> None:
        registry = SourceRegistry(delay=0)
        self.assertIsInstance(
            registry.for_url(f"https://mangadex.org/title/{MANGA_ID}"), MangaDexSource
        )
        self.assertIsInstance(
            registry.for_url(f"https://mangadex.org/chapter/{CHAPTER_ID}"), MangaDexSource
        )

    def test_raw_uuid_selects_mangadex(self) -> None:
        self.assertIsInstance(SourceRegistry(delay=0).for_url(MANGA_ID), MangaDexSource)

    def test_example_manifest_still_selects_example_source(self) -> None:
        self.assertIsInstance(
            SourceRegistry(delay=0).for_url("https://public.example/manifest.json"),
            ExampleSource,
        )

    def test_unsupported_input_has_a_useful_error(self) -> None:
        with self.assertRaisesRegex(SourceError, "No source adapter"):
            SourceRegistry(delay=0).for_url("not-a-url")
        source = SourceRegistry(delay=0).for_url("https://mangadex.org/user/me")
        with self.assertRaisesRegex(MangaDexUrlError, "Unsupported MangaDex path"):
            source.get_manga("https://mangadex.org/user/me")


class MangaDexConfigTest(unittest.TestCase):
    def test_source_config_is_loaded_and_normalized(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "config.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "storage:",
                        f"  raw: '{root / 'raw'}'",
                        f"  translated: '{root / 'translated'}'",
                        "download: {}",
                        "sources:",
                        "  mangadex:",
                        "    translated_languages: [EN, vi, en]",
                        "    data_saver: true",
                        "translation: {}",
                        "database:",
                        f"  path: '{root / 'state.db'}'",
                    ]
                ),
                encoding="utf-8",
            )

            config = load_config(config_path)

            self.assertEqual(config.sources.mangadex.translated_languages, ("en", "vi"))
            self.assertTrue(config.sources.mangadex.data_saver)

    def test_invalid_language_config_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "config.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "storage:",
                        f"  raw: '{root / 'raw'}'",
                        f"  translated: '{root / 'translated'}'",
                        "download: {}",
                        "sources:",
                        "  mangadex:",
                        "    translated_languages: [not_a_language]",
                        "translation: {}",
                        "database:",
                        f"  path: '{root / 'state.db'}'",
                    ]
                ),
                encoding="utf-8",
            )

            with self.assertRaises(ConfigError):
                load_config(config_path)


class ImageResponse:
    status_code = 200

    def __enter__(self) -> ImageResponse:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def raise_for_status(self) -> None:
        return None

    def iter_content(self, chunk_size: int) -> list[bytes]:
        del chunk_size
        return [JPEG]


class ImageSession:
    def __init__(self) -> None:
        self.urls: list[str] = []

    def get(self, url: str, **_kwargs: object) -> ImageResponse:
        self.urls.append(url)
        return ImageResponse()


class CopyTranslator:
    def translate_folder(self, input_path: Path, output_path: Path) -> None:
        output_path.mkdir(parents=True, exist_ok=True)
        for image in input_path.glob("*.jpg"):
            shutil.copyfile(image, output_path / image.name)


class OneSourceRegistry:
    def __init__(self, source: MangaSource) -> None:
        self.source = source

    def for_url(self, _url: str, _preferred: str | None = None) -> MangaSource:
        return self.source

    def get(self, _name: str) -> MangaSource:
        return self.source


class MangaDexPipelineIntegrationTest(unittest.TestCase):
    def test_chapter_url_reaches_downloader_database_and_translator(self) -> None:
        chapter_doc = {
            "result": "ok",
            "data": chapter_entity(CHAPTER_ID, relationships=True),
        }
        metadata_session = ScriptedSession(
            FakeResponse(200, chapter_doc),
            FakeResponse(200, manga_document({"en": "Pipeline Manga"})),
            FakeResponse(200, at_home_document(data=["1.png"])),
        )
        source = MangaDexSource(
            timeout=1,
            retries=0,
            request_delay=0,
            at_home_delay=0,
            session=metadata_session,  # type: ignore[arg-type]
        )
        image_session = ImageSession()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = AppConfig(
                storage=StorageConfig(root / "raw", root / "translated"),
                download=DownloadConfig(timeout=1, retries=0, delay=0, concurrency=1),
                translation=TranslationConfig("deepseek", "VIN"),
                database=DatabaseConfig(root / "state.db"),
                sources=SourcesConfig(MangaDexConfig(("en",), False)),
            )
            with Database(config.database.path) as database:
                pipeline = PipelineOrchestrator(
                    config,
                    database,
                    downloader=ChapterDownloader(
                        config.download, session=image_session  # type: ignore[arg-type]
                    ),
                    translator=CopyTranslator(),  # type: ignore[arg-type]
                    sources=OneSourceRegistry(source),  # type: ignore[arg-type]
                )
                summary = pipeline.run_chapter(
                    f"https://mangadex.org/chapter/{CHAPTER_ID}"
                )

                self.assertEqual((summary.translated, summary.failed), (1, 0))
                self.assertEqual(len(database.list_translated()), 1)
                self.assertEqual(database.list_resumable(), [])
                self.assertEqual(len(image_session.urls), 1)
                self.assertIn("/data/abc123/1.png", image_session.urls[0])


if __name__ == "__main__":
    unittest.main()
