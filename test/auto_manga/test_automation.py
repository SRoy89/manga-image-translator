from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import unittest
from io import BytesIO
from pathlib import Path

import requests
from PIL import Image

from auto_manga.config import (
    AppConfig,
    ConfigError,
    DatabaseConfig,
    DownloadConfig,
    StorageConfig,
    TranslationConfig,
    load_config,
)
from auto_manga.crawler.base import MangaSource, SourceError
from auto_manga.crawler.downloader import (
    ChapterDownloader,
    DownloadError,
    validate_chapter_images,
)
from auto_manga.crawler.models import Chapter, Manga, Page
from auto_manga.crawler.sources.example_source import ExampleSource
from auto_manga.main import build_parser, main
from auto_manga.pipeline.orchestrator import PipelineOrchestrator
from auto_manga.pipeline.translator import (
    MangaTranslator,
    TranslationError,
    validate_translated_output,
)
from auto_manga.storage.database import Database
from auto_manga.storage.paths import chapter_paths, sanitize_name


def jpeg_bytes(color: str = "white") -> bytes:
    output = BytesIO()
    Image.new("RGB", (4, 4), color=color).save(output, format="JPEG")
    return output.getvalue()


class FakeImageResponse:
    status_code = 200

    def __init__(self, content: bytes) -> None:
        self.content = content

    def __enter__(self) -> FakeImageResponse:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def raise_for_status(self) -> None:
        return None

    def iter_content(self, chunk_size: int) -> list[bytes]:
        return [self.content[index : index + chunk_size] for index in range(0, len(self.content), chunk_size)]


class FakeDownloadSession:
    def __init__(self, content: bytes, fail_first: bool = False) -> None:
        self.content = content
        self.fail_first = fail_first
        self.calls = 0

    def get(self, _url: str, **_kwargs: object) -> FakeImageResponse:
        self.calls += 1
        if self.fail_first and self.calls == 1:
            raise requests.Timeout("temporary timeout")
        return FakeImageResponse(self.content)


class AlwaysFailDownloadSession:
    def __init__(self) -> None:
        self.calls = 0

    def get(self, _url: str, **_kwargs: object) -> FakeImageResponse:
        self.calls += 1
        raise requests.ConnectionError("connection reset")


class MappingDownloadSession:
    def __init__(self, content_by_url: dict[str, bytes]) -> None:
        self.content_by_url = content_by_url
        self.calls: list[str] = []

    def get(self, url: str, **_kwargs: object) -> FakeImageResponse:
        self.calls.append(url)
        return FakeImageResponse(self.content_by_url[url])


class FakeJsonResponse:
    def __init__(self, document: dict[str, object]) -> None:
        self.document = document

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, object]:
        return self.document


class FakeJsonSession:
    def __init__(self, document: dict[str, object]) -> None:
        self.document = document

    def get(self, _url: str, **_kwargs: object) -> FakeJsonResponse:
        return FakeJsonResponse(self.document)


class ExampleSourceTest(unittest.TestCase):
    def test_reads_reference_manifest(self) -> None:
        manifest = {
            "id": "manga-1",
            "title": "Public Manga",
            "chapters": [
                {
                    "id": "c1",
                    "number": "1",
                    "title": "Start",
                    "pages": ["https://public.example/1.jpg", "https://public.example/2.jpg"],
                }
            ],
        }
        source = ExampleSource(session=FakeJsonSession(manifest))  # type: ignore[arg-type]
        manga = source.get_manga("https://public.example/manifest.json")
        chapters = source.get_chapters(manga)
        pages = source.get_pages(chapters[0])

        self.assertEqual(manga.title, "Public Manga")
        self.assertEqual(chapters[0].number, "1")
        self.assertEqual([page.index for page in pages], [1, 2])

    def test_rejects_duplicate_chapter_ids(self) -> None:
        manifest = {
            "id": "manga-1",
            "title": "Public Manga",
            "chapters": [
                {"id": "same", "number": "1", "pages": ["https://example.test/1.jpg"]},
                {"id": "same", "number": "2", "pages": ["https://example.test/2.jpg"]},
            ],
        }
        source = ExampleSource(session=FakeJsonSession(manifest))  # type: ignore[arg-type]

        with self.assertRaisesRegex(SourceError, "Duplicate chapter id"):
            source.get_chapters(source.get_manga("https://example.test/manifest.json"))

    def test_encoded_chapter_id_round_trips_through_fragment(self) -> None:
        manifest = {
            "id": "manga-1",
            "title": "Public Manga",
            "chapters": [
                {"id": "%2F", "number": "1", "pages": ["https://example.test/1.jpg"]}
            ],
        }
        source = ExampleSource(session=FakeJsonSession(manifest))  # type: ignore[arg-type]
        manga = source.get_manga("https://example.test/manifest.json")
        chapter = source.get_chapters(manga)[0]

        _resolved_manga, resolved_chapter = source.get_chapter(chapter.url)

        self.assertEqual(resolved_chapter.id, "%2F")

    def test_rejects_fractional_page_index_instead_of_truncating_it(self) -> None:
        manifest = {
            "id": "manga-1",
            "title": "Public Manga",
            "chapters": [
                {
                    "id": "c1",
                    "number": "1",
                    "pages": [
                        {"index": 1.5, "image_url": "https://example.test/1.jpg"}
                    ],
                }
            ],
        }
        source = ExampleSource(session=FakeJsonSession(manifest))  # type: ignore[arg-type]

        with self.assertRaisesRegex(SourceError, "Invalid index"):
            source.get_chapters(source.get_manga("https://example.test/manifest.json"))


class DownloaderTest(unittest.TestCase):
    def test_successful_download_uses_page_index_order(self) -> None:
        first_url = "https://public.example/first.jpg"
        second_url = "https://public.example/second.jpg"
        first_image = jpeg_bytes("red")
        second_image = jpeg_bytes("blue")
        session = MappingDownloadSession({first_url: first_image, second_url: second_image})
        downloader = ChapterDownloader(
            DownloadConfig(timeout=1, retries=0, delay=0, concurrency=1),
            session=session,  # type: ignore[arg-type]
        )

        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory)
            downloader.download_chapter(
                [Page(20, second_url), Page(10, first_url)], destination
            )

            self.assertEqual((destination / "001.jpg").read_bytes(), first_image)
            self.assertEqual((destination / "002.jpg").read_bytes(), second_image)
            self.assertTrue(validate_chapter_images(destination, expected_count=2))

    def test_retries_skips_existing_and_handles_duplicate_urls(self) -> None:
        session = FakeDownloadSession(jpeg_bytes(), fail_first=True)
        config = DownloadConfig(timeout=1, retries=2, delay=0, concurrency=2)
        downloader = ChapterDownloader(config, session=session)  # type: ignore[arg-type]
        pages = [
            Page(1, "https://public.example/page.jpg"),
            Page(2, "https://public.example/page.jpg"),
        ]

        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory)
            downloader.download_chapter(pages, destination)
            calls_after_first_run = session.calls
            downloader.download_chapter(pages, destination)

            self.assertTrue(validate_chapter_images(destination, expected_count=2))
            self.assertEqual(calls_after_first_run, 2)
            self.assertEqual(session.calls, calls_after_first_run)

    def test_failed_image_download_retries_then_leaves_invalid_chapter(self) -> None:
        session = AlwaysFailDownloadSession()
        downloader = ChapterDownloader(
            DownloadConfig(timeout=1, retries=2, delay=0, concurrency=1),
            session=session,  # type: ignore[arg-type]
        )

        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory)
            with self.assertRaisesRegex(DownloadError, "image 1"):
                downloader.download_chapter(
                    [Page(1, "https://public.example/page.jpg")], destination
                )

            self.assertEqual(session.calls, 3)
            self.assertFalse(validate_chapter_images(destination, expected_count=1))

    def test_existing_zero_byte_image_is_replaced(self) -> None:
        session = FakeDownloadSession(jpeg_bytes())
        downloader = ChapterDownloader(
            DownloadConfig(timeout=1, retries=0, delay=0, concurrency=1),
            session=session,  # type: ignore[arg-type]
        )

        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory)
            (destination / "001.jpg").write_bytes(b"")
            downloader.download_chapter(
                [Page(1, "https://public.example/page.jpg")], destination
            )

            self.assertEqual(session.calls, 1)
            self.assertTrue(validate_chapter_images(destination, expected_count=1))

    def test_zero_byte_response_is_retried_and_rejected(self) -> None:
        session = FakeDownloadSession(b"")
        downloader = ChapterDownloader(
            DownloadConfig(timeout=1, retries=1, delay=0, concurrency=1),
            session=session,  # type: ignore[arg-type]
        )

        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(DownloadError):
                downloader.download_chapter(
                    [Page(1, "https://public.example/page.jpg")], Path(directory)
                )
            self.assertEqual(session.calls, 2)

    def test_corrupt_nonempty_response_is_retried_and_rejected(self) -> None:
        session = FakeDownloadSession(b"this is not an image")
        downloader = ChapterDownloader(
            DownloadConfig(timeout=1, retries=1, delay=0, concurrency=1),
            session=session,  # type: ignore[arg-type]
        )

        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(DownloadError):
                downloader.download_chapter(
                    [Page(1, "https://public.example/page.jpg")], Path(directory)
                )
            self.assertEqual(session.calls, 2)

    def test_stale_extra_page_is_removed_when_resuming_download(self) -> None:
        first_url = "https://public.example/1.jpg"
        second_url = "https://public.example/2.jpg"
        session = MappingDownloadSession(
            {first_url: jpeg_bytes("red"), second_url: jpeg_bytes("blue")}
        )
        downloader = ChapterDownloader(
            DownloadConfig(timeout=1, retries=0, delay=0, concurrency=1),
            session=session,  # type: ignore[arg-type]
        )

        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory)
            for index in (1, 2, 3):
                (destination / f"{index:03d}.jpg").write_bytes(jpeg_bytes())

            downloader.download_chapter(
                [Page(1, first_url), Page(2, second_url)], destination
            )

            self.assertFalse((destination / "003.jpg").exists())
            self.assertTrue(validate_chapter_images(destination, expected_count=2))


class DatabaseResumeTest(unittest.TestCase):
    def test_recovers_interrupted_states_and_excludes_translated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with Database(root / "state.db") as database:
                manga = Manga("m1", "Manga", "https://public.example/manga.json")
                manga_id = database.upsert_manga(manga)
                statuses = ["downloading", "translating", "failed", "translated"]
                for number, status in enumerate(statuses, start=1):
                    chapter = Chapter(
                        f"c{number}",
                        "m1",
                        str(number),
                        "Chapter",
                        f"https://public.example/c{number}.json",
                    )
                    record = database.upsert_chapter(
                        manga_id,
                        chapter,
                        root / "raw" / str(number),
                        root / "translated" / str(number),
                    )
                    database.set_status(record.id, status)

                self.assertEqual(database.recover_interrupted(), 2)
                resumable = database.list_resumable()

                self.assertEqual(
                    [record.status for record in resumable],
                    ["pending", "downloaded", "failed"],
                )

    def test_rejects_two_chapters_that_share_storage_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with Database(root / "state.db") as database:
                manga = Manga("m1", "Manga", "https://public.example/manga.json")
                manga_id = database.upsert_manga(manga)
                raw = root / "raw" / "same"
                translated = root / "translated" / "same"
                database.upsert_chapter(
                    manga_id,
                    Chapter("c1", "m1", "1", "One", "https://example.test/c1"),
                    raw,
                    translated,
                )

                with self.assertRaisesRegex(ValueError, "storage path"):
                    database.upsert_chapter(
                        manga_id,
                        Chapter("c2", "m1", "1.0", "Other", "https://example.test/c2"),
                        raw,
                        translated,
                    )


class StoragePathTest(unittest.TestCase):
    def test_source_titles_cannot_escape_storage_roots(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manga = Manga("m1", "../../outside", "https://public.example/manga.json")
            chapter = Chapter("c1", "m1", "../1", "Chapter", "https://public.example/c1.json")
            raw, translated = chapter_paths(root / "raw", root / "translated", manga, chapter)

            self.assertIn((root / "raw").resolve(), raw.parents)
            self.assertIn((root / "translated").resolve(), translated.parents)

    def test_unicode_title_is_preserved_safely(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manga = Manga("m1", "日本語 Truyện tranh", "https://public.example/manga.json")
            chapter = Chapter("c1", "m1", "1", "Mở đầu", "https://example.test/c1")

            raw, _translated = chapter_paths(root / "raw", root / "translated", manga, chapter)

            self.assertEqual(raw.parent.name, "日本語 Truyện tranh")

    def test_dangerous_and_reserved_names_are_sanitized(self) -> None:
        self.assertNotIn("/", sanitize_name("../../outside"))
        self.assertNotIn("\\", sanitize_name("..\\outside"))
        self.assertNotEqual(sanitize_name("CON").upper(), "CON")


class TranslatorWrapperTest(unittest.TestCase):
    def test_builds_core_cli_config_without_shell(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "raw"
            output_path = root / "translated"
            input_path.mkdir()
            (input_path / "001.jpg").write_bytes(jpeg_bytes())
            observed: dict[str, object] = {}

            def runner(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
                observed["command"] = command
                observed["kwargs"] = kwargs
                config_path = Path(command[command.index("--config-file") + 1])
                observed["config"] = json.loads(config_path.read_text(encoding="utf-8"))
                output_path.mkdir(exist_ok=True)
                shutil.copyfile(input_path / "001.jpg", output_path / "001.jpg")
                return subprocess.CompletedProcess(command, 0, stdout="done", stderr="")

            wrapper = MangaTranslator(
                TranslationConfig("deepseek", "VIN"),
                runner=runner,
                project_root=root,
            )
            wrapper.translate_folder(input_path, output_path)

            command = observed["command"]
            kwargs = observed["kwargs"]
            self.assertIsInstance(command, list)
            self.assertNotIn("shell", kwargs)
            self.assertEqual(
                observed["config"],
                {
                    "translator": {
                        "translator": "deepseek",
                        "target_lang": "VIN",
                        "gpt_config": None,
                        "dialogue_style_guide": None,
                        "dialogue_consistency_validator": False,
                    },
                    "render": {
                        "renderer": "default",
                        "alignment": "auto",
                        "direction": "auto",
                        "uppercase": False,
                        "font_size_offset": 0,
                        "font_size_minimum": -1,
                        "no_hyphenation": False,
                        "line_spacing": None,
                        "disable_font_border": False,
                    },
                },
            )
            self.assertIn("--dialogue-consistency", command)
            context_index = command.index("--context-size")
            self.assertEqual(command[context_index + 1], "4")
            self.assertTrue(validate_translated_output(input_path, output_path))

    def test_dialogue_consistency_false_keeps_legacy_cli_scheduling(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "raw"
            output_path = root / "translated"
            input_path.mkdir()
            (input_path / "001.jpg").write_bytes(jpeg_bytes())
            observed: list[str] = []

            def runner(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
                observed.extend(command)
                output_path.mkdir(exist_ok=True)
                shutil.copyfile(input_path / "001.jpg", output_path / "001.jpg")
                return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

            wrapper = MangaTranslator(
                TranslationConfig(
                    translator="deepseek",
                    target_language="VIN",
                    context_pages=4,
                    dialogue_consistency=False,
                ),
                runner=runner,
                project_root=root,
            )
            wrapper.translate_folder(input_path, output_path)

            self.assertNotIn("--dialogue-consistency", observed)
            self.assertNotIn("--context-size", observed)

    def test_nonzero_exit_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "raw"
            input_path.mkdir()
            (input_path / "001.jpg").write_bytes(jpeg_bytes())

            def runner(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
                return subprocess.CompletedProcess(command, 7, stdout="", stderr="fatal error")

            wrapper = MangaTranslator(
                TranslationConfig("deepseek", "VIN"), runner=runner, project_root=root
            )

            with self.assertRaisesRegex(TranslationError, "code 7: fatal error"):
                wrapper.translate_folder(input_path, root / "translated")

    def test_zero_exit_without_complete_output_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "raw"
            input_path.mkdir()
            (input_path / "001.jpg").write_bytes(jpeg_bytes())

            def runner(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
                return subprocess.CompletedProcess(
                    command, 0, stdout="", stderr="MissingAPIKeyException"
                )

            wrapper = MangaTranslator(
                TranslationConfig("deepseek", "VIN"), runner=runner, project_root=root
            )

            with self.assertRaisesRegex(TranslationError, "incomplete output"):
                wrapper.translate_folder(input_path, root / "translated")

    def test_input_and_output_folder_cannot_be_the_same(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / "001.jpg").write_bytes(jpeg_bytes())
            wrapper = MangaTranslator(TranslationConfig("deepseek", "VIN"))

            with self.assertRaisesRegex(TranslationError, "must be different"):
                wrapper.translate_folder(folder, folder)

    def test_partial_output_is_cleaned_and_completed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "raw"
            output_path = root / "translated"
            input_path.mkdir()
            output_path.mkdir()
            (input_path / "001.jpg").write_bytes(jpeg_bytes("red"))
            (input_path / "002.jpg").write_bytes(jpeg_bytes("blue"))
            good_partial = jpeg_bytes("green")
            (output_path / "001.jpg").write_bytes(good_partial)
            (output_path / "002.jpg").write_bytes(b"")

            def runner(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
                self.assertEqual((output_path / "001.jpg").read_bytes(), good_partial)
                self.assertFalse((output_path / "002.jpg").exists())
                shutil.copyfile(input_path / "002.jpg", output_path / "002.jpg")
                return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

            wrapper = MangaTranslator(
                TranslationConfig("deepseek", "VIN"), runner=runner, project_root=root
            )
            wrapper.translate_folder(input_path, output_path)

            self.assertTrue(validate_translated_output(input_path, output_path))

    def test_output_validation_rejects_missing_zero_byte_and_extra_pages(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "raw"
            output_path = root / "translated"
            input_path.mkdir()
            output_path.mkdir()
            for index in (1, 2):
                (input_path / f"{index:03d}.jpg").write_bytes(jpeg_bytes())

            (output_path / "001.jpg").write_bytes(jpeg_bytes())
            self.assertFalse(validate_translated_output(input_path, output_path))
            (output_path / "002.jpg").write_bytes(b"")
            self.assertFalse(validate_translated_output(input_path, output_path))
            (output_path / "002.jpg").write_bytes(jpeg_bytes())
            self.assertTrue(validate_translated_output(input_path, output_path))
            (output_path / "003.jpg").write_bytes(jpeg_bytes())
            self.assertFalse(validate_translated_output(input_path, output_path))


class FakeSource(MangaSource):
    name = "example"

    def __init__(self, page_count: int = 1) -> None:
        self.page_count = page_count
        self.manga = Manga("m1", "Pipeline Manga", "https://public.example/manga.json")
        self.chapters = [
            Chapter("c1", "m1", "1", "One", "https://public.example/c1.json"),
            Chapter("c2", "m1", "2", "Two", "https://public.example/c2.json"),
        ]

    def can_handle(self, _url: str) -> bool:
        return True

    def get_manga(self, _url: str) -> Manga:
        return self.manga

    def get_chapters(self, _manga: Manga) -> list[Chapter]:
        return self.chapters

    def get_pages(self, chapter: Chapter) -> list[Page]:
        return [
            Page(index, f"https://public.example/{chapter.id}-{index}.jpg")
            for index in range(1, self.page_count + 1)
        ]


class FakeRegistry:
    def __init__(self, source: MangaSource) -> None:
        self.source = source

    def for_url(self, _url: str, _preferred: str | None = None) -> MangaSource:
        return self.source

    def get(self, _name: str) -> MangaSource:
        return self.source


class CopyTranslator:
    def __init__(self, fail_chapter: str | None = None) -> None:
        self.fail_chapter = fail_chapter
        self.calls: list[Path] = []

    def translate_folder(self, input_path: Path, output_path: Path) -> None:
        self.calls.append(input_path)
        if input_path.name == self.fail_chapter:
            raise RuntimeError("translation fixture failure")
        output_path.mkdir(parents=True, exist_ok=True)
        for image in input_path.glob("*.jpg"):
            shutil.copyfile(image, output_path / image.name)


def make_config(root: Path) -> AppConfig:
    return AppConfig(
        storage=StorageConfig(root / "raw", root / "translated"),
        download=DownloadConfig(timeout=1, retries=0, delay=0, concurrency=1),
        translation=TranslationConfig("deepseek", "VIN"),
        database=DatabaseConfig(root / "state.db"),
    )


def register_first_chapter(
    database: Database, config: AppConfig, source: FakeSource
):
    manga_id = database.upsert_manga(source.manga)
    chapter = source.chapters[0]
    raw, translated = chapter_paths(
        config.storage.raw, config.storage.translated, source.manga, chapter
    )
    return database.upsert_chapter(manga_id, chapter, raw, translated)


class PipelineTest(unittest.TestCase):
    def test_failed_chapter_does_not_stop_batch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = make_config(root)
            source = FakeSource()
            downloader = ChapterDownloader(
                config.download,
                session=FakeDownloadSession(jpeg_bytes()),  # type: ignore[arg-type]
            )
            with Database(config.database.path) as database:
                pipeline = PipelineOrchestrator(
                    config,
                    database,
                    downloader=downloader,
                    translator=CopyTranslator("chapter-001"),  # type: ignore[arg-type]
                    sources=FakeRegistry(source),  # type: ignore[arg-type]
                )
                summary = pipeline.run_manga(
                    "https://public.example/manga.json", chapter_range="1-2"
                )

                self.assertEqual(summary.failed, 1)
                self.assertEqual(summary.translated, 1)
                statuses = [record.status for record in database.list_resumable()]
                self.assertEqual(statuses, ["failed"])

                resumed_pipeline = PipelineOrchestrator(
                    config,
                    database,
                    downloader=downloader,
                    translator=CopyTranslator(),  # type: ignore[arg-type]
                    sources=FakeRegistry(source),  # type: ignore[arg-type]
                )
                resumed = resumed_pipeline.resume()
                self.assertEqual(resumed.translated, 1)
                self.assertEqual(resumed.failed, 0)
                self.assertEqual(database.list_resumable(), [])

    def test_case_a_resume_interrupted_partial_download(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = make_config(root)
            source = FakeSource(page_count=2)
            session = FakeDownloadSession(jpeg_bytes())
            downloader = ChapterDownloader(config.download, session=session)  # type: ignore[arg-type]
            translator = CopyTranslator()
            with Database(config.database.path) as database:
                record = register_first_chapter(database, config, source)
                database.set_page_count(record.id, 2)
                record.raw_path.mkdir(parents=True)
                (record.raw_path / "001.jpg").write_bytes(jpeg_bytes())
                database.set_status(record.id, "downloading")
                pipeline = PipelineOrchestrator(
                    config,
                    database,
                    downloader=downloader,
                    translator=translator,  # type: ignore[arg-type]
                    sources=FakeRegistry(source),  # type: ignore[arg-type]
                )

                summary = pipeline.resume()

                self.assertEqual(summary.translated, 1)
                self.assertEqual(session.calls, 1)
                self.assertTrue(validate_chapter_images(record.raw_path, 2))
                self.assertEqual(database.get_record(record.id).status, "translated")  # type: ignore[union-attr]

    def test_case_b_resume_complete_download_before_status_update(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = make_config(root)
            source = FakeSource(page_count=2)
            session = FakeDownloadSession(jpeg_bytes())
            translator = CopyTranslator()
            with Database(config.database.path) as database:
                record = register_first_chapter(database, config, source)
                database.set_page_count(record.id, 2)
                record.raw_path.mkdir(parents=True)
                for index in (1, 2):
                    (record.raw_path / f"{index:03d}.jpg").write_bytes(jpeg_bytes())
                database.set_status(record.id, "downloading")
                pipeline = PipelineOrchestrator(
                    config,
                    database,
                    downloader=ChapterDownloader(config.download, session=session),  # type: ignore[arg-type]
                    translator=translator,  # type: ignore[arg-type]
                    sources=FakeRegistry(source),  # type: ignore[arg-type]
                )

                summary = pipeline.resume()

                self.assertEqual(summary.translated, 1)
                self.assertEqual(session.calls, 0)
                self.assertEqual(len(translator.calls), 1)

    def test_case_c_resume_translation_does_not_trust_partial_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = make_config(root)
            source = FakeSource(page_count=2)
            observed: dict[str, bool] = {}
            with Database(config.database.path) as database:
                record = register_first_chapter(database, config, source)
                database.set_page_count(record.id, 2)
                record.raw_path.mkdir(parents=True)
                record.translated_path.mkdir(parents=True)
                for index in (1, 2):
                    (record.raw_path / f"{index:03d}.jpg").write_bytes(jpeg_bytes())
                (record.translated_path / "001.jpg").write_bytes(jpeg_bytes())
                (record.translated_path / "002.jpg").write_bytes(b"")
                database.set_status(record.id, "translating")

                def runner(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
                    observed["valid_page_preserved"] = (record.translated_path / "001.jpg").is_file()
                    observed["zero_byte_removed"] = not (record.translated_path / "002.jpg").exists()
                    shutil.copyfile(
                        record.raw_path / "002.jpg", record.translated_path / "002.jpg"
                    )
                    return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

                pipeline = PipelineOrchestrator(
                    config,
                    database,
                    downloader=ChapterDownloader(
                        config.download, session=FakeDownloadSession(jpeg_bytes())
                    ),  # type: ignore[arg-type]
                    translator=MangaTranslator(
                        config.translation, runner=runner, project_root=root
                    ),
                    sources=FakeRegistry(source),  # type: ignore[arg-type]
                )

                summary = pipeline.resume()

                self.assertEqual(summary.translated, 1)
                self.assertEqual(
                    observed, {"valid_page_preserved": True, "zero_byte_removed": True}
                )
                self.assertTrue(
                    validate_translated_output(record.raw_path, record.translated_path)
                )

    def test_case_d_resume_repairs_translated_record_with_missing_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = make_config(root)
            source = FakeSource()
            translator = CopyTranslator()
            with Database(config.database.path) as database:
                record = register_first_chapter(database, config, source)
                database.set_page_count(record.id, 1)
                record.raw_path.mkdir(parents=True)
                (record.raw_path / "001.jpg").write_bytes(jpeg_bytes())
                database.set_status(record.id, "translated")
                pipeline = PipelineOrchestrator(
                    config,
                    database,
                    downloader=ChapterDownloader(
                        config.download, session=FakeDownloadSession(jpeg_bytes())
                    ),  # type: ignore[arg-type]
                    translator=translator,  # type: ignore[arg-type]
                    sources=FakeRegistry(source),  # type: ignore[arg-type]
                )

                summary = pipeline.resume()

                self.assertEqual(summary.translated, 1)
                self.assertEqual(len(translator.calls), 1)
                self.assertTrue(
                    validate_translated_output(record.raw_path, record.translated_path)
                )

    def test_valid_translated_record_is_not_run_again(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = make_config(root)
            source = FakeSource()
            translator = CopyTranslator()
            with Database(config.database.path) as database:
                record = register_first_chapter(database, config, source)
                database.set_page_count(record.id, 1)
                record.raw_path.mkdir(parents=True)
                record.translated_path.mkdir(parents=True)
                (record.raw_path / "001.jpg").write_bytes(jpeg_bytes())
                (record.translated_path / "001.jpg").write_bytes(jpeg_bytes())
                database.set_status(record.id, "translated")
                pipeline = PipelineOrchestrator(
                    config,
                    database,
                    translator=translator,  # type: ignore[arg-type]
                    sources=FakeRegistry(source),  # type: ignore[arg-type]
                )

                summary = pipeline.resume()

                self.assertEqual(summary.selected, 0)
                self.assertEqual(translator.calls, [])

    def test_normal_rerun_repairs_translated_record_with_invalid_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = make_config(root)
            source = FakeSource()
            translator = CopyTranslator()
            with Database(config.database.path) as database:
                record = register_first_chapter(database, config, source)
                database.set_page_count(record.id, 1)
                record.raw_path.mkdir(parents=True)
                (record.raw_path / "001.jpg").write_bytes(jpeg_bytes())
                database.set_status(record.id, "translated")
                pipeline = PipelineOrchestrator(
                    config,
                    database,
                    translator=translator,  # type: ignore[arg-type]
                    sources=FakeRegistry(source),  # type: ignore[arg-type]
                )

                summary = pipeline.run_manga(
                    "https://public.example/manga.json", chapter_range="1"
                )

                self.assertEqual(summary.translated, 1)
                self.assertEqual(summary.skipped, 0)
                self.assertEqual(len(translator.calls), 1)


class ConfigTest(unittest.TestCase):
    @staticmethod
    def _write_config(root: Path, translation_lines: list[str]) -> Path:
        config_path = root / "config.yaml"
        config_path.write_text(
            "\n".join(
                [
                    "storage:",
                    f"  raw: '{root / 'raw'}'",
                    f"  translated: '{root / 'translated'}'",
                    "download: {}",
                    "translation:",
                    *(f"  {line}" for line in translation_lines),
                    "database:",
                    f"  path: '{root / 'state.db'}'",
                ]
            ),
            encoding="utf-8",
        )
        return config_path

    def test_gpt_config_and_style_guide_are_validated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gpt_config = root / "deepseek.yaml"
            style = root / "style.yaml"
            gpt_config.write_text("deepseek:\n  temperature: 0.1\n", encoding="utf-8")
            style.write_text(
                "relationships:\n"
                "  - speaker: A\n"
                "    listener: B\n"
                "    self: anh\n"
                "    address: em\n"
                "line_guidance:\n"
                "  'What did I do?': 'inner monologue; use mình'\n",
                encoding="utf-8",
            )
            config = load_config(
                self._write_config(
                    root,
                    [
                        f"gpt_config: '{gpt_config}'",
                        f"dialogue_style_guide: '{style}'",
                        "context_pages: 20",
                        "dialogue_consistency: true",
                        "dialogue_consistency_validator: true",
                    ],
                )
            )

            self.assertEqual(config.translation.gpt_config, gpt_config)
            self.assertEqual(config.translation.dialogue_style_guide, style)
            self.assertEqual(config.translation.context_pages, 20)
            self.assertTrue(config.translation.dialogue_consistency_validator)

    def test_missing_gpt_or_style_file_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ConfigError, "gpt_config does not exist"):
                load_config(
                    self._write_config(
                        root, [f"gpt_config: '{root / 'missing.yaml'}'"]
                    )
                )

    def test_negative_context_pages_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ConfigError, "context_pages must be between"):
                load_config(self._write_config(root, ["context_pages: -1"]))

    def test_unknown_translation_key_is_still_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ConfigError, "Unsupported translation setting"):
                load_config(self._write_config(root, ["context_pagez: 4"]))

    def test_invalid_style_guide_schema_has_clear_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            style = root / "style.yaml"
            style.write_text("relationships: wrong\n", encoding="utf-8")
            with self.assertRaisesRegex(ConfigError, "relationships must be a list"):
                load_config(
                    self._write_config(
                        root, [f"dialogue_style_guide: '{style}'"]
                    )
                )

    def test_raw_and_translated_roots_must_be_different(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "config.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "storage:",
                        f"  raw: '{root / 'same'}'",
                        f"  translated: '{root / 'same'}'",
                        "download: {}",
                        "translation: {}",
                        "database:",
                        f"  path: '{root / 'state.db'}'",
                    ]
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ConfigError, "must be different"):
                load_config(config_path)


class CliTest(unittest.TestCase):
    def test_cli_parses_requested_commands(self) -> None:
        parser = build_parser()
        manga = parser.parse_args(["manga", "https://public.example/manga.json", "--chapters", "1-5"])
        latest = parser.parse_args(["manga", "https://public.example/manga.json", "--latest"])
        chapter = parser.parse_args(["chapter", "https://public.example/chapter.json"])
        resume = parser.parse_args(["resume"])

        self.assertEqual(manga.chapters, "1-5")
        self.assertTrue(latest.latest)
        self.assertEqual(chapter.command, "chapter")
        self.assertEqual(resume.command, "resume")

    def test_empty_resume_command(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "config.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "storage:",
                        f"  raw: '{root / 'raw'}'",
                        f"  translated: '{root / 'translated'}'",
                        "download:",
                        "  timeout: 1",
                        "  retries: 0",
                        "  delay: 0",
                        "  concurrency: 1",
                        "translation:",
                        "  translator: deepseek",
                        "  target_language: VIN",
                        "database:",
                        f"  path: '{root / 'state.db'}'",
                    ]
                ),
                encoding="utf-8",
            )
            self.assertEqual(main(["resume", "--config", str(config_path)]), 0)


if __name__ == "__main__":
    unittest.main()
