from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from PIL import Image
import requests

from auto_manga.config import (
    AppConfig,
    DatabaseConfig,
    DownloadConfig,
    StorageConfig,
    TranslationConfig,
)
from auto_manga.crawler.base import MangaSource
from auto_manga.crawler.downloader import ChapterDownloader, validate_chapter_images
from auto_manga.crawler.models import Chapter, Manga, Page
from auto_manga.crawler.sources.example_source import ExampleSource
from auto_manga.main import build_parser, main
from auto_manga.pipeline.orchestrator import PipelineOrchestrator
from auto_manga.pipeline.translator import MangaTranslator, validate_translated_output
from auto_manga.storage.database import Database
from auto_manga.storage.paths import chapter_paths


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


class DownloaderTest(unittest.TestCase):
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


class DatabaseResumeTest(unittest.TestCase):
    def test_recovers_interrupted_states_and_excludes_translated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with Database(root / "state.db") as database:
                manga = Manga("m1", "Manga", "https://public.example/manga.json")
                manga_id = database.upsert_manga(manga)
                statuses = ["downloading", "translating", "failed", "translated"]
                records = []
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
                    records.append(record)

                self.assertEqual(database.recover_interrupted(), 2)
                resumable = database.list_resumable()

                self.assertEqual(
                    [record.status for record in resumable],
                    ["pending", "downloaded", "failed"],
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
                {"translator": {"translator": "deepseek", "target_lang": "VIN"}},
            )
            self.assertTrue(validate_translated_output(input_path, output_path))


class FakeSource(MangaSource):
    name = "example"

    def __init__(self) -> None:
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
        return [Page(1, f"https://public.example/{chapter.id}.jpg")]


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

    def translate_folder(self, input_path: Path, output_path: Path) -> None:
        if input_path.name == self.fail_chapter:
            raise RuntimeError("translation fixture failure")
        output_path.mkdir(parents=True, exist_ok=True)
        for image in input_path.glob("*.jpg"):
            shutil.copyfile(image, output_path / image.name)


class PipelineTest(unittest.TestCase):
    def test_failed_chapter_does_not_stop_batch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = AppConfig(
                storage=StorageConfig(root / "raw", root / "translated"),
                download=DownloadConfig(timeout=1, retries=0, delay=0, concurrency=1),
                translation=TranslationConfig("deepseek", "VIN"),
                database=DatabaseConfig(root / "state.db"),
            )
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
