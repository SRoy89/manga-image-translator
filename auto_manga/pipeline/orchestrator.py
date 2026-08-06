from __future__ import annotations

import logging
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

from auto_manga.config import AppConfig
from auto_manga.crawler.base import MangaSource, SourceError
from auto_manga.crawler.downloader import ChapterDownloader, validate_chapter_images
from auto_manga.crawler.models import Chapter, Manga, Page
from auto_manga.crawler.sources.registry import SourceRegistry
from auto_manga.storage.database import ChapterRecord, Database
from auto_manga.storage.paths import chapter_paths

from .translator import MangaTranslator, validate_translated_output

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class PipelineSummary:
    selected: int
    translated: int
    skipped: int
    failed: int


class PipelineOrchestrator:
    def __init__(
        self,
        config: AppConfig,
        database: Database,
        downloader: ChapterDownloader | None = None,
        translator: MangaTranslator | None = None,
        sources: SourceRegistry | None = None,
    ) -> None:
        self.config = config
        self.database = database
        self.downloader = downloader or ChapterDownloader(config.download)
        self.translator = translator or MangaTranslator(config.translation)
        self.sources = sources or SourceRegistry(
            timeout=config.download.timeout,
            retries=config.download.retries,
            delay=config.download.delay,
            mangadex=config.sources.mangadex,
        )

    def run_manga(
        self,
        url: str,
        chapter_range: str | None = None,
        latest: bool = False,
        source_name: str | None = None,
    ) -> PipelineSummary:
        source = self.sources.for_url(url, source_name)
        manga = source.get_manga(url)
        chapters = source.get_chapters(manga)
        LOGGER.info("Manga: %s", manga.title)
        LOGGER.info("Found %s chapters", len(chapters))
        selected = self._select_chapters(chapters, chapter_range, latest)
        return self._run_chapters(manga, selected, source)

    def run_chapter(self, url: str, source_name: str | None = None) -> PipelineSummary:
        source = self.sources.for_url(url, source_name)
        manga, chapter = source.get_chapter(url)
        LOGGER.info("Manga: %s", manga.title)
        return self._run_chapters(manga, [chapter], source)

    def resume(self) -> PipelineSummary:
        recovered = self.database.recover_interrupted()
        if recovered:
            LOGGER.info("Recovered %s interrupted chapter states", recovered)
        self._recover_invalid_translated_records()
        records = self.database.list_resumable()
        LOGGER.info("Found %s chapters to resume", len(records))

        translated = 0
        failed = 0
        for position, record in enumerate(records, start=1):
            LOGGER.info("[%s/%s] Chapter %s", position, len(records), record.chapter_number)
            try:
                source = self.sources.get(record.source)
            except SourceError as exc:
                error = f"{type(exc).__name__}: {exc}"
                self.database.set_status(record.id, "failed", error=error[:2000])
                LOGGER.error("Chapter %s: %s", record.chapter_number, error)
                failed += 1
                continue
            if self._process_record(record, source):
                translated += 1
            else:
                failed += 1
        return PipelineSummary(len(records), translated, 0, failed)

    def _run_chapters(
        self,
        manga: Manga,
        chapters: list[Chapter],
        source: MangaSource,
    ) -> PipelineSummary:
        manga_db_id = self.database.upsert_manga(manga)
        records: list[ChapterRecord] = []
        for chapter in chapters:
            raw_path, translated_path = chapter_paths(
                self.config.storage.raw,
                self.config.storage.translated,
                manga,
                chapter,
            )
            records.append(
                self.database.upsert_chapter(
                    manga_db_id, chapter, raw_path, translated_path
                )
            )

        translated = 0
        skipped = 0
        failed = 0
        for position, record in enumerate(records, start=1):
            LOGGER.info("[%s/%s] Chapter %s", position, len(records), record.chapter_number)
            if record.status == "translated" and self._translation_is_complete(record):
                LOGGER.info("Already translated; skipping")
                skipped += 1
            elif self._process_record(record, source):
                translated += 1
            else:
                failed += 1
        return PipelineSummary(len(records), translated, skipped, failed)

    def _process_record(self, record: ChapterRecord, source: MangaSource) -> bool:
        chapter = record.to_chapter()
        pages: list[Page] | None = None
        expected_count = record.page_count
        try:
            if expected_count is None:
                pages = source.get_pages(chapter)
                expected_count = len(pages)
                self.database.set_page_count(record.id, expected_count)

            if not validate_chapter_images(record.raw_path, expected_count):
                if pages is None:
                    pages = source.get_pages(chapter)
                    expected_count = len(pages)
                    self.database.set_page_count(record.id, expected_count)
                self.database.set_status(record.id, "downloading")
                LOGGER.info("[DOWNLOAD] %s pages", expected_count)
                self.downloader.download_chapter(pages, record.raw_path)
                if not validate_chapter_images(record.raw_path, expected_count):
                    raise RuntimeError("Downloaded images failed validation")
                self.database.set_status(record.id, "downloaded")
                LOGGER.info("[DOWNLOAD] completed")
            elif record.status != "downloaded":
                self.database.set_status(record.id, "downloaded")

            if validate_translated_output(
                record.raw_path, record.translated_path, expected_count
            ):
                self.database.set_status(record.id, "translated")
                LOGGER.info("[TRANSLATE] Existing output is complete")
                return True

            self.database.set_status(record.id, "translating")
            LOGGER.info(
                "[TRANSLATE] %s -> %s",
                self.config.translation.translator,
                self.config.translation.target_language,
            )
            self.translator.translate_folder(record.raw_path, record.translated_path)
            if not validate_translated_output(
                record.raw_path, record.translated_path, expected_count
            ):
                raise RuntimeError("Translated output failed validation")
            self.database.set_status(record.id, "translated")
            LOGGER.info("[TRANSLATE] completed")
            return True
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            self.database.set_status(record.id, "failed", error=error[:2000])
            LOGGER.error(
                "Chapter %s: %s",
                chapter.number,
                error,
                exc_info=LOGGER.isEnabledFor(logging.DEBUG),
            )
            return False

    def _recover_invalid_translated_records(self) -> None:
        for record in self.database.list_translated():
            if self._translation_is_complete(record):
                continue
            raw_complete = (
                record.page_count is not None
                and validate_chapter_images(record.raw_path, record.page_count)
            )
            recovered_status = "downloaded" if raw_complete else "pending"
            self.database.set_status(record.id, recovered_status)
            LOGGER.warning(
                "Chapter %s was marked translated but its files are incomplete; reset to %s",
                record.chapter_number,
                recovered_status,
            )

    @staticmethod
    def _translation_is_complete(record: ChapterRecord) -> bool:
        return record.page_count is not None and validate_translated_output(
            record.raw_path,
            record.translated_path,
            record.page_count,
        )

    @staticmethod
    def _select_chapters(
        chapters: list[Chapter],
        chapter_range: str | None,
        latest: bool,
    ) -> list[Chapter]:
        if not chapters:
            raise ValueError("The source returned no chapters")
        if latest:
            numeric = [
                (PipelineOrchestrator._number(chapter.number), chapter) for chapter in chapters
            ]
            numeric = [(number, chapter) for number, chapter in numeric if number is not None]
            return [max(numeric, key=lambda item: item[0])[1]] if numeric else [chapters[-1]]
        if not chapter_range:
            return chapters

        parts = chapter_range.split("-", maxsplit=1)
        start = PipelineOrchestrator._number(parts[0])
        end = PipelineOrchestrator._number(parts[1]) if len(parts) == 2 else start
        if start is None or end is None:
            raise ValueError("Chapter range must look like '1-30' or '12'")
        if start > end:
            raise ValueError("Chapter range start cannot be greater than its end")
        selected = [
            chapter
            for chapter in chapters
            if (number := PipelineOrchestrator._number(chapter.number)) is not None
            and start <= number <= end
        ]
        if not selected:
            raise ValueError(f"No chapters matched range {chapter_range}")
        return selected

    @staticmethod
    def _number(value: str) -> Decimal | None:
        try:
            number = Decimal(value.strip())
        except InvalidOperation:
            return None
        return number if number.is_finite() else None
