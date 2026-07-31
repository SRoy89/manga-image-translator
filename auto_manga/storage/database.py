from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sqlite3

from auto_manga.crawler.models import Chapter, Manga


CHAPTER_STATUSES = {
    "pending",
    "downloading",
    "downloaded",
    "translating",
    "translated",
    "failed",
}


@dataclass(frozen=True)
class ChapterRecord:
    id: int
    manga_db_id: int
    source: str
    source_manga_id: str
    manga_title: str
    manga_source_url: str
    source_chapter_id: str
    chapter_number: str
    chapter_title: str
    source_url: str
    raw_path: Path
    translated_path: Path
    status: str
    error: str | None
    page_count: int | None

    def to_models(self) -> tuple[Manga, Chapter]:
        manga = Manga(
            id=self.source_manga_id,
            title=self.manga_title,
            source_url=self.manga_source_url,
            source=self.source,
        )
        chapter = Chapter(
            id=self.source_chapter_id,
            manga_id=self.source_manga_id,
            number=self.chapter_number,
            title=self.chapter_title,
            url=self.source_url,
        )
        return manga, chapter


class Database:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        self._create_schema()

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> Database:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def _create_schema(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS manga (
                id INTEGER PRIMARY KEY,
                source TEXT NOT NULL,
                source_manga_id TEXT NOT NULL,
                title TEXT NOT NULL,
                source_url TEXT NOT NULL,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(source, source_manga_id)
            );

            CREATE TABLE IF NOT EXISTS chapters (
                id INTEGER PRIMARY KEY,
                manga_id INTEGER NOT NULL REFERENCES manga(id) ON DELETE CASCADE,
                source_chapter_id TEXT NOT NULL,
                chapter_number TEXT NOT NULL,
                title TEXT NOT NULL,
                source_url TEXT NOT NULL,
                raw_path TEXT NOT NULL,
                translated_path TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending'
                    CHECK(status IN ('pending', 'downloading', 'downloaded',
                                     'translating', 'translated', 'failed')),
                error TEXT,
                page_count INTEGER,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(manga_id, source_chapter_id)
            );

            CREATE INDEX IF NOT EXISTS chapters_status_idx ON chapters(status);
            """
        )
        self.connection.commit()

    def upsert_manga(self, manga: Manga) -> int:
        with self.connection:
            self.connection.execute(
                """
                INSERT INTO manga(source, source_manga_id, title, source_url)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(source, source_manga_id) DO UPDATE SET
                    title = excluded.title,
                    source_url = excluded.source_url,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (manga.source, manga.id, manga.title, manga.source_url),
            )
        row = self.connection.execute(
            "SELECT id FROM manga WHERE source = ? AND source_manga_id = ?",
            (manga.source, manga.id),
        ).fetchone()
        if row is None:
            raise RuntimeError("Failed to persist manga metadata")
        return int(row["id"])

    def upsert_chapter(
        self,
        manga_db_id: int,
        chapter: Chapter,
        raw_path: Path,
        translated_path: Path,
    ) -> ChapterRecord:
        with self.connection:
            self.connection.execute(
                """
                INSERT INTO chapters(
                    manga_id, source_chapter_id, chapter_number, title, source_url,
                    raw_path, translated_path
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(manga_id, source_chapter_id) DO UPDATE SET
                    chapter_number = excluded.chapter_number,
                    title = excluded.title,
                    source_url = excluded.source_url,
                    raw_path = excluded.raw_path,
                    translated_path = excluded.translated_path,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (
                    manga_db_id,
                    chapter.id,
                    chapter.number,
                    chapter.title,
                    chapter.url,
                    str(raw_path),
                    str(translated_path),
                ),
            )
        record = self.get_chapter(manga_db_id, chapter.id)
        if record is None:
            raise RuntimeError("Failed to persist chapter metadata")
        return record

    def get_chapter(self, manga_db_id: int, source_chapter_id: str) -> ChapterRecord | None:
        row = self.connection.execute(
            self._record_query() + " WHERE c.manga_id = ? AND c.source_chapter_id = ?",
            (manga_db_id, source_chapter_id),
        ).fetchone()
        return self._to_record(row) if row else None

    def get_record(self, chapter_db_id: int) -> ChapterRecord | None:
        row = self.connection.execute(
            self._record_query() + " WHERE c.id = ?", (chapter_db_id,)
        ).fetchone()
        return self._to_record(row) if row else None

    def set_status(self, chapter_db_id: int, status: str, error: str | None = None) -> None:
        if status not in CHAPTER_STATUSES:
            raise ValueError(f"Unknown chapter status '{status}'")
        with self.connection:
            self.connection.execute(
                """
                UPDATE chapters
                SET status = ?, error = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (status, error, chapter_db_id),
            )

    def set_page_count(self, chapter_db_id: int, page_count: int) -> None:
        if page_count < 1:
            raise ValueError("page_count must be positive")
        with self.connection:
            self.connection.execute(
                """
                UPDATE chapters
                SET page_count = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (page_count, chapter_db_id),
            )

    def recover_interrupted(self) -> int:
        with self.connection:
            downloading = self.connection.execute(
                "UPDATE chapters SET status = 'pending' WHERE status = 'downloading'"
            ).rowcount
            translating = self.connection.execute(
                "UPDATE chapters SET status = 'downloaded' WHERE status = 'translating'"
            ).rowcount
        return downloading + translating

    def list_resumable(self) -> list[ChapterRecord]:
        rows = self.connection.execute(
            self._record_query()
            + " WHERE c.status IN ('pending', 'downloaded', 'failed') ORDER BY c.id"
        ).fetchall()
        return [self._to_record(row) for row in rows]

    @staticmethod
    def _record_query() -> str:
        return """
            SELECT
                c.id, c.manga_id, m.source, m.source_manga_id,
                m.title AS manga_title, m.source_url AS manga_source_url,
                c.source_chapter_id, c.chapter_number,
                c.title AS chapter_title, c.source_url,
                c.raw_path, c.translated_path, c.status, c.error, c.page_count
            FROM chapters c
            JOIN manga m ON m.id = c.manga_id
        """

    @staticmethod
    def _to_record(row: sqlite3.Row) -> ChapterRecord:
        return ChapterRecord(
            id=int(row["id"]),
            manga_db_id=int(row["manga_id"]),
            source=str(row["source"]),
            source_manga_id=str(row["source_manga_id"]),
            manga_title=str(row["manga_title"]),
            manga_source_url=str(row["manga_source_url"]),
            source_chapter_id=str(row["source_chapter_id"]),
            chapter_number=str(row["chapter_number"]),
            chapter_title=str(row["chapter_title"]),
            source_url=str(row["source_url"]),
            raw_path=Path(row["raw_path"]),
            translated_path=Path(row["translated_path"]),
            status=str(row["status"]),
            error=str(row["error"]) if row["error"] is not None else None,
            page_count=int(row["page_count"]) if row["page_count"] is not None else None,
        )
