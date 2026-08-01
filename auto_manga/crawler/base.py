from __future__ import annotations

from abc import ABC, abstractmethod

from .models import Chapter, Manga, Page


class SourceError(RuntimeError):
    """Raised when source metadata cannot be loaded or parsed."""


class MangaSource(ABC):
    """Interface implemented by every source-specific adapter."""

    name: str

    @abstractmethod
    def can_handle(self, url: str) -> bool:
        """Return whether this adapter understands the URL."""

    @abstractmethod
    def get_manga(self, url: str) -> Manga:
        """Load manga metadata from a source URL."""

    @abstractmethod
    def get_chapters(self, manga: Manga) -> list[Chapter]:
        """Return all chapters exposed by the source."""

    @abstractmethod
    def get_pages(self, chapter: Chapter) -> list[Page]:
        """Return chapter pages in source order."""

    def get_chapter(self, url: str) -> tuple[Manga, Chapter]:
        """Resolve a direct chapter URL when the source supports it."""
        raise NotImplementedError(f"Source '{self.name}' does not support direct chapter URLs")
