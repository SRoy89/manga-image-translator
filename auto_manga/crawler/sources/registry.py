from __future__ import annotations

from collections.abc import Callable

from auto_manga.config import MangaDexConfig

from ..base import MangaSource, SourceError
from .example_source import ExampleSource
from .mangadex_source import MangaDexSource


class SourceRegistry:
    def __init__(
        self,
        timeout: float = 20.0,
        retries: int = 3,
        delay: float = 0.25,
        mangadex: MangaDexConfig | None = None,
    ) -> None:
        mangadex = mangadex or MangaDexConfig()
        self._factories: dict[str, Callable[[], MangaSource]] = {
            MangaDexSource.name: lambda: MangaDexSource(
                timeout=timeout,
                retries=retries,
                request_delay=max(delay, 0.25),
                translated_languages=mangadex.translated_languages,
                data_saver=mangadex.data_saver,
            ),
            ExampleSource.name: lambda: ExampleSource(timeout=timeout),
        }
        self._instances: dict[str, MangaSource] = {}

    def get(self, name: str) -> MangaSource:
        if name not in self._factories:
            raise SourceError(f"Unknown source adapter '{name}'")
        if name not in self._instances:
            try:
                self._instances[name] = self._factories[name]()
            except SourceError:
                raise
            except Exception as exc:
                raise SourceError(f"Cannot initialize source adapter '{name}': {exc}") from exc
        return self._instances[name]

    def for_url(self, url: str, preferred: str | None = None) -> MangaSource:
        if preferred:
            source = self.get(preferred)
            if not source.can_handle(url):
                raise SourceError(f"Source '{preferred}' cannot handle this URL")
            return source
        for name in self._factories:
            source = self.get(name)
            if source.can_handle(url):
                return source
        raise SourceError("No source adapter can handle this URL")
