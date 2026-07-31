from __future__ import annotations

from collections.abc import Callable

from ..base import MangaSource
from .example_source import ExampleSource, SourceError


class SourceRegistry:
    def __init__(self, timeout: float = 20.0) -> None:
        self._factories: dict[str, Callable[[], MangaSource]] = {
            ExampleSource.name: lambda: ExampleSource(timeout=timeout),
        }
        self._instances: dict[str, MangaSource] = {}

    def get(self, name: str) -> MangaSource:
        if name not in self._factories:
            raise SourceError(f"Unknown source adapter '{name}'")
        if name not in self._instances:
            self._instances[name] = self._factories[name]()
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
