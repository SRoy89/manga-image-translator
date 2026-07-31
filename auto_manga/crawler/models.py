from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Manga:
    id: str
    title: str
    source_url: str
    source: str = "example"


@dataclass(frozen=True)
class Chapter:
    id: str
    manga_id: str
    number: str
    title: str
    url: str


@dataclass(frozen=True)
class Page:
    index: int
    image_url: str
