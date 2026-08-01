from __future__ import annotations

import re
import unicodedata
from decimal import Decimal, InvalidOperation
from pathlib import Path

from auto_manga.crawler.models import Chapter, Manga

_UNSAFE = re.compile(r"[<>:\"/\\|?*\x00-\x1f]")
_WHITESPACE = re.compile(r"\s+")
_WINDOWS_RESERVED = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{number}" for number in range(1, 10)),
    *(f"LPT{number}" for number in range(1, 10)),
}


def sanitize_name(value: str, fallback: str = "untitled", max_length: int = 100) -> str:
    normalized = unicodedata.normalize("NFKC", value)
    normalized = _UNSAFE.sub("-", normalized)
    normalized = _WHITESPACE.sub(" ", normalized).strip(" .-")
    if normalized in {"", ".", ".."}:
        normalized = fallback
    if normalized.split(".", maxsplit=1)[0].upper() in _WINDOWS_RESERVED:
        normalized = f"_{normalized}"
    return normalized[:max_length].rstrip(" .") or fallback


def manga_slug(manga: Manga) -> str:
    return sanitize_name(manga.title, fallback=f"manga-{sanitize_name(manga.id)}")


def chapter_slug(chapter: Chapter) -> str:
    number = (chapter.storage_key or chapter.number).strip()
    if len(number) > 32:
        return f"chapter-{sanitize_name(number, fallback=sanitize_name(chapter.id))}"
    try:
        parsed = Decimal(number)
        if not parsed.is_finite():
            raise ValueError("Chapter number must be finite")
        if parsed == parsed.to_integral():
            formatted = f"{int(parsed):03d}"
        else:
            integer, fraction = format(parsed.normalize(), "f").split(".", maxsplit=1)
            formatted = f"{int(integer):03d}.{fraction}"
    except (InvalidOperation, ValueError, OverflowError):
        formatted = sanitize_name(number, fallback=sanitize_name(chapter.id))
    return f"chapter-{formatted}"


def child_path(root: Path, *parts: str) -> Path:
    root = root.resolve()
    candidate = root.joinpath(*(sanitize_name(part) for part in parts)).resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError("Resolved path escapes its storage root")
    return candidate


def chapter_paths(raw_root: Path, translated_root: Path, manga: Manga, chapter: Chapter) -> tuple[Path, Path]:
    manga_directory = manga_slug(manga)
    chapter_directory = chapter_slug(chapter)
    return (
        child_path(raw_root, manga_directory, chapter_directory),
        child_path(translated_root, manga_directory, chapter_directory),
    )
