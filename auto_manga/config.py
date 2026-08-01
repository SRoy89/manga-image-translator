from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from difflib import get_close_matches
from pathlib import Path
from typing import Any

import yaml

from auto_manga.font_support import (
    SUPPORTED_FONT_EXTENSIONS,
    FontInspectionError,
    format_missing_glyphs,
    inspect_font,
)


class ConfigError(ValueError):
    """Raised when the automation configuration is invalid."""


@dataclass(frozen=True)
class StorageConfig:
    raw: Path
    translated: Path


@dataclass(frozen=True)
class DownloadConfig:
    timeout: float = 20.0
    retries: int = 3
    delay: float = 1.0
    concurrency: int = 3


@dataclass(frozen=True)
class TranslationConfig:
    translator: str = "deepseek"
    target_language: str = "VIN"
    python_executable: str | None = None
    font_path: Path | None = None
    renderer: str = "default"
    alignment: str = "auto"
    direction: str = "auto"
    uppercase: bool = False
    font_size_offset: int = 0
    font_size_minimum: int = -1
    no_hyphenation: bool = False
    line_spacing: float | None = None
    disable_font_border: bool = False


@dataclass(frozen=True)
class DatabaseConfig:
    path: Path


@dataclass(frozen=True)
class MangaDexConfig:
    translated_languages: tuple[str, ...] = ("en", "ja", "ko", "zh", "zh-hk")
    data_saver: bool = False


@dataclass(frozen=True)
class SourcesConfig:
    mangadex: MangaDexConfig = field(default_factory=MangaDexConfig)


@dataclass(frozen=True)
class AppConfig:
    storage: StorageConfig
    download: DownloadConfig
    translation: TranslationConfig
    database: DatabaseConfig
    sources: SourcesConfig = field(default_factory=SourcesConfig)


def _required_mapping(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key)
    if not isinstance(value, dict):
        raise ConfigError(f"Config section '{key}' must be a mapping")
    return value


def _optional_mapping(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key, {})
    if not isinstance(value, dict):
        raise ConfigError(f"Config section '{key}' must be a mapping")
    return value


def _mangadex_config(data: dict[str, Any]) -> MangaDexConfig:
    sources = _optional_mapping(data, "sources")
    mangadex = _optional_mapping(sources, "mangadex")
    raw_languages = mangadex.get(
        "translated_languages", list(MangaDexConfig().translated_languages)
    )
    if not isinstance(raw_languages, list) or not raw_languages:
        raise ConfigError("sources.mangadex.translated_languages must be a non-empty list")

    languages: list[str] = []
    for value in raw_languages:
        if not isinstance(value, str):
            raise ConfigError("MangaDex translated language codes must be strings")
        language = value.strip().lower()
        if not re.fullmatch(r"[a-z]{2,3}(?:-[a-z]{2})?", language):
            raise ConfigError(f"Invalid MangaDex translated language code '{value}'")
        if language not in languages:
            languages.append(language)

    data_saver = mangadex.get("data_saver", False)
    if not isinstance(data_saver, bool):
        raise ConfigError("sources.mangadex.data_saver must be true or false")
    return MangaDexConfig(tuple(languages), data_saver)


def _path(value: Any, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"Config value '{name}' must be a non-empty path")
    return Path(value).expanduser().resolve()


_TRANSLATION_KEYS = frozenset(
    {
        "translator",
        "target_language",
        "python_executable",
        "font_path",
        "renderer",
        "alignment",
        "direction",
        "uppercase",
        "font_size_offset",
        "font_size_minimum",
        "no_hyphenation",
        "line_spacing",
        "disable_font_border",
    }
)
_RENDERERS = frozenset({"default", "manga2eng", "manga2eng_pillow", "none"})
_ALIGNMENTS = frozenset({"auto", "left", "center", "right"})
_DIRECTIONS = frozenset({"auto", "horizontal", "vertical"})


def _reject_unknown_translation_keys(translation: dict[str, Any]) -> None:
    unknown = sorted(set(translation) - _TRANSLATION_KEYS)
    if not unknown:
        return
    key = unknown[0]
    match = get_close_matches(key, _TRANSLATION_KEYS, n=1)
    suggestion = f"; did you mean '{match[0]}'?" if match else ""
    raise ConfigError(f"Unsupported translation setting '{key}'{suggestion}")


def _enum_value(
    data: dict[str, Any], key: str, default: str, choices: frozenset[str]
) -> str:
    value = data.get(key, default)
    if not isinstance(value, str) or value.strip().lower() not in choices:
        options = ", ".join(sorted(choices))
        raise ConfigError(f"translation.{key} must be one of: {options}")
    return value.strip().lower()


def _bool_value(data: dict[str, Any], key: str, default: bool) -> bool:
    value = data.get(key, default)
    if not isinstance(value, bool):
        raise ConfigError(f"translation.{key} must be true or false")
    return value


def _int_value(data: dict[str, Any], key: str, default: int) -> int:
    value = data.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"translation.{key} must be an integer")
    return value


def _line_spacing_value(data: dict[str, Any]) -> float | None:
    value = data.get("line_spacing")
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError("translation.line_spacing must be a number or null")
    result = float(value)
    if not math.isfinite(result) or not -0.5 <= result <= 2.0:
        raise ConfigError(
            "translation.line_spacing must be between -0.5 and 2.0; "
            "the core multiplies it by the font size"
        )
    return result


def _font_path_value(translation: dict[str, Any]) -> Path | None:
    value = translation.get("font_path")
    if value is None:
        return None
    font_path = _path(value, "translation.font_path")
    if not font_path.exists():
        raise ConfigError(f"translation.font_path does not exist: {font_path}")
    if not font_path.is_file():
        raise ConfigError(f"translation.font_path is not a file: {font_path}")
    if font_path.suffix.lower() not in SUPPORTED_FONT_EXTENSIONS:
        extensions = ", ".join(sorted(SUPPORTED_FONT_EXTENSIONS))
        raise ConfigError(
            f"translation.font_path must use a supported extension: {extensions}"
        )
    try:
        coverage = inspect_font(font_path)
    except FontInspectionError as exc:
        raise ConfigError(str(exc)) from exc
    if not coverage.complete:
        raise ConfigError(
            f"Font {font_path} is missing required Vietnamese glyphs: "
            f"{format_missing_glyphs(coverage.missing_glyphs)}"
        )
    return font_path


def _translation_config(translation: dict[str, Any]) -> TranslationConfig:
    _reject_unknown_translation_keys(translation)
    translator = str(translation.get("translator", "deepseek")).strip()
    target_language = str(translation.get("target_language", "VIN")).strip().upper()
    if not translator or not target_language:
        raise ConfigError("translation translator and target_language cannot be empty")

    python_executable = translation.get("python_executable")
    if python_executable is not None:
        python_executable = str(python_executable).strip() or None

    font_size_offset = _int_value(translation, "font_size_offset", 0)
    if not -1000 <= font_size_offset <= 1000:
        raise ConfigError("translation.font_size_offset must be between -1000 and 1000")
    font_size_minimum = _int_value(translation, "font_size_minimum", -1)
    if font_size_minimum != -1 and not 1 <= font_size_minimum <= 1000:
        raise ConfigError("translation.font_size_minimum must be -1 or between 1 and 1000")

    return TranslationConfig(
        translator=translator,
        target_language=target_language,
        python_executable=python_executable,
        font_path=_font_path_value(translation),
        renderer=_enum_value(translation, "renderer", "default", _RENDERERS),
        alignment=_enum_value(translation, "alignment", "auto", _ALIGNMENTS),
        direction=_enum_value(translation, "direction", "auto", _DIRECTIONS),
        uppercase=_bool_value(translation, "uppercase", False),
        font_size_offset=font_size_offset,
        font_size_minimum=font_size_minimum,
        no_hyphenation=_bool_value(translation, "no_hyphenation", False),
        line_spacing=_line_spacing_value(translation),
        disable_font_border=_bool_value(translation, "disable_font_border", False),
    )


def load_config(path: str | Path) -> AppConfig:
    config_path = Path(path).expanduser().resolve()
    try:
        with config_path.open("r", encoding="utf-8") as config_file:
            data = yaml.safe_load(config_file) or {}
    except OSError as exc:
        raise ConfigError(f"Cannot read config file {config_path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {config_path}: {exc}") from exc

    if not isinstance(data, dict):
        raise ConfigError("Config root must be a mapping")

    storage = _required_mapping(data, "storage")
    download = _required_mapping(data, "download")
    translation = _required_mapping(data, "translation")
    database = _required_mapping(data, "database")

    try:
        download_config = DownloadConfig(
            timeout=float(download.get("timeout", 20)),
            retries=int(download.get("retries", 3)),
            delay=float(download.get("delay", 1.0)),
            concurrency=int(download.get("concurrency", 3)),
        )
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"Invalid download config: {exc}") from exc

    if download_config.timeout <= 0:
        raise ConfigError("download.timeout must be greater than zero")
    if download_config.retries < 0:
        raise ConfigError("download.retries cannot be negative")
    if download_config.delay < 0:
        raise ConfigError("download.delay cannot be negative")
    if download_config.concurrency < 1:
        raise ConfigError("download.concurrency must be at least one")

    raw_path = _path(storage.get("raw"), "storage.raw")
    translated_path = _path(storage.get("translated"), "storage.translated")
    if raw_path == translated_path:
        raise ConfigError("storage.raw and storage.translated must be different paths")

    return AppConfig(
        storage=StorageConfig(raw=raw_path, translated=translated_path),
        download=download_config,
        translation=_translation_config(translation),
        database=DatabaseConfig(path=_path(database.get("path"), "database.path")),
        sources=SourcesConfig(mangadex=_mangadex_config(data)),
    )
