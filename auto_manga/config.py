from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


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


@dataclass(frozen=True)
class DatabaseConfig:
    path: Path


@dataclass(frozen=True)
class AppConfig:
    storage: StorageConfig
    download: DownloadConfig
    translation: TranslationConfig
    database: DatabaseConfig


def _required_mapping(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key)
    if not isinstance(value, dict):
        raise ConfigError(f"Config section '{key}' must be a mapping")
    return value


def _path(value: Any, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"Config value '{name}' must be a non-empty path")
    return Path(value).expanduser().resolve()


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

    translator = str(translation.get("translator", "deepseek")).strip()
    target_language = str(translation.get("target_language", "VIN")).strip().upper()
    if not translator or not target_language:
        raise ConfigError("translation translator and target_language cannot be empty")

    python_executable = translation.get("python_executable")
    if python_executable is not None:
        python_executable = str(python_executable).strip() or None

    return AppConfig(
        storage=StorageConfig(
            raw=_path(storage.get("raw"), "storage.raw"),
            translated=_path(storage.get("translated"), "storage.translated"),
        ),
        download=download_config,
        translation=TranslationConfig(
            translator=translator,
            target_language=target_language,
            python_executable=python_executable,
        ),
        database=DatabaseConfig(path=_path(database.get("path"), "database.path")),
    )
