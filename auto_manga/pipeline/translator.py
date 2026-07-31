from __future__ import annotations

from collections.abc import Callable
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from auto_manga.config import TranslationConfig
from auto_manga.crawler.downloader import chapter_images, is_valid_image


LOGGER = logging.getLogger(__name__)


class TranslationError(RuntimeError):
    """Raised when manga-image-translator cannot translate a folder."""


def validate_translated_output(input_path: Path, output_path: Path) -> bool:
    input_images = chapter_images(input_path)
    output_images = chapter_images(output_path)
    if not input_images or len(output_images) != len(input_images):
        return False
    if [image.name for image in output_images] != [image.name for image in input_images]:
        return False
    return all(is_valid_image(image) for image in output_images)


class MangaTranslator:
    """Small subprocess wrapper around manga-image-translator's public CLI."""

    def __init__(
        self,
        config: TranslationConfig,
        runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
        project_root: Path | None = None,
    ) -> None:
        self.config = config
        self.runner = runner
        self.project_root = project_root or Path(__file__).resolve().parents[2]

    def translate_folder(self, input_path: Path, output_path: Path) -> None:
        if not input_path.is_dir():
            raise TranslationError(f"Input folder does not exist: {input_path}")
        output_path.mkdir(parents=True, exist_ok=True)
        self._remove_invalid_outputs(input_path, output_path)

        core_config = {
            "translator": {
                "translator": self.config.translator,
                "target_lang": self.config.target_language,
            }
        }
        config_path = self._write_core_config(core_config)
        command = [
            self.config.python_executable or sys.executable,
            "-m",
            "manga_translator",
            "local",
            "-i",
            str(input_path),
            "-o",
            str(output_path),
            "--config-file",
            str(config_path),
        ]

        try:
            result = self.runner(
                command,
                cwd=self.project_root,
                check=False,
                capture_output=True,
                text=True,
            )
        except OSError as exc:
            raise TranslationError(f"Cannot start manga-image-translator: {exc}") from exc
        finally:
            config_path.unlink(missing_ok=True)

        if result.stdout:
            LOGGER.debug("Translator output:\n%s", result.stdout.rstrip())
        if result.returncode != 0:
            detail = self._last_output_line(result.stderr or result.stdout)
            suffix = f": {detail}" if detail else ""
            raise TranslationError(
                f"manga-image-translator exited with code {result.returncode}{suffix}"
            )
        if result.stderr:
            LOGGER.debug("Translator diagnostic output:\n%s", result.stderr.rstrip())

    def _write_core_config(self, config: dict[str, object]) -> Path:
        descriptor, filename = tempfile.mkstemp(prefix="auto-manga-", suffix=".json")
        config_path = Path(filename)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as config_file:
                json.dump(config, config_file)
        except Exception:
            config_path.unlink(missing_ok=True)
            raise
        return config_path

    @staticmethod
    def _remove_invalid_outputs(input_path: Path, output_path: Path) -> None:
        expected_names = {image.name for image in chapter_images(input_path)}
        for output in chapter_images(output_path):
            if output.name not in expected_names or not is_valid_image(output):
                LOGGER.warning("Removing invalid translated output %s", output.name)
                output.unlink()

    @staticmethod
    def _last_output_line(output: str | None) -> str:
        lines = [line.strip() for line in (output or "").splitlines() if line.strip()]
        return lines[-1][:500] if lines else ""
