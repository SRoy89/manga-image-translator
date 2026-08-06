from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path

from auto_manga.config import TranslationConfig
from auto_manga.crawler.downloader import (
    chapter_images,
    is_valid_image,
    validate_chapter_images,
)

LOGGER = logging.getLogger(__name__)


class TranslationError(RuntimeError):
    """Raised when manga-image-translator cannot translate a folder."""


def validate_translated_output(
    input_path: Path,
    output_path: Path,
    expected_count: int | None = None,
) -> bool:
    if input_path.resolve() == output_path.resolve():
        return False
    input_images = chapter_images(input_path)
    if not validate_chapter_images(input_path, expected_count):
        return False
    return validate_chapter_images(output_path, len(input_images))


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
        if input_path.resolve() == output_path.resolve():
            raise TranslationError("Input and output folders must be different")
        output_path.mkdir(parents=True, exist_ok=True)
        self._remove_invalid_outputs(input_path, output_path)

        core_config = {
            "translator": {
                "translator": self.config.translator,
                "target_lang": self.config.target_language,
                "gpt_config": str(self.config.gpt_config)
                if self.config.gpt_config
                else None,
                "dialogue_style_guide": str(self.config.dialogue_style_guide)
                if self.config.dialogue_style_guide
                else None,
                "dialogue_consistency_validator": (
                    self.config.dialogue_consistency_validator
                ),
            },
            "render": {
                "renderer": self.config.renderer,
                "alignment": self.config.alignment,
                "direction": self.config.direction,
                "uppercase": self.config.uppercase,
                "font_size_offset": self.config.font_size_offset,
                "font_size_minimum": self.config.font_size_minimum,
                "no_hyphenation": self.config.no_hyphenation,
                "line_spacing": self.config.line_spacing,
                "disable_font_border": self.config.disable_font_border,
            },
        }
        if self.config.translator == "deepseek_gemini_context":
            core_config["translator"]["pronoun_context"] = {
                "enabled": self.config.pronoun_context.enabled,
                "provider": self.config.pronoun_context.provider,
                "confidence_threshold": (
                    self.config.pronoun_context.confidence_threshold
                ),
                "one_vision_call_per_page": (
                    self.config.pronoun_context.one_vision_call_per_page
                ),
                "max_fallback_rounds": (
                    self.config.pronoun_context.max_fallback_rounds
                ),
                "previous_pages": self.config.pronoun_context.previous_pages,
                "cache_enabled": self.config.pronoun_context.cache_enabled,
                "use_proper_names_when_natural": (
                    self.config.pronoun_context.use_proper_names_when_natural
                ),
                "neutral_on_unresolved": (
                    self.config.pronoun_context.neutral_on_unresolved
                ),
                "model": self.config.pronoun_context.model,
                "timeout": self.config.pronoun_context.timeout,
            }
        config_path = self._write_core_config(core_config)
        command = [
            self.config.python_executable or sys.executable,
            "-m",
            "manga_translator",
        ]
        if self.config.font_path is not None:
            command.extend(["--font-path", str(self.config.font_path)])
        if self.config.dialogue_consistency:
            command.extend(
                [
                    "--dialogue-consistency",
                    "--context-size",
                    str(self.config.context_pages),
                ]
            )
        command.extend(
            [
                "local",
                "-i",
                str(input_path),
                "-o",
                str(output_path),
                "--config-file",
                str(config_path),
            ]
        )

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
        if not validate_translated_output(input_path, output_path):
            detail = self._last_output_line(result.stderr or result.stdout)
            suffix = f": {detail}" if detail else ""
            raise TranslationError(
                f"manga-image-translator produced incomplete output{suffix}"
            )

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
