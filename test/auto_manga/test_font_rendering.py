from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from auto_manga.config import ConfigError, TranslationConfig, load_config
from auto_manga.font_support import inspect_font
from auto_manga.pipeline.translator import MangaTranslator
from auto_manga.tools.font_preview import render_preview


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
COMPLETE_FONT = REPOSITORY_ROOT / "fonts" / "Arial-Unicode-Regular.ttf"
INCOMPLETE_FONT = REPOSITORY_ROOT / "fonts" / "anime_ace_3.ttf"


def jpeg_bytes() -> bytes:
    output = BytesIO()
    Image.new("RGB", (4, 4), color="white").save(output, format="JPEG")
    return output.getvalue()


def write_config(root: Path, translation_lines: list[str]) -> Path:
    config_path = root / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "storage:",
                f"  raw: '{root / 'raw'}'",
                f"  translated: '{root / 'translated'}'",
                "download: {}",
                "translation:",
                *(f"  {line}" for line in translation_lines),
                "database:",
                f"  path: '{root / 'state.db'}'",
            ]
        ),
        encoding="utf-8",
    )
    return config_path


class RenderingConfigTest(unittest.TestCase):
    def test_repository_defaults_preserve_mixed_case_and_use_tunable_renderer(self) -> None:
        config = load_config(REPOSITORY_ROOT / "auto_manga" / "config.yaml")
        self.assertEqual(config.translation.renderer, "default")
        self.assertFalse(config.translation.uppercase)
        self.assertEqual(config.translation.font_path, COMPLETE_FONT)

    def test_old_config_without_rendering_settings_still_loads(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = load_config(
                write_config(
                    Path(directory),
                    ["translator: deepseek", "target_language: VIN"],
                )
            )

        self.assertIsNone(config.translation.font_path)
        self.assertEqual(config.translation.renderer, "default")
        self.assertEqual(config.translation.alignment, "auto")
        self.assertEqual(config.translation.direction, "auto")
        self.assertFalse(config.translation.uppercase)

    def test_rendering_config_loads_and_relative_font_resolves_from_working_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = load_config(
                write_config(
                    Path(directory),
                    [
                        "translator: deepseek",
                        "target_language: VIN",
                        "font_path: ./fonts/Arial-Unicode-Regular.ttf",
                        "renderer: manga2eng",
                        "alignment: center",
                        "direction: horizontal",
                        "uppercase: false",
                        "font_size_offset: 2",
                        "font_size_minimum: 18",
                        "no_hyphenation: true",
                        "line_spacing: -0.05",
                        "disable_font_border: false",
                    ],
                )
            )

        self.assertEqual(config.translation.font_path, COMPLETE_FONT)
        self.assertEqual(config.translation.renderer, "manga2eng")
        self.assertEqual(config.translation.alignment, "center")
        self.assertEqual(config.translation.direction, "horizontal")
        self.assertEqual(config.translation.font_size_offset, 2)
        self.assertEqual(config.translation.font_size_minimum, 18)
        self.assertTrue(config.translation.no_hyphenation)
        self.assertEqual(config.translation.line_spacing, -0.05)

    def test_missing_font_has_a_useful_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = write_config(Path(directory), ["font_path: ./fonts/missing.ttf"])
            with self.assertRaisesRegex(ConfigError, "font_path does not exist"):
                load_config(config_path)

    def test_font_directory_and_unsupported_extension_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ConfigError, "font_path is not a file"):
                load_config(write_config(root, [f"font_path: '{root}'"]))

            invalid_font = root / "font.txt"
            invalid_font.write_text("not a font", encoding="utf-8")
            with self.assertRaisesRegex(ConfigError, "supported extension"):
                load_config(write_config(root, [f"font_path: '{invalid_font}'"]))

    def test_invalid_enum_values_are_rejected(self) -> None:
        cases = (
            ("renderer: imaginary", "translation.renderer"),
            ("alignment: middle", "translation.alignment"),
            ("direction: diagonal", "translation.direction"),
        )
        for setting, message in cases:
            with self.subTest(setting=setting), tempfile.TemporaryDirectory() as directory:
                with self.assertRaisesRegex(ConfigError, message):
                    load_config(write_config(Path(directory), [setting]))

    def test_font_size_values_are_strict_integers_and_bounded(self) -> None:
        cases = (
            "font_size_offset: 1.5",
            "font_size_offset: true",
            "font_size_minimum: 0",
            "font_size_minimum: 18.5",
        )
        for setting in cases:
            with self.subTest(setting=setting), tempfile.TemporaryDirectory() as directory:
                with self.assertRaisesRegex(ConfigError, "font_size"):
                    load_config(write_config(Path(directory), [setting]))

    def test_unsafe_line_spacing_and_misspelled_setting_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ConfigError, "between -0.5 and 2.0"):
                load_config(write_config(root, ["line_spacing: -2"]))
            with self.assertRaisesRegex(ConfigError, "did you mean 'alignment'"):
                load_config(write_config(root, ["aligment: center"]))

    def test_font_missing_vietnamese_glyphs_fails_immediately(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ConfigError, "missing required Vietnamese glyphs"):
                load_config(
                    write_config(
                        Path(directory), [f"font_path: '{INCOMPLETE_FONT}'"]
                    )
                )


class FontCoverageTest(unittest.TestCase):
    def test_checker_detects_complete_vietnamese_coverage(self) -> None:
        coverage = inspect_font(COMPLETE_FONT)
        self.assertTrue(coverage.complete)
        self.assertEqual(coverage.missing_glyphs, ())
        self.assertFalse(coverage.is_monospace)

    def test_checker_reports_missing_vietnamese_glyphs(self) -> None:
        coverage = inspect_font(INCOMPLETE_FONT)
        self.assertFalse(coverage.complete)
        self.assertIn("Đ", coverage.missing_glyphs)
        self.assertIn("Ỵ", coverage.missing_glyphs)


class RenderingWrapperTest(unittest.TestCase):
    def test_wrapper_passes_font_cli_argument_and_real_render_config(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "raw"
            output_path = root / "translated"
            input_path.mkdir()
            (input_path / "001.jpg").write_bytes(jpeg_bytes())
            observed: dict[str, object] = {}

            def runner(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
                observed["command"] = command
                observed["kwargs"] = kwargs
                config_path = Path(command[command.index("--config-file") + 1])
                observed["config"] = json.loads(config_path.read_text(encoding="utf-8"))
                output_path.mkdir(exist_ok=True)
                shutil.copyfile(input_path / "001.jpg", output_path / "001.jpg")
                return subprocess.CompletedProcess(command, 0, stdout="done", stderr="")

            config = TranslationConfig(
                "deepseek",
                "VIN",
                font_path=COMPLETE_FONT,
                renderer="manga2eng",
                alignment="center",
                direction="horizontal",
                uppercase=False,
                font_size_offset=2,
                font_size_minimum=18,
                no_hyphenation=True,
                line_spacing=-0.05,
                disable_font_border=False,
            )
            secret = "deepseek-secret-must-not-leak"
            with patch.dict(os.environ, {"DEEPSEEK_API_KEY": secret}), self.assertLogs(
                "auto_manga.pipeline.translator", level="DEBUG"
            ) as captured:
                MangaTranslator(config, runner=runner, project_root=root).translate_folder(
                    input_path, output_path
                )

            command = observed["command"]
            kwargs = observed["kwargs"]
            generated_config = observed["config"]
            self.assertIsInstance(command, list)
            self.assertNotIn("shell", kwargs)
            self.assertEqual(
                command[command.index("--font-path") + 1], str(COMPLETE_FONT)
            )
            self.assertLess(command.index("--font-path"), command.index("local"))
            self.assertEqual(
                generated_config,
                {
                    "translator": {"translator": "deepseek", "target_lang": "VIN"},
                    "render": {
                        "renderer": "manga2eng",
                        "alignment": "center",
                        "direction": "horizontal",
                        "uppercase": False,
                        "font_size_offset": 2,
                        "font_size_minimum": 18,
                        "no_hyphenation": True,
                        "line_spacing": -0.05,
                        "disable_font_border": False,
                    },
                },
            )
            self.assertNotIn(secret, json.dumps(generated_config))
            self.assertNotIn(secret, "\n".join(captured.output))


class FontPreviewTest(unittest.TestCase):
    def test_preview_renders_offline_png(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = render_preview(COMPLETE_FONT, Path(directory) / "preview.png")
            with Image.open(output) as image:
                self.assertEqual(image.format, "PNG")
                self.assertEqual(image.size, (1440, 1160))
                self.assertIsNotNone(image.getbbox())


if __name__ == "__main__":
    unittest.main()
