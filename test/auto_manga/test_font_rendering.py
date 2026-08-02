from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import tempfile
import unicodedata
import unittest
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from PIL import Image

from auto_manga.config import ConfigError, TranslationConfig, load_config
from auto_manga.font_support import inspect_font, inspect_font_directory
from auto_manga.pipeline.translator import MangaTranslator
from auto_manga.tools.font_preview import (
    render_contact_sheet,
    render_font_comparison,
    render_preview,
)
from manga_translator.config import Config
from manga_translator.manga_translator import MangaTranslator as CoreMangaTranslator
from manga_translator.rendering import text_render
from manga_translator.rendering.text_render_eng import Textline, render_lines


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
COMPLETE_FONT = REPOSITORY_ROOT / "fonts" / "Arial-Unicode-Regular.ttf"
INCOMPLETE_FONT = REPOSITORY_ROOT / "fonts" / "anime_ace_3.ttf"
MTO_FONT = REPOSITORY_ROOT / "fonts" / "MTO-Astro-City.ttf"


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
    @unittest.skipUnless(MTO_FONT.is_file(), "local MTO Astro City font is not installed")
    def test_repository_defaults_use_uppercase_manga_dialogue_preset(self) -> None:
        config = load_config(REPOSITORY_ROOT / "auto_manga" / "config.yaml")
        self.assertEqual(config.translation.renderer, "manga2eng")
        self.assertTrue(config.translation.uppercase)
        self.assertEqual(config.translation.font_path, MTO_FONT)
        self.assertEqual(config.translation.font_size_offset, 1)
        self.assertEqual(config.translation.font_size_minimum, 16)
        self.assertEqual(config.translation.line_spacing, 0.01)
        self.assertTrue(config.translation.disable_font_border)

    @unittest.skipUnless(MTO_FONT.is_file(), "local MTO Astro City font is not installed")
    def test_mto_astro_city_config_loads_from_its_real_local_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = load_config(
                write_config(
                    Path(directory),
                    [
                        f"font_path: '{MTO_FONT}'",
                        "renderer: manga2eng",
                        "uppercase: true",
                        "line_spacing: 0.01",
                        "disable_font_border: true",
                    ],
                )
            )
        self.assertEqual(config.translation.font_path, MTO_FONT)
        self.assertEqual(config.translation.renderer, "manga2eng")
        self.assertTrue(config.translation.uppercase)

    def test_relative_font_path_containing_spaces_resolves_as_one_path(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPOSITORY_ROOT) as directory:
            root = Path(directory)
            spaced_font = root / "MTO Astro City.ttf"
            shutil.copyfile(COMPLETE_FONT, spaced_font)
            relative_font = spaced_font.relative_to(REPOSITORY_ROOT)
            config = load_config(
                write_config(root, [f"font_path: './{relative_font}'"])
            )
        self.assertEqual(config.translation.font_path, spaced_font.resolve())

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
    @unittest.skipUnless(MTO_FONT.is_file(), "local MTO Astro City font is not installed")
    def test_mto_has_complete_uppercase_and_lowercase_vietnamese_coverage(self) -> None:
        coverage = inspect_font(MTO_FONT)
        self.assertIn("Astro City", coverage.family_name)
        self.assertEqual(coverage.required_count, 134)
        self.assertEqual(coverage.supported_count, 134)
        self.assertEqual(coverage.uppercase_required_count, 67)
        self.assertEqual(coverage.uppercase_supported_count, 67)
        self.assertTrue(coverage.uppercase_complete)
        self.assertTrue(coverage.complete)
        self.assertFalse(coverage.is_monospace)

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
        self.assertIn("ỵ", coverage.missing_glyphs)
        self.assertFalse(coverage.uppercase_complete)


class TranslationUnicodeNormalizationTest(unittest.TestCase):
    @staticmethod
    def uppercase_config(**translator_overrides: object) -> Config:
        translator = {
            "translator": "deepseek",
            "target_lang": "VIN",
            "enable_post_translation_check": False,
            **translator_overrides,
        }
        return Config(
            translator=translator,
            render={"uppercase": True},
        )

    @staticmethod
    def bare_translator() -> CoreMangaTranslator:
        translator = CoreMangaTranslator.__new__(CoreMangaTranslator)
        translator.prep_manual = False
        translator.load_text = False
        translator.save_text = False
        translator.post_dict = None
        translator.use_mtpe = False
        translator.ignore_errors = False
        translator._gpu_limited_memory = False
        translator.device = None
        translator.context_size = 0
        translator.batch_size = 1
        translator.batch_concurrent = False
        translator.all_page_translations = []
        translator._original_page_texts = []
        translator._model_usage_timestamps = {}
        return translator

    def test_normalize_translation_unicode_composes_vietnamese_nfd(self) -> None:
        nfd_text = "U\u031b\u0300"
        result = CoreMangaTranslator._normalize_translation_unicode(nfd_text)
        self.assertEqual(result, "Ừ")
        self.assertTrue(unicodedata.is_normalized("NFC", result))

    def test_format_translation_uppercases_then_restores_nfc(self) -> None:
        nfd_text = unicodedata.normalize("NFD", "ừ, đối với em")
        result = CoreMangaTranslator._format_translation_text(
            nfd_text, self.uppercase_config()
        )
        self.assertEqual(result, "Ừ, ĐỐI VỚI EM")
        self.assertTrue(unicodedata.is_normalized("NFC", result))
        self.assertFalse(any(unicodedata.combining(char) for char in result))

    def test_run_text_translation_formats_mocked_nfd_result(self) -> None:
        translator = self.bare_translator()
        expected = "Ừ, ĐỐI VỚI EM"
        nfd_text = unicodedata.normalize("NFD", expected.lower())
        translator._dispatch_with_context = AsyncMock(return_value=[nfd_text])
        region = SimpleNamespace(text="source", translation="")
        ctx = SimpleNamespace(text_regions=[region])

        result = asyncio.run(
            translator._run_text_translation(self.uppercase_config(), ctx)
        )

        self.assertEqual(result[0].translation, expected)
        self.assertTrue(unicodedata.is_normalized("NFC", result[0].translation))

    def test_post_dictionary_normalizes_without_reapplying_uppercase(self) -> None:
        translator = self.bare_translator()
        translator._dispatch_with_context = AsyncMock(return_value=["token"])
        region = SimpleNamespace(text="source", translation="")
        ctx = SimpleNamespace(text_regions=[region])

        with tempfile.TemporaryDirectory() as directory:
            post_dict = Path(directory) / "post-dictionary.txt"
            post_dict.write_text("TOKEN u\u031b\u0300\n", encoding="utf-8")
            translator.post_dict = str(post_dict)
            result = asyncio.run(
                translator._run_text_translation(self.uppercase_config(), ctx)
            )

        self.assertEqual(result[0].translation, "ừ")
        self.assertTrue(unicodedata.is_normalized("NFC", result[0].translation))

    def test_batch_translation_formats_nfd_with_the_shared_helper(self) -> None:
        translator = self.bare_translator()
        nfd_text = unicodedata.normalize("NFD", "ừ, đối với em")
        translator._batch_translate_texts = AsyncMock(return_value=[nfd_text])
        translator._report_progress = AsyncMock()
        region = SimpleNamespace(text="source", translation="")
        ctx = SimpleNamespace(text_regions=[region])
        translator._apply_post_translation_processing = AsyncMock(
            return_value=[region]
        )

        asyncio.run(
            translator._batch_translate_contexts(
                [(ctx, self.uppercase_config())], batch_size=1
            )
        )

        self.assertEqual(region.translation, "Ừ, ĐỐI VỚI EM")
        self.assertTrue(unicodedata.is_normalized("NFC", region.translation))

    def test_concurrent_translation_formats_nfd_with_the_shared_helper(self) -> None:
        translator = self.bare_translator()
        nfd_text = unicodedata.normalize("NFD", "ừ, đối với em")
        translator._batch_translate_texts = AsyncMock(return_value=[nfd_text])
        region = SimpleNamespace(text="source", translation="")
        ctx = SimpleNamespace(text_regions=[region])
        translator._apply_post_translation_processing = AsyncMock(
            return_value=[region]
        )

        asyncio.run(
            translator._concurrent_translate_contexts(
                [(ctx, self.uppercase_config())]
            )
        )

        self.assertEqual(region.translation, "Ừ, ĐỐI VỚI EM")
        self.assertTrue(unicodedata.is_normalized("NFC", region.translation))

    def test_shared_post_processing_normalizes_dictionary_output(self) -> None:
        translator = self.bare_translator()
        region = SimpleNamespace(text="source", translation="TOKEN")
        ctx = SimpleNamespace(text_regions=[region])

        with tempfile.TemporaryDirectory() as directory:
            post_dict = Path(directory) / "post-dictionary.txt"
            post_dict.write_text("TOKEN u\u031b\u0300\n", encoding="utf-8")
            translator.post_dict = str(post_dict)
            result = asyncio.run(
                translator._apply_post_translation_processing(
                    ctx, self.uppercase_config()
                )
            )

        self.assertEqual(result[0].translation, "ừ")
        self.assertTrue(unicodedata.is_normalized("NFC", result[0].translation))

    def test_region_retry_formats_nfd_with_the_shared_helper(self) -> None:
        translator = self.bare_translator()
        translator._validate_translation = AsyncMock(side_effect=[False, True])
        nfd_text = unicodedata.normalize("NFD", "ừ, đối với em")
        region = SimpleNamespace(text="source", translation="invalid")
        config = self.uppercase_config(post_check_max_retry_attempts=2)

        with patch(
            "manga_translator.translators.dispatch",
            new=AsyncMock(return_value=[nfd_text]),
        ):
            result = asyncio.run(
                translator._retry_translation_with_validation(region, config, object())
            )

        self.assertEqual(result, "Ừ, ĐỐI VỚI EM")
        self.assertTrue(unicodedata.is_normalized("NFC", region.translation))

    def test_rendering_entrypoint_defensively_normalizes_translation(self) -> None:
        translator = self.bare_translator()
        nfd_text = unicodedata.normalize("NFD", "Ừ, ĐỐI VỚI EM")
        region = SimpleNamespace(translation=nfd_text)
        image = object()
        ctx = SimpleNamespace(text_regions=[region], img_inpainted=image)
        config = Config(render={"renderer": "none"})

        rendered = asyncio.run(translator._run_text_rendering(config, ctx))

        self.assertIs(rendered, image)
        self.assertEqual(region.translation, "Ừ, ĐỐI VỚI EM")
        self.assertTrue(unicodedata.is_normalized("NFC", region.translation))

    @unittest.skipUnless(MTO_FONT.is_file(), "local MTO Astro City font is not installed")
    def test_pipeline_nfd_renders_like_nfc_with_only_mto_selected(self) -> None:
        translator = self.bare_translator()
        expected = "Ừ, ĐỐI VỚI EM"
        nfd_text = unicodedata.normalize("NFD", expected)
        region = SimpleNamespace(translation=nfd_text)
        ctx = SimpleNamespace(text_regions=[region], img_inpainted=object())
        asyncio.run(
            translator._run_text_rendering(
                Config(render={"renderer": "none"}), ctx
            )
        )

        text_render.set_font(str(MTO_FONT))
        self.assertEqual(len(text_render.FONT_SELECTION), 1)

        def render(text: str) -> bytes:
            image = render_lines(
                [Textline(text, pos_x=0, pos_y=0, length=520)],
                canvas_h=100,
                canvas_w=560,
                font_size=36,
                stroke_width=0,
                line_spacing=0.01,
                fg=(0, 0, 0),
                bg=None,
            )
            return image.tobytes()

        expected_render = render(expected)
        self.assertNotEqual(render(nfd_text), expected_render)
        self.assertEqual(render(region.translation), expected_render)


class RenderingWrapperTest(unittest.TestCase):
    def test_wrapper_passes_font_cli_argument_and_real_render_config(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "raw"
            output_path = root / "translated"
            input_path.mkdir()
            (input_path / "001.jpg").write_bytes(jpeg_bytes())
            spaced_font = root / "MTO Astro City.ttf"
            shutil.copyfile(COMPLETE_FONT, spaced_font)
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
                font_path=spaced_font,
                renderer="manga2eng",
                alignment="center",
                direction="horizontal",
                uppercase=True,
                font_size_offset=1,
                font_size_minimum=16,
                no_hyphenation=True,
                line_spacing=0.01,
                disable_font_border=True,
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
                command[command.index("--font-path") + 1], str(spaced_font)
            )
            self.assertIn(str(spaced_font), command)
            self.assertLess(command.index("--font-path"), command.index("local"))
            self.assertEqual(
                generated_config,
                {
                    "translator": {"translator": "deepseek", "target_lang": "VIN"},
                    "render": {
                        "renderer": "manga2eng",
                        "alignment": "center",
                        "direction": "horizontal",
                        "uppercase": True,
                        "font_size_offset": 1,
                        "font_size_minimum": 16,
                        "no_hyphenation": True,
                        "line_spacing": 0.01,
                        "disable_font_border": True,
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

    def test_contact_sheet_compares_every_repository_font(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = render_contact_sheet(
                REPOSITORY_ROOT / "fonts", Path(directory) / "contact-sheet.png"
            )
            with Image.open(output) as image:
                self.assertEqual(image.format, "PNG")
                font_count = len(inspect_font_directory(REPOSITORY_ROOT / "fonts"))
                self.assertEqual(image.size, (3190, 155 + 380 * font_count))
                self.assertIsNotNone(image.getbbox())

    def test_focused_font_comparison_renders_without_network(self) -> None:
        preferred = MTO_FONT if MTO_FONT.is_file() else COMPLETE_FONT
        fonts = [preferred, INCOMPLETE_FONT, COMPLETE_FONT]
        with tempfile.TemporaryDirectory() as directory:
            output = render_font_comparison(
                fonts, Path(directory) / "font-comparison.png"
            )
            with Image.open(output) as image:
                self.assertEqual(image.format, "PNG")
                font_count = len(dict.fromkeys(path.resolve() for path in fonts))
                self.assertEqual(image.size, (3660, 180 + 520 * font_count))
                self.assertIsNotNone(image.getbbox())

    def test_uppercase_preview_reports_settings_and_uses_one_column(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            spaced_font = Path(directory) / "MTO Astro City.ttf"
            shutil.copyfile(COMPLETE_FONT, spaced_font)
            output = render_preview(
                spaced_font,
                Path(directory) / "uppercase-preview.png",
                uppercase_only=True,
                font_size_offset=1,
                line_spacing=-1,
                disable_font_border=True,
            )
            with Image.open(output) as image:
                self.assertEqual(image.size, (720, 1160))

    def test_manga2eng_text_remains_visible_when_border_is_disabled(self) -> None:
        text_render.set_font(str(COMPLETE_FONT))
        text_render.get_char_glyph.cache_clear()
        rendered = render_lines(
            [Textline("ĐỐI VỚI EM", pos_x=0, pos_y=0, length=260)],
            canvas_h=100,
            canvas_w=400,
            font_size=36,
            stroke_width=0,
            line_spacing=0.01,
            fg=(0, 0, 0),
            bg=None,
        )
        alpha = rendered.getchannel("A")
        self.assertIsNotNone(alpha.getbbox())

    def test_explicit_core_font_disables_silent_fallback_selection(self) -> None:
        selected_font = MTO_FONT if MTO_FONT.is_file() else COMPLETE_FONT
        text_render.set_font(str(selected_font))
        self.assertEqual(len(text_render.FONT_SELECTION), 1)
        self.assertNotEqual(text_render.FONT_SELECTION[0].get_char_index("Đ"), 0)


if __name__ == "__main__":
    unittest.main()
