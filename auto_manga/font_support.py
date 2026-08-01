from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from fontTools.ttLib import TTFont, TTLibError


VIETNAMESE_GLYPHS = tuple(
    dict.fromkeys(
        """
        Ă Â Đ Ê Ô Ơ Ư
        ă â đ ê ô ơ ư
        Á À Ả Ã Ạ
        Ắ Ằ Ẳ Ẵ Ặ
        Ấ Ầ Ẩ Ẫ Ậ
        É È Ẻ Ẽ Ẹ
        Ế Ề Ể Ễ Ệ
        Í Ì Ỉ Ĩ Ị
        Ó Ò Ỏ Õ Ọ
        Ố Ồ Ổ Ỗ Ộ
        Ớ Ờ Ở Ỡ Ợ
        Ú Ù Ủ Ũ Ụ
        Ứ Ừ Ử Ữ Ự
        Ý Ỳ Ỷ Ỹ Ỵ
        """.split()
    )
)

SUPPORTED_FONT_EXTENSIONS = frozenset({".otf", ".ttc", ".ttf"})


class _KnownPostTableWarningFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "extra bytes in post.stringData array" not in record.getMessage()


class FontInspectionError(ValueError):
    """Raised when a font cannot be inspected safely."""


@dataclass(frozen=True)
class FontCoverage:
    path: Path
    missing_glyphs: tuple[str, ...]
    is_monospace: bool

    @property
    def complete(self) -> bool:
        return not self.missing_glyphs

    @property
    def suitable_as_default(self) -> bool:
        return self.complete and not self.is_monospace


def inspect_font(path: str | Path) -> FontCoverage:
    """Inspect face zero, matching how the core opens TTF/OTF/TTC font paths."""
    font_path = Path(path)
    post_logger = logging.getLogger("fontTools.ttLib.tables._p_o_s_t")
    warning_filter = _KnownPostTableWarningFilter()
    post_logger.addFilter(warning_filter)
    try:
        try:
            font = TTFont(font_path, fontNumber=0, lazy=True)
        except (OSError, TTLibError) as exc:
            raise FontInspectionError(f"Cannot inspect font {font_path}: {exc}") from exc

        try:
            codepoints: set[int] = set()
            for table in font["cmap"].tables:
                if table.isUnicode():
                    codepoints.update(table.cmap)
            missing = tuple(
                char for char in VIETNAMESE_GLYPHS if ord(char) not in codepoints
            )
            return FontCoverage(font_path, missing, _is_monospace(font, codepoints))
        except (KeyError, AttributeError, TypeError, ValueError) as exc:
            raise FontInspectionError(
                f"Cannot read Unicode glyph data from font {font_path}: {exc}"
            ) from exc
        finally:
            font.close()
    finally:
        post_logger.removeFilter(warning_filter)


def _is_monospace(font: TTFont, codepoints: set[int]) -> bool:
    cmap = font.getBestCmap() or {}
    metrics = font["hmtx"].metrics
    advances = {
        metrics[glyph_name][0]
        for codepoint in range(ord("A"), ord("Z") + 1)
        if codepoint in codepoints
        for glyph_name in [cmap.get(codepoint)]
        if glyph_name in metrics
    }
    return len(advances) == 1


def format_missing_glyphs(missing_glyphs: tuple[str, ...]) -> str:
    return " ".join(missing_glyphs)


def inspect_font_directory(directory: str | Path) -> list[FontCoverage]:
    font_directory = Path(directory)
    return [
        inspect_font(path)
        for path in sorted(font_directory.iterdir(), key=lambda item: item.name.lower())
        if path.is_file() and path.suffix.lower() in SUPPORTED_FONT_EXTENSIONS
    ]
