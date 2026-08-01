from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from auto_manga.font_support import (
    FontCoverage,
    format_missing_glyphs,
    inspect_font,
    inspect_font_directory,
)


SAMPLES = (
    "Đây là ngôi đền\nmình đã đến để\nHatsumōde.",
    "Mình không biết\nnó nằm trên đường\nđến trường.",
    "Tuyệt vời! Chúng ta\nhọc cùng lớp rồi,\nMakotsu!",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render an offline Vietnamese lettering preview"
    )
    parser.add_argument("--font", type=Path, required=True, help="TTF, TTC, or OTF font")
    parser.add_argument("--output", type=Path, required=True, help="Output PNG path")
    parser.add_argument("--font-size", type=int, default=38, help="Starting font size")
    parser.add_argument(
        "--line-spacing",
        type=float,
        default=-0.05,
        help="Core-style line-spacing multiplier (default: -0.05)",
    )
    parser.add_argument(
        "--disable-font-border",
        action="store_true",
        help="Render without the white scanlation border",
    )
    parser.add_argument(
        "--report-fonts",
        action="store_true",
        help="Print Vietnamese coverage for every repository font",
    )
    return parser


def _print_coverage(coverage: FontCoverage) -> None:
    missing = format_missing_glyphs(coverage.missing_glyphs) or "-"
    print(
        f"{coverage.path.name}\tcomplete={str(coverage.complete).lower()}"
        f"\tmonospace={str(coverage.is_monospace).lower()}"
        f"\tsuitable_default={str(coverage.suitable_as_default).lower()}"
        f"\tmissing={missing}"
    )


def _fit_font(
    draw: ImageDraw.ImageDraw,
    font_path: Path,
    text: str,
    starting_size: int,
    max_width: int,
    max_height: int,
    spacing_ratio: float,
) -> tuple[ImageFont.FreeTypeFont, int]:
    for size in range(starting_size, 11, -1):
        font = ImageFont.truetype(str(font_path), size)
        spacing = int(size * spacing_ratio)
        bounds = draw.multiline_textbbox(
            (0, 0), text, font=font, spacing=spacing, align="center"
        )
        if bounds[2] - bounds[0] <= max_width and bounds[3] - bounds[1] <= max_height:
            return font, spacing
    font = ImageFont.truetype(str(font_path), 12)
    return font, int(12 * spacing_ratio)


def render_preview(
    font_path: Path,
    output_path: Path,
    *,
    font_size: int = 38,
    line_spacing: float = -0.05,
    disable_font_border: bool = False,
) -> Path:
    coverage = inspect_font(font_path)
    if not coverage.complete:
        raise ValueError(
            f"Font {font_path} is missing required Vietnamese glyphs: "
            f"{format_missing_glyphs(coverage.missing_glyphs)}"
        )
    if font_size < 1:
        raise ValueError("font size must be at least 1")
    if not -0.5 <= line_spacing <= 2.0:
        raise ValueError("line spacing must be between -0.5 and 2.0")

    canvas = Image.new("RGB", (1440, 1160), "white")
    draw = ImageDraw.Draw(canvas)
    label_font = ImageFont.truetype(str(font_path), 24)
    draw.text((360, 36), "Mixed case", fill="black", font=label_font, anchor="mm")
    draw.text((1080, 36), "UPPERCASE", fill="black", font=label_font, anchor="mm")

    bubble_width, bubble_height = 610, 285
    for row, sample in enumerate(SAMPLES):
        top = 80 + row * 350
        for column, text in enumerate((sample, sample.upper())):
            left = 55 + column * 720
            box = (left, top, left + bubble_width, top + bubble_height)
            draw.ellipse(box, fill=(247, 247, 247), outline="black", width=4)
            font, spacing = _fit_font(
                draw,
                font_path,
                text,
                font_size,
                bubble_width - 100,
                bubble_height - 80,
                line_spacing,
            )
            stroke_width = 0 if disable_font_border else max(1, font.size // 14)
            draw.multiline_text(
                ((box[0] + box[2]) // 2, (box[1] + box[3]) // 2),
                text,
                fill="black",
                font=font,
                anchor="mm",
                align="center",
                spacing=spacing,
                stroke_width=stroke_width,
                stroke_fill="white",
            )

    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, format="PNG")
    return output_path


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    font_path = args.font.expanduser().resolve()
    if args.report_fonts:
        repository_root = Path(__file__).resolve().parents[2]
        for coverage in inspect_font_directory(repository_root / "fonts"):
            _print_coverage(coverage)
    output = render_preview(
        font_path,
        args.output,
        font_size=args.font_size,
        line_spacing=args.line_spacing,
        disable_font_border=args.disable_font_border,
    )
    print(f"Preview written to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
