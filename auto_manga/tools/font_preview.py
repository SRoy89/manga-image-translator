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

CONTACT_SAMPLE_MIXED = (
    "Em nghĩ là...\n"
    "Những lời khen đó\n"
    "đến từ ai mới là\n"
    "điều quan trọng\n\n"
    "đối với em."
)
CONTACT_SAMPLE_UPPER = (
    "EM NGHĨ LÀ...\n"
    "NHỮNG LỜI KHEN ĐÓ\n"
    "ĐẾN TỪ AI MỚI LÀ\n"
    "ĐIỀU QUAN TRỌNG\n\n"
    "ĐỐI VỚI EM."
)

CONTACT_VARIANTS = (
    ("Mixed · border · −0.05", CONTACT_SAMPLE_MIXED, True, -0.05),
    ("UPPER · border · −0.05", CONTACT_SAMPLE_UPPER, True, -0.05),
    ("UPPER · no border · −1.00", CONTACT_SAMPLE_UPPER, False, -1.0),
    ("UPPER · no border · −0.05", CONTACT_SAMPLE_UPPER, False, -0.05),
    ("UPPER · no border · +0.08", CONTACT_SAMPLE_UPPER, False, 0.08),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render an offline Vietnamese lettering preview"
    )
    parser.add_argument("--font", type=Path, help="TTF, TTC, or OTF font")
    parser.add_argument("--output", type=Path, required=True, help="Output PNG path")
    parser.add_argument("--font-size", type=int, default=38, help="Starting font size")
    parser.add_argument(
        "--line-spacing",
        type=float,
        default=0.01,
        help="Core-style line-spacing multiplier (default: 0.01)",
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
    parser.add_argument(
        "--contact-sheet",
        action="store_true",
        help="Compare every repository font without substituting missing glyphs",
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
    line_spacing: float = 0.01,
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


def _sample_missing_glyphs(coverage: FontCoverage) -> str:
    sample_characters = set(CONTACT_SAMPLE_MIXED + CONTACT_SAMPLE_UPPER)
    return format_missing_glyphs(
        tuple(char for char in coverage.missing_glyphs if char in sample_characters)
    )


def render_contact_sheet(font_directory: Path, output_path: Path) -> Path:
    """Render all candidate fonts directly; incomplete fonts receive no fallback."""
    coverages = inspect_font_directory(font_directory)
    if not coverages:
        raise ValueError(f"No supported fonts found in {font_directory}")

    panel_width, panel_height = 560, 380
    label_width, header_height = 390, 155
    canvas_width = label_width + panel_width * len(CONTACT_VARIANTS)
    canvas_height = header_height + panel_height * len(coverages)
    canvas = Image.new("RGB", (canvas_width, canvas_height), (225, 225, 225))
    draw = ImageDraw.Draw(canvas)
    ui_font_path = font_directory / "Arial-Unicode-Regular.ttf"
    if ui_font_path.is_file():
        heading_font = ImageFont.truetype(str(ui_font_path), 24)
        label_font = ImageFont.truetype(str(ui_font_path), 19)
        small_font = ImageFont.truetype(str(ui_font_path), 16)
    else:
        heading_font = ImageFont.load_default(size=24)
        label_font = ImageFont.load_default(size=19)
        small_font = ImageFont.load_default(size=16)

    draw.rectangle((0, 0, canvas_width, header_height), fill=(35, 35, 38))
    draw.text(
        (24, 22),
        "Vietnamese manga lettering contact sheet",
        fill="white",
        font=heading_font,
    )
    draw.text(
        (24, 62),
        "Direct font rendering · no fallback glyph substitution",
        fill=(205, 205, 205),
        font=small_font,
    )
    draw.text(
        (24, 92),
        "Spacing is a font-size multiplier; −1.00 intentionally tests the requested preset.",
        fill=(205, 205, 205),
        font=small_font,
    )
    for column, (title, _text, _border, _spacing) in enumerate(CONTACT_VARIANTS):
        draw.multiline_text(
            (label_width + column * panel_width + panel_width // 2, 85),
            title.replace(" · ", "\n"),
            fill="white",
            font=small_font,
            anchor="mm",
            align="center",
            spacing=3,
        )

    for row, coverage in enumerate(coverages):
        top = header_height + row * panel_height
        row_fill = (244, 244, 244) if row % 2 == 0 else (235, 235, 235)
        draw.rectangle((0, top, canvas_width, top + panel_height), fill=row_fill)
        status_color = (24, 112, 62) if coverage.complete else (175, 42, 42)
        metric = "monospace" if coverage.is_monospace else "proportional"
        status = "COMPLETE" if coverage.complete else "INCOMPLETE"
        draw.text((22, top + 35), coverage.path.name, fill="black", font=label_font)
        draw.text(
            (22, top + 75),
            f"{status} · {metric}",
            fill=status_color,
            font=small_font,
        )
        sample_missing = _sample_missing_glyphs(coverage)
        if sample_missing:
            draw.multiline_text(
                (22, top + 115),
                f"Missing in sample:\n{sample_missing}",
                fill=status_color,
                font=small_font,
                spacing=6,
            )

        for column, (_title, text, border, line_spacing) in enumerate(CONTACT_VARIANTS):
            left = label_width + column * panel_width
            panel_box = (left + 12, top + 16, left + panel_width - 12, top + panel_height - 16)
            bubble_box = (
                panel_box[0] + 22,
                panel_box[1] + 18,
                panel_box[2] - 22,
                panel_box[3] - 18,
            )
            draw.rounded_rectangle(panel_box, radius=18, fill="white", outline=(175, 175, 175))
            draw.ellipse(bubble_box, fill="white", outline=(65, 65, 65), width=3)
            font, spacing = _fit_font(
                draw,
                coverage.path,
                text,
                36,
                bubble_box[2] - bubble_box[0] - 80,
                bubble_box[3] - bubble_box[1] - 75,
                line_spacing,
            )
            stroke_width = max(1, font.size // 14) if border else 0
            draw.multiline_text(
                ((bubble_box[0] + bubble_box[2]) // 2, (bubble_box[1] + bubble_box[3]) // 2),
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
    parser = build_parser()
    args = parser.parse_args(argv)
    repository_root = Path(__file__).resolve().parents[2]
    if args.report_fonts:
        for coverage in inspect_font_directory(repository_root / "fonts"):
            _print_coverage(coverage)
    if args.contact_sheet:
        output = render_contact_sheet(repository_root / "fonts", args.output)
    else:
        if args.font is None:
            parser.error("--font is required unless --contact-sheet is used")
        output = render_preview(
            args.font.expanduser().resolve(),
            args.output,
            font_size=args.font_size,
            line_spacing=args.line_spacing,
            disable_font_border=args.disable_font_border,
        )
    print(f"Preview written to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
