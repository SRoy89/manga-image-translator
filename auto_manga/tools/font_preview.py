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

COMPARISON_SAMPLE_MIXED = (
    "Em nghĩ là...\n"
    "Những lời khen đó\n"
    "đến từ ai mới là\n"
    "điều quan trọng\n\n"
    "đối với em.\n\n"
    "Hửm?\n\n"
    "Em đang nhìn gì thế?"
)
COMPARISON_SAMPLE_UPPER = (
    "EM NGHĨ LÀ...\n"
    "NHỮNG LỜI KHEN ĐÓ\n"
    "ĐẾN TỪ AI MỚI LÀ\n"
    "ĐIỀU QUAN TRỌNG\n\n"
    "ĐỐI VỚI EM.\n\n"
    "HỬM?\n\n"
    "EM ĐANG NHÌN GÌ THẾ?"
)

CONTACT_VARIANTS = (
    ("Mixed · border · −0.05", CONTACT_SAMPLE_MIXED, True, -0.05),
    ("UPPER · border · −0.05", CONTACT_SAMPLE_UPPER, True, -0.05),
    ("UPPER · no border · −1.00", CONTACT_SAMPLE_UPPER, False, -1.0),
    ("UPPER · no border · −0.05", CONTACT_SAMPLE_UPPER, False, -0.05),
    ("UPPER · no border · +0.08", CONTACT_SAMPLE_UPPER, False, 0.08),
)

COMPARISON_VARIANTS = (
    ("Mixed · border · spacing 0 · offset 0", COMPARISON_SAMPLE_MIXED, True, 0.0, 0),
    ("UPPER · border · spacing 0 · offset +1", COMPARISON_SAMPLE_UPPER, True, 0.0, 1),
    ("UPPER · no border · spacing −2 · offset −2", COMPARISON_SAMPLE_UPPER, False, -2.0, -2),
    ("UPPER · no border · spacing −1 · offset 0", COMPARISON_SAMPLE_UPPER, False, -1.0, 0),
    ("UPPER · no border · spacing 0 · offset +1", COMPARISON_SAMPLE_UPPER, False, 0.0, 1),
    ("Mixed · no border · spacing 0 · offset +3", COMPARISON_SAMPLE_MIXED, False, 0.0, 3),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render an offline Vietnamese lettering preview"
    )
    parser.add_argument("--font", type=Path, help="TTF, TTC, or OTF font")
    parser.add_argument("--output", type=Path, required=True, help="Output PNG path")
    parser.add_argument("--font-size", type=int, default=38, help="Starting font size")
    parser.add_argument(
        "--font-size-offset",
        type=int,
        default=0,
        help="Preview-only offset added to the starting font size",
    )
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
        "--uppercase",
        action="store_true",
        help="Render only the uppercase column instead of the mixed/uppercase pair",
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
    parser.add_argument(
        "--compare-font",
        action="append",
        type=Path,
        default=[],
        help="Font to add to a focused comparison; repeat for each candidate",
    )
    return parser


def _print_coverage(coverage: FontCoverage) -> None:
    missing = format_missing_glyphs(coverage.missing_glyphs) or "-"
    print(
        f"{coverage.path.name}\tfamily={coverage.family_name}"
        f"\tcomplete={str(coverage.complete).lower()}"
        f"\tsupported={coverage.supported_count}/{coverage.required_count}"
        f"\tuppercase={coverage.uppercase_supported_count}/"
        f"{coverage.uppercase_required_count}"
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
    font_size_offset: int = 0,
    line_spacing: float = 0.01,
    disable_font_border: bool = False,
    uppercase_only: bool = False,
) -> Path:
    coverage = inspect_font(font_path)
    if not coverage.complete:
        raise ValueError(
            f"Font {font_path} is missing required Vietnamese glyphs: "
            f"{format_missing_glyphs(coverage.missing_glyphs)}"
        )
    starting_size = font_size + font_size_offset
    if starting_size < 1:
        raise ValueError("font size plus offset must be at least 1")
    if not -2.0 <= line_spacing <= 2.0:
        raise ValueError("preview line spacing must be between -2.0 and 2.0")

    columns = (("UPPERCASE", True),) if uppercase_only else (
        ("Mixed case", False),
        ("UPPERCASE", True),
    )
    canvas = Image.new("RGB", (720 * len(columns), 1160), "white")
    draw = ImageDraw.Draw(canvas)
    label_font = ImageFont.truetype(str(font_path), 24)
    settings = (
        f"{coverage.family_name} · {font_path.name} · size {font_size}"
        f" · offset {font_size_offset:+d} · spacing {line_spacing:+.2f}"
        f" · border {'off' if disable_font_border else 'on'}"
    )
    settings_font = ImageFont.truetype(str(font_path), 16)
    draw.multiline_text(
        (canvas.width // 2, 22),
        settings.replace(" · size", "\nsize", 1),
        fill="black",
        font=settings_font,
        anchor="mm",
        align="center",
        spacing=2,
    )
    for column, (heading, _uppercase) in enumerate(columns):
        draw.text(
            (360 + column * 720, 58), heading, fill="black", font=label_font, anchor="mm"
        )

    bubble_width, bubble_height = 610, 285
    for row, sample in enumerate(SAMPLES):
        top = 90 + row * 350
        for column, (_heading, uppercase) in enumerate(columns):
            text = sample.upper() if uppercase else sample
            left = 55 + column * 720
            box = (left, top, left + bubble_width, top + bubble_height)
            draw.ellipse(box, fill=(247, 247, 247), outline="black", width=4)
            font, spacing = _fit_font(
                draw,
                font_path,
                text,
                starting_size,
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


def _missing_glyphs_in_text(coverage: FontCoverage, text: str) -> str:
    sample_characters = set(text)
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
        sample_missing = _missing_glyphs_in_text(
            coverage, CONTACT_SAMPLE_MIXED + CONTACT_SAMPLE_UPPER
        )
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


def render_font_comparison(font_paths: list[Path], output_path: Path) -> Path:
    """Compare selected fonts directly without applying any glyph fallback."""
    unique_paths = list(dict.fromkeys(path.expanduser().resolve() for path in font_paths))
    if not unique_paths:
        raise ValueError("At least one comparison font is required")
    coverages = [inspect_font(path) for path in unique_paths]

    panel_width, panel_height = 540, 520
    label_width, header_height = 420, 180
    canvas_width = label_width + panel_width * len(COMPARISON_VARIANTS)
    canvas_height = header_height + panel_height * len(coverages)
    canvas = Image.new("RGB", (canvas_width, canvas_height), (228, 228, 228))
    draw = ImageDraw.Draw(canvas)

    ui_path = next((coverage.path for coverage in coverages if coverage.complete), None)
    if ui_path is None:
        ui_path = unique_paths[0]
    heading_font = ImageFont.truetype(str(ui_path), 24)
    label_font = ImageFont.truetype(str(ui_path), 18)
    small_font = ImageFont.truetype(str(ui_path), 14)

    draw.rectangle((0, 0, canvas_width, header_height), fill=(35, 35, 38))
    draw.text(
        (24, 22),
        "MTO Astro City Vietnamese comparison",
        fill="white",
        font=heading_font,
    )
    draw.text(
        (24, 60),
        "Direct Pillow rendering · no fallback · offsets are preview starting-size offsets",
        fill=(205, 205, 205),
        font=small_font,
    )
    draw.text(
        (24, 88),
        "Spacing −2 and −1 are intentionally destructive test values.",
        fill=(205, 205, 205),
        font=small_font,
    )
    for column, (title, _text, _border, _spacing, _offset) in enumerate(
        COMPARISON_VARIANTS
    ):
        draw.multiline_text(
            (label_width + column * panel_width + panel_width // 2, 94),
            title.replace(" · ", "\n"),
            fill="white",
            font=small_font,
            anchor="mm",
            align="center",
            spacing=2,
        )

    comparison_text = COMPARISON_SAMPLE_MIXED + COMPARISON_SAMPLE_UPPER
    for row, coverage in enumerate(coverages):
        top = header_height + row * panel_height
        row_fill = (245, 245, 245) if row % 2 == 0 else (235, 235, 235)
        draw.rectangle((0, top, canvas_width, top + panel_height), fill=row_fill)
        status_color = (24, 112, 62) if coverage.complete else (175, 42, 42)
        metric = "monospace" if coverage.is_monospace else "proportional"
        status = "COMPLETE" if coverage.complete else "INCOMPLETE"
        draw.text((22, top + 30), coverage.family_name, fill="black", font=label_font)
        draw.text((22, top + 65), coverage.path.name, fill="black", font=small_font)
        draw.text(
            (22, top + 95),
            f"{status} · {metric} · {coverage.supported_count}/"
            f"{coverage.required_count} Vietnamese",
            fill=status_color,
            font=small_font,
        )
        sample_missing = _missing_glyphs_in_text(coverage, comparison_text)
        if sample_missing:
            draw.multiline_text(
                (22, top + 130),
                f"Missing in sample:\n{sample_missing}",
                fill=status_color,
                font=small_font,
                spacing=5,
            )

        for column, (_title, text, border, line_spacing, offset) in enumerate(
            COMPARISON_VARIANTS
        ):
            panel = Image.new("RGB", (panel_width, panel_height), "white")
            panel_draw = ImageDraw.Draw(panel)
            panel_box = (12, 14, panel_width - 12, panel_height - 14)
            bubble_box = (35, 35, panel_width - 35, panel_height - 35)
            panel_draw.rounded_rectangle(
                panel_box, radius=18, fill="white", outline=(175, 175, 175)
            )
            panel_draw.ellipse(bubble_box, fill="white", outline=(65, 65, 65), width=3)
            font, spacing = _fit_font(
                panel_draw,
                coverage.path,
                text,
                31 + offset,
                bubble_box[2] - bubble_box[0] - 65,
                bubble_box[3] - bubble_box[1] - 65,
                line_spacing,
            )
            stroke_width = max(1, font.size // 14) if border else 0
            panel_draw.multiline_text(
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
            canvas.paste(panel, (label_width + column * panel_width, top))

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
    if args.compare_font:
        output = render_font_comparison(args.compare_font, args.output)
    elif args.contact_sheet:
        output = render_contact_sheet(repository_root / "fonts", args.output)
    else:
        if args.font is None:
            parser.error("--font is required unless --contact-sheet is used")
        output = render_preview(
            args.font.expanduser().resolve(),
            args.output,
            font_size=args.font_size,
            font_size_offset=args.font_size_offset,
            line_spacing=args.line_spacing,
            disable_font_border=args.disable_font_border,
            uppercase_only=args.uppercase,
        )
    print(f"Preview written to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
