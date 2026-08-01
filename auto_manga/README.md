# Automatic manga pipeline

This package is a separate automation layer around `manga-image-translator`. It only
supports sources that the user is allowed to access and download. It does not implement
authentication, paywall, CAPTCHA, anti-bot, or access-control bypasses.

## Supported sources

### MangaDex

`MangaDexSource` uses only the public official API and MangaDex@Home endpoints. It
accepts canonical title/chapter URLs and raw UUIDs:

```bash
python main.py manga 'https://mangadex.org/title/MANGA_UUID/optional-slug' --chapters 1-5
python main.py chapter 'https://mangadex.org/chapter/CHAPTER_UUID'
python main.py manga 'MANGA_UUID' --latest
python main.py chapter 'CHAPTER_UUID'
```

Configure feed languages and image quality in `auto_manga/config.yaml`:

```yaml
sources:
  mangadex:
    translated_languages: [en, ja, ko, zh, zh-hk]
    data_saver: false
```

Feed pagination is automatic. For duplicate releases with the same volume, chapter,
and language, the adapter chooses the highest entity version, then the newest readable
date, then the lexicographically smallest UUID. External chapters and releases without
public pages are skipped. Direct chapter commands process the explicitly requested
chapter regardless of the feed language list.

MangaDex requires a real User-Agent and conservative request rates; the adapter provides
both and honors `Retry-After` on HTTP 429. It does not authenticate or bypass unavailable
or restricted chapters. MangaDex availability does not grant permission to redistribute
or translate a work, so users remain responsible for the rights to process it and for
MangaDex/scanlation-group attribution requirements.

### Example JSON manifest

`ExampleSource` accepts a public HTTP(S) JSON manifest. It is a reference adapter, not a
general website scraper:

```json
{
  "id": "public-manga-id",
  "title": "Public domain manga",
  "chapters": [
    {
      "id": "chapter-1",
      "number": "1",
      "title": "Chapter 1",
      "pages": [
        "https://public.example/manga/chapter-1/page-1.jpg",
        "https://public.example/manga/chapter-1/page-2.jpg"
      ]
    }
  ]
}
```

Page entries can also be objects such as
`{"index": 1, "image_url": "https://..."}`. A direct chapter manifest uses top-level
`manga`, `chapter`, and `pages` fields.

## Install and run

The core project currently supports Python 3.10 or 3.11. Python 3.11 is recommended:

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
export DEEPSEEK_API_KEY="..."
```

Keys are read by the existing translator core. They are never stored in YAML or SQLite.

## Vietnamese lettering and fonts

The repository configuration selects compact uppercase dialogue with the core's
bubble-aware renderer:

```yaml
translation:
  translator: "deepseek"
  target_language: "VIN"

  font_path: "./fonts/Arial-Unicode-Regular.ttf"
  renderer: "manga2eng"
  alignment: "center"
  direction: "horizontal"

  uppercase: true
  font_size_offset: 1
  font_size_minimum: 16
  no_hyphenation: true
  line_spacing: 0.01
  disable_font_border: true
```

Relative font and storage paths follow the existing project convention: they resolve
from the current working directory. Run the commands from the repository root, where
`./fonts/...` is predictable. The wrapper passes `font_path` to the core as the global
`--font-path` CLI argument and writes the remaining values to the core JSON `render`
section. It uses a subprocess argument list and never invokes a shell.

`anime_ace_3.ttf` is visually the closest existing candidate to clean uppercase
scanlation lettering, but it is not valid for Vietnamese: the sample itself needs glyphs
such as `Đ`, `Ữ`, `Ờ`, `Ế`, `Ề`, `Ọ`, and `Ố` that the font does not contain. It is never
silently combined with fallback glyphs. `Arial-Unicode-Regular.ttf` remains the default
because it is the only proportional font already in this repository with complete
coverage of the required Vietnamese glyphs. It is a compatibility fallback, not an exact
match for Anime Ace, CC Wild Words, or any font in a supplied reference image. No new font
file is bundled by this automation layer.

For a closer result, provide another licensed, medium-weight, moderately condensed
TTF/OTF/TTC font with complete Vietnamese coverage. A true monospace font is not
recommended for dialogue: equal character widths make short words and punctuation look
mechanical and waste space inside speech bubbles. Small handwritten or italic side
comments are not classified into a second font automatically.

Font validation is fail-fast. A configured font that is missing, is a directory, has an
unsupported extension, cannot be parsed, or lacks any required Vietnamese glyph stops
configuration loading with a specific error. The core therefore cannot silently mix in
fallback glyphs or render missing-character boxes for a configured font.

The checked-in font audit (face zero, matching how the core opens a font path) is:

| Font | Complete Vietnamese | Missing glyphs | Suitable default |
| --- | --- | --- | --- |
| `Arial-Unicode-Regular.ttf` | yes | none | yes; proportional compatibility fallback |
| `NotoSansMonoCJK-VF.ttf.ttc` | yes | none | no; true monospace |
| `anime_ace.ttf` | no | `Ă Đ Ơ Ư ă đ ơ ư Ả Ạ Ắ Ằ Ẳ Ẵ Ặ Ấ Ầ Ẩ Ẫ Ậ Ẻ Ẽ Ẹ Ế Ề Ể Ễ Ệ Ỉ Ĩ Ị Ỏ Ọ Ố Ồ Ổ Ỗ Ộ Ớ Ờ Ở Ỡ Ợ Ủ Ũ Ụ Ứ Ừ Ử Ữ Ự Ỳ Ỷ Ỹ Ỵ` | no |
| `anime_ace_3.ttf` | no | `Ă Đ Ơ Ư ă đ ơ ư Ả Ạ Ắ Ằ Ẳ Ẵ Ặ Ấ Ầ Ẩ Ẫ Ậ Ẻ Ẽ Ẹ Ế Ề Ể Ễ Ệ Ỉ Ĩ Ị Ỏ Ọ Ố Ồ Ổ Ỗ Ộ Ớ Ờ Ở Ỡ Ợ Ủ Ũ Ụ Ứ Ừ Ử Ữ Ự Ỳ Ỷ Ỹ Ỵ` | no |
| `comic shanns 2.ttf` | no | `Đ Ơ Ư đ ơ ư Ả Ạ Ắ Ằ Ẳ Ẵ Ặ Ấ Ầ Ẩ Ẫ Ậ Ẻ Ẽ Ẹ Ế Ề Ể Ễ Ệ Ỉ Ị Ỏ Ọ Ố Ồ Ổ Ỗ Ộ Ớ Ờ Ở Ỡ Ợ Ủ Ụ Ứ Ừ Ử Ữ Ự Ỷ Ỹ Ỵ` | no; fixed-width Latin metrics and incomplete coverage |
| `msgothic.ttc` | no | `Ơ Ư ơ ư Ả Ạ Ắ Ằ Ẳ Ẵ Ặ Ấ Ầ Ẩ Ẫ Ậ Ẻ Ẽ Ẹ Ế Ề Ể Ễ Ệ Ỉ Ị Ỏ Ọ Ố Ồ Ổ Ỗ Ộ Ớ Ờ Ở Ỡ Ợ Ủ Ụ Ứ Ừ Ử Ữ Ự Ỷ Ỹ Ỵ` | no |
| `msyh.ttc` | no | `Ơ Ư ơ ư Ả Ạ Ắ Ằ Ẳ Ẵ Ặ Ấ Ầ Ẩ Ẫ Ậ Ẻ Ẽ Ẹ Ế Ề Ể Ễ Ệ Ỉ Ĩ Ị Ỏ Ọ Ố Ồ Ổ Ỗ Ộ Ớ Ờ Ở Ỡ Ợ Ủ Ụ Ứ Ừ Ử Ữ Ự Ỷ Ỹ Ỵ` | no |

The rendering controls have these effects:

- `renderer: manga2eng` is selected for the new deliberate all-caps target. It detects
  the available bubble area, wraps whole words, centers the lines, and downscales per
  region rather than forcing one fixed font size.
- This core version uppercases inside `seg_eng` and ignores `font_size_offset`,
  `font_size_minimum`, alignment, direction, and hyphenation when `manga2eng` is active.
  The configured `1` and `16` remain useful if switching back to `renderer: default`, but
  they do not tune `manga2eng` itself. `manga2eng_pillow` has similar constraints.
- Use `renderer: default` when mixed case or direct control over alignment, direction,
  minimum size, offset, and hyphenation is more important than bubble-aware wrapping.
- `alignment` and `direction` force the default renderer's region behavior.
- `font_size_offset` adjusts each detected size without imposing one fixed size.
  `font_size_minimum` prevents unreadably small output. These two settings affect the
  core `default` renderer; `manga2eng` performs its own bubble fitting and downscaling.
- `no_hyphenation` disables the normal renderer's dictionary-based word splitting.
  `manga2eng` already wraps whole segmented words.
- `line_spacing` is a multiplier of the current font size, not a pixel count. `0.01`
  keeps the block compact while giving stacked uppercase Vietnamese marks reliable
  clearance in the real `manga2eng` renderer. `-0.05` was readable but left some marks
  too close at full size, `-0.10` crowded them, and the proposed `-1` collapses successive
  baselines completely. Values outside `-0.5..2.0` are rejected.
- `uppercase: true` matches the main-dialogue target. The selected font was checked for
  both uppercase and lowercase Vietnamese coverage before enabling it.
- `disable_font_border: true` produces clean black text in white bubbles. Set it to
  `false` for patterned or uncertain backgrounds that need a white contrast outline.

Generate a local, network-free comparison of mixed case and uppercase lettering:

```bash
python3 -m auto_manga.tools.font_preview \
  --font ./fonts/Arial-Unicode-Regular.ttf \
  --output ./font-preview.png \
  --report-fonts
```

The report lists coverage, missing glyphs, monospace detection, and default suitability
for every font in `fonts/`. Add `--disable-font-border`, adjust `--font-size`, or adjust
the core-style `--line-spacing` multiplier to compare another preview. The utility only
uses Pillow/fontTools; it never invokes OCR, a crawler, MangaDex, or a translator API.

Generate the all-font contact sheet used to select the preset:

```bash
python3 -m auto_manga.tools.font_preview \
  --contact-sheet \
  --output ./vietnamese-font-contact-sheet.png \
  --report-fonts
```

Every row renders directly with that font, including visible missing-glyph boxes for
incomplete candidates; samples never borrow characters from Arial or another fallback.
The columns compare mixed case, uppercase, border on/off, and spacing multipliers
`-1.00`, `-0.05`, and `+0.08`. The intentionally broken `-1.00` column documents why
that proposed value is not used in production.

Changing a font does not invalidate an already valid translated folder. `resume` will
skip that output. To rerender one chapter safely, move its translated folder to a backup
name and run the same direct chapter command again:

```bash
mv "data/translated/MANGA/chapter-001" \
  "data/translated/MANGA/chapter-001.before-font-change"
python main.py chapter "MANGADEX_CHAPTER_URL"
```

The missing output causes the existing record to be repaired and translated again;
the raw source images and SQLite database remain intact. After inspecting the new
result, keep or remove the backup manually. Do not delete raw images or edit database
statuses to force a rerender.

## Run the pipeline

Edit `auto_manga/config.yaml`, then run:

```bash
python main.py manga https://public.example/manifest.json --chapters 1-5
python main.py manga https://public.example/manifest.json --latest
python main.py chapter 'https://public.example/manifest.json#chapter=chapter-1'
python main.py resume
```

Source selection is automatic. Use `--source mangadex` or `--source example` to force an
adapter. Without `--chapters` or `--latest`, `manga` processes every discovered chapter.
The wrapper invokes the existing CLI with an argument list (never a shell command):

```text
python -m manga_translator --font-path FONT local \
  -i RAW_FOLDER -o OUTPUT_FOLDER --config-file TEMP.json
```

Valid existing input and output images are reused. `resume` resets interrupted
`downloading` work to `pending`, resets interrupted `translating` work to `downloaded`,
and retries `pending`, `downloaded`, and `failed` chapters. It also revalidates chapters
marked `translated` and repairs their state if the raw or translated files are incomplete.
