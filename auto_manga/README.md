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

This installation uses the local MTO Astro City Regular file for compact uppercase
dialogue with the core's bubble-aware renderer:

```yaml
translation:
  translator: "deepseek"
  target_language: "VIN"

  font_path: "./fonts/MTO-Astro-City.ttf"
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

The exact file found locally is `fonts/MTO-Astro-City.ttf`; the family metadata says
`MTO Astro City Regular`. The binary is intentionally ignored by Git. Its metadata only
states `Copyright (c) 2011 by Kandy Pham. All rights reserved.` and contains no license
grant or license URL, so this repository does not redistribute it. Each installation must
provide its own lawfully licensed local copy. A differently named local file, including a
path such as `./fonts/MTO Astro City.ttf`, also works when quoted in YAML.

Font validation is fail-fast. A configured font that is missing, is a directory, has an
unsupported extension, cannot be parsed, or lacks any required Vietnamese glyph stops
configuration loading with a specific error. When an explicit font is configured, the
core now loads only that face; its system fallback chain is used only when no font path is
provided. A sentence therefore cannot silently mix MTO with Arial, MS Gothic, or another
fallback face.

The audit checks 134 precomposed Vietnamese characters: 67 uppercase and their 67
lowercase equivalents. Face zero is inspected, matching how the core opens font paths:

| Font | Coverage | Uppercase | Metrics | Production choice |
| --- | --- | --- | --- | --- |
| `MTO-Astro-City.ttf` | 134/134 | 67/67 | proportional | preferred local default |
| `Arial-Unicode-Regular.ttf` | 134/134 | 67/67 | proportional | compatible generic fallback |
| `NotoSansMonoCJK-VF.ttf.ttc` | 134/134 | 67/67 | monospace | rejected for dialogue style |
| `anime_ace_3.ttf` | 32/134 | 16/67 | proportional | rejected; missing Vietnamese |

Run `--report-fonts` for exact missing-character lists for every local candidate.

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
- `line_spacing` is a multiplier of the current font size, not a pixel count. MTO was
  tested at `-1`, `-0.10`, `-0.05`, `0.01`, and `0.08` through the real renderer. `-1`
  collapses the lines, negative values crowd stacked accents, and `0.01` is the compact
  readable choice. Production values outside `-0.5..2.0` are rejected; the preview tool
  permits `-2/-1` only to make those failure modes visible.
- `uppercase: true` matches the main-dialogue target. To preserve mixed case, set it to
  `false` and switch to `renderer: default`; `manga2eng` itself always uppercases text.
- `disable_font_border: true` produces clean black text in white bubbles. Set it to
  `false` for patterned or uncertain backgrounds that need a white contrast outline.

Generate a local, network-free MTO preview. Paths containing spaces are passed as one
argument when quoted:

```bash
python3 -m auto_manga.tools.font_preview \
  --font "./fonts/MTO-Astro-City.ttf" \
  --output ./font-preview-mto-astro-city.png \
  --uppercase \
  --font-size-offset 1 \
  --line-spacing 0.01 \
  --disable-font-border \
  --report-fonts
```

The report lists coverage, missing glyphs, monospace detection, and default suitability
for every font in `fonts/`. Omit `--uppercase` to see mixed case and uppercase side by
side. Adjust `--font-size`, `--font-size-offset`, or `--line-spacing`; omit
`--disable-font-border` to inspect the white outline used on noisy backgrounds. The
utility only uses Pillow/fontTools and never invokes OCR, MangaDex, or a translator API.

Generate the focused MTO/Anime Ace/Arial comparison:

```bash
python3 -m auto_manga.tools.font_preview \
  --compare-font "./fonts/MTO-Astro-City.ttf" \
  --compare-font "./fonts/anime_ace_3.ttf" \
  --compare-font "./fonts/Arial-Unicode-Regular.ttf" \
  --output ./font-comparison-mto-astro-city.png \
  --report-fonts
```

Every row renders directly with that file and never borrows missing glyphs. The columns
show mixed case, uppercase, border on/off, spacing `-2/-1/0`, and offsets `-2/0/+1/+3`.
These offsets affect only the development preview: the production `manga2eng` renderer
performs its own automatic fitting. The broader `--contact-sheet` mode remains available
to compare every font in `fonts/`.

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
