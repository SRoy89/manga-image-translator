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
Edit `auto_manga/config.yaml`, then run:

```bash
python main.py manga https://public.example/manifest.json --chapters 1-5
python main.py manga https://public.example/manifest.json --latest
python main.py chapter 'https://public.example/manifest.json#chapter=chapter-1'
python main.py resume
```

Source selection is automatic. Use `--source mangadex` or `--source example` to force an
adapter. Without `--chapters` or `--latest`, `manga` processes every discovered chapter.
Relative storage paths in YAML are resolved from the current working directory.

The wrapper invokes the existing CLI with an argument list (never a shell command):

```text
python -m manga_translator local -i RAW_FOLDER -o OUTPUT_FOLDER --config-file TEMP.json
```

Valid existing input and output images are reused. `resume` resets interrupted
`downloading` work to `pending`, resets interrupted `translating` work to `downloaded`,
and retries `pending`, `downloaded`, and `failed` chapters. It also revalidates chapters
marked `translated` and repairs their state if the raw or translated files are incomplete.
