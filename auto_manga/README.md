# Automatic manga pipeline

This package is a separate automation layer around `manga-image-translator`. It only
supports sources that the user is allowed to access and download. It does not implement
authentication, paywall, CAPTCHA, anti-bot, or access-control bypasses.

## Reference source format

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
`manga`, `chapter`, and `pages` fields. To add a real permitted source later, implement
`MangaSource` in `crawler/sources/` and register it in `registry.py`.

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

The default source is `ExampleSource`; use `--source example` to select it explicitly.
Without `--chapters` or `--latest`, `manga` processes every chapter in the manifest.
Relative storage paths in YAML are resolved from the current working directory.

The wrapper invokes the existing CLI with an argument list (never a shell command):

```text
python -m manga_translator local -i RAW_FOLDER -o OUTPUT_FOLDER --config-file TEMP.json
```

Valid existing input and output images are reused. `resume` resets interrupted
`downloading` work to `pending`, resets interrupted `translating` work to `downloaded`,
and retries `pending`, `downloaded`, and `failed` chapters.
