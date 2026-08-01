from __future__ import annotations

import hashlib
import logging
import re
import shutil
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlparse

import requests
from PIL import Image, UnidentifiedImageError

from auto_manga.config import DownloadConfig

from .models import Page

LOGGER = logging.getLogger(__name__)
IMAGE_NAME = re.compile(r"^(\d{3,})\.jpg$")
USER_AGENT = (
    "auto-manga/0.1 "
    "(+https://github.com/zyddnys/manga-image-translator)"
)


class DownloadError(RuntimeError):
    """Raised when a chapter image cannot be downloaded safely."""


def is_valid_image(path: Path) -> bool:
    try:
        if not path.is_file() or path.stat().st_size == 0:
            return False
        with Image.open(path) as image:
            image.verify()
    except (OSError, UnidentifiedImageError):
        return False
    return True


def chapter_images(folder: Path) -> list[Path]:
    try:
        if not folder.is_dir():
            return []
        images = [path for path in folder.iterdir() if IMAGE_NAME.match(path.name)]
    except OSError:
        return []
    return sorted(images, key=lambda path: int(path.stem))


def validate_chapter_images(folder: Path, expected_count: int | None = None) -> bool:
    images = chapter_images(folder)
    if not images:
        return False
    if expected_count is not None and len(images) != expected_count:
        return False
    expected_names = [f"{index:03d}.jpg" for index in range(1, len(images) + 1)]
    if [path.name for path in images] != expected_names:
        return False
    return all(is_valid_image(path) for path in images)


class _RateLimiter:
    def __init__(self, delay: float) -> None:
        self.delay = delay
        self._lock = threading.Lock()
        self._next_request = 0.0

    def wait(self) -> None:
        if self.delay <= 0:
            return
        with self._lock:
            now = time.monotonic()
            wait_for = max(0.0, self._next_request - now)
            if wait_for:
                time.sleep(wait_for)
            self._next_request = time.monotonic() + self.delay


class ChapterDownloader:
    def __init__(
        self,
        config: DownloadConfig,
        session: requests.Session | None = None,
    ) -> None:
        self.config = config
        self.session = session or requests.Session()
        if session is None:
            self.session.headers["User-Agent"] = USER_AGENT
        self._rate_limiter = _RateLimiter(config.delay)

    @staticmethod
    def _ordered_pages(pages: list[Page]) -> list[Page]:
        if not pages:
            raise DownloadError("A chapter must contain at least one page")
        ordered = sorted(pages, key=lambda page: page.index)
        indexes = [page.index for page in ordered]
        if any(index < 0 for index in indexes) or len(indexes) != len(set(indexes)):
            raise DownloadError("Page indexes must be unique non-negative integers")
        for page in ordered:
            if urlparse(page.image_url).scheme.lower() not in {"http", "https"}:
                raise DownloadError("Image URLs must use HTTP(S)")
        return ordered

    def download_chapter(self, pages: list[Page], destination: Path) -> list[Path]:
        ordered = self._ordered_pages(pages)
        destination.mkdir(parents=True, exist_ok=True)
        targets = [destination / f"{position:03d}.jpg" for position in range(1, len(ordered) + 1)]
        expected_targets = set(targets)
        for existing in chapter_images(destination):
            if existing not in expected_targets:
                LOGGER.warning("Removing stale chapter image %s", existing.name)
                existing.unlink()

        first_by_url: dict[str, int] = {}
        duplicate_positions: dict[int, list[int]] = defaultdict(list)
        unique_positions: list[int] = []
        for position, page in enumerate(ordered):
            first = first_by_url.setdefault(page.image_url, position)
            if first == position:
                unique_positions.append(position)
            else:
                duplicate_positions[first].append(position)

        failures: list[str] = []
        with ThreadPoolExecutor(max_workers=self.config.concurrency) as executor:
            future_positions = {
                executor.submit(self._download_page, ordered[position], targets[position]): position
                for position in unique_positions
            }
            for future in as_completed(future_positions):
                position = future_positions[future]
                try:
                    future.result()
                except DownloadError as exc:
                    failures.append(f"image {position + 1}: {exc}")

        if failures:
            raise DownloadError("; ".join(failures))

        for original, duplicates in duplicate_positions.items():
            for duplicate in duplicates:
                if (
                    not is_valid_image(targets[duplicate])
                    or targets[duplicate].read_bytes() != targets[original].read_bytes()
                ):
                    shutil.copyfile(targets[original], targets[duplicate])

        if not validate_chapter_images(destination, len(ordered)):
            raise DownloadError("Downloaded chapter failed image sequence validation")
        self._log_content_duplicates(targets)
        return targets

    def _download_page(self, page: Page, destination: Path) -> None:
        if is_valid_image(destination):
            LOGGER.debug("Skipping existing valid image %s", destination.name)
            return

        last_error: Exception | None = None
        for attempt in range(self.config.retries + 1):
            try:
                self._rate_limiter.wait()
                self._download_once(page.image_url, destination)
                return
            except (requests.Timeout, requests.ConnectionError) as exc:
                last_error = exc
            except requests.HTTPError as exc:
                last_error = exc
                status = exc.response.status_code if exc.response is not None else None
                if status not in {408, 429} and (status is None or status < 500):
                    break
            except requests.RequestException as exc:
                last_error = exc
                break
            except UnidentifiedImageError as exc:
                last_error = exc
            except OSError as exc:
                last_error = exc
                break

            if attempt < self.config.retries:
                LOGGER.warning(
                    "Temporary download error for image %s; retry %s/%s",
                    page.index,
                    attempt + 1,
                    self.config.retries,
                )

        detail = type(last_error).__name__ if last_error else "unknown error"
        raise DownloadError(f"download failed after retries ({detail})") from last_error

    def _download_once(self, url: str, destination: Path) -> None:
        temporary = destination.with_suffix(destination.suffix + ".part")
        try:
            with self.session.get(url, timeout=self.config.timeout, stream=True) as response:
                response.raise_for_status()
                with temporary.open("wb") as output:
                    for chunk in response.iter_content(chunk_size=64 * 1024):
                        if chunk:
                            output.write(chunk)
            if not is_valid_image(temporary):
                raise UnidentifiedImageError("Response is not a valid non-empty image")
            temporary.replace(destination)
        finally:
            if temporary.exists():
                temporary.unlink()

    @staticmethod
    def _log_content_duplicates(paths: list[Path]) -> None:
        hashes: dict[str, str] = {}
        for path in paths:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest in hashes:
                LOGGER.warning("Duplicate image content: %s and %s", hashes[digest], path.name)
            else:
                hashes[digest] = path.name
