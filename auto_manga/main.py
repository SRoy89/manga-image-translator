from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

if __package__ in {None, ""}:
    print(
        "Run Auto Manga from the repository root: python3 main.py <command> ...",
        file=sys.stderr,
    )
    raise SystemExit(2)

from auto_manga.config import ConfigError, load_config
from auto_manga.crawler.base import SourceError
from auto_manga.pipeline.orchestrator import PipelineOrchestrator, PipelineSummary
from auto_manga.storage.database import Database

DEFAULT_CONFIG = Path(__file__).with_name("config.yaml")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download permitted manga sources and translate chapter folders"
    )
    parser.add_argument("--verbose", action="store_true", help="Show debug logging")
    subparsers = parser.add_subparsers(dest="command", required=True)

    manga = subparsers.add_parser("manga", help="Process chapters from a supported manga source")
    manga.add_argument("url")
    selection = manga.add_mutually_exclusive_group()
    selection.add_argument("--chapters", help="Chapter number or range, for example 1-30")
    selection.add_argument("--latest", action="store_true", help="Process the latest chapter")
    _add_common_options(manga)

    chapter = subparsers.add_parser("chapter", help="Process one chapter from a supported source")
    chapter.add_argument("url")
    _add_common_options(chapter)

    resume = subparsers.add_parser("resume", help="Continue pending, interrupted, or failed work")
    _add_common_options(resume, include_source=False)
    return parser


def _add_common_options(parser: argparse.ArgumentParser, include_source: bool = True) -> None:
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"YAML config path (default: {DEFAULT_CONFIG})",
    )
    if include_source:
        parser.add_argument("--source", default=None, help="Force a registered source adapter")


def _log_summary(summary: PipelineSummary) -> None:
    logging.info(
        "Summary: selected=%s translated=%s skipped=%s failed=%s",
        summary.selected,
        summary.translated,
        summary.skipped,
        summary.failed,
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="[%(levelname)s] %(message)s",
    )

    database: Database | None = None
    try:
        config = load_config(args.config)
        database = Database(config.database.path)
        orchestrator = PipelineOrchestrator(config, database)
        if args.command == "manga":
            summary = orchestrator.run_manga(
                args.url,
                chapter_range=args.chapters,
                latest=args.latest,
                source_name=args.source,
            )
        elif args.command == "chapter":
            summary = orchestrator.run_chapter(args.url, source_name=args.source)
        else:
            summary = orchestrator.resume()
        _log_summary(summary)
        return 1 if summary.failed else 0
    except (ConfigError, SourceError, ValueError, NotImplementedError) as exc:
        logging.error("%s", exc)
        return 2
    finally:
        if database is not None:
            database.close()


if __name__ == "__main__":
    raise SystemExit(main())
