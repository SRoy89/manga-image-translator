from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


STYLE_GUIDE_MAX_BYTES = 64 * 1024
_TOP_LEVEL_KEYS = {
    "default",
    "characters",
    "relationships",
    "terminology",
    "line_guidance",
}


class DialogueStyleGuideError(ValueError):
    """Raised when a manual manga dialogue style guide is invalid."""


@dataclass
class PageTranslationContext:
    """One page of aligned source/translation history in manga reading order."""

    source_lines: list[str]
    translated_lines: list[str]
    page_name: str | None = None
    source_image_sha256: str | None = None

    def __post_init__(self) -> None:
        if len(self.source_lines) != len(self.translated_lines):
            raise ValueError("Page source and translation line counts must match")

    def is_empty(self) -> bool:
        return not any(
            source.strip() or translation.strip()
            for source, translation in zip(self.source_lines, self.translated_lines)
        )


def _require_string(value: Any, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DialogueStyleGuideError(f"{location} must be a non-empty string")
    return value


def _reject_keys(value: dict[str, Any], allowed: set[str], location: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise DialogueStyleGuideError(
            f"{location} contains unsupported key '{unknown[0]}'"
        )


def validate_style_guide(document: Any) -> dict[str, Any]:
    if not isinstance(document, dict):
        raise DialogueStyleGuideError("Style guide root must be a mapping")
    _reject_keys(document, _TOP_LEVEL_KEYS, "Style guide")

    default = document.get("default", {})
    if not isinstance(default, dict):
        raise DialogueStyleGuideError("default must be a mapping")
    _reject_keys(
        default,
        {
            "peer_pair",
            "unknown_strategy",
            "narration_first_person",
            "inner_monologue_first_person",
            "narration_guidance",
        },
        "default",
    )
    peer_pair = default.get("peer_pair", {})
    if not isinstance(peer_pair, dict):
        raise DialogueStyleGuideError("default.peer_pair must be a mapping")
    _reject_keys(peer_pair, {"first_person", "second_person"}, "default.peer_pair")
    for key, value in peer_pair.items():
        _require_string(value, f"default.peer_pair.{key}")
    for key in (
        "unknown_strategy",
        "narration_first_person",
        "inner_monologue_first_person",
        "narration_guidance",
    ):
        if key in default:
            _require_string(default[key], f"default.{key}")

    characters = document.get("characters", [])
    if not isinstance(characters, list):
        raise DialogueStyleGuideError("characters must be a list")
    seen_names: set[str] = set()
    for index, character in enumerate(characters):
        location = f"characters[{index}]"
        if not isinstance(character, dict):
            raise DialogueStyleGuideError(f"{location} must be a mapping")
        _reject_keys(
            character,
            {"name", "aliases", "voice", "self_pronoun", "third_person"},
            location,
        )
        name = _require_string(character.get("name"), f"{location}.name")
        if name in seen_names:
            raise DialogueStyleGuideError(f"Duplicate character name '{name}'")
        seen_names.add(name)
        aliases = character.get("aliases", [])
        if not isinstance(aliases, list) or any(
            not isinstance(alias, str) or not alias.strip() for alias in aliases
        ):
            raise DialogueStyleGuideError(f"{location}.aliases must be a list of strings")
        for key in ("voice", "self_pronoun", "third_person"):
            if key in character:
                _require_string(character[key], f"{location}.{key}")

    relationships = document.get("relationships", [])
    if not isinstance(relationships, list):
        raise DialogueStyleGuideError("relationships must be a list")
    seen_relationships: set[tuple[str, str]] = set()
    for index, relationship in enumerate(relationships):
        location = f"relationships[{index}]"
        if not isinstance(relationship, dict):
            raise DialogueStyleGuideError(f"{location} must be a mapping")
        required = {"speaker", "listener", "self", "address"}
        _reject_keys(relationship, required, location)
        missing = sorted(required - set(relationship))
        if missing:
            raise DialogueStyleGuideError(f"{location} is missing '{missing[0]}'")
        values = {
            key: _require_string(relationship[key], f"{location}.{key}")
            for key in required
        }
        direction = (values["speaker"], values["listener"])
        if direction in seen_relationships:
            raise DialogueStyleGuideError(
                f"Duplicate relationship direction '{direction[0]}' -> '{direction[1]}'"
            )
        seen_relationships.add(direction)

    terminology = document.get("terminology", {})
    if not isinstance(terminology, dict) or any(
        not isinstance(source, str)
        or not source.strip()
        or not isinstance(translation, str)
        or not translation.strip()
        for source, translation in terminology.items()
    ):
        raise DialogueStyleGuideError("terminology must map non-empty strings to strings")

    line_guidance = document.get("line_guidance", {})
    if not isinstance(line_guidance, dict) or any(
        not isinstance(source, str)
        or not source.strip()
        or not isinstance(guidance, str)
        or not guidance.strip()
        for source, guidance in line_guidance.items()
    ):
        raise DialogueStyleGuideError(
            "line_guidance must map exact non-empty source lines to instructions"
        )
    return document


def load_style_guide(path: str | Path) -> str:
    style_path = Path(path)
    try:
        raw = style_path.read_bytes()
    except OSError as exc:
        raise DialogueStyleGuideError(f"Cannot read style guide {style_path}: {exc}") from exc
    if len(raw) > STYLE_GUIDE_MAX_BYTES:
        raise DialogueStyleGuideError(
            f"Style guide exceeds {STYLE_GUIDE_MAX_BYTES} bytes: {style_path}"
        )
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise DialogueStyleGuideError(f"Style guide must be UTF-8: {style_path}") from exc
    try:
        document = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise DialogueStyleGuideError(f"Invalid YAML in style guide {style_path}: {exc}") from exc
    validated = validate_style_guide(document)
    return yaml.safe_dump(
        validated,
        allow_unicode=True,
        sort_keys=False,
        default_flow_style=False,
    ).strip()
