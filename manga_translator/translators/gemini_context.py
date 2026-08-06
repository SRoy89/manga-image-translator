from __future__ import annotations

import asyncio
import hashlib
import io
import json
import logging
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from PIL import Image

try:
    from google import genai
    from google.genai import types
except ImportError:  # pragma: no cover - dependency errors are handled at call time.
    genai = None
    types = None

from ..config import PronounContextConfig, TranslatorConfig
from ..utils import BASE_PATH, Context
from .common import CommonTranslator
from .deepseek import DeepseekTranslator
from .keys import GEMINI_API_KEY, GEMINI_MODEL, GEMINI_VISION_MODEL


LOGGER = logging.getLogger("manga_translator")
PROMPT_VERSION = "deepseek-gemini-pronoun-context-v1"

AMBIGUITY_TYPES = frozenset(
    {
        "none",
        "speaker",
        "addressee",
        "relationship",
        "age_or_rank",
        "gender",
        "proper_name",
        "kinship",
        "missing_subject",
        "conflicting_context",
    }
)

_UNKNOWN_VALUES = {"", "unknown", "null", "none", "unclear", "không rõ"}


class GeminiContextError(RuntimeError):
    """A recoverable visual-context failure."""


@dataclass(frozen=True)
class RegionContext:
    id: int
    bbox: tuple[int, int, int, int]
    source_text: str
    draft_translation: str

    def to_prompt_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "bbox": list(self.bbox),
            "source_text": self.source_text,
            "draft_translation": self.draft_translation,
        }


@dataclass(frozen=True)
class UncertaintyItem:
    id: int
    translation: str
    confidence: float
    needs_vision: bool
    ambiguity_type: str
    possible_forms: tuple[str, ...]
    reason: str

    def to_prompt_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["possible_forms"] = list(self.possible_forms)
        return value


@dataclass(frozen=True)
class ResolvedContextItem:
    id: int
    speaker: str
    addressee: str
    speaker_visible_name: str | None
    addressee_visible_name: str | None
    relationship: str
    recommended_self_reference: str
    recommended_address: str
    prefer_proper_name: bool
    confidence: float
    visual_evidence: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["visual_evidence"] = list(self.visual_evidence)
        return value

    def is_resolved(self) -> bool:
        values = (
            self.speaker,
            self.addressee,
            self.speaker_visible_name or "",
            self.addressee_visible_name or "",
            self.relationship,
            self.recommended_self_reference,
            self.recommended_address,
        )
        return any(value.strip().casefold() not in _UNKNOWN_VALUES for value in values)


@dataclass
class RelationshipRecord:
    relationship: str
    self_reference: str
    address: str
    confidence: float
    last_seen_page: int
    conflict: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RelationshipMemory:
    """Chapter-scoped directional character relationship memory."""

    def __init__(self, records: dict[str, RelationshipRecord] | None = None) -> None:
        self._records = records or {}

    @staticmethod
    def _known(value: str | None) -> bool:
        return bool(value and value.strip().casefold() not in _UNKNOWN_VALUES)

    @classmethod
    def _identity(cls, internal_id: str, visible_name: str | None) -> str | None:
        if cls._known(internal_id):
            return internal_id.strip()
        if cls._known(visible_name):
            return f"name:{visible_name.strip()}"
        return None

    def update(self, items: Iterable[ResolvedContextItem], page_number: int) -> None:
        for item in items:
            speaker = self._identity(item.speaker, item.speaker_visible_name)
            addressee = self._identity(item.addressee, item.addressee_visible_name)
            if not speaker or not addressee:
                continue
            key = f"{speaker}->{addressee}"
            incoming = RelationshipRecord(
                relationship=item.relationship,
                self_reference=item.recommended_self_reference,
                address=item.recommended_address,
                confidence=item.confidence,
                last_seen_page=page_number,
            )
            current = self._records.get(key)
            if current is None:
                self._records[key] = incoming
                continue

            current_values = (
                current.relationship,
                current.self_reference,
                current.address,
            )
            incoming_values = (
                incoming.relationship,
                incoming.self_reference,
                incoming.address,
            )
            if current_values == incoming_values:
                current.confidence = max(current.confidence, incoming.confidence)
                current.last_seen_page = max(current.last_seen_page, page_number)
                continue

            # A conflict is always surfaced to the next Gemini request. The more
            # confident record remains authoritative; a weaker observation never
            # silently overwrites it.
            if incoming.confidence > current.confidence:
                incoming.conflict = True
                self._records[key] = incoming
            else:
                current.conflict = True
                current.last_seen_page = max(current.last_seen_page, page_number)

    def to_dict(self) -> dict[str, dict[str, Any]]:
        return {
            key: record.to_dict()
            for key, record in sorted(self._records.items())
        }

    @classmethod
    def from_dict(cls, value: Any) -> "RelationshipMemory":
        if value is None:
            return cls()
        if not isinstance(value, dict):
            raise ValueError("relationships must be an object")
        records: dict[str, RelationshipRecord] = {}
        allowed = {
            "relationship",
            "self_reference",
            "address",
            "confidence",
            "last_seen_page",
            "conflict",
        }
        for key, raw in value.items():
            if not isinstance(key, str) or "->" not in key or not isinstance(raw, dict):
                raise ValueError("invalid relationship memory entry")
            if set(raw) - allowed:
                raise ValueError(f"unsupported relationship fields for {key}")
            relationship = raw.get("relationship")
            self_reference = raw.get("self_reference")
            address = raw.get("address")
            confidence = raw.get("confidence")
            last_seen_page = raw.get("last_seen_page")
            conflict = raw.get("conflict", False)
            if (
                not all(
                    isinstance(item, str)
                    for item in (relationship, self_reference, address)
                )
                or isinstance(confidence, bool)
                or not isinstance(confidence, (int, float))
                or not 0 <= float(confidence) <= 1
                or isinstance(last_seen_page, bool)
                or not isinstance(last_seen_page, int)
                or last_seen_page < 0
                or not isinstance(conflict, bool)
            ):
                raise ValueError(f"invalid relationship values for {key}")
            records[key] = RelationshipRecord(
                relationship=relationship,
                self_reference=self_reference,
                address=address,
                confidence=float(confidence),
                last_seen_page=last_seen_page,
                conflict=conflict,
            )
        return cls(records)


class GeminiContextResolver:
    """Resolve only visual speaker/addressee context for one complete page."""

    _SYSTEM_PROMPT = """You resolve visual dialogue context for Vietnamese manga translation.

The current manga page is attached as an actual image. OCR region IDs and bounding boxes use that image's pixel coordinates. Previous-page images, when attached, are references only.

For each requested ambiguous region:
- infer only speaker, addressee, relative age/rank, gender when visually supported, relationship, kinship, and whether a confirmed proper name is the natural form of address;
- use only evidence visible in the images and the supplied chapter relationship memory;
- never invent a name, age, gender, identity, kinship, or relationship;
- use the literal string "unknown" whenever evidence is insufficient;
- do not translate, rewrite, summarize, or stylistically edit dialogue;
- do not change dialogue meaning;
- return one result for every requested ID, in the same order.

Character identity must be a confirmed visible name or a stable chapter-internal ID such as main_character_a. Never derive character identity from an image filename. If a memory entry has conflict=true, inspect it again instead of silently choosing a new relationship.

Return only a JSON object matching the requested schema, without markdown or commentary."""

    _RESPONSE_SCHEMA: dict[str, Any] = {
        "type": "object",
        "properties": {
            "resolved_items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "integer"},
                        "speaker": {"type": "string"},
                        "addressee": {"type": "string"},
                        "speaker_visible_name": {
                            "type": "string",
                            "nullable": True,
                        },
                        "addressee_visible_name": {
                            "type": "string",
                            "nullable": True,
                        },
                        "relationship": {"type": "string"},
                        "recommended_self_reference": {"type": "string"},
                        "recommended_address": {"type": "string"},
                        "prefer_proper_name": {"type": "boolean"},
                        "confidence": {"type": "number"},
                        "visual_evidence": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                    "required": [
                        "id",
                        "speaker",
                        "addressee",
                        "speaker_visible_name",
                        "addressee_visible_name",
                        "relationship",
                        "recommended_self_reference",
                        "recommended_address",
                        "prefer_proper_name",
                        "confidence",
                        "visual_evidence",
                    ],
                },
            }
        },
        "required": ["resolved_items"],
    }

    def __init__(
        self,
        config: PronounContextConfig,
        *,
        client: Any | None = None,
        cache_dir: Path | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.config = config
        self.model = (
            config.model
            or os.getenv("GEMINI_VISION_MODEL")
            or os.getenv("GEMINI_MODEL")
            or GEMINI_VISION_MODEL
            or GEMINI_MODEL
        )
        self._client = client
        self.cache_dir = cache_dir or Path(BASE_PATH) / "result" / "gemini_context_cache"
        self.logger = logger or LOGGER

    @staticmethod
    def _image_bytes(image: Any) -> tuple[bytes, str]:
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        if not isinstance(image, Image.Image):
            raise GeminiContextError("Gemini context requires an in-memory page image")
        output = io.BytesIO()
        image.convert("RGB").save(output, format="PNG")
        return output.getvalue(), "image/png"

    @staticmethod
    def _strip_json_fence(value: str) -> str:
        value = value.strip()
        if not value.startswith("```"):
            return value
        lines = value.splitlines()
        if len(lines) < 3 or lines[-1].strip() != "```":
            return value
        body = "\n".join(lines[1:-1]).strip()
        if body.startswith("json"):
            body = body[4:].lstrip()
        return body

    @classmethod
    def parse_response(
        cls, response_text: str, expected_ids: Sequence[int]
    ) -> list[ResolvedContextItem]:
        try:
            document = json.loads(cls._strip_json_fence(response_text))
        except (json.JSONDecodeError, TypeError) as exc:
            raise GeminiContextError("Gemini returned invalid JSON") from exc
        if not isinstance(document, dict) or set(document) != {"resolved_items"}:
            raise GeminiContextError("Gemini JSON must contain only resolved_items")
        raw_items = document["resolved_items"]
        if not isinstance(raw_items, list):
            raise GeminiContextError("Gemini resolved_items must be a list")

        required = {
            "id",
            "speaker",
            "addressee",
            "speaker_visible_name",
            "addressee_visible_name",
            "relationship",
            "recommended_self_reference",
            "recommended_address",
            "prefer_proper_name",
            "confidence",
            "visual_evidence",
        }
        parsed: list[ResolvedContextItem] = []
        for raw in raw_items:
            if not isinstance(raw, dict) or set(raw) != required:
                raise GeminiContextError("Gemini resolved item has invalid fields")
            identifier = raw["id"]
            confidence = raw["confidence"]
            visible_names = (
                raw["speaker_visible_name"],
                raw["addressee_visible_name"],
            )
            strings = (
                raw["speaker"],
                raw["addressee"],
                raw["relationship"],
                raw["recommended_self_reference"],
                raw["recommended_address"],
            )
            evidence = raw["visual_evidence"]
            if (
                isinstance(identifier, bool)
                or not isinstance(identifier, int)
                or not all(isinstance(value, str) and value.strip() for value in strings)
                or any(value is not None and not isinstance(value, str) for value in visible_names)
                or not isinstance(raw["prefer_proper_name"], bool)
                or isinstance(confidence, bool)
                or not isinstance(confidence, (int, float))
                or not 0 <= float(confidence) <= 1
                or not isinstance(evidence, list)
                or any(not isinstance(value, str) or not value.strip() for value in evidence)
            ):
                raise GeminiContextError("Gemini resolved item has invalid values")
            parsed.append(
                ResolvedContextItem(
                    id=identifier,
                    speaker=raw["speaker"].strip(),
                    addressee=raw["addressee"].strip(),
                    speaker_visible_name=(
                        raw["speaker_visible_name"].strip()
                        if isinstance(raw["speaker_visible_name"], str)
                        and raw["speaker_visible_name"].strip()
                        else None
                    ),
                    addressee_visible_name=(
                        raw["addressee_visible_name"].strip()
                        if isinstance(raw["addressee_visible_name"], str)
                        and raw["addressee_visible_name"].strip()
                        else None
                    ),
                    relationship=raw["relationship"].strip(),
                    recommended_self_reference=raw[
                        "recommended_self_reference"
                    ].strip(),
                    recommended_address=raw["recommended_address"].strip(),
                    prefer_proper_name=raw["prefer_proper_name"],
                    confidence=float(confidence),
                    visual_evidence=tuple(value.strip() for value in evidence),
                )
            )
        if [item.id for item in parsed] != list(expected_ids):
            raise GeminiContextError(
                "Gemini changed ambiguous region IDs, count, or order"
            )
        return parsed

    def _cache_key(
        self,
        image_bytes: bytes,
        regions: Sequence[RegionContext],
        ambiguous_items: Sequence[UncertaintyItem],
        memory: RelationshipMemory,
        previous_image_bytes: Sequence[bytes],
    ) -> str:
        material = {
            "image_sha256": hashlib.sha256(image_bytes).hexdigest(),
            "ocr_texts": [region.source_text for region in regions],
            "bboxes": [list(region.bbox) for region in regions],
            "model": self.model,
            "prompt_version": PROMPT_VERSION,
            "ambiguous_items": [item.to_prompt_dict() for item in ambiguous_items],
            "relationship_memory": memory.to_dict(),
            "previous_image_sha256": [
                hashlib.sha256(value).hexdigest() for value in previous_image_bytes
            ],
        }
        encoded = json.dumps(
            material, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def _read_cache(
        self, key: str, expected_ids: Sequence[int]
    ) -> list[ResolvedContextItem] | None:
        path = self._cache_path(key)
        try:
            raw = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        except OSError as exc:
            self.logger.warning("Cannot read Gemini context cache %s: %s", path, exc)
            return None
        try:
            document = json.loads(raw)
            if (
                not isinstance(document, dict)
                or document.get("prompt_version") != PROMPT_VERSION
                or document.get("model") != self.model
            ):
                return None
            return self.parse_response(
                json.dumps(
                    {"resolved_items": document.get("resolved_items")},
                    ensure_ascii=False,
                ),
                expected_ids,
            )
        except (json.JSONDecodeError, GeminiContextError, TypeError) as exc:
            self.logger.warning("Ignoring invalid Gemini context cache %s: %s", path, exc)
            return None

    def _write_cache(self, key: str, items: Sequence[ResolvedContextItem]) -> None:
        path = self._cache_path(key)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temporary_path = tempfile.mkstemp(
                prefix=f".{key}-", suffix=".tmp", dir=path.parent
            )
            try:
                with os.fdopen(descriptor, "w", encoding="utf-8") as cache_file:
                    json.dump(
                        {
                            "prompt_version": PROMPT_VERSION,
                            "model": self.model,
                            "resolved_items": [item.to_dict() for item in items],
                        },
                        cache_file,
                        ensure_ascii=False,
                        indent=2,
                    )
                    cache_file.flush()
                    os.fsync(cache_file.fileno())
                os.replace(temporary_path, path)
            except Exception:
                try:
                    os.unlink(temporary_path)
                except OSError:
                    pass
                raise
        except OSError as exc:
            self.logger.warning("Cannot write Gemini context cache %s: %s", path, exc)

    def _client_or_raise(self) -> Any:
        if self._client is not None:
            return self._client
        if genai is None:
            raise GeminiContextError("google-genai is not installed")
        api_key = os.getenv("GEMINI_API_KEY") or GEMINI_API_KEY
        if not api_key:
            raise GeminiContextError(
                "GEMINI_API_KEY is missing; using the initial DeepSeek translations"
            )
        self._client = genai.Client(api_key=api_key)
        return self._client

    def _request_prompt(
        self,
        regions: Sequence[RegionContext],
        ambiguous_items: Sequence[UncertaintyItem],
        memory: RelationshipMemory,
    ) -> str:
        payload = {
            "prompt_version": PROMPT_VERSION,
            "all_ocr_regions": [region.to_prompt_dict() for region in regions],
            "ambiguous_regions": [item.to_prompt_dict() for item in ambiguous_items],
            "chapter_relationship_memory": memory.to_dict(),
        }
        return (
            "Analyze the attached page and this JSON input:\n"
            + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        )

    async def _request(
        self,
        image_bytes: bytes,
        image_mime: str,
        previous_images: Sequence[tuple[bytes, str]],
        prompt: str,
    ) -> str:
        if types is None:
            raise GeminiContextError("google-genai is not installed")
        client = self._client_or_raise()
        parts = [
            types.Part.from_text(text="CURRENT_PAGE_IMAGE"),
            types.Part.from_bytes(data=image_bytes, mime_type=image_mime),
        ]
        for index, (previous_bytes, previous_mime) in enumerate(
            previous_images, start=1
        ):
            parts.extend(
                [
                    types.Part.from_text(text=f"PREVIOUS_PAGE_IMAGE_{index}"),
                    types.Part.from_bytes(
                        data=previous_bytes, mime_type=previous_mime
                    ),
                ]
            )
        parts.append(types.Part.from_text(text=prompt))
        request = client.aio.models.generate_content(
            model=self.model,
            contents=[types.Content(role="user", parts=parts)],
            config=types.GenerateContentConfig(
                system_instruction=self._SYSTEM_PROMPT,
                temperature=0,
                max_output_tokens=4096,
                response_mime_type="application/json",
                response_schema=self._RESPONSE_SCHEMA,
            ),
        )
        try:
            response = await asyncio.wait_for(request, timeout=self.config.timeout)
        except asyncio.TimeoutError as exc:
            raise GeminiContextError(
                f"Gemini visual context timed out after {self.config.timeout:g}s"
            ) from exc
        response_text = getattr(response, "text", None)
        if not isinstance(response_text, str) or not response_text.strip():
            raise GeminiContextError("Gemini returned an empty visual-context response")
        return response_text

    async def resolve_page(
        self,
        image: Any,
        regions: Sequence[RegionContext],
        ambiguous_items: Sequence[UncertaintyItem],
        memory: RelationshipMemory,
        previous_images: Sequence[Any] = (),
    ) -> list[ResolvedContextItem]:
        """Make at most one Gemini request for all ambiguous regions on a page."""
        if not ambiguous_items:
            return []
        expected_ids = [item.id for item in ambiguous_items]
        image_bytes, image_mime = self._image_bytes(image)
        encoded_previous = [
            self._image_bytes(value)
            for value in list(previous_images)[-self.config.previous_pages :]
        ] if self.config.previous_pages else []
        key = self._cache_key(
            image_bytes,
            regions,
            ambiguous_items,
            memory,
            [value[0] for value in encoded_previous],
        )
        if self.config.cache_enabled:
            cached = self._read_cache(key, expected_ids)
            if cached is not None:
                self.logger.info("Gemini visual-context cache hit")
                return cached

        prompt = self._request_prompt(regions, ambiguous_items, memory)
        response_text = await self._request(
            image_bytes, image_mime, encoded_previous, prompt
        )
        items = self.parse_response(response_text, expected_ids)
        if self.config.cache_enabled:
            self._write_cache(key, items)
        return items


class DeepseekGeminiContextTranslator(CommonTranslator):
    """DeepSeek translator with one conditional Gemini Vision fallback per page."""

    _LANGUAGE_CODE_MAP = DeepseekTranslator._LANGUAGE_CODE_MAP

    _CLASSIFIER_SYSTEM = """You are a strict uncertainty classifier for Vietnamese manga dialogue translations. DeepSeek has already produced a draft translation. Do not translate again and do not change the draft.

For every input item, return id, the unchanged translation, confidence from 0 to 1, needs_vision, one ambiguity_type, possible_forms, and a short reason. ambiguity_type must be one of: none, speaker, addressee, relationship, age_or_rank, gender, proper_name, kinship, missing_subject, conflicting_context.

Set needs_vision=true when any of these applies: first/second-person reference without a known relationship; omitted subject or listener; several plausible Vietnamese address forms; conflict with chapter relationship memory; a proper name may be direct address or third-person reference; confidence is below the supplied threshold; or the same identity has inconsistent address forms. Do not set needs_vision merely because a line is short. Clear sound effects, signs, narration/monologue, and lines with no address-form ambiguity stay on the fast path.

Return only strict JSON: {"items":[...]}. Preserve every ID and order."""

    _REVISION_SYSTEM = """You revise only Vietnamese pronouns and directly related forms of address using supplied visual context.

Rules:
1. Preserve the draft meaning. Change only pronouns, address forms, omitted/explicit subjects, titles, kinship terms, or a directly relevant confirmed proper name.
2. Return only the requested ambiguous IDs, unchanged and in the same order. Do not rewrite unrelated wording.
3. Never contradict Gemini context. When it says unknown, prefer a confirmed proper name, a natural sentence with no pronoun, or neutral wording instead of inventing a relationship.
4. Use tớ/cậu only for a pair confirmed as same-age intimate peers. Never apply tớ/cậu to everyone speaking with a main character.
5. Use bác/chú/cô/ông/bà/anh/chị only with sufficient evidence of age, rank, gender, or relationship.
6. Keep a confirmed proper name when direct name address is more natural. Do not infer gender from a name alone.
7. Do not add a Vietnamese pronoun when the sentence is natural without one.

Return only strict JSON: {"items":[{"id":0,"translation":"..."}]} and never return an empty translation."""

    def __init__(
        self,
        *,
        deepseek: DeepseekTranslator | None = None,
        resolver: GeminiContextResolver | None = None,
    ) -> None:
        super().__init__()
        self.deepseek = deepseek or DeepseekTranslator()
        self._resolver = resolver
        self.config: TranslatorConfig | None = None

    def parse_args(self, args: TranslatorConfig) -> None:
        self.config = args
        self.deepseek.parse_args(args)

    def set_prev_context(self, context: str | None) -> None:
        self.deepseek.set_prev_context(context)

    def set_dialogue_style_guide(self, style_guide: str | None) -> None:
        self.deepseek.set_dialogue_style_guide(style_guide)

    def supports_languages(
        self, from_lang: str, to_lang: str, fatal: bool = False
    ) -> bool:
        return self.deepseek.supports_languages(from_lang, to_lang, fatal)

    async def unload(self, device: str | None = None) -> None:
        await self.deepseek.unload(device)

    async def _translate(
        self, from_lang: str, to_lang: str, queries: list[str]
    ) -> list[str]:
        # Direct generic-dispatch use has no image Context, so it deliberately
        # degrades to the existing DeepSeek path.
        return await self.deepseek._translate(from_lang, to_lang, queries)

    @staticmethod
    def _safe_json(value: str) -> Any:
        value = GeminiContextResolver._strip_json_fence(value)
        return json.loads(value)

    @classmethod
    def _parse_uncertainty(
        cls,
        response: str,
        drafts: Sequence[str],
        threshold: float,
    ) -> list[UncertaintyItem]:
        try:
            document = cls._safe_json(response)
        except (json.JSONDecodeError, TypeError) as exc:
            raise ValueError("DeepSeek uncertainty classifier returned invalid JSON") from exc
        if not isinstance(document, dict) or set(document) != {"items"}:
            raise ValueError("DeepSeek uncertainty JSON must contain only items")
        raw_items = document["items"]
        if not isinstance(raw_items, list) or len(raw_items) != len(drafts):
            raise ValueError("DeepSeek uncertainty classifier changed item count")
        required = {
            "id",
            "translation",
            "confidence",
            "needs_vision",
            "ambiguity_type",
            "possible_forms",
            "reason",
        }
        items: list[UncertaintyItem] = []
        for expected_id, (raw, draft) in enumerate(zip(raw_items, drafts)):
            if not isinstance(raw, dict) or set(raw) != required:
                raise ValueError("DeepSeek uncertainty item has invalid fields")
            identifier = raw["id"]
            confidence = raw["confidence"]
            ambiguity_type = raw["ambiguity_type"]
            possible_forms = raw["possible_forms"]
            if (
                isinstance(identifier, bool)
                or identifier != expected_id
                or not isinstance(raw["translation"], str)
                or raw["translation"] != draft
                or isinstance(confidence, bool)
                or not isinstance(confidence, (int, float))
                or not 0 <= float(confidence) <= 1
                or not isinstance(raw["needs_vision"], bool)
                or ambiguity_type not in AMBIGUITY_TYPES
                or not isinstance(possible_forms, list)
                or any(not isinstance(value, str) for value in possible_forms)
                or not isinstance(raw["reason"], str)
            ):
                raise ValueError("DeepSeek uncertainty item has invalid values")
            confidence_value = float(confidence)
            needs_vision = raw["needs_vision"] or confidence_value < threshold
            if needs_vision and ambiguity_type == "none":
                ambiguity_type = "conflicting_context"
            items.append(
                UncertaintyItem(
                    id=identifier,
                    translation=draft,
                    confidence=confidence_value,
                    needs_vision=needs_vision,
                    ambiguity_type=ambiguity_type,
                    possible_forms=tuple(value.strip() for value in possible_forms),
                    reason=raw["reason"].strip(),
                )
            )
        return items

    async def _classify_uncertainty(
        self,
        regions: Sequence[RegionContext],
        previous_context: str,
        memory: RelationshipMemory,
        config: PronounContextConfig,
    ) -> list[UncertaintyItem]:
        payload = {
            "confidence_threshold": config.confidence_threshold,
            "items": [region.to_prompt_dict() for region in regions],
            "previous_bilingual_context": previous_context,
            "chapter_relationship_memory": memory.to_dict(),
        }
        response = await self.deepseek._request_chat(
            [
                {"role": "system", "content": self._CLASSIFIER_SYSTEM},
                {
                    "role": "user",
                    "content": json.dumps(payload, ensure_ascii=False),
                },
            ],
            max_tokens=min(4000, max(1200, len(regions) * 260)),
            temperature=0,
        )
        return self._parse_uncertainty(
            response,
            [region.draft_translation for region in regions],
            config.confidence_threshold,
        )

    @classmethod
    def _parse_revisions(
        cls, response: str, expected_ids: Sequence[int]
    ) -> dict[int, str]:
        try:
            document = cls._safe_json(response)
        except (json.JSONDecodeError, TypeError) as exc:
            raise ValueError("DeepSeek revision returned invalid JSON") from exc
        if not isinstance(document, dict) or set(document) != {"items"}:
            raise ValueError("DeepSeek revision JSON must contain only items")
        raw_items = document["items"]
        if not isinstance(raw_items, list):
            raise ValueError("DeepSeek revision items must be a list")
        revisions: dict[int, str] = {}
        for raw in raw_items:
            if not isinstance(raw, dict) or set(raw) != {"id", "translation"}:
                raise ValueError("DeepSeek revision item has invalid fields")
            identifier = raw["id"]
            translation = raw["translation"]
            if (
                isinstance(identifier, bool)
                or not isinstance(identifier, int)
                or not isinstance(translation, str)
                or not translation.strip()
                or identifier in revisions
            ):
                raise ValueError("DeepSeek revision item has invalid values")
            revisions[identifier] = translation.strip()
        if list(revisions) != list(expected_ids):
            raise ValueError("DeepSeek revision changed region IDs, count, or order")
        return revisions

    async def _revise_with_context(
        self,
        regions: Sequence[RegionContext],
        ambiguous_items: Sequence[UncertaintyItem],
        resolved_items: Sequence[ResolvedContextItem],
        memory: RelationshipMemory,
        config: PronounContextConfig,
    ) -> dict[int, str]:
        by_id = {region.id: region for region in regions}
        payload = {
            "use_proper_names_when_natural": config.use_proper_names_when_natural,
            "neutral_on_unresolved": config.neutral_on_unresolved,
            "chapter_relationship_memory": memory.to_dict(),
            "items": [
                {
                    "id": assessment.id,
                    "source_text": by_id[assessment.id].source_text,
                    "draft_translation": assessment.translation,
                    "uncertainty": assessment.to_prompt_dict(),
                    "gemini_context": resolved.to_dict(),
                }
                for assessment, resolved in zip(ambiguous_items, resolved_items)
            ],
        }
        response = await self.deepseek._request_chat(
            [
                {"role": "system", "content": self._REVISION_SYSTEM},
                {
                    "role": "user",
                    "content": json.dumps(payload, ensure_ascii=False),
                },
            ],
            max_tokens=min(4000, max(1000, len(ambiguous_items) * 300)),
            temperature=0,
        )
        return self._parse_revisions(
            response, [item.id for item in ambiguous_items]
        )

    @staticmethod
    def _regions(ctx: Context, queries: Sequence[str], drafts: Sequence[str]) -> list[RegionContext]:
        text_regions = list(getattr(ctx, "text_regions", None) or [])
        if len(text_regions) != len(queries):
            raise ValueError(
                "Hybrid translator requires one OCR region for every source line"
            )
        result: list[RegionContext] = []
        for identifier, (region, query, draft) in enumerate(
            zip(text_regions, queries, drafts)
        ):
            setattr(region, "translation_context_id", identifier)
            raw_bbox = getattr(region, "xyxy", (0, 0, 0, 0))
            bbox_values = tuple(int(value) for value in raw_bbox)
            if len(bbox_values) != 4:
                raise ValueError(f"OCR region {identifier} has an invalid bounding box")
            raw_source = getattr(region, "text_raw", None)
            source = raw_source if isinstance(raw_source, str) and raw_source.strip() else query
            result.append(
                RegionContext(
                    id=identifier,
                    bbox=bbox_values,
                    source_text=source,
                    draft_translation=draft,
                )
            )
        return result

    async def translate_page(
        self,
        from_lang: str,
        to_lang: str,
        queries: list[str],
        ctx: Context,
        *,
        previous_context: str,
        style_guide: str,
        memory: RelationshipMemory,
        page_number: int,
        page_label: str,
        previous_images: Sequence[Any] = (),
    ) -> list[str]:
        if self.config is None:
            raise RuntimeError("Hybrid translator configuration was not parsed")
        self.deepseek.set_prev_context(previous_context)
        self.deepseek.set_dialogue_style_guide(style_guide)
        drafts = await self.deepseek.translate(from_lang, to_lang, queries, False)
        if len(drafts) != len(queries):
            raise RuntimeError("DeepSeek changed the number of OCR regions")
        safe_drafts = [
            draft if isinstance(draft, str) and draft.strip() else source
            for source, draft in zip(queries, drafts)
        ]
        if safe_drafts != drafts:
            self.logger.warning(
                "Page %s: DeepSeek returned empty text; preserving the source region",
                page_label,
            )
        self.logger.info(
            "Page %s: %s regions translated by DeepSeek",
            page_label,
            len(safe_drafts),
        )

        context_config = self.config.pronoun_context
        if not context_config.enabled or context_config.max_fallback_rounds == 0:
            return safe_drafts
        if to_lang != "VIN":
            self.logger.warning(
                "Page %s: pronoun context is Vietnamese-specific; using initial DeepSeek translations for %s",
                page_label,
                to_lang,
            )
            return safe_drafts

        regions = self._regions(ctx, queries, safe_drafts)
        try:
            assessments = await self._classify_uncertainty(
                regions, previous_context, memory, context_config
            )
        except Exception as exc:
            self.logger.warning(
                "Page %s: DeepSeek pronoun uncertainty detection failed (%s); using initial translations",
                page_label,
                exc,
            )
            return safe_drafts
        ambiguous = [item for item in assessments if item.needs_vision]
        self.logger.info(
            "Page %s: %s regions require visual context",
            page_label,
            len(ambiguous),
        )
        if not ambiguous:
            return safe_drafts

        if getattr(ctx, "_gemini_context_attempted", False):
            self.logger.warning(
                "Page %s: Gemini fallback already attempted once; using the current DeepSeek translations",
                page_label,
            )
            return safe_drafts
        ctx._gemini_context_attempted = True

        resolver = self._resolver or GeminiContextResolver(
            context_config, logger=self.logger
        )
        try:
            # This is the only Gemini invocation in the page orchestrator. There
            # is intentionally no retry/fallback loop around it.
            page_image = getattr(ctx, "input", None)
            if page_image is None:
                page_image = getattr(ctx, "img_rgb", None)
            resolved = await resolver.resolve_page(
                page_image,
                regions,
                ambiguous,
                memory,
                previous_images,
            )
        except Exception as exc:
            self.logger.warning(
                "Page %s: Gemini visual context failed (%s); using initial DeepSeek translations",
                page_label,
                exc,
            )
            return safe_drafts
        resolved_count = sum(item.is_resolved() for item in resolved)
        self.logger.info(
            "Page %s: Gemini context resolved %s/%s regions",
            page_label,
            resolved_count,
            len(ambiguous),
        )
        memory.update(resolved, page_number)

        try:
            revisions = await self._revise_with_context(
                regions,
                ambiguous,
                resolved,
                memory,
                context_config,
            )
        except Exception as exc:
            self.logger.warning(
                "Page %s: DeepSeek context revision failed (%s); using initial translations",
                page_label,
                exc,
            )
            return safe_drafts

        final = list(safe_drafts)
        for identifier, translation in revisions.items():
            final[identifier] = translation
        self.logger.info(
            "Page %s: %s regions revised by DeepSeek",
            page_label,
            len(revisions),
        )
        return final
