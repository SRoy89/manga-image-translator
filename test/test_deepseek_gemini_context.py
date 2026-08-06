from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import numpy as np
import pytest
from PIL import Image

from manga_translator.config import Config, PronounContextConfig, Translator
from manga_translator.dialogue import PageTranslationContext
from manga_translator.manga_translator import MangaTranslator
from manga_translator.translators import TRANSLATORS
from manga_translator.translators.deepseek import DeepseekTranslator
from manga_translator.translators.gemini_context import (
    DeepseekGeminiContextTranslator,
    GeminiContextError,
    GeminiContextResolver,
    RegionContext,
    RelationshipMemory,
    ResolvedContextItem,
    UncertaintyItem,
)
from manga_translator.utils import Context


class FakeDeepseek:
    def __init__(self, drafts: Sequence[str]) -> None:
        self.drafts = list(drafts)
        self.previous_context = ""
        self.style_guide = ""
        self.config = None

    def parse_args(self, config: Any) -> None:
        self.config = config

    def set_prev_context(self, value: str) -> None:
        self.previous_context = value

    def set_dialogue_style_guide(self, value: str) -> None:
        self.style_guide = value

    def supports_languages(self, *_args: Any, **_kwargs: Any) -> bool:
        return True

    async def unload(self, _device: str | None = None) -> None:
        return None

    async def translate(
        self, _from_lang: str, _to_lang: str, queries: list[str], _mtpe: bool
    ) -> list[str]:
        assert len(queries) == len(self.drafts)
        return list(self.drafts)


class FakeResolver:
    def __init__(self, items: Sequence[ResolvedContextItem] = (), error: Exception | None = None) -> None:
        self.items = list(items)
        self.error = error
        self.calls = 0
        self.ambiguous_counts: list[int] = []

    async def resolve_page(
        self,
        _image: Any,
        _regions: Sequence[RegionContext],
        ambiguous: Sequence[UncertaintyItem],
        _memory: RelationshipMemory,
        _previous_images: Sequence[Any],
    ) -> list[ResolvedContextItem]:
        self.calls += 1
        self.ambiguous_counts.append(len(ambiguous))
        if self.error:
            raise self.error
        return list(self.items)


def assessment(
    identifier: int,
    draft: str,
    *,
    needs_vision: bool = True,
    ambiguity_type: str = "relationship",
    confidence: float = 0.5,
) -> UncertaintyItem:
    return UncertaintyItem(
        id=identifier,
        translation=draft,
        confidence=confidence,
        needs_vision=needs_vision,
        ambiguity_type=ambiguity_type,
        possible_forms=("cậu", "bác", "tên riêng"),
        reason="OCR text does not establish the relationship",
    )


def resolved(
    identifier: int,
    *,
    speaker: str = "main_character_a",
    addressee: str = "main_character_b",
    relationship: str = "close_same_age_friends",
    self_reference: str = "tớ",
    address: str = "cậu",
    addressee_name: str | None = None,
    prefer_name: bool = False,
    confidence: float = 0.9,
) -> ResolvedContextItem:
    return ResolvedContextItem(
        id=identifier,
        speaker=speaker,
        addressee=addressee,
        speaker_visible_name=None,
        addressee_visible_name=addressee_name,
        relationship=relationship,
        recommended_self_reference=self_reference,
        recommended_address=address,
        prefer_proper_name=prefer_name,
        confidence=confidence,
        visual_evidence=("The balloon points from the visible speaker.",),
    )


def page_context(texts: Sequence[str]) -> Context:
    regions = []
    for index, text in enumerate(texts):
        regions.append(
            SimpleNamespace(
                text=text,
                text_raw=text,
                xyxy=np.array([index * 10, 0, index * 10 + 8, 8]),
            )
        )
    return Context(
        input=Image.new("RGB", (64, 64), "white"),
        img_rgb=np.full((64, 64, 3), 255, dtype=np.uint8),
        text_regions=regions,
    )


def hybrid(
    drafts: Sequence[str],
    resolver: FakeResolver,
    *,
    enabled: bool = True,
) -> DeepseekGeminiContextTranslator:
    translator = DeepseekGeminiContextTranslator(
        deepseek=FakeDeepseek(drafts),  # type: ignore[arg-type]
        resolver=resolver,  # type: ignore[arg-type]
    )
    translator.parse_args(
        Config(
            translator={
                "translator": "deepseek_gemini_context",
                "target_lang": "VIN",
                "pronoun_context": {"enabled": enabled},
                "enable_post_translation_check": False,
            }
        ).translator
    )
    return translator


async def run_page(
    translator: DeepseekGeminiContextTranslator,
    texts: list[str],
) -> list[str]:
    return await translator.translate_page(
        "auto",
        "VIN",
        texts,
        page_context(texts),
        previous_context="",
        style_guide="",
        memory=RelationshipMemory(),
        page_number=1,
        page_label="1",
    )


@pytest.mark.parametrize(
    ("draft", "vision", "revision"),
    [
        (
            "Tôi sẽ đi với bạn.",
            resolved(0),
            "Tớ sẽ đi với cậu.",
        ),
        (
            "Cậu đang làm gì vậy?",
            resolved(
                0,
                addressee="older_male_character",
                relationship="younger_person_to_older_acquaintance",
                self_reference="cháu",
                address="bác",
            ),
            "Bác đang làm gì vậy?",
        ),
        (
            "Cậu đợi tôi nhé, Hiroshi.",
            resolved(
                0,
                addressee="hiroshi",
                relationship="known_person",
                self_reference="unknown",
                address="Hiroshi",
                addressee_name="Hiroshi",
                prefer_name=True,
            ),
            "Hiroshi, đợi một chút nhé.",
        ),
    ],
)
def test_visual_relationship_selects_peer_older_or_proper_name(
    draft: str, vision: ResolvedContextItem, revision: str
) -> None:
    resolver = FakeResolver([vision])
    translator = hybrid([draft], resolver)

    async def classify(*_args: Any, **_kwargs: Any) -> list[UncertaintyItem]:
        return [assessment(0, draft)]

    async def revise(
        _regions: Sequence[RegionContext],
        _ambiguous: Sequence[UncertaintyItem],
        visual: Sequence[ResolvedContextItem],
        *_args: Any,
        **_kwargs: Any,
    ) -> dict[int, str]:
        assert visual == [vision]
        return {0: revision}

    translator._classify_uncertainty = classify  # type: ignore[method-assign]
    translator._revise_with_context = revise  # type: ignore[method-assign]
    result = asyncio.run(run_page(translator, ["I will go with you."]))

    assert result == [revision]
    if vision.recommended_address == "bác":
        assert "cậu" not in result[0].casefold()
    if vision.prefer_proper_name:
        assert "Hiroshi" in result[0]


def test_clear_line_does_not_call_gemini() -> None:
    resolver = FakeResolver()
    translator = hybrid(["Trời bắt đầu mưa."], resolver)

    async def classify(*_args: Any, **_kwargs: Any) -> list[UncertaintyItem]:
        return [
            assessment(
                0,
                "Trời bắt đầu mưa.",
                needs_vision=False,
                ambiguity_type="none",
                confidence=0.97,
            )
        ]

    translator._classify_uncertainty = classify  # type: ignore[method-assign]
    result = asyncio.run(run_page(translator, ["It started raining."]))

    assert result == ["Trời bắt đầu mưa."]
    assert resolver.calls == 0


def test_multiple_ambiguous_regions_use_one_gemini_page_request() -> None:
    drafts = ["Cậu đi đâu?", "Tớ đi cùng.", "Đợi tớ với."]
    visual = [resolved(index) for index in range(3)]
    resolver = FakeResolver(visual)
    translator = hybrid(drafts, resolver)

    async def classify(*_args: Any, **_kwargs: Any) -> list[UncertaintyItem]:
        return [assessment(index, draft) for index, draft in enumerate(drafts)]

    async def revise(*_args: Any, **_kwargs: Any) -> dict[int, str]:
        return {index: f"revised-{index}" for index in range(3)}

    translator._classify_uncertainty = classify  # type: ignore[method-assign]
    translator._revise_with_context = revise  # type: ignore[method-assign]
    result = asyncio.run(run_page(translator, ["Where?", "Me too.", "Wait."]))

    assert result == ["revised-0", "revised-1", "revised-2"]
    assert resolver.calls == 1
    assert resolver.ambiguous_counts == [3]


def test_post_translation_retry_cannot_trigger_a_second_gemini_round() -> None:
    draft = "Cậu chờ tớ."
    resolver = FakeResolver([resolved(0)])
    translator = hybrid([draft], resolver)
    ctx = page_context(["Wait for me."])

    async def classify(*_args: Any, **_kwargs: Any) -> list[UncertaintyItem]:
        return [assessment(0, draft)]

    async def revise(*_args: Any, **_kwargs: Any) -> dict[int, str]:
        return {0: "Cậu chờ tớ."}

    translator._classify_uncertainty = classify  # type: ignore[method-assign]
    translator._revise_with_context = revise  # type: ignore[method-assign]

    async def twice() -> None:
        for _ in range(2):
            await translator.translate_page(
                "auto",
                "VIN",
                ["Wait for me."],
                ctx,
                previous_context="",
                style_guide="",
                memory=RelationshipMemory(),
                page_number=1,
                page_label="1",
            )

    asyncio.run(twice())
    assert resolver.calls == 1


def test_unknown_visual_context_is_revised_neutrally() -> None:
    draft = "Cậu chờ tớ với."
    unknown = resolved(
        0,
        speaker="unknown",
        addressee="unknown",
        relationship="unknown",
        self_reference="unknown",
        address="unknown",
        confidence=0.2,
    )
    resolver = FakeResolver([unknown])
    translator = hybrid([draft], resolver)

    async def classify(*_args: Any, **_kwargs: Any) -> list[UncertaintyItem]:
        return [assessment(0, draft)]

    async def revise(
        _regions: Sequence[RegionContext],
        _ambiguous: Sequence[UncertaintyItem],
        visual: Sequence[ResolvedContextItem],
        *_args: Any,
        **_kwargs: Any,
    ) -> dict[int, str]:
        assert not visual[0].is_resolved()
        return {0: "Chờ một chút."}

    translator._classify_uncertainty = classify  # type: ignore[method-assign]
    translator._revise_with_context = revise  # type: ignore[method-assign]

    assert asyncio.run(run_page(translator, ["Wait for me."])) == ["Chờ một chút."]


@pytest.mark.parametrize(
    "error",
    [
        asyncio.TimeoutError("timeout"),
        GeminiContextError("Gemini returned invalid JSON"),
    ],
)
def test_gemini_timeout_or_invalid_json_preserves_deepseek_output(error: Exception) -> None:
    draft = "Bản dịch DeepSeek ban đầu."
    resolver = FakeResolver(error=error)
    translator = hybrid([draft], resolver)

    async def classify(*_args: Any, **_kwargs: Any) -> list[UncertaintyItem]:
        return [assessment(0, draft)]

    translator._classify_uncertainty = classify  # type: ignore[method-assign]

    assert asyncio.run(run_page(translator, ["Original."])) == [draft]
    assert resolver.calls == 1


def test_cache_hit_does_not_make_a_second_request(tmp_path: Path) -> None:
    config = PronounContextConfig(cache_enabled=True, model="gemini-test")

    class CachedResolver(GeminiContextResolver):
        def __init__(self) -> None:
            super().__init__(config, cache_dir=tmp_path)
            self.requests = 0

        async def _request(self, *_args: Any, **_kwargs: Any) -> str:
            self.requests += 1
            return json.dumps(
                {"resolved_items": [resolved(0).to_dict()]}, ensure_ascii=False
            )

    resolver = CachedResolver()
    region = RegionContext(0, (0, 0, 10, 10), "You?", "Cậu à?")
    ambiguous = assessment(0, "Cậu à?")
    image = Image.new("RGB", (16, 16), "white")
    memory = RelationshipMemory()

    first = asyncio.run(
        resolver.resolve_page(image, [region], [ambiguous], memory)
    )
    second = asyncio.run(
        resolver.resolve_page(image, [region], [ambiguous], memory)
    )

    assert first == second
    assert resolver.requests == 1
    assert len(list(tmp_path.glob("*.json"))) == 1


def test_output_count_order_and_stable_region_ids_are_preserved() -> None:
    sources = ["zero", "one", "two", "three"]
    drafts = ["draft-0", "draft-1", "draft-2", "draft-3"]
    resolver = FakeResolver([resolved(1), resolved(3)])
    translator = hybrid(drafts, resolver)
    ctx = page_context(sources)

    async def classify(*_args: Any, **_kwargs: Any) -> list[UncertaintyItem]:
        return [
            assessment(
                index,
                draft,
                needs_vision=index in (1, 3),
                ambiguity_type="relationship" if index in (1, 3) else "none",
                confidence=0.5 if index in (1, 3) else 0.95,
            )
            for index, draft in enumerate(drafts)
        ]

    async def revise(*_args: Any, **_kwargs: Any) -> dict[int, str]:
        return {1: "revised-1", 3: "revised-3"}

    translator._classify_uncertainty = classify  # type: ignore[method-assign]
    translator._revise_with_context = revise  # type: ignore[method-assign]
    result = asyncio.run(
        translator.translate_page(
            "auto",
            "VIN",
            sources,
            ctx,
            previous_context="",
            style_guide="",
            memory=RelationshipMemory(),
            page_number=1,
            page_label="1",
        )
    )

    assert result == ["draft-0", "revised-1", "draft-2", "revised-3"]
    assert len(result) == len(sources)
    assert [region.translation_context_id for region in ctx.text_regions] == [0, 1, 2, 3]


def test_disabled_hybrid_matches_deepseek_fast_path_and_old_registration_is_unchanged() -> None:
    drafts = ["Giữ nguyên bản dịch DeepSeek."]
    resolver = FakeResolver()
    translator = hybrid(drafts, resolver, enabled=False)

    async def must_not_classify(*_args: Any, **_kwargs: Any) -> list[UncertaintyItem]:
        raise AssertionError("classifier must be disabled")

    translator._classify_uncertainty = must_not_classify  # type: ignore[method-assign]

    assert asyncio.run(run_page(translator, ["Source."])) == drafts
    assert resolver.calls == 0
    assert TRANSLATORS[Translator.deepseek] is DeepseekTranslator
    assert TRANSLATORS[Translator.deepseek_gemini_context] is DeepseekGeminiContextTranslator


def test_high_confidence_relationship_is_not_overwritten_and_conflict_is_exposed() -> None:
    memory = RelationshipMemory()
    memory.update([resolved(0, address="cậu", confidence=0.95)], page_number=1)
    memory.update(
        [
            resolved(
                0,
                relationship="younger_to_older_acquaintance",
                self_reference="cháu",
                address="bác",
                confidence=0.4,
            )
        ],
        page_number=2,
    )

    record = memory.to_dict()["main_character_a->main_character_b"]
    assert record["address"] == "cậu"
    assert record["confidence"] == 0.95
    assert record["conflict"] is True
    assert record["last_seen_page"] == 2


def test_relationship_memory_round_trips_through_chapter_resume_sidecar(
    tmp_path: Path,
) -> None:
    raw_dir = tmp_path / "raw"
    output_dir = tmp_path / "translated"
    raw_dir.mkdir()
    output_dir.mkdir()
    source_page = raw_dir / "001.jpg"
    source_page.write_bytes(b"source-image")
    config = Config(
        translator={
            "translator": "deepseek_gemini_context",
            "target_lang": "VIN",
        }
    )

    writer = MangaTranslator.__new__(MangaTranslator)
    writer.dialogue_consistency = False
    writer._chapter_output_dir = str(output_dir)
    writer._chapter_page_order = ["001.jpg"]
    writer._sidecar_entries_by_page = {
        "001.jpg": PageTranslationContext(
            ["You?"],
            ["Cậu à?"],
            page_name="001.jpg",
            source_image_sha256=MangaTranslator._sha256_file(str(source_page)),
        )
    }
    writer._relationship_memory = RelationshipMemory()
    writer._relationship_memory.update([resolved(0, confidence=0.94)], page_number=1)
    writer._persist_translation_context(config)

    resumed = MangaTranslator.__new__(MangaTranslator)
    resumed.dialogue_consistency = False
    resumed._begin_chapter_context(
        str(raw_dir), str(output_dir), config, ["001.jpg"]
    )

    record = resumed._relationship_memory.to_dict()[
        "main_character_a->main_character_b"
    ]
    assert record["relationship"] == "close_same_age_friends"
    assert record["address"] == "cậu"
    assert record["confidence"] == 0.94


def test_hybrid_context_alone_forces_chapter_order_but_disabled_mode_does_not() -> None:
    core = MangaTranslator.__new__(MangaTranslator)
    core.dialogue_consistency = False
    enabled = Config(
        translator={"translator": "deepseek_gemini_context", "target_lang": "VIN"}
    )
    disabled = Config(
        translator={
            "translator": "deepseek_gemini_context",
            "target_lang": "VIN",
            "pronoun_context": {"enabled": False},
        }
    )

    assert core._requires_sequential_translation(enabled)
    assert not core._requires_sequential_translation(disabled)


def test_core_dispatches_hybrid_with_existing_image_and_regions(monkeypatch: pytest.MonkeyPatch) -> None:
    observed: dict[str, Any] = {}

    class CoreHybrid:
        def parse_args(self, config: Any) -> None:
            observed["config"] = config

        def set_prev_context(self, value: str) -> None:
            observed["previous"] = value

        def set_dialogue_style_guide(self, value: str) -> None:
            observed["style"] = value

        async def translate_page(
            self,
            _from_lang: str,
            _to_lang: str,
            queries: list[str],
            ctx: Context,
            **kwargs: Any,
        ) -> list[str]:
            observed["ctx"] = ctx
            observed["kwargs"] = kwargs
            return [f"translated:{query}" for query in queries]

    monkeypatch.setitem(
        TRANSLATORS, Translator.deepseek_gemini_context, CoreHybrid
    )
    core = MangaTranslator.__new__(MangaTranslator)
    core.context_size = 0
    core.dialogue_consistency = False
    core.page_translation_history = []
    core.all_page_translations = []
    core._original_page_texts = []
    core._style_guide_cache_path = None
    core._style_guide_cache_text = ""
    core._relationship_memory = RelationshipMemory()
    core._hybrid_previous_images = []
    core._hybrid_page_counter = 0
    core.use_mtpe = False
    ctx = page_context(["Source"])
    ctx.page_name = "015.jpg"
    config = Config(
        translator={
            "translator": "deepseek_gemini_context",
            "target_lang": "VIN",
            "pronoun_context": {"previous_pages": 0},
        }
    )

    result = asyncio.run(core._translate_context_aware(config, ["Source"], ctx))

    assert result == ["translated:Source"]
    assert observed["ctx"] is ctx
    assert observed["kwargs"]["page_label"] == "015.jpg"
    assert ctx.input is observed["ctx"].input


def test_gemini_request_contains_real_image_part_not_a_path(tmp_path: Path) -> None:
    captured: dict[str, Any] = {}

    class Models:
        async def generate_content(self, **kwargs: Any) -> Any:
            captured.update(kwargs)
            return SimpleNamespace(
                text=json.dumps(
                    {"resolved_items": [resolved(0).to_dict()]}, ensure_ascii=False
                )
            )

    client = SimpleNamespace(aio=SimpleNamespace(models=Models()))
    resolver = GeminiContextResolver(
        PronounContextConfig(cache_enabled=False, model="gemini-test"),
        client=client,
        cache_dir=tmp_path,
    )
    asyncio.run(
        resolver.resolve_page(
            Image.new("RGB", (8, 8), "white"),
            [RegionContext(0, (0, 0, 8, 8), "You?", "Cậu à?")],
            [assessment(0, "Cậu à?")],
            RelationshipMemory(),
        )
    )

    parts = captured["contents"][0].parts
    assert parts[1].inline_data.mime_type == "image/png"
    assert isinstance(parts[1].inline_data.data, bytes)
    assert parts[1].inline_data.data.startswith(b"\x89PNG")
