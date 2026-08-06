from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

from omegaconf import OmegaConf

from manga_translator.config import Config, Translator
from manga_translator.dialogue import PageTranslationContext
from manga_translator.dialogue import validate_style_guide
from manga_translator.manga_translator import MangaTranslator
from manga_translator.translators.common_gpt import CommonGPTTranslator
from manga_translator.translators.deepseek import DeepseekTranslator


class CharacterTokenCounter:
    def count_tokens(self, text: str) -> int:
        return len(text)


def deepseek_translator() -> DeepseekTranslator:
    translator = DeepseekTranslator.__new__(DeepseekTranslator)
    CommonGPTTranslator.__init__(translator, config_key="deepseek.deepseek-chat")
    translator.tokenizer = CharacterTokenCounter()
    translator.token_count = 0
    translator.token_count_last = 0
    translator.config = OmegaConf.create(
        {
            "deepseek": {
                "temperature": 0.1,
                "top_p": 0.8,
                "include_template": True,
                "prompt_template": "Translate current lines only.",
                "chat_system_template": "Translate to {to_lang}; preserve every marker.",
                "chat_sample": {},
            }
        }
    )
    return translator


def bare_core(context_size: int = 4) -> MangaTranslator:
    core = MangaTranslator.__new__(MangaTranslator)
    core.context_size = context_size
    core.dialogue_consistency = True
    core.page_translation_history = []
    core.all_page_translations = []
    core._original_page_texts = []
    core._sidecar_entries_by_page = {}
    core._style_guide_cache_path = None
    core._style_guide_cache_text = ""
    core._chapter_source_dir = None
    core._chapter_output_dir = None
    core._chapter_page_order = []
    return core


def bilingual_context(*pages: PageTranslationContext) -> str:
    core = bare_core()
    core.page_translation_history.extend(pages)
    return core._build_prev_context()


def test_page_two_request_contains_aligned_page_one_but_only_current_ids() -> None:
    previous = bilingual_context(
        PageTranslationContext(
            source_lines=["そうだね", "君はどうする？"],
            translated_lines=["Ừ, tớ cũng nghĩ vậy.", "Còn cậu định làm gì?"],
            page_name="001.jpg",
        )
    )
    translator = deepseek_translator()
    translator.set_prev_context(previous)
    captured: list[str] = []

    async def request(_to_lang: str, prompt: str) -> str:
        captured.append(prompt)
        return "<|1|>Tớ sẽ về trước."

    translator._request_translation = request  # type: ignore[method-assign]
    result = asyncio.run(translator._translate("Japanese", "Vietnamese", ["先に帰るよ"]))

    assert result == ["Tớ sẽ về trước."]
    assert "<SOURCE_1> そうだね" in captured[0]
    assert "<VI_1> Ừ, tớ cũng nghĩ vậy." in captured[0]
    assert captured[0].count("<|1|>") == 1
    assert "<CURRENT_TEXT>\n<|1|>先に帰るよ\n</CURRENT_TEXT>" in captured[0]


def test_to_cau_choice_is_reused_from_the_actual_request_context() -> None:
    translator = deepseek_translator()
    translator.set_prev_context(
        bilingual_context(
            PageTranslationContext(
                source_lines=["I'll go first.", "Wait for me."],
                translated_lines=["Tớ sẽ đi trước.", "Cậu đợi tớ với."],
            )
        )
    )

    async def context_dependent_response(_to_lang: str, prompt: str) -> str:
        assert "<VI_1> Tớ sẽ đi trước." in prompt
        assert "<VI_2> Cậu đợi tớ với." in prompt
        return "<|1|>Cậu đi cùng tớ nhé."

    translator._request_translation = context_dependent_response  # type: ignore[method-assign]
    result = asyncio.run(
        translator._translate("English", "Vietnamese", ["Come with me."])
    )
    assert result == ["Cậu đi cùng tớ nhé."]


def test_directional_anh_em_style_guide_is_separate_from_current_queries() -> None:
    translator = deepseek_translator()
    translator.set_dialogue_style_guide(
        "relationships:\n"
        "  - speaker: A\n    listener: B\n    self: anh\n    address: em\n"
        "  - speaker: B\n    listener: A\n    self: em\n    address: anh"
    )
    captured: list[str] = []

    async def request(_to_lang: str, prompt: str) -> str:
        captured.append(prompt)
        return "<|1|>Anh đợi em nhé."

    translator._request_translation = request  # type: ignore[method-assign]
    asyncio.run(translator._translate("English", "Vietnamese", ["Wait for me."]))

    prompt = captured[0]
    assert "speaker: A\n    listener: B\n    self: anh\n    address: em" in prompt
    assert "speaker: B\n    listener: A\n    self: em\n    address: anh" in prompt
    assert prompt.count("<|1|>") == 1
    assert "<MANUAL_DIALOGUE_STYLE_GUIDE_DO_NOT_TRANSLATE>" in prompt


def test_narration_toi_does_not_replace_established_friend_dialogue() -> None:
    translator = deepseek_translator()
    translator.set_prev_context(
        bilingual_context(
            PageTranslationContext(
                source_lines=["I had admired her.", "Are you coming?"],
                translated_lines=["Tôi đã luôn ngưỡng mộ cô ấy.", "Cậu có đi không?"],
            )
        )
    )

    async def request(_to_lang: str, prompt: str) -> str:
        assert "<VI_1> Tôi đã luôn ngưỡng mộ cô ấy." in prompt
        assert "<VI_2> Cậu có đi không?" in prompt
        return "<|1|>Tớ đi cùng cậu."

    translator._request_translation = request  # type: ignore[method-assign]
    result = asyncio.run(
        translator._translate("English", "Vietnamese", ["I'll come with you."])
    )
    assert result == ["Tớ đi cùng cậu."]


def test_style_guide_can_lock_inner_monologue_separately_from_dialogue() -> None:
    document = validate_style_guide(
        {
            "default": {
                "peer_pair": {"first_person": "tớ", "second_person": "cậu"},
                "narration_first_person": "tôi",
                "inner_monologue_first_person": "tôi",
            },
            "characters": [
                {"name": "Mei", "self_pronoun": "tớ", "third_person": "cô ấy"}
            ],
            "line_guidance": {
                "What did I do?": "inner monologue; use mình"
            },
        }
    )

    assert document["default"]["peer_pair"]["first_person"] == "tớ"
    assert document["default"]["inner_monologue_first_person"] == "tôi"
    assert document["characters"][0]["third_person"] == "cô ấy"
    assert document["line_guidance"]["What did I do?"].endswith("mình")


def test_token_split_keeps_style_and_context_in_every_child_request() -> None:
    translator = deepseek_translator()
    translator.set_prev_context(
        bilingual_context(
            PageTranslationContext(
                source_lines=["Previous source"],
                translated_lines=["Tớ đã đến rồi."],
            )
        )
    )
    translator.set_dialogue_style_guide("default:\n  unknown_strategy: omit_pronoun_when_natural")
    first_query = "Current line one " + ("x" * 400)
    second_query = "Current line two " + ("y" * 400)
    one_prompt, _ = translator._assemble_prompt_with_context(
        "English", "Vietnamese", [first_query]
    )
    one_tokens = translator._prompt_token_count("Vietnamese", one_prompt)
    # One current line plus full context fits, while both current lines do not fit
    # even after the oldest context page is reduced.
    translator._MAX_TOKENS_IN = one_tokens + 50
    captured: list[str] = []

    async def request(_to_lang: str, prompt: str) -> str:
        captured.append(prompt)
        return "<|1|>Bản dịch"

    translator._request_translation = request  # type: ignore[method-assign]
    result = asyncio.run(
        translator._translate(
            "English", "Vietnamese", [first_query, second_query]
        )
    )

    assert result == ["Bản dịch", "Bản dịch"]
    assert len(captured) == 2
    for prompt in captured:
        assert "<SOURCE_1> Previous source" in prompt
        assert "<VI_1> Tớ đã đến rồi." in prompt
        assert "unknown_strategy: omit_pronoun_when_natural" in prompt
        assert prompt.count("<|1|>") == 1


def test_retry_reuses_the_identical_context_prompt(monkeypatch) -> None:
    translator = deepseek_translator()
    translator.set_prev_context(
        bilingual_context(
            PageTranslationContext(["Source"], ["Tớ hiểu rồi."])
        )
    )
    captured: list[str] = []

    real_sleep = asyncio.sleep

    async def no_sleep(_delay: float) -> None:
        await real_sleep(0)

    async def flaky_request(_to_lang: str, prompt: str) -> str:
        captured.append(prompt)
        if len(captured) == 1:
            raise RuntimeError("temporary failure")
        return "<|1|>Cậu nói tiếp đi."

    monkeypatch.setattr("manga_translator.translators.deepseek.asyncio.sleep", no_sleep)
    translator._request_translation = flaky_request  # type: ignore[method-assign]
    result = asyncio.run(translator._translate("English", "Vietnamese", ["Continue."]))

    assert result == ["Cậu nói tiếp đi."]
    assert len(captured) == 2
    assert captured[0] == captured[1]
    assert "<VI_1> Tớ hiểu rồi." in captured[1]


def test_dialogue_consistency_scheduler_updates_history_before_next_page() -> None:
    core = bare_core()
    core._format_translation_text = MangaTranslator._format_translation_text
    first_region = SimpleNamespace(text="I'll go first.", translation="")
    second_region = SimpleNamespace(text="Wait for me.", translation="")
    contexts = [
        SimpleNamespace(
            text_regions=[first_region],
            page_name="001.jpg",
            source_image_sha256="hash-1",
        ),
        SimpleNamespace(
            text_regions=[second_region],
            page_name="002.jpg",
            source_image_sha256="hash-2",
        ),
    ]
    config = Config(
        translator={
            "translator": "deepseek",
            "target_lang": "VIN",
            "enable_post_translation_check": False,
        },
        render={"uppercase": False},
    )
    contexts_seen: list[str] = []

    async def translate_texts(texts, _config, _ctx):
        contexts_seen.append(core._build_prev_context())
        return ["Tớ đi trước đây."] if texts[0].startswith("I'll") else ["Cậu đợi tớ với."]

    async def post_process(ctx, _config):
        return ctx.text_regions

    core._batch_translate_texts = translate_texts  # type: ignore[method-assign]
    core._apply_post_translation_processing = post_process  # type: ignore[method-assign]
    asyncio.run(
        core._sequential_translate_contexts(
            [(contexts[0], config), (contexts[1], config)]
        )
    )

    assert contexts_seen[0] == ""
    assert "<SOURCE_1> I'll go first." in contexts_seen[1]
    assert "<VI_1> Tớ đi trước đây." in contexts_seen[1]


def test_resume_loads_page_context_only_when_source_hash_matches(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    translated = tmp_path / "translated"
    raw.mkdir()
    translated.mkdir()
    source_page = raw / "001.jpg"
    source_page.write_bytes(b"source-image-one")
    (translated / "001.jpg").write_bytes(b"translated-image-one")

    writer = bare_core()
    writer._chapter_output_dir = str(translated)
    writer._chapter_page_order = ["001.jpg", "002.jpg"]
    source_hash = writer._sha256_file(str(source_page))
    writer._sidecar_entries_by_page["001.jpg"] = PageTranslationContext(
        ["Source page one"],
        ["Tớ đã đến rồi."],
        page_name="001.jpg",
        source_image_sha256=source_hash,
    )
    config = Config(translator={"translator": "deepseek", "target_lang": "VIN"})
    writer._persist_translation_context(config)

    document = json.loads(
        (translated / ".translation-context.json").read_text(encoding="utf-8")
    )
    assert document["pages"][0]["source"] == ["Source page one"]
    assert document["pages"][0]["translation"] == ["Tớ đã đến rồi."]

    resumed = bare_core()
    resumed._begin_chapter_context(
        str(raw), str(translated), config, ["001.jpg", "002.jpg"]
    )
    assert resumed._reuse_persisted_page_context(str(source_page))
    page_two_context = resumed._build_prev_context()
    assert "<SOURCE_1> Source page one" in page_two_context
    assert "<VI_1> Tớ đã đến rồi." in page_two_context

    source_page.write_bytes(b"changed-source-image")
    stale = bare_core()
    stale._begin_chapter_context(
        str(raw), str(translated), config, ["001.jpg", "002.jpg"]
    )
    assert not stale._reuse_persisted_page_context(str(source_page))


def test_overwrite_run_does_not_load_existing_sidecar(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    translated = tmp_path / "translated"
    raw.mkdir()
    translated.mkdir()
    source_page = raw / "001.jpg"
    source_page.write_bytes(b"source")
    (translated / "001.jpg").write_bytes(b"old-output")

    writer = bare_core()
    writer._chapter_output_dir = str(translated)
    writer._chapter_page_order = ["001.jpg"]
    writer._sidecar_entries_by_page["001.jpg"] = PageTranslationContext(
        ["Old source"],
        ["Tôi/anh"],
        page_name="001.jpg",
        source_image_sha256=writer._sha256_file(str(source_page)),
    )
    config = Config(translator={"translator": "deepseek", "target_lang": "VIN"})
    writer._persist_translation_context(config)

    overwritten = bare_core()
    overwritten._begin_chapter_context(
        str(raw),
        str(translated),
        config,
        ["001.jpg"],
        reuse_sidecar=False,
    )

    assert not overwritten._reuse_persisted_page_context(str(source_page))
    assert overwritten._build_prev_context() == ""


def test_response_parser_rejects_changed_marker_count_or_order() -> None:
    assert DeepseekTranslator._parse_marked_response("<|1|>A\n<|2|>B", 2) == ["A", "B"]
    assert DeepseekTranslator._parse_marked_response("<|2|>B\n<|1|>A", 2) is None
    assert DeepseekTranslator._parse_marked_response("<|1|>A\n<|2|>B\n<|3|>C", 2) is None


def test_consistency_validator_retranslates_only_flagged_lines_once() -> None:
    translator = deepseek_translator()
    translator.dialogue_consistency_validator = True
    translator.set_dialogue_style_guide(
        "default:\n"
        "  peer_pair: {first_person: tớ, second_person: cậu}\n"
        "  inner_monologue_first_person: mình"
    )
    translation_prompts: list[str] = []

    async def request_translation(_to_lang: str, prompt: str) -> str:
        translation_prompts.append(prompt)
        if len(translation_prompts) == 1:
            return "<|1|>Tớ đã làm gì cô ấy à...\n<|2|>Cậu chờ tớ với."
        assert "<REJECTED_TRANSLATION> Tớ đã làm gì cô ấy à..." in prompt
        assert "dùng 'mình' cho độc thoại" in prompt
        assert "<|1|>What did I do to her..." in prompt
        assert "Wait for me." not in prompt.split("<CURRENT_TEXT>", 1)[1]
        return "<|1|>Mình đã làm gì cô ấy à..."

    async def validate(_prompt: str) -> str:
        return json.dumps(
            {
                "consistent": False,
                "issues": [
                    {
                        "id": 1,
                        "reason": "Đây là độc thoại nội tâm.",
                        "instruction": "dùng 'mình' cho độc thoại",
                    }
                ],
            },
            ensure_ascii=False,
        )

    translator._request_translation = request_translation  # type: ignore[method-assign]
    translator._request_consistency_validation = validate  # type: ignore[method-assign]
    result = asyncio.run(
        translator._translate(
            "English",
            "Vietnamese",
            ["What did I do to her...", "Wait for me."],
        )
    )

    assert result == ["Mình đã làm gì cô ấy à...", "Cậu chờ tớ với."]
    assert len(translation_prompts) == 2


def test_consistency_validator_json_requires_valid_current_ids() -> None:
    valid = '{"consistent": false, "issues": [{"id": 2, "reason": "r", "instruction": "i"}]}'
    invalid = '{"consistent": false, "issues": [{"id": 3, "reason": "r", "instruction": "i"}]}'

    assert DeepseekTranslator._parse_consistency_validation(valid, 2) == [
        {"id": 2, "reason": "r", "instruction": "i"}
    ]
    assert DeepseekTranslator._parse_consistency_validation(invalid, 2) is None


def test_sparse_vietnamese_page_rejects_chinese_but_ignores_preserved_sfx() -> None:
    core = bare_core()
    chinese = [SimpleNamespace(text="Translate me", translation="遠崎芽衣，和我同班的女孩子")]
    vietnamese = [
        SimpleNamespace(text="SFX", translation="SFX"),
        SimpleNamespace(text="Are you coming?", translation="Cậu có đi cùng tớ không?"),
    ]
    short_vietnamese = [
        SimpleNamespace(text="EHHH!? WHAT!?", translation="HẢ!? GÌ CƠ!? Ơ!?!?!")
    ]

    assert not asyncio.run(core._check_target_language_ratio(chinese, "VIN"))
    assert asyncio.run(core._check_target_language_ratio(vietnamese, "VIN"))
    assert asyncio.run(core._check_target_language_ratio(short_vietnamese, "VIN"))
