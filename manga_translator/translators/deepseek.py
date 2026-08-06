import json
import re

from ..config import TranslatorConfig

try:
    import openai
except ImportError:
    openai = None
import asyncio
import time
from typing import List
from .common import MissingAPIKeyException
from .common_gpt import CommonGPTTranslator
from .keys import DEEPSEEK_API_KEY, DEEPSEEK_API_BASE, DEEPSEEK_MODEL
from .tokenizers.token_counters import deepseekTokenCounter


class DeepseekTranslator(CommonGPTTranslator):
    _INVALID_REPEAT_COUNT = 0  # 现在这个参数没意义了
    _MAX_REQUESTS_PER_MINUTE = 9999  # 无RPM限制
    _TIMEOUT = 40  # 在重试之前等待服务器响应的时间（秒）
    _RETRY_ATTEMPTS = 3  # 在放弃之前重试错误请求的次数
    _TIMEOUT_RETRY_ATTEMPTS = 3  # 在放弃之前重试超时请求的次数
    _RATELIMIT_RETRY_ATTEMPTS = 3  # 在放弃之前重试速率限制请求的次数

    # 最大令牌数量，用于控制处理的文本长度
    # Maximum token count for controlling the length of text processed
    # 
    # 最大输出长度: 8K
    # MAX OUTPUT TOKENS: 8K
    # -- https://api-docs.deepseek.com/quick_start/pricing
    _MAX_TOKENS = 8000

    # 将每个 prompt 限制为最大输出 tokens 的 50％。
    # （这是一个任意比率，用于解释语言之间的差异。）
    # 
    # Limit each prompt to 50% max output tokens. 
    # (This is an arbitrary ratio to account for variance between languages.)
    _MAX_TOKENS_IN = _MAX_TOKENS // 2

    # 是否返回原始提示，用于控制输出内容
    _RETURN_PROMPT = False

    # 是否包含模板，用于决定是否使用预设的提示模板
    _INCLUDE_TEMPLATE = False

    _CONSISTENCY_VALIDATOR_SYSTEM = """Bạn là bộ kiểm tra tính nhất quán xưng hô cho bản dịch manga tiếng Việt.

Đối chiếu manual style guide, previous bilingual context, current source và current translation. Manual style guide có ưu tiên cao nhất. Phân biệt lời thoại, lời kể và độc thoại nội tâm. self_pronoun trong characters CHỈ dùng cho lời thoại trực tiếp có người nghe; nó không được ghi đè narration_first_person hoặc inner_monologue_first_person. Chuỗi câu trần thuật giới thiệu nhân vật ở ngôi ba và kể lại quá khứ, không hướng tới người nghe, phải được coi là lời kể. Chỉ báo lỗi khi một dòng thực sự vi phạm guide/context; không yêu cầu đổi chỉ để đa dạng văn phong.

Chỉ trả một JSON object hợp lệ, không markdown, không giải thích ngoài JSON:
{"consistent": true, "issues": []}
hoặc:
{"consistent": false, "issues": [{"id": 1, "reason": "...", "instruction": "..."}]}

id là số dòng trong CURRENT_TEXT. instruction phải là chỉ dẫn dịch lại cụ thể nhưng không tự viết bản dịch thay thế. Không được đưa ID của previous context vào issues."""

    def __init__(self, check_openai_key=True):
        # CommonGPTTranslator 的初始化
        # CommonGPTTranslator initialization 
        _CONFIG_KEY = 'deepseek.' + DEEPSEEK_MODEL
        CommonGPTTranslator.__init__(self, config_key=_CONFIG_KEY)

        # Initialize the token counter
        self.tokenizer = deepseekTokenCounter()

        self.client = openai.AsyncOpenAI(api_key=openai.api_key or DEEPSEEK_API_KEY)
        if not self.client.api_key and check_openai_key:
            raise MissingAPIKeyException('DEEPSEEK_API_KEY environment variable required')
            
        self.client.base_url = DEEPSEEK_API_BASE
        self.token_count = 0
        self.token_count_last = 0
        self.config = None

    def count_tokens(self, text: str):
        """
        通过字符估计标记很困难，并且因语言而异:
        - 1 个英文字符 ≈ 0.3 个 token。
        - 1 个中文字符 ≈ 0.6 个 token。
        -- https://api-docs.deepseek.com/zh-cn/quick_start/token_usage
        
        因此：使用 deepseek 的 tokenizer 来准确计算 token 的数量。
        
        Estimating tokens by characters is tricky and varies by language:
        - 1 English character ≈ 0.3 token.
        - 1 Chinese character ≈ 0.6 token.
        -- https://api-docs.deepseek.com/quick_start/token_usage
        
        Thus: Use deepseek's tokenizer to accurately count tokens.
        """
        return self.tokenizer.count_tokens(text)


    def _format_prompt_log(self, to_lang: str, prompt: str) -> str:
        prompt = prompt.strip()  
        if to_lang in self.chat_sample:
            return '\n'.join([
                'System:',
                self.chat_system_template.format(to_lang=to_lang),
                'User:',
                self.chat_sample[to_lang][0],
                'Assistant:',
                self.chat_sample[to_lang][1],
                'User:',
                prompt,
            ])
        return '\n'.join([
            'System:',
            self.chat_system_template.format(to_lang=to_lang),
            'User:',
            prompt,
        ])

    @staticmethod
    def _context_pages(context: str) -> list[str]:
        """Return page blocks in reading order without exposing current IDs."""
        if not context:
            return []
        context = re.sub(
            r"^\s*<PREVIOUS_CONTEXT_DO_NOT_TRANSLATE>\s*|\s*</PREVIOUS_CONTEXT_DO_NOT_TRANSLATE>\s*$",
            "",
            context,
        )
        return re.findall(r"\[PAGE -\d+\].*?(?=\n\[PAGE -\d+\]|\Z)", context, re.DOTALL)

    def _prompt_token_count(self, to_lang: str, prompt: str) -> int:
        """Count every fixed and variable component sent to DeepSeek."""
        parts = [self.chat_system_template.format(to_lang=to_lang), prompt]
        sample = self.get_chat_sample(to_lang)
        if sample:
            parts.extend(sample)
        return self.count_tokens("\n".join(parts))

    def _assemble_prompt_with_context(
        self,
        from_lang: str,
        to_lang: str,
        prompt_queries: List[str],
    ) -> tuple[str, int]:
        """Build a prompt while dropping oldest context pages before current text."""
        del from_lang  # Language detection is handled by the common translator layer.
        current = "\n".join(
            f"<|{identifier}|>{query.strip()}"
            for identifier, query in enumerate(prompt_queries, start=1)
        )
        prefix = self.prompt_template.format(to_lang=to_lang) if self.include_template else ""
        style_section = ""
        if self.dialogue_style_guide:
            style_section = (
                "<MANUAL_DIALOGUE_STYLE_GUIDE_DO_NOT_TRANSLATE>\n"
                f"{self.dialogue_style_guide}\n"
                "</MANUAL_DIALOGUE_STYLE_GUIDE_DO_NOT_TRANSLATE>\n\n"
            )

        page_blocks = self._context_pages(self.prev_context)

        def render(blocks: list[str]) -> str:
            context_section = ""
            if blocks:
                context_section = (
                    "<PREVIOUS_CONTEXT_DO_NOT_TRANSLATE>\n"
                    + "\n\n".join(blocks)
                    + "\n</PREVIOUS_CONTEXT_DO_NOT_TRANSLATE>\n\n"
                )
            return (
                f"{prefix.strip()}\n\n" if prefix else ""
            ) + style_section + context_section + (
                "<CURRENT_TEXT>\n"
                f"{current}\n"
                "</CURRENT_TEXT>"
            )

        prompt = render(page_blocks)
        dropped = 0
        while page_blocks and self._prompt_token_count(to_lang, prompt) > self._MAX_TOKENS_IN:
            page_blocks.pop(0)
            dropped += 1
            prompt = render(page_blocks)
        if dropped:
            self.logger.info(
                "Dropped %s oldest context page(s) to fit DeepSeek's token limit.",
                dropped,
            )
        return prompt, len(prompt_queries)

    def _queries_within_token_limit(
        self, from_lang: str, to_lang: str, queries: List[str]
    ) -> bool:
        prompt, _ = self._assemble_prompt_with_context(from_lang, to_lang, queries)
        return self._prompt_token_count(to_lang, prompt) <= self._MAX_TOKENS_IN

    def _split_current_queries(
        self, from_lang: str, to_lang: str, queries: List[str]
    ) -> list[tuple[List[str], list[int]]]:
        """Split only between current lines; context is reassembled for every child."""
        batches: list[tuple[List[str], list[int]]] = []
        current_queries: list[str] = []
        current_indices: list[int] = []
        for index, query in enumerate(queries):
            candidate = current_queries + [query]
            if current_queries and not self._queries_within_token_limit(
                from_lang, to_lang, candidate
            ):
                batches.append((current_queries, current_indices))
                current_queries = []
                current_indices = []
            current_queries.append(query)
            current_indices.append(index)
        if current_queries:
            batches.append((current_queries, current_indices))
        return batches

    @staticmethod
    def _parse_marked_response(response: str, expected_count: int) -> list[str] | None:
        """Accept only an exact, ordered 1:1 marker mapping."""
        matches = list(re.finditer(r"<\|(\d+)\|>", response))
        if [int(match.group(1)) for match in matches] != list(
            range(1, expected_count + 1)
        ):
            return None
        translations: list[str] = []
        for position, match in enumerate(matches):
            start = match.end()
            end = matches[position + 1].start() if position + 1 < len(matches) else len(response)
            value = response[start:end].strip()
            if not value:
                return None
            translations.append(value)
        return translations

    @staticmethod
    def _parse_consistency_validation(
        response: str, expected_count: int
    ) -> list[dict[str, str | int]] | None:
        """Parse the validator's JSON without modifying any translated text."""
        value = response.strip()
        if value.startswith("```"):
            lines = value.splitlines()
            if len(lines) >= 3 and lines[-1].strip() == "```":
                value = "\n".join(lines[1:-1])
                if value.lstrip().startswith("json"):
                    value = value.lstrip()[4:].lstrip()
        try:
            document = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return None
        if not isinstance(document, dict) or not isinstance(
            document.get("consistent"), bool
        ):
            return None
        raw_issues = document.get("issues")
        if not isinstance(raw_issues, list):
            return None
        if document["consistent"]:
            return [] if not raw_issues else None

        issues: list[dict[str, str | int]] = []
        seen_ids: set[int] = set()
        for raw_issue in raw_issues:
            if not isinstance(raw_issue, dict):
                return None
            identifier = raw_issue.get("id")
            reason = raw_issue.get("reason")
            instruction = raw_issue.get("instruction")
            if (
                isinstance(identifier, bool)
                or not isinstance(identifier, int)
                or not 1 <= identifier <= expected_count
                or identifier in seen_ids
                or not isinstance(reason, str)
                or not reason.strip()
                or not isinstance(instruction, str)
                or not instruction.strip()
            ):
                return None
            seen_ids.add(identifier)
            issues.append(
                {
                    "id": identifier,
                    "reason": reason.strip(),
                    "instruction": instruction.strip(),
                }
            )
        return issues or None

    def _trim_validator_prompt(self, to_lang: str, prompt: str) -> str | None:
        """Drop oldest reference pages before allowing validator input overflow."""
        system = self._CONSISTENCY_VALIDATOR_SYSTEM
        trimmed = prompt
        for block in self._context_pages(self.prev_context):
            if self.count_tokens(system + "\n" + trimmed) <= self._MAX_TOKENS_IN:
                return trimmed
            trimmed = trimmed.replace(block, "", 1)
        if self.count_tokens(system + "\n" + trimmed) <= self._MAX_TOKENS_IN:
            return trimmed
        self.logger.warning(
            "Skipping dialogue consistency validator because current lines alone exceed the input token limit."
        )
        return None

    async def _validate_and_retry_consistency(
        self,
        from_lang: str,
        to_lang: str,
        prompt_queries: List[str],
        current_translations: List[str],
        translation_prompt: str,
    ) -> List[str]:
        """Validate once and retranslate only flagged lines once; never text-replace."""
        if not self.dialogue_consistency_validator or not (
            self.dialogue_style_guide or self.prev_context
        ):
            return current_translations

        translated_lines = "\n".join(
            f"<TRANSLATION_{identifier}> {translation}"
            for identifier, translation in enumerate(current_translations, start=1)
        )
        validator_prompt = (
            f"{translation_prompt}\n\n"
            "<CURRENT_TRANSLATIONS_TO_VALIDATE>\n"
            f"{translated_lines}\n"
            "</CURRENT_TRANSLATIONS_TO_VALIDATE>"
        )
        validator_prompt = self._trim_validator_prompt(to_lang, validator_prompt)
        if validator_prompt is None:
            return current_translations

        try:
            response = await self._request_consistency_validation(validator_prompt)
        except Exception as exc:
            self.logger.warning("Dialogue consistency validator failed: %s", exc)
            return current_translations
        self.logger.debug("-- Consistency Validator Response --\n%s", response)
        issues = self._parse_consistency_validation(response, len(prompt_queries))
        if issues is None:
            self.logger.warning(
                "Ignoring malformed dialogue consistency validator response"
            )
            return current_translations
        if not issues:
            return current_translations

        issue_ids = [int(issue["id"]) for issue in issues]
        retry_queries = [prompt_queries[identifier - 1] for identifier in issue_ids]
        retry_prompt, retry_size = self._assemble_prompt_with_context(
            from_lang, to_lang, retry_queries
        )
        corrections = []
        for retry_id, (original_id, issue) in enumerate(zip(issue_ids, issues), start=1):
            corrections.extend(
                [
                    f"[RETRY_LINE {retry_id}]",
                    f"<REJECTED_TRANSLATION> {current_translations[original_id - 1]}",
                    f"<VALIDATOR_REASON> {issue['reason']}",
                    f"<RETRANSLATION_INSTRUCTION> {issue['instruction']}",
                ]
            )
        correction_section = (
            "<CONSISTENCY_CORRECTIONS_DO_NOT_COPY>\n"
            "Đây là lần sửa tính nhất quán duy nhất. Dịch lại chỉ các dòng CURRENT_TEXT "
            "bên dưới theo chỉ dẫn cùng số thứ tự; chỉ trả marker và bản dịch cuối.\n"
            + "\n".join(corrections)
            + "\n</CONSISTENCY_CORRECTIONS_DO_NOT_COPY>\n\n"
        )
        retry_prompt = retry_prompt.replace(
            "<CURRENT_TEXT>", correction_section + "<CURRENT_TEXT>", 1
        )
        retry_prompt = self._trim_validator_prompt(to_lang, retry_prompt)
        if retry_prompt is None:
            return current_translations

        self.logger.info(
            "Dialogue consistency validator flagged current line ID(s): %s; retranslating once",
            ", ".join(map(str, issue_ids)),
        )
        try:
            retry_response = await self._request_translation(to_lang, retry_prompt)
        except Exception as exc:
            self.logger.warning("Dialogue consistency retranslation failed: %s", exc)
            return current_translations
        replacements = self._parse_marked_response(retry_response, retry_size)
        if replacements is None:
            self.logger.warning(
                "Ignoring consistency retranslation with invalid markers/count"
            )
            return current_translations

        corrected = list(current_translations)
        for original_id, replacement in zip(issue_ids, replacements):
            corrected[original_id - 1] = replacement
        return corrected

    async def _translate(self, from_lang: str, to_lang: str, queries: List[str]) -> List[str]:  
        translations = [''] * len(queries)  
        self.logger.debug(f'Temperature: {self.temperature}, TopP: {self.top_p}')  
        MAX_SPLIT_ATTEMPTS = 5  # Default max split attempts  
        RETRY_ATTEMPTS = self._RETRY_ATTEMPTS  

        async def translate_batch(prompt_queries, prompt_query_indices, split_level=0):  
            nonlocal MAX_SPLIT_ATTEMPTS
            split_prefix = ' (split)' if split_level > 0 else ''  

            # Assemble prompt for the current batch  
            prompt, query_size = self._assemble_prompt_with_context(
                from_lang, to_lang, prompt_queries
            )
            self.logger.debug(f'-- GPT Prompt{split_prefix} --\n' + self._format_prompt_log(to_lang, prompt))  

            for attempt in range(RETRY_ATTEMPTS):  
                try:  
                    # Start the translation request with timeout handling
                    request_task = asyncio.create_task(self._request_translation(to_lang, prompt))
                    started = time.time()
                    timeout_attempt = 0
                    while not request_task.done():
                        await asyncio.sleep(0.1)
                        if time.time() - started > self._TIMEOUT + (timeout_attempt * self._TIMEOUT / 2):
                            # Server takes too long to respond
                            if timeout_attempt >= self._TIMEOUT_RETRY_ATTEMPTS:
                                raise Exception('deepseek servers did not respond quickly enough.')
                            timeout_attempt += 1
                            self.logger.warning(f'Restarting request due to timeout. Attempt: {timeout_attempt}')
                            request_task.cancel()
                            request_task = asyncio.create_task(self._request_translation(to_lang, prompt))
                            started = time.time()

                    # Get the response
                    response = await request_task  
                    self.logger.debug(f'-- GPT Response{split_prefix} --\n' + response)  

                    new_translations = self._parse_marked_response(response, query_size)
                    if new_translations is None:
                        remaining_attempts = RETRY_ATTEMPTS - attempt - 1  
                        self.logger.warning(
                            'DeepSeek output markers/count mismatch; %s retry attempt(s) remain before splitting.',
                            remaining_attempts,
                        )
                        continue  

                    new_translations = await self._validate_and_retry_consistency(
                        from_lang,
                        to_lang,
                        prompt_queries,
                        new_translations,
                        prompt,
                    )

                    # Store the translations in the correct indices  
                    for idx, translation in zip(prompt_query_indices, new_translations):  
                        translations[idx] = translation  

                    # Log progress  
                    self.logger.info(f'Batch translated: {len([t for t in translations if t])}/{len(queries)} completed.')  
                    self.logger.debug(f'Completed translations: {[t if t else queries[i] for i, t in enumerate(translations)]}')        
                    return True  # Successfully translated this batch  
                    
                except openai.APIError:
                    if attempt >= RETRY_ATTEMPTS - 1:
                        self.logger.error(
                            'Deepseek encountered a server error, possibly due to high server load. Use a different translator or try again later.')
                        raise
                    self.logger.warning(
                        'Restarting request due to a server error. Attempt: %s',
                        attempt + 1,
                    )
                    await asyncio.sleep(1)
                except Exception as e:  
                    self.logger.error(f'Error during translation attempt: {e}')  
                    if attempt == RETRY_ATTEMPTS - 1:  
                        raise  
                    await asyncio.sleep(1)  

            # If retries exhausted and still not successful, proceed to split if allowed  
            if len(prompt_queries) == 1:
                raise RuntimeError(
                    'DeepSeek returned an invalid marker mapping for a single current line'
                )
            if split_level < MAX_SPLIT_ATTEMPTS:  
                if split_level == 0:  
                    self.logger.warning('Retry limit reached. Starting to split the translation batch.')  
                else:  
                    self.logger.warning('Further splitting the translation batch due to persistent errors.')  
                mid_index = len(prompt_queries) // 2  
                futures = []  
                # Split the batch into two halves  
                for sub_queries, sub_indices in [   
                    (prompt_queries[:mid_index], prompt_query_indices[:mid_index]),  
                    (prompt_queries[mid_index:], prompt_query_indices[mid_index:]),  
                ]:  
                    if sub_queries:  
                        futures.append(translate_batch(sub_queries, sub_indices, split_level + 1))  
                results = await asyncio.gather(*futures)  
                return all(results)  
            else:  
                self.logger.error('Maximum split attempts reached. Unable to translate the following queries:')  
                for idx in prompt_query_indices:  
                    self.logger.error(f'Query: {queries[idx]}')  
                return False  # Indicate failure for this batch   

        # Reduce context first, then split current lines only at line boundaries.
        for prompt_queries, prompt_query_indices in self._split_current_queries(
            from_lang, to_lang, queries
        ):
            await translate_batch(prompt_queries, prompt_query_indices)

        self.logger.debug(translations)  
        if self.token_count_last:  
            self.logger.info(f'Used {self.token_count_last} tokens (Total: {self.token_count})')  
        return translations

    async def _request_translation(self, to_lang: str, prompt: str) -> str:
        system_message = self.chat_system_template.format(to_lang=to_lang)
        messages = [  
            {'role': 'system', 'content': system_message},  
        ]  
        lang_chat_samples = self.get_chat_sample(to_lang)
        if lang_chat_samples:
            messages.append({'role': 'user', 'content': lang_chat_samples[0]})
            messages.append({'role': 'assistant', 'content': lang_chat_samples[1]})
        messages.append({"role": "user", "content": prompt})

        return await self._request_chat(messages)

    async def _request_consistency_validation(self, prompt: str) -> str:
        messages = [
            {"role": "system", "content": self._CONSISTENCY_VALIDATOR_SYSTEM},
            {"role": "user", "content": prompt},
        ]
        return await self._request_chat(messages, max_tokens=1200, temperature=0)

    async def _request_chat(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:

        kwargs = {
            'model': DEEPSEEK_MODEL,
            'messages': messages,
            
            # `max_tokens` only affects output token length. Set to max.
            'max_tokens': max_tokens or self._MAX_TOKENS,
            
            'temperature': self.temperature if temperature is None else temperature,
            'top_p': self.top_p,
        }
        try:
            response = await self.client.beta.chat.completions.parse(**kwargs)
            
            # 添加错误处理和日志
            if not hasattr(response, 'usage') or not hasattr(response.usage, 'total_tokens'):
                self.logger.warning("Response does not contain usage information")
                self.token_count_last = 0
            else:
                self.token_count += response.usage.total_tokens
                self.token_count_last = response.usage.total_tokens
            
            # 获取响应文本
            # Get the response text
            for choice in response.choices:
                if 'text' in choice:
                    return choice.text

            # 如果响应中包含推理内容，记录下来
            # Log reasoning content if available
            if hasattr(response.choices[0].message, 'reasoning_content'):
                self.logger.debug("-- GPT Reasoning --\n" +
                                response.choices[0].message.reasoning_content +
                                "\n------------------\n"
                            )
                
            # If no response with text is found, return the first response's content (which may be empty)
            # 如果没有找到包含文本的响应，则返回第一个响应的内容（可能为空）
            return response.choices[0].message.content
        
        except Exception as e:
            self.logger.error(f"Error in _request_translation: {str(e)}")
            raise
