"""Provider-abstracted LLM client supporting Gemini and Groq with on-disk caching,
exponential backoff retry on 429, per-run call budget enforcement, structured JSON output validation,
and automatic schema repair.
"""

import asyncio
import json
import logging
import random
import re
from typing import Any, Dict, List, Optional, Type, TypeVar, Union
from pydantic import BaseModel, ValidationError

from app.config import settings
from app.clients.cache import llm_cache

logger = logging.getLogger("researchos.llm")

T = TypeVar("T", bound=BaseModel)


class BudgetExceededError(Exception):
    """Raised when the per-run LLM call budget is exhausted."""
    pass


class CallBudgetTracker:
    def __init__(self, max_calls: Optional[int] = None):
        self.max_calls = max_calls if max_calls is not None else settings.MAX_LLM_CALLS_PER_RUN
        self.call_count = 0
        self.cached_count = 0

    def check_budget(self) -> None:
        if self.call_count >= self.max_calls:
            raise BudgetExceededError(
                f"LLM call budget exceeded ({self.call_count}/{self.max_calls} calls used)."
            )

    def record_call(self, is_cached: bool = False) -> None:
        if is_cached:
            self.cached_count += 1
        else:
            self.check_budget()
            self.call_count += 1


def _extract_json_block(text: str) -> str:
    """Extracts JSON string from markdown code block or raw text."""
    text = text.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if match:
        return match.group(1).strip()
    # Try finding the first '{' or '[' and last '}' or ']'
    start_brace = text.find("{")
    start_bracket = text.find("[")
    if start_brace != -1 and (start_bracket == -1 or start_brace < start_bracket):
        end_brace = text.rfind("}")
        if end_brace != -1 and end_brace > start_brace:
            return text[start_brace : end_brace + 1]
    elif start_bracket != -1:
        end_bracket = text.rfind("]")
        if end_bracket != -1 and end_bracket > start_bracket:
            return text[start_bracket : end_bracket + 1]
    return text


class LLMClient:
    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        budget_tracker: Optional[CallBudgetTracker] = None,
        force_mock: bool = False,
    ):
        self.provider = provider or settings.LLM_PROVIDER
        self.model = model or settings.LLM_MODEL
        self.budget_tracker = budget_tracker or CallBudgetTracker()
        self.force_mock = force_mock

    async def _call_gemini_raw(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: float = 0.2,
    ) -> str:
        """Executes raw call to Gemini API using google-genai SDK."""
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=settings.GEMINI_API_KEY)
        config = types.GenerateContentConfig(
            temperature=temperature,
            system_instruction=system_instruction,
        )

        resp = await client.aio.models.generate_content(
            model=self.model,
            contents=prompt,
            config=config,
        )
        return resp.text or ""

    async def _call_groq_raw(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: float = 0.2,
    ) -> str:
        """Executes raw call to Groq API using OpenAI-compatible interface."""
        from groq import AsyncGroq

        client = AsyncGroq(api_key=settings.GROQ_API_KEY)
        messages: List[Dict[str, str]] = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})

        # Default Groq model if Gemini model was set in env
        groq_model = self.model if "gemini" not in self.model.lower() else "llama-3.3-70b-versatile"

        resp = await client.chat.completions.create(
            model=groq_model,
            messages=messages,
            temperature=temperature,
        )
        return resp.choices[0].message.content or ""

    async def _execute_with_retry(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: float = 0.2,
        max_retries: int = 3,
    ) -> str:
        """Executes LLM call with exponential backoff on 429/transient errors."""
        delay = 2.0
        last_error = None

        for attempt in range(max_retries):
            try:
                if self.provider == "groq" and settings.GROQ_API_KEY:
                    return await self._call_groq_raw(prompt, system_instruction, temperature)
                elif settings.GEMINI_API_KEY:
                    return await self._call_gemini_raw(prompt, system_instruction, temperature)
                else:
                    raise ValueError(
                        f"No API key configured for provider '{self.provider}' (GEMINI_API_KEY or GROQ_API_KEY required)."
                    )
            except Exception as e:
                err_str = str(e).lower()
                last_error = e
                # Check for rate limit (429) or temporary server errors
                if any(x in err_str for x in ["429", "resource_exhausted", "rate_limit", "quota", "503"]):
                    logger.warning(
                        f"Rate limit or 503 encountered (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {delay}s..."
                    )
                    await asyncio.sleep(delay + random.uniform(0.1, 0.5))
                    delay *= 2.0
                else:
                    logger.error(f"Non-retryable LLM error: {e}")
                    raise

        raise last_error or RuntimeError("Max retries exceeded for LLM call.")

    async def generate_text(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: float = 0.2,
    ) -> str:
        """Generates plain text response with caching and budget enforcement."""
        cache_key = {
            "provider": self.provider,
            "model": self.model,
            "prompt": prompt,
            "system_instruction": system_instruction,
            "temperature": temperature,
        }

        # Check disk cache
        cached = llm_cache.get(cache_key)
        if cached is not None:
            self.budget_tracker.record_call(is_cached=True)
            return cached

        # Check mock mode or missing keys
        if self.force_mock or (not settings.GEMINI_API_KEY and not settings.GROQ_API_KEY):
            from app.clients.mock_llm import get_mock_text_response
            resp_text = get_mock_text_response(prompt, system_instruction)
            llm_cache.set(cache_key, resp_text)
            self.budget_tracker.record_call(is_cached=False)
            return resp_text

        # Enforce budget before live API call
        self.budget_tracker.check_budget()

        # Real API call
        resp_text = await self._execute_with_retry(prompt, system_instruction, temperature)
        llm_cache.set(cache_key, resp_text)
        self.budget_tracker.record_call(is_cached=False)
        return resp_text

    async def generate_structured(
        self,
        prompt: str,
        schema: Type[T],
        system_instruction: Optional[str] = None,
        temperature: float = 0.1,
    ) -> T:
        """Generates structured response validated against a Pydantic schema.
        Includes disk caching and ONE automatic repair retry if JSON parsing fails.
        """
        schema_json_str = json.dumps(schema.model_json_schema(), indent=2)
        full_system = (
            f"{system_instruction or 'You are a precise scientific research agent.'}\n\n"
            f"You MUST output ONLY a valid JSON object strictly complying with this JSON Schema:\n"
            f"{schema_json_str}\n"
            f"Do not include any markdown fences or conversational prelude outside the JSON."
        )

        cache_key = {
            "provider": self.provider,
            "model": self.model,
            "prompt": prompt,
            "system_instruction": full_system,
            "schema": schema.__name__,
            "temperature": temperature,
        }

        # Check cache
        cached = llm_cache.get(cache_key)
        if cached is not None:
            try:
                parsed = schema.model_validate(cached)
                self.budget_tracker.record_call(is_cached=True)
                return parsed
            except ValidationError:
                pass  # Fall through if schema changed

        # Check mock mode or missing keys
        if self.force_mock or (not settings.GEMINI_API_KEY and not settings.GROQ_API_KEY):
            from app.clients.mock_llm import get_mock_structured_response
            mock_data = get_mock_structured_response(prompt, schema)
            llm_cache.set(cache_key, mock_data.model_dump())
            self.budget_tracker.record_call(is_cached=False)
            return mock_data

        # Enforce budget
        self.budget_tracker.check_budget()

        # Initial call
        raw_resp = await self._execute_with_retry(prompt, full_system, temperature)
        self.budget_tracker.record_call(is_cached=False)

        json_str = _extract_json_block(raw_resp)
        try:
            parsed = schema.model_validate_json(json_str)
            llm_cache.set(cache_key, parsed.model_dump())
            return parsed
        except (ValidationError, json.JSONDecodeError) as parse_err:
            logger.warning(
                f"Initial structured output parsing failed ({parse_err}). Attempting 1 automatic repair retry..."
            )

        # Automatic repair retry (exactly one retry as per prompt specs)
        repair_prompt = (
            f"The previous output was invalid JSON or did not match the required schema.\n"
            f"Error details: {parse_err}\n"
            f"Previous output:\n{raw_resp}\n\n"
            f"Please output strictly valid JSON conforming to the schema without commentary."
        )

        self.budget_tracker.check_budget()
        repair_resp = await self._execute_with_retry(repair_prompt, full_system, temperature=0.0)
        self.budget_tracker.record_call(is_cached=False)

        repaired_json = _extract_json_block(repair_resp)
        parsed = schema.model_validate_json(repaired_json)
        llm_cache.set(cache_key, parsed.model_dump())
        return parsed
