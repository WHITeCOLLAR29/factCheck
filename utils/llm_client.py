"""Official Google GenAI SDK wrapper using the Gemini Interactions API (`client.interactions.create`)
with Pydantic structured output validation, rate-limit/timeout retries, and safe fallback.
"""

from __future__ import annotations

import json
import logging
from typing import Optional, Tuple, Type, TypeVar

try:
    from google import genai
    from google.genai import types
    from google.genai.errors import APIError
    _HAS_GENAI = True
except ImportError:
    genai = None  # type: ignore[assignment]
    types = None  # type: ignore[assignment]
    APIError = Exception  # type: ignore[assignment,misc]
    _HAS_GENAI = False

from pydantic import BaseModel, ValidationError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from config.settings import CONFIG, get_secret
from utils.security import scrub_secrets_from_message

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class LLMServiceError(Exception):
    """Custom exception raised when Gemini API fails or returns invalid structured data."""


class MalformedLLMJSONError(Exception):
    """Internal exception triggering automatic retry when Gemini returns malformed JSON or schema mismatch."""


def _strip_code_fences(raw_text: str) -> str:
    """Remove accidental markdown code fences around JSON output."""
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    return cleaned


class GeminiClient:
    """Thread-safe wrapper around the official `google-genai` SDK using the Gemini Interactions API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = CONFIG.DEFAULT_GEMINI_MODEL,
        temperature: float = CONFIG.LLM_TEMPERATURE,
        timeout_seconds: float = float(CONFIG.API_TIMEOUT_SECONDS),
    ) -> None:
        if not _HAS_GENAI:
            raise LLMServiceError(
                "The `google-genai` package is not installed. Please run `pip install -r requirements.txt`."
            )
        resolved_key = get_secret("GEMINI_API_KEY", session_override=api_key)
        if not resolved_key:
            raise LLMServiceError(
                "Gemini API Key is missing. Please configure `GEMINI_API_KEY` in `.streamlit/secrets.toml`, environment variables, or the sidebar."
            )
        self._api_key = resolved_key
        self.model_name = model_name
        self.temperature = temperature
        self.timeout_seconds = max(15.0, float(timeout_seconds))
        try:
            self.client = genai.Client(
                api_key=self._api_key,
                http_options=types.HttpOptions(api_version="v1beta"),
            )
        except Exception as exc:
            safe_msg = scrub_secrets_from_message(str(exc), [self._api_key])
            raise LLMServiceError(
                f"Failed to initialize Google GenAI client: {safe_msg}"
            ) from exc

    def _extract_interaction_text(self, interaction: object) -> str:
        """Extract the final text output from a Gemini Interactions API response."""
        outputs = getattr(interaction, "outputs", None) or []
        for output in reversed(outputs):
            text_val = getattr(output, "text", None)
            if isinstance(text_val, str) and text_val.strip():
                return text_val.strip()
        return ""

    def test_connection(self) -> Tuple[bool, str]:
        """Verify Gemini API connectivity with a minimal request without exposing credentials."""
        try:
            if hasattr(self.client, "interactions"):
                interaction = self.client.interactions.create(
                    model=self.model_name,
                    input="Reply with the single word OK.",
                    generation_config={"temperature": 0.0},
                    timeout=self.timeout_seconds,
                )
                text_out = self._extract_interaction_text(interaction)
                if text_out:
                    return True, f"Connected to Gemini (`{self.model_name}`) via Interactions API."
                return False, "Gemini returned an empty response."

            response = self.client.models.generate_content(
                model=self.model_name,
                contents="Reply with OK.",
                config=types.GenerateContentConfig(max_output_tokens=5, temperature=0.0),
            )
            if response is not None:
                return True, f"Connected to Gemini (`{self.model_name}`)."
            return False, "Gemini returned an empty response."
        except Exception as exc:
            safe_msg = scrub_secrets_from_message(str(exc), [self._api_key])
            return False, f"Gemini connection failed: {safe_msg}"

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception_type(
            (APIError, ConnectionError, TimeoutError, MalformedLLMJSONError)
        ),
    )
    def _invoke_structured(
        self,
        system_instruction: str,
        user_prompt: str,
        response_schema: Type[T],
    ) -> T:
        """Execute a structured JSON generation call via Gemini Interactions API (with models.generate_content fallback)."""
        if hasattr(self.client, "interactions"):
            interaction = self.client.interactions.create(
                model=self.model_name,
                input=user_prompt,
                system_instruction=system_instruction,
                response_format=response_schema.model_json_schema(),
                generation_config={"temperature": self.temperature},
                timeout=self.timeout_seconds,
            )
            raw_text = _strip_code_fences(self._extract_interaction_text(interaction))
            if not raw_text:
                raise MalformedLLMJSONError("Gemini Interactions API returned an empty response body.")
            try:
                return response_schema.model_validate_json(raw_text)
            except (ValidationError, json.JSONDecodeError) as parse_err:
                logger.warning("Malformed JSON or schema mismatch from Interactions API; retrying...")
                raise MalformedLLMJSONError(str(parse_err)) from parse_err

        # Fallback for older google-genai SDKs without `client.interactions`
        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=self.temperature,
            response_mime_type="application/json",
            response_schema=response_schema,
        )
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=user_prompt,
            config=config,
        )

        parsed_obj = getattr(response, "parsed", None)
        if isinstance(parsed_obj, response_schema):
            return parsed_obj
        if isinstance(parsed_obj, dict):
            try:
                return response_schema.model_validate(parsed_obj)
            except ValidationError as val_err:
                raise MalformedLLMJSONError(str(val_err)) from val_err

        raw_text = _strip_code_fences(response.text or "")
        if not raw_text:
            raise MalformedLLMJSONError("Gemini returned an empty response body.")

        try:
            return response_schema.model_validate_json(raw_text)
        except (ValidationError, json.JSONDecodeError) as parse_err:
            logger.warning("Malformed JSON or schema mismatch from LLM; retrying...")
            raise MalformedLLMJSONError(str(parse_err)) from parse_err

    def generate_structured(
        self,
        system_instruction: str,
        user_prompt: str,
        response_schema: Type[T],
    ) -> T:
        """Public method to generate and validate structured Pydantic output with safe error handling."""
        try:
            return self._invoke_structured(
                system_instruction=system_instruction,
                user_prompt=user_prompt,
                response_schema=response_schema,
            )
        except MalformedLLMJSONError as mal_err:
            raise LLMServiceError(
                f"LLM response failed schema validation after retries: {mal_err}"
            ) from mal_err
        except Exception as exc:
            safe_msg = scrub_secrets_from_message(str(exc), [self._api_key])
            raise LLMServiceError(f"Gemini API request failed: {safe_msg}") from exc
