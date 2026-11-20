"""Official Google GenAI SDK wrapper using the Gemini Interactions API (`client.interactions.create`)
with granular connect/read/overall timeouts, elapsed-time logging, Pydantic structured validation,
and disabled SDK internal retries on timeouts / 429 quota errors.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Optional, Tuple, Type, TypeVar

try:
    import httpx
    _HAS_HTTPX = True
except ImportError:
    httpx = None  # type: ignore[assignment]
    _HAS_HTTPX = False

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
    """Internal exception triggering a single safe retry when Gemini returns malformed JSON or schema mismatch."""


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


def _classify_timeout_error(exc: Exception) -> Optional[str]:
    """Distinguish connection timeout, read timeout, and overall request timeout."""
    exc_chain = [exc]
    cause = getattr(exc, "__cause__", None)
    if cause is not None:
        exc_chain.append(cause)
        if getattr(cause, "__cause__", None) is not None:
            exc_chain.append(cause.__cause__)

    names = " ".join(type(e).__name__.lower() for e in exc_chain)
    msg = str(exc).lower()

    if "connecttimeout" in names or "connect timeout" in msg:
        return "connection timeout"
    if "readtimeout" in names or "read operation timed out" in msg or "read timeout" in msg:
        return "read timeout"
    if "timeout" in names or "timed out" in msg:
        return "overall request timeout"
    return None


class GeminiClient:
    """Thread-safe wrapper around the official `google-genai` SDK using the Gemini Interactions API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = CONFIG.DEFAULT_GEMINI_MODEL,
        temperature: float = CONFIG.LLM_TEMPERATURE,
        connect_timeout_seconds: float = CONFIG.API_CONNECT_TIMEOUT_SECONDS,
        read_timeout_seconds: float = CONFIG.API_READ_TIMEOUT_SECONDS,
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
        # Bounded granular timeouts: fast connect (10s), generous read (60-90s, default 75s)
        self.connect_timeout_seconds = max(5.0, min(float(connect_timeout_seconds), 20.0))
        self.read_timeout_seconds = max(30.0, min(float(read_timeout_seconds), 90.0))
        self.timeout_seconds = self.read_timeout_seconds

        try:
            # Disable SDK internal retries (attempts=1) so timeouts and 429 quota errors are never retried silently
            http_opts_kwargs: dict = {
                "api_version": "v1beta",
                "timeout": int(self.read_timeout_seconds * 1000),
            }
            if hasattr(types, "HttpRetryOptions"):
                http_opts_kwargs["retry_options"] = types.HttpRetryOptions(attempts=1)

            self.client = genai.Client(
                api_key=self._api_key,
                http_options=types.HttpOptions(**http_opts_kwargs),
            )
        except Exception as exc:
            safe_msg = scrub_secrets_from_message(str(exc), [self._api_key])
            raise LLMServiceError(
                f"Failed to initialize Google GenAI client: {safe_msg}"
            ) from exc

    def _build_request_timeout(self):
        """Return an `httpx.Timeout` separating connect, read, write, and pool timeouts when httpx is available."""
        if _HAS_HTTPX and httpx is not None:
            return httpx.Timeout(
                connect=self.connect_timeout_seconds,
                read=self.read_timeout_seconds,
                write=15.0,
                pool=10.0,
            )
        return self.read_timeout_seconds

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
        t0 = time.perf_counter()
        try:
            if hasattr(self.client, "interactions"):
                interaction = self.client.interactions.create(
                    model=self.model_name,
                    input="Reply with the single word OK.",
                    generation_config={"temperature": 0.0},
                    timeout=self._build_request_timeout(),
                )
                elapsed = time.perf_counter() - t0
                text_out = self._extract_interaction_text(interaction)
                if text_out:
                    return True, f"Connected to Gemini (`{self.model_name}`) via Interactions API in {elapsed:.1f}s."
                return False, f"Gemini returned an empty response after {elapsed:.1f}s."

            response = self.client.models.generate_content(
                model=self.model_name,
                contents="Reply with OK.",
                config=types.GenerateContentConfig(max_output_tokens=5, temperature=0.0),
            )
            elapsed = time.perf_counter() - t0
            if response is not None:
                return True, f"Connected to Gemini (`{self.model_name}`) in {elapsed:.1f}s."
            return False, "Gemini returned an empty response."
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            timeout_kind = _classify_timeout_error(exc)
            safe_msg = scrub_secrets_from_message(str(exc), [self._api_key])
            if timeout_kind:
                return False, f"Gemini {timeout_kind} after {elapsed:.1f}s: {safe_msg}"
            return False, f"Gemini connection failed after {elapsed:.1f}s: {safe_msg}"

    @retry(
        reraise=True,
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=2),
        # Strictly retry ONLY malformed JSON responses—never retry timeouts or 429 quota errors
        retry=retry_if_exception_type((MalformedLLMJSONError,)),
    )
    def _invoke_structured(
        self,
        system_instruction: str,
        user_prompt: str,
        response_schema: Type[T],
    ) -> T:
        """Execute a structured JSON generation call via Gemini Interactions API (with models.generate_content fallback)."""
        t0 = time.perf_counter()
        logger.info(
            "Gemini structured request started | model=%s | schema=%s | prompt_chars=%d | connect_timeout=%.1fs | read_timeout=%.1fs",
            self.model_name,
            response_schema.__name__,
            len(user_prompt),
            self.connect_timeout_seconds,
            self.read_timeout_seconds,
        )
        if hasattr(self.client, "interactions"):
            try:
                interaction = self.client.interactions.create(
                    model=self.model_name,
                    input=user_prompt,
                    system_instruction=system_instruction,
                    response_format=response_schema.model_json_schema(),
                    generation_config={"temperature": self.temperature},
                    timeout=self._build_request_timeout(),
                )
            except Exception as exc:
                elapsed = time.perf_counter() - t0
                timeout_kind = _classify_timeout_error(exc)
                if timeout_kind:
                    logger.error(
                        "Gemini Interactions API %s after %.2fs | model=%s | schema=%s | prompt_chars=%d",
                        timeout_kind,
                        elapsed,
                        self.model_name,
                        response_schema.__name__,
                        len(user_prompt),
                    )
                else:
                    logger.error(
                        "Gemini Interactions API error after %.2fs | model=%s | type=%s",
                        elapsed,
                        self.model_name,
                        type(exc).__name__,
                    )
                raise

            elapsed = time.perf_counter() - t0
            raw_text = _strip_code_fences(self._extract_interaction_text(interaction))
            if not raw_text:
                raise MalformedLLMJSONError("Gemini Interactions API returned an empty response body.")
            try:
                parsed = response_schema.model_validate_json(raw_text)
                logger.info(
                    "Gemini Interactions API succeeded in %.2fs | model=%s | schema=%s",
                    elapsed,
                    self.model_name,
                    response_schema.__name__,
                )
                return parsed
            except (ValidationError, json.JSONDecodeError) as parse_err:
                logger.warning("Malformed JSON or schema mismatch from Interactions API after %.2fs; retrying once...", elapsed)
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
            logger.warning("Malformed JSON or schema mismatch from LLM; retrying once...")
            raise MalformedLLMJSONError(str(parse_err)) from parse_err

    def generate_structured(
        self,
        system_instruction: str,
        user_prompt: str,
        response_schema: Type[T],
    ) -> T:
        """Public method to generate and validate structured Pydantic output with clear elapsed-time & timeout errors."""
        t0 = time.perf_counter()
        try:
            return self._invoke_structured(
                system_instruction=system_instruction,
                user_prompt=user_prompt,
                response_schema=response_schema,
            )
        except MalformedLLMJSONError as mal_err:
            elapsed = time.perf_counter() - t0
            raise LLMServiceError(
                f"LLM response failed schema validation after {elapsed:.1f}s: {mal_err}"
            ) from mal_err
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            timeout_kind = _classify_timeout_error(exc)
            safe_msg = scrub_secrets_from_message(str(exc), [self._api_key])
            if timeout_kind:
                raise LLMServiceError(
                    f"Gemini API {timeout_kind} after {elapsed:.1f}s "
                    f"(connect={self.connect_timeout_seconds:.0f}s, read={self.read_timeout_seconds:.0f}s). "
                    f"Try reducing 'Max Claims to Extract' or retrying shortly. Details: {safe_msg}"
                ) from exc
            if "429" in safe_msg or "resource_exhausted" in safe_msg.lower() or "quota" in safe_msg.lower():
                raise LLMServiceError(
                    f"Gemini API rate limit / quota exceeded (429) after {elapsed:.1f}s (not retried): {safe_msg}"
                ) from exc
            raise LLMServiceError(
                f"Gemini API request failed after {elapsed:.1f}s: {safe_msg}"
            ) from exc
