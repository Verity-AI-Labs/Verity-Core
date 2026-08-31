"""LLM access for the audit tools, against any OpenAI-compatible endpoint.

Verity serves open-source models through vLLM, so this is a thin wrapper over
``POST /v1/chat/completions`` rather than a multi-provider abstraction.

Two features exist specifically because these calls drive audits:

* **A disk cache**, so re-running an audit does not re-sample the model. Audits are
  compared against each other over time, and a re-run that quietly draws new samples
  would make two scorecards incomparable.
* **A usage accumulator**, so every scorecard can state what it cost to produce.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Any

import httpx

DEFAULT_BASE_URL = "http://localhost:8000/v1"
DEFAULT_CACHE_DIR = Path(".verity_cache") / "models"
DEFAULT_TIMEOUT_SECONDS = 120.0
DEFAULT_MAX_RETRIES = 2
RETRYABLE_STATUS_CODES = frozenset({408, 409, 429, 500, 502, 503, 504})

__all__ = [
    "DEFAULT_BASE_URL",
    "ModelClient",
    "ModelError",
    "ModelResponse",
    "ResponseCache",
    "TokenUsage",
]


class ModelError(RuntimeError):
    """Raised when the endpoint rejects a request or returns an unusable body."""


@dataclass(frozen=True, slots=True)
class TokenUsage:
    """Token counts for one call, or the running total across a session."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def __add__(self, other: TokenUsage) -> TokenUsage:
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
        )

    def to_dict(self) -> dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> TokenUsage:
        """Build usage from an API payload, deriving ``total_tokens`` when omitted.

        Not every OpenAI-compatible server reports a total, so it is reconstructed
        from the parts instead of being left at zero.
        """
        if not data:
            return cls()
        prompt = int(data.get("prompt_tokens") or 0)
        completion = int(data.get("completion_tokens") or 0)
        total = int(data.get("total_tokens") or 0) or prompt + completion
        return cls(prompt_tokens=prompt, completion_tokens=completion, total_tokens=total)


@dataclass(slots=True)
class ModelResponse:
    """One completion, plus whether it was served from cache."""

    content: str
    model: str
    usage: TokenUsage = field(default_factory=TokenUsage)
    cached: bool = False
    finish_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "content": self.content,
            "model": self.model,
            "usage": self.usage.to_dict(),
            "cached": self.cached,
            "finish_reason": self.finish_reason,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelResponse:
        return cls(
            content=data.get("content", ""),
            model=data.get("model", ""),
            usage=TokenUsage.from_dict(data.get("usage")),
            cached=bool(data.get("cached", False)),
            finish_reason=data.get("finish_reason", ""),
        )


class ResponseCache:
    """A content-addressed JSON file store for completions.

    Keys are sharded into subdirectories by their first two hex characters, which
    keeps a long-running audit from producing a single directory with tens of
    thousands of entries.
    """

    def __init__(self, directory: Path | str) -> None:
        self.directory = Path(directory).expanduser()

    @staticmethod
    def make_key(
        *,
        base_url: str,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float,
        max_tokens: int,
        extra: dict[str, Any] | None = None,
    ) -> str:
        """Hash everything that could change the response into one cache key.

        ``base_url`` is part of the key because two endpoints can serve different
        weights under the same model name; without it, pointing at a new vLLM server
        would silently return the previous server's answers.
        """
        payload = {
            "base_url": base_url,
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "extra": extra or {},
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def path_for(self, key: str) -> Path:
        return self.directory / key[:2] / f"{key}.json"

    def get(self, key: str) -> ModelResponse | None:
        """Return the cached response, treating an unreadable entry as a miss.

        A truncated or hand-edited cache file should cost one extra API call, not
        crash an audit mid-run.
        """
        path = self.path_for(key)
        if not path.is_file():
            return None
        try:
            with path.open("r", encoding="utf-8") as handle:
                record = json.load(handle)
            return ModelResponse.from_dict(record["response"])
        except (json.JSONDecodeError, KeyError, OSError, TypeError, ValueError):
            return None

    def set(self, key: str, response: ModelResponse, request: dict[str, Any] | None = None) -> None:
        """Write an entry atomically so a crash cannot leave a half-written file."""
        path = self.path_for(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "key": key,
            "created_at": datetime.now(UTC).isoformat(),
            "request": request or {},
            "response": response.to_dict(),
        }
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump(record, handle, indent=2, sort_keys=True)
        os.replace(tmp, path)


class ModelClient:
    """Calls an OpenAI-compatible chat completions endpoint, with caching.

    ``http_client`` may be injected to supply a preconfigured or mocked
    :class:`httpx.Client`; tests pass one backed by ``httpx.MockTransport``.
    """

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        *,
        api_key: str | None = None,
        cache_dir: Path | str | None = DEFAULT_CACHE_DIR,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        max_retries: int = DEFAULT_MAX_RETRIES,
        http_client: httpx.Client | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.cache = ResponseCache(cache_dir) if cache_dir is not None else None
        self._owns_client = http_client is None
        self._http = http_client or httpx.Client(timeout=timeout)
        self._total_usage = TokenUsage()
        self._api_calls = 0
        self._cache_hits = 0

    def __enter__(self) -> ModelClient:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        """Close the HTTP client, but only if this instance created it."""
        if self._owns_client:
            self._http.close()

    @property
    def total_usage(self) -> TokenUsage:
        """Tokens consumed by real API calls in this session; cache hits cost nothing."""
        return self._total_usage

    @property
    def api_calls(self) -> int:
        return self._api_calls

    @property
    def cache_hits(self) -> int:
        return self._cache_hits

    @classmethod
    def from_config(cls, config: Any, **kwargs: Any) -> ModelClient:
        """Build a client from a :class:`~verity_core.config.VerityConfig`."""
        return cls(
            base_url=config.model_base_url,
            cache_dir=Path(config.cache_dir) / "models",
            **kwargs,
        )

    def complete(
        self,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float = 0.0,
        max_tokens: int = 4096,
        *,
        use_cache: bool = True,
        **kwargs: Any,
    ) -> ModelResponse:
        """Return a completion, serving it from cache when an identical call was made.

        Cache hits are not added to :attr:`total_usage`: that counter reports what the
        session actually spent, which is what a scorecard needs to cite.
        """
        cache_key: str | None = None
        if self.cache is not None and use_cache:
            cache_key = self.cache.make_key(
                base_url=self.base_url,
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                extra=kwargs,
            )
            hit = self.cache.get(cache_key)
            if hit is not None:
                self._cache_hits += 1
                hit.cached = True
                return hit

        request_body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs,
        }
        response = self._post_chat_completion(request_body)
        self._api_calls += 1
        self._total_usage = self._total_usage + response.usage

        if self.cache is not None and cache_key is not None:
            self.cache.set(cache_key, response, request=request_body)
        return response

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _post_chat_completion(self, body: dict[str, Any]) -> ModelResponse:
        """POST the request, retrying transient failures with exponential backoff."""
        url = f"{self.base_url}/chat/completions"
        last_error: Exception | None = None

        for attempt in range(self.max_retries + 1):
            try:
                http_response = self._http.post(url, json=body, headers=self._headers())
            except httpx.HTTPError as exc:
                last_error = ModelError(f"request to {url} failed: {exc}")
            else:
                if http_response.status_code < 400:
                    return self._parse(http_response, requested_model=body["model"])
                last_error = ModelError(
                    f"{url} returned HTTP {http_response.status_code}: {http_response.text[:500]}"
                )
                if http_response.status_code not in RETRYABLE_STATUS_CODES:
                    break

            if attempt < self.max_retries:
                time.sleep(2**attempt)

        raise last_error if last_error else ModelError(f"request to {url} failed")

    @staticmethod
    def _parse(http_response: httpx.Response, *, requested_model: str) -> ModelResponse:
        try:
            payload = http_response.json()
        except ValueError as exc:
            raise ModelError(f"response body was not JSON: {http_response.text[:500]}") from exc

        choices = payload.get("choices")
        if not choices:
            raise ModelError(f"response contained no choices: {payload}")

        message = choices[0].get("message") or {}
        return ModelResponse(
            # A refusal or a length cutoff can leave content null; an empty string is
            # a truthful representation and keeps callers off a None path.
            content=message.get("content") or "",
            model=payload.get("model") or requested_model,
            usage=TokenUsage.from_dict(payload.get("usage")),
            cached=False,
            finish_reason=choices[0].get("finish_reason") or "",
        )
