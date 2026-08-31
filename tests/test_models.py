"""Tests for the model client, with the HTTP layer mocked out.

The cache is the part worth testing hardest: an audit that silently re-samples the
model produces scorecards that cannot be compared with earlier ones.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from verity_core.config import VerityConfig
from verity_core.models import (
    ModelClient,
    ModelError,
    ModelResponse,
    ResponseCache,
    TokenUsage,
)

MESSAGES: list[dict[str, Any]] = [{"role": "user", "content": "What is 2 + 2?"}]


def completion_payload(
    content: str = "4",
    *,
    model: str = "test-model",
    prompt_tokens: int = 11,
    completion_tokens: int = 3,
    total_tokens: int | None = 14,
) -> dict[str, Any]:
    usage: dict[str, Any] = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
    }
    if total_tokens is not None:
        usage["total_tokens"] = total_tokens
    return {
        "model": model,
        "choices": [
            {"message": {"role": "assistant", "content": content}, "finish_reason": "stop"}
        ],
        "usage": usage,
    }


class Recorder:
    """A mock transport that records requests and replays queued responses."""

    def __init__(self, *responses: httpx.Response) -> None:
        self.queue = list(responses) or [httpx.Response(200, json=completion_payload())]
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self.queue.pop(0) if len(self.queue) > 1 else self.queue[0]

    @property
    def call_count(self) -> int:
        return len(self.requests)

    def client(self) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(self))


@pytest.fixture
def recorder() -> Recorder:
    return Recorder()


@pytest.fixture
def client(recorder: Recorder, tmp_path: Path) -> ModelClient:
    return ModelClient(
        "http://localhost:8000/v1",
        cache_dir=tmp_path / "cache",
        http_client=recorder.client(),
    )


class TestTokenUsage:
    def test_adds_componentwise(self) -> None:
        total = TokenUsage(1, 2, 3) + TokenUsage(10, 20, 30)
        assert total == TokenUsage(11, 22, 33)

    def test_derives_a_missing_total(self) -> None:
        usage = TokenUsage.from_dict({"prompt_tokens": 7, "completion_tokens": 5})
        assert usage.total_tokens == 12

    def test_keeps_a_reported_total(self) -> None:
        usage = TokenUsage.from_dict(
            {"prompt_tokens": 7, "completion_tokens": 5, "total_tokens": 99}
        )
        assert usage.total_tokens == 99

    def test_absent_usage_is_all_zeros(self) -> None:
        assert TokenUsage.from_dict(None) == TokenUsage(0, 0, 0)

    def test_round_trips_through_a_dict(self) -> None:
        usage = TokenUsage(3, 4, 7)
        assert TokenUsage.from_dict(usage.to_dict()) == usage


class TestCompletion:
    def test_returns_content_model_and_usage(self, client: ModelClient) -> None:
        response = client.complete("test-model", MESSAGES)
        assert response.content == "4"
        assert response.model == "test-model"
        assert response.usage == TokenUsage(11, 3, 14)
        assert response.cached is False
        assert response.finish_reason == "stop"

    def test_posts_to_the_chat_completions_endpoint(
        self, client: ModelClient, recorder: Recorder
    ) -> None:
        client.complete("test-model", MESSAGES)
        request = recorder.requests[0]
        assert str(request.url) == "http://localhost:8000/v1/chat/completions"
        assert request.method == "POST"

    def test_sends_the_documented_request_body(
        self, client: ModelClient, recorder: Recorder
    ) -> None:
        client.complete("test-model", MESSAGES, temperature=0.3, max_tokens=256, top_p=0.9)
        body = json.loads(recorder.requests[0].content)
        assert body == {
            "model": "test-model",
            "messages": MESSAGES,
            "temperature": 0.3,
            "max_tokens": 256,
            "top_p": 0.9,
        }

    def test_defaults_are_greedy_decoding(self, client: ModelClient, recorder: Recorder) -> None:
        client.complete("test-model", MESSAGES)
        body = json.loads(recorder.requests[0].content)
        assert body["temperature"] == 0.0
        assert body["max_tokens"] == 4096

    def test_omits_the_auth_header_without_an_api_key(
        self, client: ModelClient, recorder: Recorder
    ) -> None:
        client.complete("test-model", MESSAGES)
        assert "authorization" not in recorder.requests[0].headers

    def test_sends_a_bearer_token_when_given_one(self, recorder: Recorder) -> None:
        client = ModelClient(cache_dir=None, api_key="secret", http_client=recorder.client())
        client.complete("test-model", MESSAGES)
        assert recorder.requests[0].headers["authorization"] == "Bearer secret"

    def test_trailing_slash_in_base_url_does_not_double_up(self, recorder: Recorder) -> None:
        client = ModelClient("http://host:8000/v1/", cache_dir=None, http_client=recorder.client())
        client.complete("m", MESSAGES)
        assert str(recorder.requests[0].url) == "http://host:8000/v1/chat/completions"

    def test_null_content_becomes_an_empty_string(self) -> None:
        payload = completion_payload()
        payload["choices"][0]["message"]["content"] = None
        recorder = Recorder(httpx.Response(200, json=payload))
        client = ModelClient(cache_dir=None, http_client=recorder.client())
        assert client.complete("m", MESSAGES).content == ""

    def test_falls_back_to_the_requested_model_name(self) -> None:
        payload = completion_payload()
        del payload["model"]
        recorder = Recorder(httpx.Response(200, json=payload))
        client = ModelClient(cache_dir=None, http_client=recorder.client())
        assert client.complete("requested-model", MESSAGES).model == "requested-model"


class TestCaching:
    def test_first_call_hits_the_api_and_second_is_served_from_cache(
        self, client: ModelClient, recorder: Recorder
    ) -> None:
        first = client.complete("test-model", MESSAGES)
        second = client.complete("test-model", MESSAGES)
        assert recorder.call_count == 1
        assert first.cached is False
        assert second.cached is True
        assert second.content == first.content

    def test_counters_separate_api_calls_from_cache_hits(self, client: ModelClient) -> None:
        client.complete("test-model", MESSAGES)
        client.complete("test-model", MESSAGES)
        assert client.api_calls == 1
        assert client.cache_hits == 1

    def test_a_cache_hit_does_not_inflate_usage(self, client: ModelClient) -> None:
        client.complete("test-model", MESSAGES)
        client.complete("test-model", MESSAGES)
        assert client.total_usage == TokenUsage(11, 3, 14)

    def test_usage_accumulates_across_distinct_calls(self, client: ModelClient) -> None:
        client.complete("test-model", MESSAGES)
        client.complete("test-model", [{"role": "user", "content": "different"}])
        assert client.total_usage == TokenUsage(22, 6, 28)

    def test_the_cache_survives_a_new_client_on_the_same_directory(
        self, tmp_path: Path, recorder: Recorder
    ) -> None:
        shared = tmp_path / "cache"
        ModelClient(cache_dir=shared, http_client=recorder.client()).complete("m", MESSAGES)
        second = ModelClient(cache_dir=shared, http_client=recorder.client())
        assert second.complete("m", MESSAGES).cached is True
        assert recorder.call_count == 1

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"temperature": 0.7},
            {"max_tokens": 128},
            {"top_p": 0.5},
            {"messages": [{"role": "user", "content": "other"}]},
        ],
    )
    def test_changing_any_request_parameter_misses_the_cache(
        self, client: ModelClient, recorder: Recorder, kwargs: dict[str, Any]
    ) -> None:
        client.complete("test-model", MESSAGES)
        messages = kwargs.pop("messages", MESSAGES)
        client.complete("test-model", messages, **kwargs)
        assert recorder.call_count == 2

    def test_changing_the_model_misses_the_cache(
        self, client: ModelClient, recorder: Recorder
    ) -> None:
        client.complete("model-a", MESSAGES)
        client.complete("model-b", MESSAGES)
        assert recorder.call_count == 2

    def test_a_different_endpoint_does_not_reuse_cached_answers(
        self, tmp_path: Path, recorder: Recorder
    ) -> None:
        # Two servers can serve different weights under one model name, so the
        # endpoint has to be part of the key.
        shared = tmp_path / "cache"
        ModelClient("http://a:8000/v1", cache_dir=shared, http_client=recorder.client()).complete(
            "m", MESSAGES
        )
        other = ModelClient("http://b:8000/v1", cache_dir=shared, http_client=recorder.client())
        assert other.complete("m", MESSAGES).cached is False
        assert recorder.call_count == 2

    def test_use_cache_false_always_calls_the_api(
        self, client: ModelClient, recorder: Recorder
    ) -> None:
        client.complete("test-model", MESSAGES)
        response = client.complete("test-model", MESSAGES, use_cache=False)
        assert recorder.call_count == 2
        assert response.cached is False

    def test_cache_dir_none_disables_caching(self, recorder: Recorder) -> None:
        client = ModelClient(cache_dir=None, http_client=recorder.client())
        client.complete("test-model", MESSAGES)
        client.complete("test-model", MESSAGES)
        assert recorder.call_count == 2
        assert client.cache is None


class TestResponseCache:
    def test_a_miss_returns_none(self, tmp_path: Path) -> None:
        assert ResponseCache(tmp_path).get("deadbeef") is None

    def test_stores_and_retrieves_a_response(self, tmp_path: Path) -> None:
        cache = ResponseCache(tmp_path)
        response = ModelResponse(content="hi", model="m", usage=TokenUsage(1, 2, 3))
        cache.set("abc123", response)
        assert cache.get("abc123") == response

    def test_entries_are_sharded_by_key_prefix(self, tmp_path: Path) -> None:
        cache = ResponseCache(tmp_path)
        cache.set("abcdef", ModelResponse(content="x", model="m"))
        assert (tmp_path / "ab" / "abcdef.json").is_file()

    def test_an_unreadable_entry_degrades_to_a_miss(self, tmp_path: Path) -> None:
        cache = ResponseCache(tmp_path)
        cache.set("abcdef", ModelResponse(content="x", model="m"))
        cache.path_for("abcdef").write_text("{ truncated", encoding="utf-8")
        assert cache.get("abcdef") is None

    def test_a_corrupt_entry_costs_one_extra_call_not_a_crash(
        self, client: ModelClient, recorder: Recorder
    ) -> None:
        client.complete("test-model", MESSAGES)
        assert client.cache is not None
        key = client.cache.make_key(
            base_url=client.base_url,
            model="test-model",
            messages=MESSAGES,
            temperature=0.0,
            max_tokens=4096,
            extra={},
        )
        client.cache.path_for(key).write_text("not json", encoding="utf-8")
        assert client.complete("test-model", MESSAGES).content == "4"
        assert recorder.call_count == 2

    def test_the_key_is_stable_across_calls(self, tmp_path: Path) -> None:
        args: dict[str, Any] = {
            "base_url": "http://h/v1",
            "model": "m",
            "messages": MESSAGES,
            "temperature": 0.0,
            "max_tokens": 10,
        }
        assert ResponseCache.make_key(**args) == ResponseCache.make_key(**args)

    def test_the_key_ignores_mapping_order(self) -> None:
        first = ResponseCache.make_key(
            base_url="u",
            model="m",
            messages=[],
            temperature=0.0,
            max_tokens=1,
            extra={"a": 1, "b": 2},
        )
        second = ResponseCache.make_key(
            base_url="u",
            model="m",
            messages=[],
            temperature=0.0,
            max_tokens=1,
            extra={"b": 2, "a": 1},
        )
        assert first == second


class TestErrorHandling:
    def test_a_client_error_raises_with_the_status_and_body(self) -> None:
        recorder = Recorder(httpx.Response(400, text="bad request"))
        client = ModelClient(cache_dir=None, max_retries=0, http_client=recorder.client())
        with pytest.raises(ModelError, match="HTTP 400: bad request"):
            client.complete("m", MESSAGES)

    def test_a_client_error_is_not_retried(self) -> None:
        recorder = Recorder(httpx.Response(400, text="nope"))
        client = ModelClient(cache_dir=None, max_retries=3, http_client=recorder.client())
        with pytest.raises(ModelError):
            client.complete("m", MESSAGES)
        assert recorder.call_count == 1

    def test_a_transient_error_is_retried_then_succeeds(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("verity_core.models.time.sleep", lambda _: None)
        recorder = Recorder(
            httpx.Response(503, text="unavailable"),
            httpx.Response(200, json=completion_payload()),
        )
        client = ModelClient(cache_dir=None, max_retries=2, http_client=recorder.client())
        assert client.complete("m", MESSAGES).content == "4"
        assert recorder.call_count == 2

    def test_retries_are_bounded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("verity_core.models.time.sleep", lambda _: None)
        recorder = Recorder(httpx.Response(429, text="slow down"))
        client = ModelClient(cache_dir=None, max_retries=2, http_client=recorder.client())
        with pytest.raises(ModelError):
            client.complete("m", MESSAGES)
        assert recorder.call_count == 3

    def test_a_transport_failure_becomes_a_model_error(self) -> None:
        def explode(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("connection refused")

        client = ModelClient(
            cache_dir=None,
            max_retries=0,
            http_client=httpx.Client(transport=httpx.MockTransport(explode)),
        )
        with pytest.raises(ModelError, match="connection refused"):
            client.complete("m", MESSAGES)

    def test_an_empty_choices_list_is_an_error(self) -> None:
        recorder = Recorder(httpx.Response(200, json={"choices": []}))
        client = ModelClient(cache_dir=None, max_retries=0, http_client=recorder.client())
        with pytest.raises(ModelError, match="no choices"):
            client.complete("m", MESSAGES)

    def test_a_non_json_body_is_an_error(self) -> None:
        recorder = Recorder(httpx.Response(200, text="<html>oops</html>"))
        client = ModelClient(cache_dir=None, max_retries=0, http_client=recorder.client())
        with pytest.raises(ModelError, match="not JSON"):
            client.complete("m", MESSAGES)

    def test_a_failed_call_is_not_cached(self, tmp_path: Path) -> None:
        recorder = Recorder(httpx.Response(500, text="boom"))
        failing = ModelClient(
            cache_dir=tmp_path / "cache", max_retries=0, http_client=recorder.client()
        )
        with pytest.raises(ModelError):
            failing.complete("m", MESSAGES)
        assert not list((tmp_path / "cache").rglob("*.json"))


class TestLifecycle:
    def test_from_config_derives_the_endpoint_and_cache_directory(self, tmp_path: Path) -> None:
        config = VerityConfig(model_base_url="http://gpu:8000/v1", cache_dir=tmp_path)
        client = ModelClient.from_config(config)
        assert client.base_url == "http://gpu:8000/v1"
        assert client.cache is not None
        assert client.cache.directory == tmp_path / "models"

    def test_does_not_close_an_injected_http_client(self, recorder: Recorder) -> None:
        http = recorder.client()
        client = ModelClient(cache_dir=None, http_client=http)
        client.close()
        assert http.is_closed is False

    def test_closes_a_client_it_created(self) -> None:
        client = ModelClient(cache_dir=None)
        client.close()
        assert client._http.is_closed is True

    def test_works_as_a_context_manager(self, recorder: Recorder) -> None:
        with ModelClient(cache_dir=None, http_client=recorder.client()) as client:
            assert client.complete("m", MESSAGES).content == "4"
