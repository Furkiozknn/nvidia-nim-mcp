"""Fallback logic: _chat_with_fallback (per-tool NVIDIA model list) and
_multi_provider_chat (NVIDIA -> cross-provider chain).

_chat_with_fallback talks to NVIDIA's endpoints directly, so every test of
it takes the `nvidia_key` fixture: without a key it short-circuits to None
by design, leaving the caller's cross-provider fallback to run."""
from unittest.mock import AsyncMock

import pytest

import nvidia_image
from conftest import FakeResponse


MESSAGES = [{"role": "user", "content": "hello"}]


# --- _chat_with_fallback -------------------------------------------------

@pytest.mark.asyncio
async def test_first_model_success_is_used_and_second_is_not_tried(nvidia_key):
    client = AsyncMock()
    client.post = AsyncMock(
        return_value=FakeResponse(200, {"choices": [{"message": {"content": "hi there"}}]})
    )

    result = await nvidia_image._chat_with_fallback(client, ["model-a", "model-b"], MESSAGES)

    assert result == ("hi there", "model-a")
    assert client.post.call_count == 1


@pytest.mark.asyncio
async def test_first_model_http_error_falls_back_to_second(nvidia_key):
    client = AsyncMock()
    client.post = AsyncMock(
        side_effect=[
            FakeResponse(429, text="rate limited"),
            FakeResponse(200, {"choices": [{"message": {"content": "second answered"}}]}),
        ]
    )

    result = await nvidia_image._chat_with_fallback(client, ["model-a", "model-b"], MESSAGES)

    assert result == ("second answered", "model-b")
    assert client.post.call_count == 2


@pytest.mark.asyncio
async def test_first_model_timeout_falls_back_to_second(nvidia_key):
    client = AsyncMock()
    client.post = AsyncMock(
        side_effect=[
            nvidia_image.httpx2.TimeoutException("timed out"),
            FakeResponse(200, {"choices": [{"message": {"content": "second answered"}}]}),
        ]
    )

    result = await nvidia_image._chat_with_fallback(client, ["model-a", "model-b"], MESSAGES)

    assert result == ("second answered", "model-b")


@pytest.mark.asyncio
async def test_all_models_failing_returns_none(nvidia_key):
    client = AsyncMock()
    client.post = AsyncMock(
        side_effect=[
            FakeResponse(500, text="server error"),
            nvidia_image.httpx2.TimeoutException("timed out"),
        ]
    )

    result = await nvidia_image._chat_with_fallback(client, ["model-a", "model-b"], MESSAGES)

    assert result is None


@pytest.mark.asyncio
async def test_malformed_success_response_is_treated_as_failure(nvidia_key):
    client = AsyncMock()
    client.post = AsyncMock(
        side_effect=[
            FakeResponse(200, {"unexpected": "shape"}),
            FakeResponse(200, {"choices": [{"message": {"content": "ok now"}}]}),
        ]
    )

    result = await nvidia_image._chat_with_fallback(client, ["model-a", "model-b"], MESSAGES)

    assert result == ("ok now", "model-b")


# --- _multi_provider_chat --------------------------------------------------

@pytest.mark.asyncio
async def test_multi_provider_chat_success(nvidia_key, monkeypatch):
    class FakeMessage:
        content = "the answer"

    class FakeChoice:
        message = FakeMessage()

    class FakeCompletionResponse:
        choices = [FakeChoice()]
        model = "nvidia/llama-3.3-nemotron-super-49b-v1.5"

    monkeypatch.setattr(
        nvidia_image.litellm, "acompletion", AsyncMock(return_value=FakeCompletionResponse())
    )

    result = await nvidia_image._multi_provider_chat(["nvidia/llama-3.3-nemotron-super-49b-v1.5"], MESSAGES)

    assert result == ("the answer", "nvidia/llama-3.3-nemotron-super-49b-v1.5")


@pytest.mark.asyncio
async def test_multi_provider_chat_returns_none_when_everything_fails(nvidia_key, monkeypatch):
    monkeypatch.setattr(
        nvidia_image.litellm,
        "acompletion",
        AsyncMock(side_effect=RuntimeError("all providers in chain failed")),
    )

    result = await nvidia_image._multi_provider_chat(["nvidia/llama-3.3-nemotron-super-49b-v1.5"], MESSAGES)

    assert result is None


@pytest.mark.asyncio
async def test_multi_provider_chat_passes_built_chain_as_fallbacks(nvidia_key, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setenv("GROQ_API_KEY", "groq-key")
    for env in ("MISTRAL_API_KEY", "GEMINI_API_KEY", "CEREBRAS_API_KEY"):
        monkeypatch.delenv(env, raising=False)

    captured = {}

    class FakeMessage:
        content = "ok"

    class FakeChoice:
        message = FakeMessage()

    class FakeCompletionResponse:
        choices = [FakeChoice()]
        model = "openai/nvidia/model-a"

    async def fake_acompletion(**kwargs):
        captured.update(kwargs)
        return FakeCompletionResponse()

    monkeypatch.setattr(nvidia_image.litellm, "acompletion", fake_acompletion)

    await nvidia_image._multi_provider_chat(["nvidia/model-a"], MESSAGES)

    assert captured["model"] == "openai/nvidia/model-a"
    assert len(captured["fallbacks"]) == 1
    assert captured["fallbacks"][0]["model"] == "groq/openai/gpt-oss-120b"


@pytest.mark.asyncio
async def test_chat_with_fallback_returns_none_without_a_nvidia_key_and_sends_nothing(no_nvidia_key):
    """No key means no NVIDIA tier to try - and, crucially, no request. The
    caller (describe_image, check_content_safety) then reaches its own
    cross-provider fallback instead of being blocked behind a 401."""
    client = AsyncMock()
    client.post = AsyncMock()

    result = await nvidia_image._chat_with_fallback(client, ["model-a", "model-b"], MESSAGES)

    assert result is None
    assert client.post.await_count == 0
