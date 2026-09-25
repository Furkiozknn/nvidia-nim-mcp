"""Regressions for the provider chain and the Pollinations URL.

The chain bug these pin: the chain used to be handed to litellm as
`acompletion(**primary, fallbacks=rest)`. litellm copies the primary call's
kwargs into every fallback, so with NVIDIA first, the Groq/Mistral entries
inherited NVIDIA's `api_base`. Their keys were sent to NVIDIA's endpoint and
the cross-provider fallback never answered. Every test here runs offline.
"""
import logging
from unittest.mock import AsyncMock

import litellm
import pytest

import nvidia_image

MESSAGES = [{"role": "user", "content": "hi"}]
NVIDIA_BASE = nvidia_image.CHAT_URL.rsplit("/chat/completions", 1)[0]


def _only_keys(monkeypatch, **keys):
    for env in ("NVIDIA_API_KEY", "GROQ_API_KEY", "MISTRAL_API_KEY", "GEMINI_API_KEY", "CEREBRAS_API_KEY"):
        monkeypatch.delenv(env, raising=False)
    for env, value in keys.items():
        monkeypatch.setenv(env, value)


class _Reply:
    def __init__(self, content, model):
        message = type("M", (), {"content": content})()
        self.choices = [type("C", (), {"message": message})()]
        self.model = model


def _spy(monkeypatch, answer):
    """Replace litellm.acompletion with a recorder that never touches the
    network. A call that carries `fallbacks=` is handed to litellm's real
    fallback runner, which calls litellm.acompletion (this spy) once per
    entry with the kwargs litellm itself merged. That keeps litellm's own
    merge behaviour inside the test, which is where the bug lived."""
    real = litellm.acompletion
    calls = []

    async def spy(**kwargs):
        if kwargs.get("fallbacks"):
            return await real(**kwargs)
        calls.append(kwargs)
        return answer(kwargs)

    monkeypatch.setattr(nvidia_image.litellm, "acompletion", spy)
    return calls


async def test_a_groq_key_is_never_sent_to_nvidias_endpoint(monkeypatch):
    _only_keys(monkeypatch, NVIDIA_API_KEY="nv-key", GROQ_API_KEY="groq-key")

    def answer(kwargs):
        if kwargs["model"].startswith("openai/"):
            raise RuntimeError("NVIDIA model down")
        return _Reply("from groq", kwargs["model"])

    calls = _spy(monkeypatch, answer)

    result = await nvidia_image._multi_provider_chat(["nvidia/model-a"], MESSAGES)

    assert result == ("from groq", "groq/openai/gpt-oss-120b")
    groq_calls = [c for c in calls if c["model"] == "groq/openai/gpt-oss-120b"]
    assert len(groq_calls) == 1
    assert groq_calls[0]["api_key"] == "groq-key"
    assert "api_base" not in groq_calls[0]
    for call in calls:
        if call.get("api_base") == NVIDIA_BASE:
            assert call["api_key"] == "nv-key", f"{call['model']} sent a non-NVIDIA key to NVIDIA"


async def test_each_entry_is_one_attempt(monkeypatch):
    """The chain is the retry: litellm's default retried each entry twice
    more before the next model was tried."""
    _only_keys(monkeypatch, NVIDIA_API_KEY="nv-key", GROQ_API_KEY="groq-key")
    calls = _spy(monkeypatch, lambda kw: _Reply("ok", kw["model"]))

    await nvidia_image._multi_provider_chat(["nvidia/model-a"], MESSAGES)

    assert calls and all(c.get("max_retries") == 0 for c in calls)
    assert all("fallbacks" not in c for c in calls)


async def test_an_empty_reply_moves_on_to_the_next_provider(monkeypatch):
    """A reasoning model can use its whole max_tokens budget before writing
    an answer; the tool used to return the literal text "None"."""
    _only_keys(monkeypatch, GROQ_API_KEY="groq-key", MISTRAL_API_KEY="mistral-key")

    def answer(kwargs):
        if kwargs["model"].startswith("groq/"):
            return _Reply(None, kwargs["model"])
        return _Reply("real answer", kwargs["model"])

    _spy(monkeypatch, answer)

    result = await nvidia_image.ask_llm(question="hi")

    assert result.startswith("real answer")
    assert "mistral/mistral-small-latest" in result
    assert "None" not in result


async def test_a_failing_provider_does_not_log_any_key(monkeypatch, caplog):
    _only_keys(monkeypatch, NVIDIA_API_KEY="nv-secret-1", GROQ_API_KEY="groq-secret-2")

    async def boom(**kwargs):
        raise RuntimeError("401 Incorrect API key provided: nv-secret-1 / groq-secret-2")

    monkeypatch.setattr(nvidia_image.litellm, "acompletion", boom)

    with caplog.at_level(logging.WARNING, logger="nvidia_image"):
        result = await nvidia_image._multi_provider_chat(["nvidia/model-a"], MESSAGES)

    assert result is None
    assert caplog.records, "a failed provider should still be logged"
    assert "nv-secret-1" not in caplog.text
    assert "groq-secret-2" not in caplog.text


@pytest.mark.parametrize(
    "prompt",
    ["AC/DC poster", "../../models", "..", "a cat?width=9999#x"],
)
async def test_the_prompt_stays_one_path_segment_under_prompt(no_nvidia_key, fake_async_client, prompt, tmp_path, monkeypatch):
    import httpx2

    monkeypatch.setattr(nvidia_image, "OUTPUT_DIR", tmp_path)
    client = fake_async_client()

    result = await nvidia_image.generate_image(prompt=prompt)

    assert "Image saved to" in result
    url = str(httpx2.URL(client.get.call_args.args[0]))
    assert url.startswith(nvidia_image.POLLINATIONS_BASE + "/")
    segment = url[len(nvidia_image.POLLINATIONS_BASE) + 1 :]
    assert "/" not in segment and "?" not in segment and "#" not in segment
    assert segment not in ("", ".", "..")


async def test_describe_image_reports_an_unreadable_file_instead_of_crashing(tmp_path, monkeypatch):
    _only_keys(monkeypatch, GROQ_API_KEY="groq-key")
    image = tmp_path / "photo.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 16)

    def denied(path):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(nvidia_image, "_read_image_bytes", denied)
    monkeypatch.setattr(nvidia_image.litellm, "acompletion", AsyncMock(side_effect=AssertionError("no upload")))

    result = await nvidia_image.describe_image(image_path=str(image))

    assert result == f"Cannot read {image}: Permission denied"


async def test_generate_image_rejects_an_empty_prompt_through_its_schema():
    """An empty prompt is an empty path segment: Pollinations would be asked
    for /prompt/ itself."""
    tools = {t.name: t for t in await nvidia_image.mcp.list_tools()}
    assert tools["generate_image"].input_schema["properties"]["prompt"]["minLength"] == 1
