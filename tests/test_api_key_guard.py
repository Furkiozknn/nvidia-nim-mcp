"""What each tool does when NVIDIA_API_KEY isn't set.

The old contract - every tool short-circuiting on `if not API_KEY` - ran
*before* every documented fallback, so a caller holding only a GROQ_API_KEY,
or no key at all, got nothing: not the keyless Pollinations image tier, not
the free-tier chat chain, not the local sentence-transformers embedding
path. The README's whole "automatic fallback" promise was unreachable
without an NVIDIA key.

The contract asserted here instead:

* a tool with a keyless or cross-provider fallback PROCEEDS to it;
* a tool with no such tier available fails with a message naming what the
  user actually needs (NVIDIA's key *or* any free-tier key), not one that
  names NVIDIA's alone.
"""
import pytest

import nvidia_image

# Tools that must never be blocked by a missing NVIDIA key, because they own
# a tier that needs no key at all.
KEYLESS_FALLBACK_TOOLS = {
    # Pollinations.ai: free, no auth, no quota on our side.
    "generate_image": lambda: nvidia_image.generate_image(prompt="a cat"),
    # sentence-transformers, running locally.
    "create_embedding": lambda: nvidia_image.create_embedding(text="hello"),
}

# Tools with no keyless tier, but which still work off any one of several
# free-tier provider keys - so the NVIDIA key is not what they require.
CROSS_PROVIDER_TOOLS = {
    "translate_text": (
        lambda: nvidia_image.translate_text(text="hello", target_language="Turkish"),
        nvidia_image.EXTRA_PROVIDERS,
    ),
    "ask_llm": (lambda: nvidia_image.ask_llm(question="what is 2+2?"), nvidia_image.EXTRA_PROVIDERS),
    "check_content_safety": (
        lambda: nvidia_image.check_content_safety(text="hello"),
        nvidia_image.EXTRA_PROVIDERS,
    ),
    "describe_image": (
        lambda: nvidia_image.describe_image(image_path="/nonexistent/file.jpg"),
        nvidia_image.VISION_PROVIDERS,
    ),
}

ALL_PROVIDER_ENVS = sorted(
    {p["env"] for p in (*nvidia_image.EXTRA_PROVIDERS, *nvidia_image.VISION_PROVIDERS)}
)

OLD_GUARD_MESSAGE = "NVIDIA_API_KEY not set in .env - can't call the API."


@pytest.fixture
def no_provider_keys(monkeypatch):
    """No NVIDIA key and no free-tier key either - nothing is configured."""
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    for env in ALL_PROVIDER_ENVS:
        monkeypatch.delenv(env, raising=False)


# --- tools with a keyless tier: they must actually reach it ---------------

@pytest.mark.asyncio
async def test_generate_image_falls_through_to_pollinations_without_a_nvidia_key(
    no_provider_keys, fake_async_client, tmp_path, monkeypatch
):
    monkeypatch.setattr(nvidia_image, "OUTPUT_DIR", tmp_path)
    client = fake_async_client()  # `get` defaults to a valid tiny JPEG

    result = await nvidia_image.generate_image(prompt="a cat")

    assert "model: pollinations" in result
    assert client.get.await_count == 1
    # And no NVIDIA request was attempted with a missing key.
    assert client.post.await_count == 0
    assert len(list(tmp_path.glob("pollinations_*.jpg"))) == 1


@pytest.mark.asyncio
async def test_create_embedding_falls_through_to_the_local_model_without_a_nvidia_key(
    no_provider_keys, fake_async_client, tmp_path, monkeypatch
):
    monkeypatch.setattr(nvidia_image, "OUTPUT_DIR", tmp_path)
    client = fake_async_client()
    monkeypatch.setattr(nvidia_image, "_local_embedding", lambda text: [0.1, 0.2])

    result = await nvidia_image.create_embedding(text="hello")

    assert "2 dimensions" in result
    assert f"model: local:{nvidia_image._LOCAL_EMBED_MODEL_NAME}" in result
    assert client.post.await_count == 0


@pytest.mark.asyncio
async def test_create_embedding_without_any_key_and_without_the_extra_says_so(
    no_provider_keys, tmp_path, monkeypatch
):
    monkeypatch.setattr(nvidia_image, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(nvidia_image, "_local_embedding", lambda text: None)

    result = await nvidia_image.create_embedding(text="hello")

    # It names the actual remedy - install the local extra - and reports the
    # skipped NVIDIA tier as context, not as the headline.
    assert "no local fallback available" in result
    assert "local-embeddings" in result


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_name", sorted(KEYLESS_FALLBACK_TOOLS))
async def test_keyless_tools_never_return_the_old_blanket_guard(
    tool_name, no_provider_keys, fake_async_client, tmp_path, monkeypatch
):
    monkeypatch.setattr(nvidia_image, "OUTPUT_DIR", tmp_path)
    fake_async_client()
    monkeypatch.setattr(nvidia_image, "_local_embedding", lambda text: [0.1])

    result = await KEYLESS_FALLBACK_TOOLS[tool_name]()

    assert result != OLD_GUARD_MESSAGE


# --- tools with only cross-provider fallbacks ----------------------------

@pytest.mark.asyncio
@pytest.mark.parametrize("tool_name", sorted(CROSS_PROVIDER_TOOLS))
async def test_cross_provider_tool_names_every_usable_key_when_nothing_is_configured(
    tool_name, no_provider_keys
):
    call, providers = CROSS_PROVIDER_TOOLS[tool_name]

    result = await call()

    assert result != OLD_GUARD_MESSAGE
    assert "no provider configured" in result
    # It must name NVIDIA's key AND every free-tier alternative - the point
    # of the fix is that NVIDIA's is not the only thing that would work.
    assert "NVIDIA_API_KEY" in result
    for provider in providers:
        assert provider["env"] in result


@pytest.mark.asyncio
async def test_translate_text_uses_the_free_tier_when_only_groq_is_configured(
    no_provider_keys, monkeypatch
):
    """The exact case the old guard broke: a user with GROQ_API_KEY and no
    NVIDIA key."""
    monkeypatch.setenv("GROQ_API_KEY", "groq-key")

    captured = {}

    class FakeMessage:
        content = "merhaba"

    class FakeChoice:
        message = FakeMessage()

    class FakeCompletionResponse:
        choices = [FakeChoice()]
        model = "groq/openai/gpt-oss-120b"

    async def fake_acompletion(**kwargs):
        captured.update(kwargs)
        return FakeCompletionResponse()

    monkeypatch.setattr(nvidia_image.litellm, "acompletion", fake_acompletion)

    result = await nvidia_image.translate_text(text="hello", target_language="Turkish")

    assert "merhaba" in result
    # Groq is the primary, not a fallback behind a dead NVIDIA entry.
    assert captured["model"] == "groq/openai/gpt-oss-120b"
    assert not captured.get("fallbacks")


@pytest.mark.asyncio
async def test_describe_image_reaches_its_vision_provider_without_a_nvidia_key(
    no_provider_keys, fake_async_client, tmp_path, monkeypatch
):
    image = tmp_path / "test.jpg"
    image.write_bytes(b"\xff\xd8\xff\xe0fake-jpeg-bytes")
    monkeypatch.setenv("GROQ_API_KEY", "groq-key")
    client = fake_async_client()

    class FakeMessage:
        content = "a cat, via the free tier"

    class FakeChoice:
        message = FakeMessage()

    class FakeCompletionResponse:
        choices = [FakeChoice()]
        model = "groq/qwen/qwen3.6-27b"

    async def fake_acompletion(**kwargs):
        return FakeCompletionResponse()

    monkeypatch.setattr(nvidia_image.litellm, "acompletion", fake_acompletion)

    result = await nvidia_image.describe_image(image_path=str(image))

    assert "a cat, via the free tier" in result
    assert client.post.await_count == 0  # no NVIDIA request attempted


# --- the diagnostic tool ---------------------------------------------------

@pytest.mark.asyncio
async def test_check_provider_health_still_runs_and_reports_nvidia_as_unconfigured(
    no_provider_keys, fake_async_client
):
    """A health check that refuses to run without a key is useless to exactly
    the caller who needs it. It must report, not refuse."""
    client = fake_async_client()

    report = await nvidia_image.check_provider_health()

    assert report != OLD_GUARD_MESSAGE
    assert "NVIDIA NIM provider health check:" in report
    assert "not configured (NVIDIA_API_KEY not set)" in report
    # The keyless Pollinations tier is still genuinely probed.
    assert "pollinations (fallback)" in report
    assert client.get.await_count == 1
    # No NVIDIA model was probed with a missing key.
    assert client.post.await_count == 0


# --- an emptied-out .env value is the same as an unset one ----------------

@pytest.mark.asyncio
@pytest.mark.parametrize("tool_name", sorted(CROSS_PROVIDER_TOOLS))
async def test_empty_string_key_is_treated_as_unset(tool_name, no_provider_keys, monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "")
    call, _providers = CROSS_PROVIDER_TOOLS[tool_name]

    result = await call()

    assert "no provider configured" in result


def test_every_tool_is_covered_exactly_once():
    """check_provider_health has its own test above; the other six are
    partitioned between the two tables."""
    assert set(KEYLESS_FALLBACK_TOOLS) | set(CROSS_PROVIDER_TOOLS) == {
        "generate_image",
        "translate_text",
        "ask_llm",
        "describe_image",
        "check_content_safety",
        "create_embedding",
    }
    assert not set(KEYLESS_FALLBACK_TOOLS) & set(CROSS_PROVIDER_TOOLS)
