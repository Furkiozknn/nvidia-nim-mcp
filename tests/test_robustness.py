"""Failure modes that used to escape the fallback chain as an unhandled
exception instead of moving on to the next tier, plus the guards around
what describe_image uploads and how retired (HTTP 410) models surface."""
import base64
import json
from unittest.mock import AsyncMock

import pytest

import nvidia_image
from conftest import FakeResponse


MESSAGES = [{"role": "user", "content": "hello"}]
CHOICE = {"choices": [{"message": {"content": "hi there"}}]}


class NotJsonResponse(FakeResponse):
    """A 200 whose body is not JSON - e.g. an HTML page from a proxy."""

    def json(self):
        raise json.JSONDecodeError("Expecting value", "<html>", 0)


def _connect_error(*a, **kw):
    raise nvidia_image.httpx2.ConnectError("connection refused")


# --- _chat_with_fallback: transport errors and retired models -------------

@pytest.mark.asyncio
async def test_connection_error_on_first_model_falls_back_to_second(nvidia_key):
    client = AsyncMock()
    client.post = AsyncMock(side_effect=[nvidia_image.httpx2.ConnectError("refused"), FakeResponse(200, CHOICE)])

    result = await nvidia_image._chat_with_fallback(client, ["model-a", "model-b"], MESSAGES)

    assert result == ("hi there", "model-b")


@pytest.mark.asyncio
async def test_retired_model_http_410_falls_back_to_second(nvidia_key):
    client = AsyncMock()
    client.post = AsyncMock(
        side_effect=[FakeResponse(410, text="model has reached its end of life"), FakeResponse(200, CHOICE)]
    )

    result = await nvidia_image._chat_with_fallback(client, ["retired-model", "model-b"], MESSAGES)

    assert result == ("hi there", "model-b")


@pytest.mark.asyncio
async def test_health_probe_labels_http_410_as_retired(nvidia_key):
    client = AsyncMock()
    client.post = AsyncMock(return_value=FakeResponse(410, text="end of life"))

    ok, detail = await nvidia_image._probe_nvidia_chat_model(client, "retired-model")

    assert ok is False
    assert "HTTP 410" in detail
    assert "retired" in detail


@pytest.mark.asyncio
async def test_describe_image_survives_a_connection_error_and_uses_the_vision_fallback(
    nvidia_key, fake_async_client, tmp_path, monkeypatch
):
    image = tmp_path / "shot.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\nrest")
    monkeypatch.setenv("GROQ_API_KEY", "groq-key")
    fake_async_client(post_side_effect=_connect_error)
    fallback = AsyncMock(return_value=("a screenshot", "groq/some-vision-model"))
    monkeypatch.setattr(nvidia_image, "_chat_via_provider_chain", fallback)

    result = await nvidia_image.describe_image(str(image))

    assert "a screenshot" in result
    fallback.assert_awaited_once()


@pytest.mark.asyncio
async def test_content_safety_survives_a_connection_error_and_uses_the_fallback(
    nvidia_key, fake_async_client, monkeypatch
):
    monkeypatch.setenv("GROQ_API_KEY", "groq-key")
    fake_async_client(post_side_effect=_connect_error)
    monkeypatch.setattr(
        nvidia_image, "_chat_via_provider_chain", AsyncMock(return_value=("SAFE - fine", "groq/x"))
    )

    result = await nvidia_image.check_content_safety("hello")

    assert "SAFE" in result
    assert "best-effort fallback" in result


# --- generate_image: every NVIDIA failure mode reaches Pollinations -------

@pytest.mark.asyncio
@pytest.mark.parametrize(
    "post_side_effect",
    [
        _connect_error,
        lambda *a, **kw: NotJsonResponse(200, text="<html>gateway</html>"),
        lambda *a, **kw: FakeResponse(200, json_data={"artifacts": [{"finishReason": "CONTENT_FILTERED"}]}),
        lambda *a, **kw: FakeResponse(200, json_data={"artifacts": [{"base64": "not base64!!"}]}),
    ],
    ids=["connect-error", "non-json-200", "artifact-without-base64", "invalid-base64"],
)
async def test_generate_image_falls_back_to_pollinations_on_any_nvidia_failure(
    post_side_effect, nvidia_key, fake_async_client, tmp_path, monkeypatch
):
    monkeypatch.setattr(nvidia_image, "OUTPUT_DIR", tmp_path)
    client = fake_async_client(post_side_effect=post_side_effect)

    result = await nvidia_image.generate_image(prompt="a cat")

    assert "model: pollinations" in result
    assert client.post.await_count == len(nvidia_image.IMAGE_MODELS)
    assert list(tmp_path.glob("pollinations_*.jpg"))


@pytest.mark.asyncio
async def test_generate_image_connection_error_detail_is_redacted(
    nvidia_key, fake_async_client, tmp_path, monkeypatch
):
    monkeypatch.setattr(nvidia_image, "OUTPUT_DIR", tmp_path)

    def _leaky(*a, **kw):
        raise nvidia_image.httpx2.ConnectError(f"proxy said: Bearer {nvidia_key}")

    fake_async_client(post_side_effect=_leaky, get_side_effect=RuntimeError("pollinations down"))

    result = await nvidia_image.generate_image(prompt="a cat")

    assert "All image models failed" in result
    assert nvidia_key not in result
    assert "***" in result


# --- create_embedding: malformed 200 reaches the local tier ---------------

@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response",
    [NotJsonResponse(200, text="<html>"), FakeResponse(200, json_data={"data": []})],
    ids=["non-json-200", "empty-data"],
)
async def test_create_embedding_malformed_success_falls_back_to_local(
    response, nvidia_key, fake_async_client, tmp_path, monkeypatch
):
    monkeypatch.setattr(nvidia_image, "OUTPUT_DIR", tmp_path)
    fake_async_client(post_side_effect=lambda *a, **kw: response)
    monkeypatch.setattr(nvidia_image, "_local_embedding", lambda text: [0.1, 0.2])

    result = await nvidia_image.create_embedding(text="hello")

    assert "2 dimensions" in result
    assert "local:" in result


# --- describe_image: the bytes must actually be an allowed image ----------

@pytest.mark.asyncio
async def test_describe_image_refuses_a_non_image_behind_an_image_extension(
    nvidia_key, fake_async_client, tmp_path, monkeypatch
):
    disguised = tmp_path / "report.png"
    disguised.write_bytes(b"%PDF-1.7 a renamed document, not an image")
    client = fake_async_client(post_side_effect=AssertionError("no bytes may leave the machine"))
    monkeypatch.setattr(nvidia_image.litellm, "acompletion", AsyncMock(side_effect=AssertionError("no upload")))

    result = await nvidia_image.describe_image(str(disguised))

    assert "not a JPEG, PNG or WebP image" in result
    assert client.post.await_count == 0


@pytest.mark.asyncio
async def test_describe_image_uses_the_sniffed_format_for_the_data_url(
    nvidia_key, fake_async_client, tmp_path
):
    # A JPEG saved with a .png extension is common; the data URL must say
    # what the bytes are, not what the filename claims.
    mislabelled = tmp_path / "photo.png"
    payload = b"\xff\xd8\xff\xe0jpeg-body"
    mislabelled.write_bytes(payload)
    client = fake_async_client(post_side_effect=lambda *a, **kw: FakeResponse(200, CHOICE))

    await nvidia_image.describe_image(str(mislabelled))

    sent = client.post.await_args.kwargs["json"]["messages"][0]["content"][1]["image_url"]["url"]
    assert sent == "data:image/jpeg;base64," + base64.b64encode(payload).decode()


def test_read_image_is_bounded_even_if_the_file_grows_after_the_size_check(tmp_path, monkeypatch):
    monkeypatch.setattr(nvidia_image, "DESCRIBE_IMAGE_MAX_BYTES", 16)
    grown = tmp_path / "grown.png"
    grown.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

    assert nvidia_image._read_image_bytes(grown) is None


# --- tool schemas ----------------------------------------------------------

@pytest.mark.asyncio
async def test_generate_image_schema_bounds_width_and_height():
    tools = {t.name: t for t in await nvidia_image.mcp.list_tools()}
    props = tools["generate_image"].input_schema["properties"]

    for dim in ("width", "height"):
        assert props[dim]["minimum"] == nvidia_image.IMAGE_MIN_SIDE
        assert props[dim]["maximum"] == nvidia_image.IMAGE_MAX_SIDE
    assert props["seed"]["minimum"] == 0
    for tool in tools.values():
        for name, prop in tool.input_schema.get("properties", {}).items():
            assert prop.get("description"), f"{tool.name}.{name} has no description"
