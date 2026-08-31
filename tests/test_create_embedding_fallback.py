"""create_embedding: NVIDIA's embedding endpoint first, then a fully local
sentence-transformers fallback if it fails and the optional dependency is
available (mocked here - the real package is an opt-in extra, not installed
in the base test environment)."""
import json

import pytest

import nvidia_image
from conftest import FakeResponse


@pytest.mark.asyncio
async def test_succeeds_on_nvidia_embedding_without_trying_local_fallback(
    nvidia_key, fake_async_client, tmp_path, monkeypatch
):
    monkeypatch.setattr(nvidia_image, "OUTPUT_DIR", tmp_path)
    fake_async_client(
        post_side_effect=lambda *a, **kw: FakeResponse(200, {"data": [{"embedding": [0.1, 0.2, 0.3]}]})
    )
    monkeypatch.setattr(
        nvidia_image, "_local_embedding", lambda text: (_ for _ in ()).throw(AssertionError("should not be called"))
    )

    result = await nvidia_image.create_embedding(text="hello world")

    assert "3 dimensions" in result
    assert f"model: {nvidia_image.EMBED_MODEL}" in result


@pytest.mark.asyncio
async def test_falls_back_to_local_embedding_when_nvidia_fails(nvidia_key, fake_async_client, tmp_path, monkeypatch):
    monkeypatch.setattr(nvidia_image, "OUTPUT_DIR", tmp_path)
    fake_async_client(post_side_effect=lambda *a, **kw: FakeResponse(500, text="server error"))
    monkeypatch.setattr(nvidia_image, "_local_embedding", lambda text: [0.5, 0.6, 0.7, 0.8])

    result = await nvidia_image.create_embedding(text="hello world")

    assert "4 dimensions" in result
    assert f"model: local:{nvidia_image._LOCAL_EMBED_MODEL_NAME}" in result
    saved = list(tmp_path.glob("embedding_*.json"))
    assert len(saved) == 1
    data = json.loads(saved[0].read_text())
    assert data["vector"] == [0.5, 0.6, 0.7, 0.8]
    assert data["model"] == f"local:{nvidia_image._LOCAL_EMBED_MODEL_NAME}"


@pytest.mark.asyncio
async def test_falls_back_to_local_embedding_on_nvidia_timeout(nvidia_key, fake_async_client, tmp_path, monkeypatch):
    monkeypatch.setattr(nvidia_image, "OUTPUT_DIR", tmp_path)
    fake_async_client(post_side_effect=nvidia_image.httpx2.TimeoutException("slow"))
    monkeypatch.setattr(nvidia_image, "_local_embedding", lambda text: [0.1])

    result = await nvidia_image.create_embedding(text="hello world")

    assert "1 dimensions" in result


@pytest.mark.asyncio
async def test_falls_back_to_local_embedding_on_connection_error_not_just_timeout(
    nvidia_key, fake_async_client, tmp_path, monkeypatch
):
    # A dropped connection, DNS failure, or TLS error is not a
    # httpx2.TimeoutException - the fallback must still trigger for these,
    # not propagate as an unhandled exception.
    monkeypatch.setattr(nvidia_image, "OUTPUT_DIR", tmp_path)
    fake_async_client(post_side_effect=RuntimeError("connection refused"))
    monkeypatch.setattr(nvidia_image, "_local_embedding", lambda text: [0.2, 0.3])

    result = await nvidia_image.create_embedding(text="hello world")

    assert "2 dimensions" in result
    assert f"model: local:{nvidia_image._LOCAL_EMBED_MODEL_NAME}" in result


@pytest.mark.asyncio
async def test_reports_clear_error_when_nvidia_fails_and_no_local_fallback_installed(
    nvidia_key, fake_async_client, tmp_path, monkeypatch
):
    monkeypatch.setattr(nvidia_image, "OUTPUT_DIR", tmp_path)
    fake_async_client(post_side_effect=lambda *a, **kw: FakeResponse(500, text="server error"))
    # Simulate the real, undecorated behavior of the optional dependency being
    # absent: _local_embedding catches the ImportError internally and returns None.
    monkeypatch.setattr(nvidia_image, "_local_embedding", lambda text: None)

    result = await nvidia_image.create_embedding(text="hello world")

    assert "no local fallback available" in result
    assert "local-embeddings" in result
    assert "HTTP 500" in result


def test_local_embedding_returns_none_when_dependency_missing(monkeypatch):
    """The real function, unmocked: if sentence-transformers truly isn't
    installed, it must fail closed (return None) rather than raise."""
    monkeypatch.setattr(nvidia_image, "_local_embed_model", None)

    # In this repo's base test environment sentence-transformers is not
    # installed (it's the optional local-embeddings extra), so this exercises
    # the real ImportError path without needing to mock anything.
    result = nvidia_image._local_embedding("hello world")

    assert result is None
