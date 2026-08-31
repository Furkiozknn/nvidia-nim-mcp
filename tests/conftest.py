"""Shared pytest fixtures and setup for the nvidia-nim-mcp test suite.

No test in this suite makes a real network call or needs NVIDIA_API_KEY (or
any other provider key) to be set - every HTTP/litellm call is mocked. See
the individual test modules for how each dependency is faked.
"""
import sys
import typing
from unittest.mock import AsyncMock

import pytest

# --- CPython 3.14 pre-release compatibility shim -----------------------
# Some 3.14 release-candidate builds ship a `typing._eval_type()` that does
# not yet accept the `prefer_fwd_module` keyword argument that pydantic 2.x
# unconditionally passes on Python >= 3.14. Without this shim, simply
# importing `mcp` (and therefore `nvidia_image`) raises a TypeError before a
# single test can collect. The wrapper only changes behavior when that exact
# TypeError occurs, so it's a no-op - and safe to leave in place - on any
# interpreter (including final 3.14 releases) where the original call
# already succeeds.
_orig_eval_type = typing._eval_type


def _eval_type_compat(*args, **kwargs):
    try:
        return _orig_eval_type(*args, **kwargs)
    except TypeError as e:
        if "prefer_fwd_module" in str(e):
            kwargs.pop("prefer_fwd_module", None)
            return _orig_eval_type(*args, **kwargs)
        raise


typing._eval_type = _eval_type_compat
# -------------------------------------------------------------------------


@pytest.fixture
def nvidia_key(monkeypatch):
    """Make the module believe NVIDIA_API_KEY is set, without touching .env."""
    import nvidia_image

    monkeypatch.setattr(nvidia_image, "API_KEY", "test-nvidia-key")
    monkeypatch.setitem(nvidia_image.HEADERS, "Authorization", "Bearer test-nvidia-key")
    return "test-nvidia-key"


@pytest.fixture
def no_nvidia_key(monkeypatch):
    """Make the module believe NVIDIA_API_KEY is NOT set."""
    import nvidia_image

    monkeypatch.setattr(nvidia_image, "API_KEY", None)
    return None


class FakeResponse:
    """Minimal stand-in for an httpx2.Response."""

    def __init__(self, status_code=200, json_data=None, text="", content=b""):
        self.status_code = status_code
        self._json_data = json_data if json_data is not None else {}
        self.text = text
        self.content = content

    def json(self):
        return self._json_data

    def raise_for_status(self):
        # Real httpx raises httpx.HTTPStatusError; production code only ever
        # catches this generically (`except Exception`), so a plain
        # RuntimeError is an equally valid stand-in for tests.
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class FakeAsyncClient:
    """Minimal stand-in for httpx2.AsyncClient, usable as an async context
    manager. `post_side_effect`/`get_side_effect` are passed straight to an
    AsyncMock, so each can be a return value, an exception, a list of either,
    or a callable. `get` defaults to a successful tiny-image response so
    existing tests that only care about `post` don't need to know about the
    Pollinations fallback path at all."""

    def __init__(self, post_side_effect=None, get_side_effect=None):
        self.post = AsyncMock(side_effect=post_side_effect)
        self.get = AsyncMock(
            side_effect=get_side_effect
            if get_side_effect is not None
            else (lambda *a, **kw: FakeResponse(200, content=b"fake-image-bytes"))
        )

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return False


@pytest.fixture
def fake_async_client(monkeypatch):
    """Patch nvidia_image.httpx2.AsyncClient so `async with httpx2.AsyncClient()`
    yields a FakeAsyncClient. Returns a factory: call it with the desired
    `post_side_effect`/`get_side_effect` before the code under test opens the
    client."""
    import nvidia_image

    holder = {}

    def _factory(post_side_effect=None, get_side_effect=None):
        client = FakeAsyncClient(post_side_effect, get_side_effect)
        holder["client"] = client
        monkeypatch.setattr(nvidia_image.httpx2, "AsyncClient", lambda *a, **kw: client)
        return client

    return _factory
