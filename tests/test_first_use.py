"""First-use behaviour: start-up cost, the real stdio handshake, `--help`, and
telling a rejected key apart from every other failure.

Nothing here reaches a real provider. The one test that starts the server as a
subprocess only does `initialize` + `tools/list`, which need no key and no
network.
"""
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

import nvidia_image
from conftest import FakeResponse

ROOT = Path(__file__).resolve().parent.parent
KEYS = ("NVIDIA_API_KEY", "GROQ_API_KEY", "MISTRAL_API_KEY", "GEMINI_API_KEY", "CEREBRAS_API_KEY")
TOOLS = {
    "generate_image",
    "translate_text",
    "ask_llm",
    "describe_image",
    "check_content_safety",
    "create_embedding",
    "check_provider_health",
}
UVX_COMMAND = "uvx --from git+https://github.com/Furkiozknn/nvidia-nim-mcp nvidia-nim-mcp"


def _clean_env():
    env = {k: v for k, v in os.environ.items() if k not in KEYS}
    env["PYTHONIOENCODING"] = "utf-8"
    return env


# --- start-up ------------------------------------------------------------

def test_importing_the_server_does_not_import_litellm():
    """`import litellm` alone took ~10 s of a 13 s start; MCP clients give
    `initialize` a limited time. It must stay off the import path."""
    code = "import sys, nvidia_image; assert 'litellm' not in sys.modules, 'litellm imported at start-up'"
    done = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=_clean_env(), capture_output=True, text=True)
    assert done.returncode == 0, done.stderr


def test_lazy_litellm_reads_and_writes_reach_the_real_module(monkeypatch):
    real = nvidia_image._load_litellm()
    marker = object()
    monkeypatch.setattr(nvidia_image.litellm, "acompletion", marker)
    assert real.acompletion is marker
    assert nvidia_image.litellm.acompletion is marker


# --- --help / --version ---------------------------------------------------

def test_help_prints_the_registration_command_and_exits_zero(capsys):
    nvidia_image.main(["--help"])
    out = capsys.readouterr().out
    assert UVX_COMMAND in out
    assert "NVIDIA_API_KEY" in out and "build.nvidia.com" in out
    assert "output" in out


def test_help_registration_command_matches_the_readme():
    """The command a person copies from --help and from the README must be the same."""
    assert UVX_COMMAND in (ROOT / "README.md").read_text(encoding="utf-8")
    assert UVX_COMMAND in nvidia_image.HELP


def test_version_prints_and_exits_zero(capsys):
    nvidia_image.main(["--version"])
    assert capsys.readouterr().out.startswith("nvidia-nim-mcp ")


def test_unknown_argument_exits_two_and_points_at_help(capsys):
    with pytest.raises(SystemExit) as exc:
        nvidia_image.main(["--bogus"])
    assert exc.value.code == 2
    assert "--help" in capsys.readouterr().err


def test_startup_line_names_keys_but_never_their_values(monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "secret-value-1")
    monkeypatch.setenv("GROQ_API_KEY", "secret-value-2")
    line = nvidia_image._startup_line()
    assert "NVIDIA_API_KEY set" in line and "GROQ_API_KEY" in line
    assert "secret-value" not in line


def test_startup_line_without_keys_says_keyless_tiers_only(monkeypatch):
    for k in KEYS:
        monkeypatch.delenv(k, raising=False)
    assert "NOT set (keyless tiers only)" in nvidia_image._startup_line()


# --- no key vs wrong key ----------------------------------------------------

def test_no_provider_message_says_where_to_get_and_where_to_put_the_key():
    msg = nvidia_image._no_provider_message("ask_llm", nvidia_image.EXTRA_PROVIDERS)
    assert "https://build.nvidia.com/" in msg
    assert "environment of this MCP server" in msg
    assert ".env" in msg


class _Rejected(Exception):
    status_code = 401


@pytest.fixture
def only_nvidia(nvidia_key, monkeypatch):
    for k in KEYS[1:]:
        monkeypatch.delenv(k, raising=False)


@pytest.mark.asyncio
async def test_translate_with_a_rejected_key_says_the_key_was_rejected(only_nvidia, monkeypatch):
    monkeypatch.setattr(nvidia_image.litellm, "acompletion", AsyncMock(side_effect=_Rejected("401 unauthorized")))
    result = await nvidia_image.translate_text(text="hi", target_language="Turkish")
    assert result.startswith("All translation models/providers failed or timed out.")
    assert "rejected its API key" in result


@pytest.mark.asyncio
async def test_translate_failing_for_another_reason_keeps_the_plain_message(only_nvidia, monkeypatch):
    monkeypatch.setattr(nvidia_image.litellm, "acompletion", AsyncMock(side_effect=RuntimeError("rate limited")))
    result = await nvidia_image.translate_text(text="hi", target_language="Turkish")
    assert result == "All translation models/providers failed or timed out."


@pytest.mark.asyncio
async def test_ask_llm_with_a_rejected_key_says_so(only_nvidia, monkeypatch):
    monkeypatch.setattr(nvidia_image.litellm, "acompletion", AsyncMock(side_effect=_Rejected("403")))
    result = await nvidia_image.ask_llm(question="hi")
    assert "rejected its API key" in result


@pytest.mark.asyncio
async def test_content_safety_with_a_rejected_key_says_so(only_nvidia, fake_async_client):
    fake_async_client(post_side_effect=lambda *a, **kw: FakeResponse(401, text="unauthorized"))
    result = await nvidia_image.check_content_safety(text="hello")
    assert result.startswith("Content safety check failed.")
    assert "rejected its API key" in result


@pytest.mark.asyncio
async def test_describe_image_with_a_rejected_key_says_so(only_nvidia, fake_async_client, tmp_path):
    image = tmp_path / "a.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\nfake")
    fake_async_client(post_side_effect=lambda *a, **kw: FakeResponse(403, text="forbidden"))
    result = await nvidia_image.describe_image(image_path=str(image))
    assert result.startswith("All vision models failed or timed out.")
    assert "rejected its API key" in result


@pytest.mark.asyncio
async def test_generate_image_with_a_rejected_key_says_so_and_shows_the_status(
    nvidia_key, fake_async_client, tmp_path, monkeypatch
):
    monkeypatch.setattr(nvidia_image, "OUTPUT_DIR", tmp_path)
    fake_async_client(
        post_side_effect=lambda *a, **kw: FakeResponse(401, text="unauthorized"),
        get_side_effect=RuntimeError("pollinations down"),
    )
    result = await nvidia_image.generate_image(prompt="a cat")
    assert "All image models failed" in result
    assert "HTTP 401 (key rejected" in result
    assert "rejected its API key" in result


@pytest.mark.asyncio
async def test_generate_image_failing_with_503_has_no_key_hint(nvidia_key, fake_async_client, tmp_path, monkeypatch):
    monkeypatch.setattr(nvidia_image, "OUTPUT_DIR", tmp_path)
    fake_async_client(
        post_side_effect=lambda *a, **kw: FakeResponse(503, text="overloaded"),
        get_side_effect=RuntimeError("pollinations down"),
    )
    result = await nvidia_image.generate_image(prompt="a cat")
    assert "rejected" not in result


@pytest.mark.asyncio
async def test_embedding_with_a_rejected_key_says_so(nvidia_key, fake_async_client, monkeypatch):
    monkeypatch.setattr(nvidia_image, "_local_embedding", lambda text: None)
    fake_async_client(post_side_effect=lambda *a, **kw: FakeResponse(401, text="unauthorized"))
    result = await nvidia_image.create_embedding(text="hello")
    assert "rejected its API key" in result
    assert "local-embeddings" in result


@pytest.mark.asyncio
async def test_health_probe_labels_a_rejected_key():
    client = AsyncMock()
    client.post = AsyncMock(return_value=FakeResponse(401, text="unauthorized"))
    ok, detail = await nvidia_image._probe_nvidia_chat_model(client, "nvidia/x")
    assert ok is False
    assert detail.startswith("HTTP 401") and "key rejected" in detail


# --- scripts/sonda.py (the stdio probe the README and the demo use) ------------

def _sonda(*args, cwd=None):
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "sonda.py"), *args],
        cwd=cwd, env=_clean_env(), capture_output=True, text=True, timeout=120,
    )


def test_real_stdio_handshake_lists_all_seven_tools_and_calls_answer(tmp_path):
    """One real server start (they cost ~4 s each): the handshake, all seven
    tools with a description, a keyless call, and an unknown tool -> exit 1."""
    done = _sonda(
        "--no-keys", "--call", "ask_llm", '{"question": "hi"}', "--call", "nope", "{}",
        "--", sys.executable, str(ROOT / "nvidia_image.py"), cwd=tmp_path,
    )
    assert done.returncode == 1, done.stderr  # the unknown tool is an error result
    out = done.stdout
    assert "initialize ok: nvidia-nim" in out
    assert "tools/list: 7 tools" in out
    assert all(name in out for name in TOOLS)
    assert "(no description)" not in out
    assert "no provider configured" in out
    assert "Unknown tool: nope" in out


def test_sonda_usage_and_missing_command_exit_codes(tmp_path):
    assert _sonda().returncode == 2
    assert _sonda("--call", "x", "notjson", "--", "nvidia-nim-mcp").returncode == 2
    missing = _sonda("--", "no-such-command-zzz", cwd=tmp_path)
    assert missing.returncode == 3 and "not found on PATH" in missing.stderr
