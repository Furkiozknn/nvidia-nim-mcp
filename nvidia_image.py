import asyncio
import base64
import json
import logging
import os
import threading
import urllib.parse
from datetime import datetime
from pathlib import Path
from typing import Annotated

import httpx2
import litellm
from dotenv import load_dotenv
from mcp.server import MCPServer
from pydantic import Field

load_dotenv()

logger = logging.getLogger(__name__)

mcp = MCPServer("nvidia-nim")

NVIDIA_API_KEY_ENV = "NVIDIA_API_KEY"
GENAI_BASE = "https://ai.api.nvidia.com/v1/genai"
CHAT_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
EMBED_URL = "https://integrate.api.nvidia.com/v1/embeddings"

# Path(__file__).parent would write inside the installed package: for a
# pip-installed user that is site-packages. Default to the current working
# directory instead, overridable with NVIDIA_NIM_OUTPUT_DIR. Same fix as
# voice-io-mcp and mini-creative-toolkit, which had the identical bug.
OUTPUT_DIR = Path(os.environ.get("NVIDIA_NIM_OUTPUT_DIR") or Path.cwd() / "output")


def _api_key() -> str | None:
    """The NVIDIA key as it is *right now*, or None.

    Read per request rather than snapshotted at import: a module-level
    snapshot ignores key rotation entirely, and a `.env` written after the
    server process started never takes effect. Empty string normalizes to
    None - an emptied-out .env value means "no key", not "the empty key".
    """
    return os.environ.get(NVIDIA_API_KEY_ENV) or None


def _headers() -> dict[str, str]:
    """Auth headers for NVIDIA's endpoints, built per request from the
    current environment. A module-level HEADERS dict built at import time
    produced a literal `Bearer None` when no key was set - a request
    guaranteed to 401 confusingly instead of the NVIDIA tier being skipped.
    Only called from paths already gated on _api_key() being set.
    """
    return {
        "Authorization": f"Bearer {_api_key()}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


def _stamp() -> str:
    """Microsecond-precision timestamp for output filenames. Plain
    second-precision let two calls landing in the same wall-clock second
    silently overwrite each other's file - the same bug, and the same fix,
    as voice-io-mcp's _stamp()."""
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def _redact(text: str, secret: str | None) -> str:
    """Scrub a known secret out of an error/log string before it's returned
    to the caller. Ported from voice-io-mcp. Defense-in-depth: a provider's
    error body is attacker-influenced text this server hands straight back,
    and a 401 echoing the offending Authorization header is not ruled out."""
    if not secret:
        return text
    return text.replace(secret, "***")


def _read_image_bytes(path: Path) -> bytes | None:
    """Read at most DESCRIBE_IMAGE_MAX_BYTES of an image file, or None if it
    is larger. The size is enforced on the read itself, not only on an
    earlier stat(): a file that grows between the check and the read must
    not turn into an unbounded upload. Blocking disk I/O - always called
    through asyncio.to_thread, never inline in an async def."""
    with path.open("rb") as f:
        data = f.read(DESCRIBE_IMAGE_MAX_BYTES + 1)
    return None if len(data) > DESCRIBE_IMAGE_MAX_BYTES else data


def _http_failure(status_code: int) -> str:
    """Short failure detail for a non-200 NVIDIA response. HTTP 410 is how
    NVIDIA reports a model that has reached end of life - that one needs a
    code change (remove or replace the model), not a retry, so say so."""
    if status_code == 410:
        return "HTTP 410 - model retired by the provider; remove or replace it"
    return f"HTTP {status_code}"

# Every model below was confirmed working with a real request (2026-08-22)
# before being wired in. Each capability has 2+ models so one being
# rate-limited/slow/congested doesn't take the tool down; the caller never
# needs to know which one answered.

# Bounds on generate_image's width/height, enforced by the tool's input
# schema. A request far outside them is never a real image size, and the
# keyless Pollinations tier would otherwise be asked for it verbatim.
IMAGE_MIN_SIDE = 64
IMAGE_MAX_SIDE = 2048

IMAGE_MODELS = [
    {"slug": "black-forest-labs/flux.1-dev", "cfg_scale": 3.5, "steps": 25},
    {"slug": "black-forest-labs/flux.2-klein-4b", "cfg_scale": 1, "steps": 4},
]

# Last-resort image fallback if every NVIDIA model above fails or is
# rate-limited: Pollinations.ai, free and keyless (no auth, no quota on our
# side). Same backend and technique already validated in
# mini-creative-toolkit's own generate_image_free tool - reused here as a
# fallback tier rather than duplicated as a second free-standing tool, since
# this MCP server's generate_image already owns the "try several backends"
# contract.
POLLINATIONS_BASE = "https://image.pollinations.ai/prompt"

# Ceiling for one Pollinations image download. The response is streamed and
# counted, not trusted from Content-Length; without this a misbehaving or
# compromised endpoint could make the fallback buffer arbitrarily many bytes.
POLLINATIONS_MAX_DOWNLOAD_BYTES = 20 * 1024 * 1024

# describe_image reads a caller-supplied local file and uploads its bytes
# (base64, inside the prompt) to NVIDIA or a fallback provider - a third
# party either way. The allowlist and cap make a wrong or malicious path
# ("describe /etc/shadow", a multi-GB file) fail fast and locally instead of
# exfiltrating whatever happens to be at the path. Same reasoning and shape
# as voice-io-mcp's speech_to_text limits.
DESCRIBE_IMAGE_ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}
DESCRIBE_IMAGE_ALLOWED_FORMATS = {"jpeg", "png", "webp"}
DESCRIBE_IMAGE_MAX_BYTES = 10 * 1024 * 1024

# Passed to every litellm chain call. litellm's own default is 600 seconds -
# a wedged provider would otherwise hold a tool call for ten minutes, far
# past any MCP client's patience. Two minutes is generous for chat-sized
# completions while still guaranteeing the tool returns.
LLM_TIMEOUT = 120.0

TRANSLATE_MODELS = [
    "nvidia/riva-translate-4b-instruct-v2",
    "nvidia/llama-3.3-nemotron-super-49b-v1.5",
    "openai/gpt-oss-120b",
]

LLM_MODELS = [
    # glm-5.2 and deepseek-v4-flash were removed 2026-08-22 - both confirmed
    # permanently retired on NVIDIA's platform (HTTP 410, "reached its end
    # of life"), not just rate-limited. check_provider_health labels a 410
    # as "retired" so the next one is easy to spot.
    "nvidia/llama-3.3-nemotron-super-49b-v1.5",
    "openai/gpt-oss-120b",
]

# Providers beyond NVIDIA (added 2026-08-22). Each is skipped automatically
# if its API key isn't set - set a key and it joins the fallback chain on the
# next call, no code changes needed. OpenAI has no ongoing free tier as of
# 2026 (only an expiring trial credit) so it's deliberately not here.
EXTRA_PROVIDERS = [
    # Ordered by what was confirmed working with a real call on 2026-08-22.
    # Model names drift fast; verify against the provider's live /models
    # endpoint before changing one.
    {"env": "GROQ_API_KEY", "model": "groq/openai/gpt-oss-120b"},  # free tier, confirmed working
    {"env": "MISTRAL_API_KEY", "model": "mistral/mistral-small-latest"},  # free tier, confirmed working
    # Optional, not verified end to end: on the Google Cloud project used for
    # testing, the Gemini API answered PERMISSION_DENIED until the project is
    # moved to a paid plan. Keys from other projects may work. A failing
    # entry costs one failed call; litellm then moves to the next.
    {"env": "GEMINI_API_KEY", "model": "gemini/gemini-flash-latest"},
    # Optional, paid: Cerebras answers "Payment required" until a payment
    # method is on the account, so it is not a free tier. Kept last so it is
    # only reached when everything else has failed.
    {"env": "CEREBRAS_API_KEY", "model": "cerebras/gpt-oss-120b"},
]


def _build_chat_chain(nvidia_models: list[str]) -> list[dict]:
    """NVIDIA models first (they need the custom api_base) - but only when
    NVIDIA_API_KEY is actually set - then any extra free-tier provider whose
    key is present in the environment right now.

    With no NVIDIA key the chain starts at the free tier instead of being
    empty. That is the entire point of the documented fallback chain, and a
    blanket "NVIDIA_API_KEY not set" guard in front of every tool made it
    unreachable for anyone holding only a GROQ/MISTRAL/GEMINI key."""
    nvidia_key = _api_key()
    chain = [
        {"model": f"openai/{m}", "api_base": CHAT_URL.rsplit("/chat/completions", 1)[0], "api_key": nvidia_key}
        for m in nvidia_models
    ] if nvidia_key else []
    for provider in EXTRA_PROVIDERS:
        key = os.environ.get(provider["env"])
        if key:
            chain.append({"model": provider["model"], "api_key": key})
    return chain


def _redact_known_keys(text: str) -> str:
    """_redact for every provider key currently set, not just one. For
    messages that may come from any provider in a chain (litellm exception
    text), where the caller cannot know which key could be echoed."""
    for secret in (_api_key(), *(os.environ.get(p["env"]) for p in (*EXTRA_PROVIDERS, *VISION_PROVIDERS))):
        text = _redact(text, secret)
    return text


async def _run_chat_chain(chain: list[dict], messages: list[dict], max_tokens: int = 1024) -> tuple[str, str] | None:
    """Try each entry of a pre-built provider chain in order and return
    (content, model) from the first one that answers with text, or None.
    Shared by _multi_provider_chat (NVIDIA-prefixed) and
    _chat_via_provider_chain (cross-provider only) so the actual
    execution/error-handling logic exists in exactly one place.

    Each entry is its own litellm call with only its own kwargs. litellm's
    `fallbacks=` parameter is deliberately not used: it copies the primary
    call's kwargs into every fallback, so with NVIDIA first, the Groq and
    Mistral entries inherited NVIDIA's api_base - their keys were sent to
    NVIDIA's endpoint and the cross-provider fallback could never answer.

    max_retries=0: the chain is the retry. litellm's default retried each
    NVIDIA entry twice more (with backoff) before the next model got a turn.
    """
    for entry in chain:
        try:
            response = await litellm.acompletion(
                messages=messages,
                max_tokens=max_tokens,
                timeout=LLM_TIMEOUT,
                max_retries=0,
                **entry,
            )
            content = response.choices[0].message.content
        except Exception as e:
            logger.warning("%s failed: %s", entry["model"], _redact_known_keys(str(e)))
            continue
        if not content:
            # A reasoning model can spend the whole max_tokens budget before
            # writing any answer; an empty reply is a failed tier, not a result.
            logger.warning("%s returned no text", entry["model"])
            continue
        return content, response.model
    return None


async def _multi_provider_chat(nvidia_models: list[str], messages: list[dict], max_tokens: int = 1024) -> tuple[str, str] | None:
    """Try NVIDIA models, then any configured free-tier provider, in order."""
    return await _run_chat_chain(_build_chat_chain(nvidia_models), messages, max_tokens)

VISION_MODELS = [
    "nvidia/nemotron-nano-12b-v2-vl",
    "meta/llama-3.2-11b-vision-instruct",
]

# Vision-capable free-tier fallback for describe_image, tried only after both
# NVIDIA_MODELS above fail. A separate list from EXTRA_PROVIDERS because not
# every model there is vision-capable (Groq's gpt-oss-120b and Mistral's
# mistral-small are text-only) - these are vision-specific model names for
# the same three providers, confirmed 2026-08-31. Model names on free/preview
# tiers drift fast (Groq alone has retired two prior vision picks this year) -
# expect to revisit.
VISION_PROVIDERS = [
    {"env": "GROQ_API_KEY", "model": "groq/qwen/qwen3.6-27b"},
    {"env": "MISTRAL_API_KEY", "model": "mistral/pixtral-12b-2409"},
    {"env": "GEMINI_API_KEY", "model": "gemini/gemini-flash-latest"},
]

EMBED_MODEL = "nvidia/nemotron-3-embed-1b"
SAFETY_MODEL = "nvidia/nemotron-3.5-content-safety"

# Fully local, keyless embedding fallback for create_embedding - only used if
# the NVIDIA endpoint fails. Deliberately an *optional* dependency (`pip
# install .[local-embeddings]` / `uv sync --extra local-embeddings`): the
# sentence-transformers package pulls in torch, ~1GB+, too heavy to force on
# every install just for a fallback path most calls will never need.
_LOCAL_EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
_local_embed_model = None  # lazy singleton - only loaded the first time it's actually needed
# create_embedding runs _local_embedding via asyncio.to_thread, so concurrent
# calls can land on different real OS threads - a plain "is None" check alone
# would let two threads both start loading the ~90MB model at once. A
# threading.Lock (not asyncio.Lock, which wouldn't cross thread-pool workers)
# makes the load-once check race-free.
_local_embed_lock = threading.Lock()


def _local_embedding(text: str) -> list[float] | None:
    """Compute an embedding locally via sentence-transformers, if installed.
    Returns None (never raises) if the optional dependency is missing or the
    model fails to load/run, so the caller can report a clean error."""
    global _local_embed_model
    try:
        if _local_embed_model is None:
            with _local_embed_lock:
                if _local_embed_model is None:
                    from sentence_transformers import SentenceTransformer

                    _local_embed_model = SentenceTransformer(_LOCAL_EMBED_MODEL_NAME)
        return _local_embed_model.encode(text).tolist()
    except Exception as e:
        logger.warning("local embedding fallback unavailable: %s", e)
        return None


def _build_provider_chain(providers: list[dict]) -> list[dict]:
    """Chain of {model, api_key} entries for whichever of `providers` has its
    API key actually present in the environment right now - same
    skip-silently-if-unconfigured rule as _build_chat_chain, but for a
    provider list with no NVIDIA-model prefix of its own."""
    chain = []
    for provider in providers:
        key = os.environ.get(provider["env"])
        if key:
            chain.append({"model": provider["model"], "api_key": key})
    return chain


async def _chat_via_provider_chain(providers: list[dict], messages: list[dict], max_tokens: int = 1024) -> tuple[str, str] | None:
    """Try each configured provider in `providers`, in declared order, via
    litellm. Returns None if none are configured or all fail - never raises."""
    return await _run_chat_chain(_build_provider_chain(providers), messages, max_tokens)


async def _chat_with_fallback(client: httpx2.AsyncClient, models: list[str], messages: list[dict], max_tokens: int = 1024) -> tuple[str, str] | None:
    """Try each model in order, return (content, model_used) from the first
    success. These are NVIDIA-only endpoints, so with no NVIDIA_API_KEY there
    is nothing here to try: return None immediately and let the caller move
    on to its cross-provider fallback, rather than firing a request that can
    only 401."""
    if not _api_key():
        return None
    headers = _headers()
    for model in models:
        body = {"model": model, "messages": messages, "max_tokens": max_tokens}
        try:
            resp = await client.post(CHAT_URL, headers=headers, json=body, timeout=18.0)
        except httpx2.HTTPError as e:
            # Timeouts and every other transport failure (connection refused,
            # DNS, TLS, ...) move on to the next model. Catching only
            # TimeoutException let a ConnectError escape the tool entirely,
            # skipping the cross-provider fallback the caller still had.
            logger.warning("%s: %s", model, _redact(str(e), _api_key()))
            continue
        if resp.status_code != 200:
            continue
        try:
            content = resp.json()["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError, ValueError):
            # ValueError covers json.JSONDecodeError (a non-JSON 200 body).
            continue
        return content, model
    return None


# Short timeout for health probes specifically - these are meant to be a
# quick "is it alive" check, not a real request, so failing fast is correct.
HEALTH_PROBE_TIMEOUT = 8.0

# check_provider_health fires every probe concurrently via asyncio.gather with
# no cap of its own - fine today with ~12 total models/providers, but nothing
# stops that from becoming a self-inflicted burst against NVIDIA's (and every
# extra provider's) rate limits if the model lists grow. Bound how many probes
# are ever in flight at once, independent of how many models exist.
HEALTH_PROBE_CONCURRENCY = 6


async def _bounded(sem: asyncio.Semaphore, coro):
    """Run one probe coroutine under a concurrency cap."""
    async with sem:
        return await coro


async def _probe_nvidia_chat_model(client: httpx2.AsyncClient, model: str) -> tuple[bool, str]:
    """Cheap liveness probe for one NVIDIA chat-completions model: a
    single-token reply, not a real generation."""
    body = {"model": model, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 1}
    try:
        resp = await client.post(CHAT_URL, headers=_headers(), json=body, timeout=HEALTH_PROBE_TIMEOUT)
    except httpx2.TimeoutException:
        return False, "timed out"
    except Exception as e:
        return False, f"error: {_redact(str(e), _api_key())}"
    if resp.status_code != 200:
        return False, _http_failure(resp.status_code)
    return True, "ok"


async def _probe_nvidia_image_model(client: httpx2.AsyncClient, slug: str) -> tuple[bool, str]:
    """Cheap liveness probe for one NVIDIA image model: a single-step,
    tiny-resolution generation instead of a full-quality image."""
    body = {"prompt": "hi", "steps": 1, "cfg_scale": 1, "seed": 0, "width": 64, "height": 64}
    try:
        resp = await client.post(f"{GENAI_BASE}/{slug}", headers=_headers(), json=body, timeout=HEALTH_PROBE_TIMEOUT)
    except httpx2.TimeoutException:
        return False, "timed out"
    except Exception as e:
        return False, f"error: {_redact(str(e), _api_key())}"
    if resp.status_code != 200:
        return False, _http_failure(resp.status_code)
    return True, "ok"


async def _probe_nvidia_embed_model(client: httpx2.AsyncClient, model: str) -> tuple[bool, str]:
    """Cheap liveness probe for the NVIDIA embedding model: a one-word input."""
    body = {"input": ["hi"], "model": model, "input_type": "query"}
    try:
        resp = await client.post(EMBED_URL, headers=_headers(), json=body, timeout=HEALTH_PROBE_TIMEOUT)
    except httpx2.TimeoutException:
        return False, "timed out"
    except Exception as e:
        return False, f"error: {_redact(str(e), _api_key())}"
    if resp.status_code != 200:
        return False, _http_failure(resp.status_code)
    return True, "ok"


def _sniff_image_format(body: bytes) -> str | None:
    """The format the magic bytes claim, or None for not-an-image. A cheap
    stand-in for a full decode (this server deliberately has no image
    library): it catches the real failure mode - an HTML error page or
    empty body served with a 200 - though not a truncated transfer."""
    if body.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    if body.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if body[:4] == b"RIFF" and body[8:12] == b"WEBP":
        return "webp"
    if body.startswith((b"GIF87a", b"GIF89a")):
        return "gif"
    return None


def _prompt_path_segment(prompt: str) -> str:
    """The prompt as exactly one URL path segment under /prompt/.

    urllib.parse.quote leaves "/" alone by default, so "AC/DC poster" became
    two segments and "../../models" climbed out of /prompt/ to another path
    on the host. Everything is escaped here, and a prompt that is only dots
    (".", "..") is escaped too, because the URL parser would otherwise treat
    it as a dot segment."""
    encoded = urllib.parse.quote(prompt, safe="")
    if encoded.strip(".") == "":
        encoded = encoded.replace(".", "%2E")
    return encoded


async def _generate_image_pollinations(client: httpx2.AsyncClient, prompt: str, seed: int, width: int, height: int) -> bytes:
    """Generate via the free Pollinations.ai backend - only called from
    generate_image after every NVIDIA model has already failed.

    The body is streamed with a byte ceiling and verified to actually be an
    image (Content-Type and magic bytes) before it is returned - a free
    endpoint under load happily serves HTML error pages with a 200, and
    those must not be written to disk as a .jpg."""
    encoded = _prompt_path_segment(prompt)
    params = {"width": width, "height": height, "nologo": "true", "seed": seed}
    async with client.stream(
        "GET", f"{POLLINATIONS_BASE}/{encoded}", params=params, timeout=60.0, follow_redirects=True
    ) as resp:
        resp.raise_for_status()
        content_type = resp.headers.get("content-type", "").split(";")[0].strip().lower()
        if not content_type.startswith("image/"):
            raise RuntimeError(
                f"pollinations returned Content-Type {content_type or '(none)'!r}, not an image "
                "(usually an error page served with a success status)"
            )
        buffer = bytearray()
        async for chunk in resp.aiter_bytes():
            buffer.extend(chunk)
            if len(buffer) > POLLINATIONS_MAX_DOWNLOAD_BYTES:
                raise RuntimeError(
                    f"pollinations response exceeded the {POLLINATIONS_MAX_DOWNLOAD_BYTES} byte "
                    "download limit and was aborted"
                )
    body = bytes(buffer)
    if _sniff_image_format(body) is None:
        raise RuntimeError(
            f"pollinations returned {len(body)} bytes that are not a recognizable image"
        )
    return body


async def _probe_pollinations(client: httpx2.AsyncClient) -> tuple[bool, str]:
    """Cheap liveness probe for the Pollinations.ai fallback: a tiny image
    request. Streamed and never read - only the status line matters, so a
    misbehaving endpoint cannot make the probe buffer a large body."""
    try:
        async with client.stream(
            "GET",
            f"{POLLINATIONS_BASE}/hi",
            params={"width": 8, "height": 8, "nologo": "true"},
            timeout=HEALTH_PROBE_TIMEOUT,
            follow_redirects=True,
        ) as resp:
            status_code = resp.status_code
    except httpx2.TimeoutException:
        return False, "timed out"
    except Exception as e:
        return False, f"error: {e}"
    if status_code != 200:
        return False, f"HTTP {status_code}"
    return True, "ok"


async def _probe_extra_provider(provider: dict) -> tuple[bool, str]:
    """Cheap liveness probe for one cross-provider fallback model. Skipped
    (not an error) if its API key isn't set in .env."""
    key = os.environ.get(provider["env"])
    if not key:
        return False, "not configured"
    try:
        await litellm.acompletion(
            model=provider["model"],
            api_key=key,
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=1,
            timeout=HEALTH_PROBE_TIMEOUT,
        )
    except Exception as e:
        return False, f"error: {_redact(str(e), key)}"
    return True, "ok"


async def _skipped(detail: str) -> tuple[bool, str]:
    """A probe result for a tier that was deliberately never contacted -
    lets check_provider_health keep one uniform asyncio.gather shape."""
    return False, detail


def _any_provider_configured(providers: list[dict]) -> bool:
    """Whether a chat-backed tool has anything at all to try right now:
    NVIDIA's key, or any one of `providers`' keys."""
    return bool(_api_key()) or any(os.environ.get(p["env"]) for p in providers)


def _no_provider_message(action: str, providers: list[dict]) -> str:
    """The one 'nothing is configured' message shape every tool with no
    keyless tier returns - factored out once so they can't drift apart.

    It names what the user actually needs. The old blanket
    "NVIDIA_API_KEY not set" guard did not: these tools run on any one of
    several free-tier keys, and NVIDIA's is only the first of them."""
    return (
        f"{action}: no provider configured. Set {NVIDIA_API_KEY_ENV} in .env, "
        f"or any of {', '.join(p['env'] for p in providers)} for a free-tier fallback."
    )


@mcp.tool()
async def generate_image(
    prompt: Annotated[str, Field(min_length=1, description="Description of the image to generate.")],
    seed: Annotated[int, Field(ge=0, description="Seed for reproducibility.")] = 0,
    width: Annotated[
        int, Field(ge=IMAGE_MIN_SIDE, le=IMAGE_MAX_SIDE, description="Image width in pixels.")
    ] = 1024,
    height: Annotated[
        int, Field(ge=IMAGE_MIN_SIDE, le=IMAGE_MAX_SIDE, description="Image height in pixels.")
    ] = 1024,
) -> str:
    """Generate an image from a text prompt using NVIDIA NIM image models.

    Tries multiple models in order (flux.1-dev, then flux.2-klein-4b), then
    falls back to the free Pollinations.ai backend if both NVIDIA models
    fail - a provider outage no longer blocks generation entirely, it just
    silently drops the caller to a lower-quality free tier.

    Works with NO API key at all: without NVIDIA_API_KEY the NVIDIA tier is
    skipped and the keyless Pollinations tier is used directly.

    Args:
        prompt: Description of the image to generate.
        seed: Seed for reproducibility.
        width: Image width in pixels.
        height: Image height in pixels.
    """
    # No NVIDIA_API_KEY is not a reason to fail: the Pollinations tier below
    # is free and keyless, so an unkeyed caller simply starts there.
    api_key = _api_key()
    errors = []
    async with httpx2.AsyncClient() as client:
        if api_key:
            headers = _headers()
            for model in IMAGE_MODELS:
                body = {
                    "prompt": prompt,
                    "steps": model["steps"],
                    "cfg_scale": model["cfg_scale"],
                    "seed": seed,
                    "width": width,
                    "height": height,
                }
                try:
                    resp = await client.post(f"{GENAI_BASE}/{model['slug']}", headers=headers, json=body, timeout=45.0)
                except httpx2.TimeoutException:
                    errors.append(f"{model['slug']}: timed out")
                    continue
                except httpx2.HTTPError as e:
                    # Connection refused, DNS, TLS... - the next model or the
                    # Pollinations tier may still answer, so record and move on.
                    errors.append(f"{model['slug']}: {_redact(str(e), api_key)}")
                    continue
                if resp.status_code != 200:
                    errors.append(
                        f"{model['slug']}: {_http_failure(resp.status_code)} - {_redact(resp.text[:150], api_key)}"
                    )
                    continue
                try:
                    artifacts = resp.json().get("artifacts")
                    image_bytes = base64.b64decode(artifacts[0]["base64"], validate=True) if artifacts else None
                except (AttributeError, KeyError, IndexError, TypeError, ValueError):
                    # ValueError covers a non-JSON body and invalid base64.
                    errors.append(f"{model['slug']}: malformed response")
                    continue
                if not image_bytes:
                    errors.append(f"{model['slug']}: no artifacts in response")
                    continue
                OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
                filepath = OUTPUT_DIR / f"{model['slug'].split('/')[-1]}_{_stamp()}.jpg"
                # Blocking disk write - off the event loop, not inline here.
                await asyncio.to_thread(filepath.write_bytes, image_bytes)
                return f"Image saved to {filepath} (model: {model['slug']})"
        else:
            errors.append(f"NVIDIA models skipped: {NVIDIA_API_KEY_ENV} not set in .env")

        try:
            image_bytes = await _generate_image_pollinations(client, prompt, seed, width, height)
        except Exception as e:
            errors.append(f"pollinations (fallback): {e}")
        else:
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            filepath = OUTPUT_DIR / f"pollinations_{_stamp()}.jpg"
            await asyncio.to_thread(filepath.write_bytes, image_bytes)
            return f"Image saved to {filepath} (model: pollinations, fallback after NVIDIA models failed)"

    return "All image models failed:\n" + "\n".join(errors)


@mcp.tool()
async def translate_text(
    text: Annotated[str, Field(description="The text to translate.")],
    target_language: Annotated[str, Field(description='Language to translate into, e.g. "Turkish", "Spanish".')],
) -> str:
    """Translate text to a target language using NVIDIA NIM models.

    Tries a dedicated translation model first, falling back to general NVIDIA
    chat models, then Groq/Mistral/Gemini/Cerebras if those API keys are
    configured.

    Needs NVIDIA_API_KEY *or* any one free-tier key (GROQ/MISTRAL/GEMINI/
    CEREBRAS_API_KEY); without NVIDIA's, the chain simply starts at whichever
    free-tier provider is configured.

    Args:
        text: The text to translate.
        target_language: Language to translate into, e.g. "Turkish", "Spanish".
    """
    if not _any_provider_configured(EXTRA_PROVIDERS):
        return _no_provider_message("translate_text", EXTRA_PROVIDERS)

    messages = [{"role": "user", "content": f"Translate to {target_language}: {text}"}]
    result = await _multi_provider_chat(TRANSLATE_MODELS, messages)

    if result is None:
        return "All translation models/providers failed or timed out."
    content, model = result
    return f"{content}\n\n(model: {model})"


@mcp.tool()
async def ask_llm(
    question: Annotated[str, Field(description="The question or task to send.")],
    system_prompt: Annotated[str | None, Field(description="Optional system instruction.")] = None,
) -> str:
    """Ask a question to an alternative LLM (not Claude) via NVIDIA NIM, with
    automatic fallback to Groq/Mistral/Gemini/Cerebras if those API keys are set.

    Useful for a second opinion, or when you specifically want a non-Anthropic
    model's answer. Tries several strong models/providers in order as fallback.

    Needs NVIDIA_API_KEY *or* any one free-tier key (GROQ/MISTRAL/GEMINI/
    CEREBRAS_API_KEY).

    Args:
        question: The question or task to send.
        system_prompt: Optional system instruction.
    """
    if not _any_provider_configured(EXTRA_PROVIDERS):
        return _no_provider_message("ask_llm", EXTRA_PROVIDERS)

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": question})

    result = await _multi_provider_chat(LLM_MODELS, messages, max_tokens=2048)

    if result is None:
        return "All LLM fallback models/providers failed or timed out."
    content, model = result
    return f"{content}\n\n(model: {model})"


@mcp.tool()
async def describe_image(
    image_path: Annotated[
        str, Field(description="Absolute path to a local JPEG, PNG or WebP image (at most 10 MB).")
    ],
    question: Annotated[str, Field(description="What to ask about the image.")] = "Describe this image in detail.",
) -> str:
    """Analyze/describe a local image using an NVIDIA NIM vision-language model.

    Falls back to a free-tier vision-capable provider (Groq/Mistral/Gemini,
    whichever is configured in .env) if both NVIDIA vision models fail.

    Needs NVIDIA_API_KEY *or* one of GROQ/MISTRAL/GEMINI_API_KEY - there is no
    keyless vision tier, so with none of them set this fails fast and says so.

    Args:
        image_path: Absolute path to a local image file (jpg/png/webp, at
            most 10 MB). The bytes must really be one of those formats.
        question: What to ask about the image.
    """
    # Checked before the file is even opened: there is no keyless vision tier,
    # so with nothing configured the honest answer is "configure a provider",
    # not a pointless read of the caller's file.
    if not _any_provider_configured(VISION_PROVIDERS):
        return _no_provider_message("describe_image", VISION_PROVIDERS)

    path = Path(image_path)
    if not path.is_file():
        return f"File not found: {image_path}"

    ext = path.suffix.lstrip(".").lower()
    if ext not in DESCRIBE_IMAGE_ALLOWED_EXTENSIONS:
        return (
            f"Not an image file this tool will upload: {image_path} "
            f"(allowed extensions: {', '.join(sorted(DESCRIBE_IMAGE_ALLOWED_EXTENSIONS))}). "
            "The file's bytes would be sent to a third-party API, so only "
            "recognized image types are accepted."
        )
    size = path.stat().st_size
    if size > DESCRIBE_IMAGE_MAX_BYTES:
        return (
            f"File too large to upload: {image_path} is {size} bytes; the limit is "
            f"{DESCRIBE_IMAGE_MAX_BYTES} (base64 inflates it by a third on top)."
        )
    # Blocking disk read - off the event loop, same rule as every other
    # file access in this module.
    try:
        data = await asyncio.to_thread(_read_image_bytes, path)
    except OSError as e:
        return f"Cannot read {image_path}: {e.strerror or e}"
    if data is None:
        return f"File too large to upload: {image_path} grew past {DESCRIBE_IMAGE_MAX_BYTES} bytes while being read."
    # The extension alone is only a filename: check the bytes before they
    # leave the machine, so a renamed key or document is never uploaded.
    # The sniffed format also names the data URL correctly for the common
    # case of a JPEG saved as .png.
    mime = _sniff_image_format(data)
    if mime not in DESCRIBE_IMAGE_ALLOWED_FORMATS:
        return (
            f"Not uploading {image_path}: its contents are not a JPEG, PNG or WebP image "
            "(checked by file signature, not by extension)."
        )
    img_b64 = base64.b64encode(data).decode()

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": question},
                {"type": "image_url", "image_url": {"url": f"data:image/{mime};base64,{img_b64}"}},
            ],
        }
    ]

    async with httpx2.AsyncClient() as client:
        result = await _chat_with_fallback(client, VISION_MODELS, messages, max_tokens=512)

    if result is None:
        result = await _chat_via_provider_chain(VISION_PROVIDERS, messages, max_tokens=512)

    if result is None:
        return "All vision models failed or timed out."
    content, model = result
    return f"{content}\n\n(model: {model})"


@mcp.tool()
async def check_content_safety(text: Annotated[str, Field(description="The text to check.")]) -> str:
    """Check whether text is safe/appropriate using NVIDIA's content-safety NIM.

    Useful before publishing user-generated content, comments, or chat messages
    in a project. Returns the model's safe/unsafe verdict.

    Falls back to a best-effort classification prompt sent to a general-purpose
    free-tier chat model (Groq/Mistral/Gemini/Cerebras, whichever is configured)
    if the dedicated NVIDIA safety model fails - clearly labeled as a fallback
    verdict, since a general chat model is less calibrated for this than a
    purpose-built classifier.

    Needs NVIDIA_API_KEY *or* any one free-tier key (GROQ/MISTRAL/GEMINI/
    CEREBRAS_API_KEY).

    Args:
        text: The text to check.
    """
    if not _any_provider_configured(EXTRA_PROVIDERS):
        return _no_provider_message("check_content_safety", EXTRA_PROVIDERS)

    messages = [{"role": "user", "content": text}]
    async with httpx2.AsyncClient() as client:
        result = await _chat_with_fallback(client, [SAFETY_MODEL], messages, max_tokens=100)

    if result is not None:
        content, _ = result
        return content

    fallback_messages = [
        {
            "role": "system",
            "content": (
                "Classify the following user-supplied text as SAFE or UNSAFE "
                "(hate speech, harassment, sexual content involving minors, "
                "violent extremism, or comparable serious harms). Respond "
                "with only the verdict word and a one-line reason."
            ),
        },
        {"role": "user", "content": text},
    ]
    fallback_result = await _chat_via_provider_chain(EXTRA_PROVIDERS, fallback_messages, max_tokens=100)
    if fallback_result is None:
        return "Content safety check failed."
    content, model = fallback_result
    return f"{content}\n\n(best-effort fallback verdict from {model}, not the dedicated NVIDIA safety model)"


@mcp.tool()
async def create_embedding(text: Annotated[str, Field(description="The text to embed.")]) -> str:
    """Create a semantic embedding vector for text, for search/RAG use cases.

    Saves the vector to a local JSON file (too large to return inline) and
    reports its dimensionality.

    Falls back to a fully local, keyless embedding model (sentence-transformers'
    all-MiniLM-L6-v2) if the NVIDIA endpoint fails - only available if the
    optional `local-embeddings` extra is installed (`uv sync --extra
    local-embeddings`); otherwise the failure is reported plainly.

    Works with NO API key at all once that extra is installed: without
    NVIDIA_API_KEY the hosted tier is skipped and the local model is used
    directly.

    Args:
        text: The text to embed.
    """
    # No NVIDIA_API_KEY is not a reason to fail: the local sentence-
    # transformers tier below is keyless, so an unkeyed caller starts there.
    api_key = _api_key()
    body = {"input": [text], "model": EMBED_MODEL, "input_type": "query"}
    resp = None
    request_error = None
    if api_key:
        async with httpx2.AsyncClient() as client:
            try:
                resp = await client.post(EMBED_URL, headers=_headers(), json=body, timeout=30.0)
            except Exception as e:
                # Broad on purpose: a timeout is only one of several ways this
                # request can fail (connection refused, DNS failure, TLS error,
                # ...) and every one of them should still trigger the local
                # fallback below, not crash the tool.
                request_error = _redact(str(e), api_key)
    else:
        request_error = f"{NVIDIA_API_KEY_ENV} not set in .env - NVIDIA tier skipped"

    vector = None
    if resp is not None and resp.status_code == 200:
        try:
            vector = resp.json()["data"][0]["embedding"]
        except (KeyError, IndexError, TypeError, ValueError):
            # A 200 that is not the documented shape (or not JSON at all)
            # is a failed tier like any other: fall through to local.
            request_error = "malformed response from the NVIDIA embedding endpoint"
            resp = None
    if vector is not None:
        model_used = EMBED_MODEL
    else:
        vector = await asyncio.to_thread(_local_embedding, text)
        model_used = f"local:{_LOCAL_EMBED_MODEL_NAME}"

    if vector is None:
        detail = (
            f"HTTP {resp.status_code} - {_redact(resp.text[:300], api_key)}"
            if resp is not None
            else request_error
        )
        return (
            f"NVIDIA embedding failed ({detail}) and no local fallback available "
            "(run `uv sync --extra local-embeddings` to enable one)."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    filepath = OUTPUT_DIR / f"embedding_{_stamp()}.json"
    # Blocking disk write - off the event loop, like the _local_embedding
    # call above already is.
    await asyncio.to_thread(
        filepath.write_text, json.dumps({"text": text, "model": model_used, "vector": vector})
    )

    return f"Embedding saved to {filepath} ({len(vector)} dimensions, model: {model_used})"


@mcp.tool()
async def check_provider_health() -> str:
    """Check which configured NVIDIA models and cross-provider fallbacks are
    currently reachable, without generating any real content.

    Runs a minimal, single-token liveness probe against every unique NVIDIA
    model used across the other six tools' fallback chains (plus every
    free-tier provider - Groq/Mistral/Gemini/Cerebras - that has an API key
    set, generate_image's Pollinations.ai tier, and describe_image's
    vision-specific provider chain), all concurrently with a short timeout.
    Failures are caught and reported per model instead of raising, so one
    dead model never hides the status of the others. Models get silently
    rate-limited or retired on NVIDIA's platform without notice - use this
    to see what's actually alive right now instead of only discovering a
    dead model when a real request from one of the other tools fails.

    Runs with or without NVIDIA_API_KEY: unset, the NVIDIA rows report
    "not configured" and only the keyless/free-tier probes actually fire.
    """
    # Deliberately NOT gated on NVIDIA_API_KEY: a diagnostic that refuses to
    # run without a key is useless to exactly the caller who needs it. The
    # NVIDIA rows report "not configured" instead of being probed.
    api_key = _api_key()

    chat_models = sorted(set(TRANSLATE_MODELS) | set(LLM_MODELS) | set(VISION_MODELS) | {SAFETY_MODEL})
    image_slugs = [m["slug"] for m in IMAGE_MODELS]
    sem = asyncio.Semaphore(HEALTH_PROBE_CONCURRENCY)

    # EXTRA_PROVIDERS and VISION_PROVIDERS share one entry (Gemini's
    # gemini-flash-latest is both the text and the vision fallback for that
    # provider) - dedupe by model so it's probed once, not twice against the
    # same free-tier rate limit for the same answer.
    free_providers = list({p["model"]: p for p in (*EXTRA_PROVIDERS, *VISION_PROVIDERS)}.values())

    async with httpx2.AsyncClient() as client:
        if api_key:
            chat_probes = [_bounded(sem, _probe_nvidia_chat_model(client, m)) for m in chat_models]
            image_probes = [_bounded(sem, _probe_nvidia_image_model(client, s)) for s in image_slugs]
            embed_probe = _bounded(sem, _probe_nvidia_embed_model(client, EMBED_MODEL))
        else:
            # Probing without a key would send "Bearer None" and report the
            # same uninformative 401 for every model - say what is actually
            # wrong instead, and spend no requests doing it.
            detail = f"not configured ({NVIDIA_API_KEY_ENV} not set)"
            chat_probes = [_skipped(detail) for _ in chat_models]
            image_probes = [_skipped(detail) for _ in image_slugs]
            embed_probe = _skipped(detail)

        chat_results, image_results, embed_result, pollinations_result, free_provider_results = await asyncio.gather(
            asyncio.gather(*chat_probes),
            asyncio.gather(*image_probes),
            embed_probe,
            _bounded(sem, _probe_pollinations(client)),
            asyncio.gather(*(_bounded(sem, _probe_extra_provider(p)) for p in free_providers)),
        )

    chat_status = dict(zip(chat_models, chat_results))
    image_status = dict(zip(image_slugs, image_results))
    free_provider_status = dict(zip((p["model"] for p in free_providers), free_provider_results))

    def fmt_group(lines: list[str], title: str, models: list[str], status: dict) -> None:
        lines.append(f"{title}:")
        for m in models:
            ok, detail = status[m]
            lines.append(f"  {'OK ' if ok else 'FAIL'} {m} - {detail}")

    lines: list[str] = ["NVIDIA NIM provider health check:", ""]
    fmt_group(lines, "generate_image", image_slugs, image_status)
    poll_ok, poll_detail = pollinations_result
    lines.append(f"  {'OK ' if poll_ok else 'FAIL'} pollinations (fallback) - {poll_detail}")
    fmt_group(lines, "translate_text", TRANSLATE_MODELS, chat_status)
    fmt_group(lines, "ask_llm", LLM_MODELS, chat_status)
    fmt_group(lines, "describe_image", VISION_MODELS, chat_status)
    fmt_group(lines, "check_content_safety", [SAFETY_MODEL], chat_status)

    embed_ok, embed_detail = embed_result
    lines.append("create_embedding:")
    lines.append(f"  {'OK ' if embed_ok else 'FAIL'} {EMBED_MODEL} - {embed_detail}")

    lines.append("")
    lines.append("Cross-provider fallback (translate_text / ask_llm / check_content_safety):")
    for provider in EXTRA_PROVIDERS:
        ok, detail = free_provider_status[provider["model"]]
        lines.append(f"  {'OK ' if ok else 'FAIL'} {provider['model']} ({provider['env']}) - {detail}")

    lines.append("")
    lines.append("Cross-provider fallback (describe_image only):")
    for provider in VISION_PROVIDERS:
        ok, detail = free_provider_status[provider["model"]]
        lines.append(f"  {'OK ' if ok else 'FAIL'} {provider['model']} ({provider['env']}) - {detail}")

    return "\n".join(lines)


def main() -> None:
    """Console entry point.

    A separate function because `[project.scripts]` wants a CALLABLE, not a
    module. Without it the package installs but cannot be run: the user would
    have to clone the repository and point at the file, which defeats the
    point of publishing it.
    """
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
