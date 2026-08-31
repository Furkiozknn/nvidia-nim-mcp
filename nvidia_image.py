import asyncio
import base64
import json
import logging
import os
from datetime import datetime
from pathlib import Path

import httpx2
import litellm
from dotenv import load_dotenv
from mcp.server import MCPServer

load_dotenv()

logger = logging.getLogger(__name__)

mcp = MCPServer("nvidia-nim")

API_KEY = os.environ.get("NVIDIA_API_KEY")
GENAI_BASE = "https://ai.api.nvidia.com/v1/genai"
CHAT_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
EMBED_URL = "https://integrate.api.nvidia.com/v1/embeddings"

OUTPUT_DIR = Path(__file__).parent / "output"

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
    "Accept": "application/json",
}

# Every model below was confirmed working with a real request (2026-08-22)
# before being wired in - see new-mcp-server skill for why that matters.
# Each capability has 2+ models so one being rate-limited/slow/congested
# doesn't take the tool down; the caller never needs to know which one answered.

IMAGE_MODELS = [
    {"slug": "black-forest-labs/flux.1-dev", "cfg_scale": 3.5, "steps": 25},
    {"slug": "black-forest-labs/flux.2-klein-4b", "cfg_scale": 1, "steps": 4},
]

TRANSLATE_MODELS = [
    "nvidia/riva-translate-4b-instruct-v2",
    "nvidia/llama-3.3-nemotron-super-49b-v1.5",
    "openai/gpt-oss-120b",
]

LLM_MODELS = [
    # glm-5.2 and deepseek-v4-flash were removed 2026-08-22 - both confirmed
    # permanently retired on NVIDIA's platform (HTTP 410, "reached its end
    # of life"), not just rate-limited. Verified via litellm before removing -
    # see systematic-debugging skill for why that check matters here.
    "nvidia/llama-3.3-nemotron-super-49b-v1.5",
    "openai/gpt-oss-120b",
]

# Free-tier providers beyond NVIDIA (added 2026-08-22). Each is skipped
# automatically if its API key isn't set in .env yet - paste a key in and it
# joins the fallback chain on the next call, no code changes needed. Picked
# because each has a genuinely permanent free tier, no credit card:
# Gemini (aistudio.google.com/apikey), Groq (console.groq.com/keys),
# Mistral (console.mistral.ai/api-keys). OpenAI has no ongoing free tier as
# of 2026 (only an expiring trial credit) so it's deliberately not here.
EXTRA_PROVIDERS = [
    # Ordered by what was actually confirmed working with a real call on
    # 2026-08-22 - model names drift fast, always verify against the
    # provider's live /models endpoint before trusting a name from memory.
    {"env": "GROQ_API_KEY", "model": "groq/openai/gpt-oss-120b"},  # confirmed working
    {"env": "MISTRAL_API_KEY", "model": "mistral/mistral-small-latest"},  # confirmed working
    # Blocked: Google wants a paid plan upgrade on this project to lift the
    # PERMISSION_DENIED. Not a code bug - user deliberately declined the
    # upgrade (2026-08-22), fine with Groq+Mistral as the free backup for now.
    # Left in harmlessly (litellm just skips a failing entry and moves on);
    # revisit only if the user brings it up again.
    {"env": "GEMINI_API_KEY", "model": "gemini/gemini-flash-latest"},
    # Cerebras now requires billing ("Payment required") - not actually a
    # free tier despite earlier research suggesting otherwise. Kept last as
    # a deliberate last resort; remove entirely if the user doesn't want to
    # add a payment method there.
    {"env": "CEREBRAS_API_KEY", "model": "cerebras/gpt-oss-120b"},
]


def _build_chat_chain(nvidia_models: list[str]) -> list[dict]:
    """NVIDIA models first (need the custom api_base), then any extra free-tier
    provider whose key is actually present in the environment right now."""
    chain = [
        {"model": f"openai/{m}", "api_base": CHAT_URL.rsplit("/chat/completions", 1)[0], "api_key": API_KEY}
        for m in nvidia_models
    ]
    for provider in EXTRA_PROVIDERS:
        key = os.environ.get(provider["env"])
        if key:
            chain.append({"model": provider["model"], "api_key": key})
    return chain


async def _multi_provider_chat(nvidia_models: list[str], messages: list[dict], max_tokens: int = 1024) -> tuple[str, str] | None:
    """Try NVIDIA models, then any configured free-tier provider, in order."""
    chain = _build_chat_chain(nvidia_models)
    if not chain:
        return None
    primary, fallbacks = chain[0], chain[1:]
    try:
        response = await litellm.acompletion(
            messages=messages,
            max_tokens=max_tokens,
            fallbacks=fallbacks or None,
            **primary,
        )
    except Exception as e:
        logger.warning("all providers in chain failed: %s", e)
        return None
    return response.choices[0].message.content, response.model

VISION_MODELS = [
    "nvidia/nemotron-nano-12b-v2-vl",
    "meta/llama-3.2-11b-vision-instruct",
]

EMBED_MODEL = "nvidia/nemotron-3-embed-1b"
SAFETY_MODEL = "nvidia/nemotron-3.5-content-safety"


async def _chat_with_fallback(client: httpx2.AsyncClient, models: list[str], messages: list[dict], max_tokens: int = 1024) -> tuple[str, str] | None:
    """Try each model in order, return (content, model_used) from the first success."""
    for model in models:
        body = {"model": model, "messages": messages, "max_tokens": max_tokens}
        try:
            resp = await client.post(CHAT_URL, headers=HEADERS, json=body, timeout=18.0)
        except httpx2.TimeoutException:
            continue
        if resp.status_code != 200:
            continue
        try:
            content = resp.json()["choices"][0]["message"]["content"]
        except (KeyError, IndexError, json.JSONDecodeError):
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
        resp = await client.post(CHAT_URL, headers=HEADERS, json=body, timeout=HEALTH_PROBE_TIMEOUT)
    except httpx2.TimeoutException:
        return False, "timed out"
    except Exception as e:
        return False, f"error: {e}"
    if resp.status_code != 200:
        return False, f"HTTP {resp.status_code}"
    return True, "ok"


async def _probe_nvidia_image_model(client: httpx2.AsyncClient, slug: str) -> tuple[bool, str]:
    """Cheap liveness probe for one NVIDIA image model: a single-step,
    tiny-resolution generation instead of a full-quality image."""
    body = {"prompt": "hi", "steps": 1, "cfg_scale": 1, "seed": 0, "width": 64, "height": 64}
    try:
        resp = await client.post(f"{GENAI_BASE}/{slug}", headers=HEADERS, json=body, timeout=HEALTH_PROBE_TIMEOUT)
    except httpx2.TimeoutException:
        return False, "timed out"
    except Exception as e:
        return False, f"error: {e}"
    if resp.status_code != 200:
        return False, f"HTTP {resp.status_code}"
    return True, "ok"


async def _probe_nvidia_embed_model(client: httpx2.AsyncClient, model: str) -> tuple[bool, str]:
    """Cheap liveness probe for the NVIDIA embedding model: a one-word input."""
    body = {"input": ["hi"], "model": model, "input_type": "query"}
    try:
        resp = await client.post(EMBED_URL, headers=HEADERS, json=body, timeout=HEALTH_PROBE_TIMEOUT)
    except httpx2.TimeoutException:
        return False, "timed out"
    except Exception as e:
        return False, f"error: {e}"
    if resp.status_code != 200:
        return False, f"HTTP {resp.status_code}"
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
        return False, f"error: {e}"
    return True, "ok"


@mcp.tool()
async def generate_image(prompt: str, seed: int = 0, width: int = 1024, height: int = 1024) -> str:
    """Generate an image from a text prompt using NVIDIA NIM image models.

    Tries multiple models in order (flux.1-dev, then flux.2-klein-4b) so a
    single model being rate-limited or slow doesn't block generation.

    Args:
        prompt: Description of the image to generate.
        seed: Seed for reproducibility.
        width: Image width in pixels.
        height: Image height in pixels.
    """
    if not API_KEY:
        return "NVIDIA_API_KEY not set in .env - can't call the API."

    errors = []
    async with httpx2.AsyncClient() as client:
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
                resp = await client.post(f"{GENAI_BASE}/{model['slug']}", headers=HEADERS, json=body, timeout=45.0)
            except httpx2.TimeoutException:
                errors.append(f"{model['slug']}: timed out")
                continue
            if resp.status_code != 200:
                errors.append(f"{model['slug']}: HTTP {resp.status_code} - {resp.text[:150]}")
                continue
            data = resp.json()
            artifacts = data.get("artifacts")
            if not artifacts:
                errors.append(f"{model['slug']}: no artifacts in response")
                continue
            img_b64 = artifacts[0]["base64"]
            OUTPUT_DIR.mkdir(exist_ok=True)
            filename = f"{model['slug'].split('/')[-1]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            filepath = OUTPUT_DIR / filename
            filepath.write_bytes(base64.b64decode(img_b64))
            return f"Image saved to {filepath} (model: {model['slug']})"

    return "All image models failed:\n" + "\n".join(errors)


@mcp.tool()
async def translate_text(text: str, target_language: str) -> str:
    """Translate text to a target language using NVIDIA NIM models.

    Tries a dedicated translation model first, falling back to general NVIDIA
    chat models, then Gemini/Groq/Mistral if those API keys are configured.

    Args:
        text: The text to translate.
        target_language: Language to translate into, e.g. "Turkish", "Spanish".
    """
    if not API_KEY:
        return "NVIDIA_API_KEY not set in .env - can't call the API."

    messages = [{"role": "user", "content": f"Translate to {target_language}: {text}"}]
    result = await _multi_provider_chat(TRANSLATE_MODELS, messages)

    if result is None:
        return "All translation models/providers failed or timed out."
    content, model = result
    return f"{content}\n\n(model: {model})"


@mcp.tool()
async def ask_llm(question: str, system_prompt: str | None = None) -> str:
    """Ask a question to an alternative LLM (not Claude) via NVIDIA NIM, with
    automatic fallback to Gemini/Groq/Mistral if those API keys are set.

    Useful for a second opinion, or when you specifically want a non-Anthropic
    model's answer. Tries several strong models/providers in order as fallback.

    Args:
        question: The question or task to send.
        system_prompt: Optional system instruction.
    """
    if not API_KEY:
        return "NVIDIA_API_KEY not set in .env - can't call the API."

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
async def describe_image(image_path: str, question: str = "Describe this image in detail.") -> str:
    """Analyze/describe a local image using an NVIDIA NIM vision-language model.

    Args:
        image_path: Absolute path to a local image file (jpg/png).
        question: What to ask about the image.
    """
    if not API_KEY:
        return "NVIDIA_API_KEY not set in .env - can't call the API."

    path = Path(image_path)
    if not path.is_file():
        return f"File not found: {image_path}"

    ext = path.suffix.lstrip(".").lower()
    mime = "jpeg" if ext in ("jpg", "jpeg") else ext
    img_b64 = base64.b64encode(path.read_bytes()).decode()

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
        return "All vision models failed or timed out."
    content, model = result
    return f"{content}\n\n(model: {model})"


@mcp.tool()
async def check_content_safety(text: str) -> str:
    """Check whether text is safe/appropriate using NVIDIA's content-safety NIM.

    Useful before publishing user-generated content, comments, or chat messages
    in a project. Returns the model's safe/unsafe verdict.

    Args:
        text: The text to check.
    """
    if not API_KEY:
        return "NVIDIA_API_KEY not set in .env - can't call the API."

    messages = [{"role": "user", "content": text}]
    async with httpx2.AsyncClient() as client:
        result = await _chat_with_fallback(client, [SAFETY_MODEL], messages, max_tokens=100)

    if result is None:
        return "Content safety check failed."
    content, _ = result
    return content


@mcp.tool()
async def create_embedding(text: str) -> str:
    """Create a semantic embedding vector for text, for search/RAG use cases.

    Saves the vector to a local JSON file (too large to return inline) and
    reports its dimensionality.

    Args:
        text: The text to embed.
    """
    if not API_KEY:
        return "NVIDIA_API_KEY not set in .env - can't call the API."

    body = {"input": [text], "model": EMBED_MODEL, "input_type": "query"}
    async with httpx2.AsyncClient() as client:
        try:
            resp = await client.post(EMBED_URL, headers=HEADERS, json=body, timeout=30.0)
        except httpx2.TimeoutException:
            return f"{EMBED_MODEL}: timed out"

    if resp.status_code != 200:
        return f"Request failed: HTTP {resp.status_code} - {resp.text[:300]}"

    data = resp.json()
    vector = data["data"][0]["embedding"]

    OUTPUT_DIR.mkdir(exist_ok=True)
    filename = f"embedding_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    filepath = OUTPUT_DIR / filename
    filepath.write_text(json.dumps({"text": text, "model": EMBED_MODEL, "vector": vector}))

    return f"Embedding saved to {filepath} ({len(vector)} dimensions)"


@mcp.tool()
async def check_provider_health() -> str:
    """Check which configured NVIDIA models and cross-provider fallbacks are
    currently reachable, without generating any real content.

    Runs a minimal, single-token liveness probe against every unique NVIDIA
    model used across the other six tools' fallback chains (plus every
    free-tier provider - Groq/Mistral/Gemini/Cerebras - that has an API key
    set), all concurrently with a short timeout. Failures are caught and
    reported per model instead of raising, so one dead model never hides the
    status of the others. Models get silently rate-limited or retired on
    NVIDIA's platform without notice - use this to see what's actually alive
    right now instead of only discovering a dead model when a real request
    from one of the other tools fails.
    """
    if not API_KEY:
        return "NVIDIA_API_KEY not set in .env - can't call the API."

    chat_models = sorted(set(TRANSLATE_MODELS) | set(LLM_MODELS) | set(VISION_MODELS) | {SAFETY_MODEL})
    image_slugs = [m["slug"] for m in IMAGE_MODELS]
    sem = asyncio.Semaphore(HEALTH_PROBE_CONCURRENCY)

    async with httpx2.AsyncClient() as client:
        chat_results, image_results, embed_result, extra_results = await asyncio.gather(
            asyncio.gather(*(_bounded(sem, _probe_nvidia_chat_model(client, m)) for m in chat_models)),
            asyncio.gather(*(_bounded(sem, _probe_nvidia_image_model(client, s)) for s in image_slugs)),
            _bounded(sem, _probe_nvidia_embed_model(client, EMBED_MODEL)),
            asyncio.gather(*(_bounded(sem, _probe_extra_provider(p)) for p in EXTRA_PROVIDERS)),
        )

    chat_status = dict(zip(chat_models, chat_results))
    image_status = dict(zip(image_slugs, image_results))

    def fmt_group(lines: list[str], title: str, models: list[str], status: dict) -> None:
        lines.append(f"{title}:")
        for m in models:
            ok, detail = status[m]
            lines.append(f"  {'OK ' if ok else 'FAIL'} {m} - {detail}")

    lines: list[str] = ["NVIDIA NIM provider health check:", ""]
    fmt_group(lines, "generate_image", image_slugs, image_status)
    fmt_group(lines, "translate_text", TRANSLATE_MODELS, chat_status)
    fmt_group(lines, "ask_llm", LLM_MODELS, chat_status)
    fmt_group(lines, "describe_image", VISION_MODELS, chat_status)
    fmt_group(lines, "check_content_safety", [SAFETY_MODEL], chat_status)

    embed_ok, embed_detail = embed_result
    lines.append("create_embedding:")
    lines.append(f"  {'OK ' if embed_ok else 'FAIL'} {EMBED_MODEL} - {embed_detail}")

    lines.append("")
    lines.append("Cross-provider fallback (translate_text / ask_llm only):")
    for provider, (ok, detail) in zip(EXTRA_PROVIDERS, extra_results):
        lines.append(f"  {'OK ' if ok else 'FAIL'} {provider['model']} ({provider['env']}) - {detail}")

    return "\n".join(lines)


if __name__ == "__main__":
    mcp.run(transport="stdio")
