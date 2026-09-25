# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-09-25

First tagged release: the first version published to PyPI (`nvidia-nim-mcp`)
and to the MCP Registry (`io.github.Furkiozknn/nvidia-nim-mcp`).

### Tools

- `generate_image`: FLUX on NVIDIA (`flux.1-dev`, then `flux.2-klein-4b`),
  then keyless Pollinations.ai. Works with no API key.
- `translate_text`, `ask_llm`: NVIDIA chat models, then Groq, Mistral,
  Gemini and Cerebras, each used only when its key is set.
- `describe_image`: NVIDIA vision models, then Groq, Mistral or Gemini vision
  models.
- `check_content_safety`: NVIDIA's content-safety model, then a labelled
  best-effort verdict from a chat model.
- `create_embedding`: NVIDIA embeddings, then a local sentence-transformers
  model (`local-embeddings` extra). Works with no API key once the extra is
  installed.
- `check_provider_health`: a cheap liveness probe of every model and fallback
  above. A retired model is reported as `HTTP 410 - model retired`.

### Fixed before release

- Cross-provider fallback: with `NVIDIA_API_KEY` set, litellm's `fallbacks=`
  copied NVIDIA's `api_base` into the Groq/Mistral/Gemini/Cerebras entries.
  Their keys were sent to NVIDIA's endpoint, and the fallback never answered.
  Each chain entry is now its own call with its own settings.
- Each chain entry is tried once (`max_retries=0`). Before, litellm retried
  every NVIDIA entry twice with backoff before the next model got a turn.
- An empty reply (a reasoning model that used its whole token budget) moves
  on to the next provider instead of returning the text `None`.
- A connection, DNS or TLS error, a non-JSON `200` or a malformed `200` counts
  as a failed model and falls through to the next tier. Before, it crashed the
  tool.
- Provider keys are scrubbed from error text and from the server's log,
  whichever provider's error it is.
- `generate_image`: a prompt containing `/` (or `../`) is sent as one URL path
  segment. Before, `../../models` reached a different path on the Pollinations
  host. An empty prompt and sizes outside 64-2048 are rejected by the input
  schema.
- `describe_image` checks the file signature, not only the extension, so a
  renamed non-image is never uploaded. The 10 MB cap is enforced on the read
  itself, and an unreadable file returns a message instead of crashing.
- Documentation: `claude mcp add ... uv run --directory` (not `--project`);
  `.env` is read once at startup (the README claimed changes took effect
  without a restart); Gemini and Cerebras are documented as optional, and
  Cerebras as not free.
- `server.json` fits the MCP Registry schema (description at most 100
  characters), and `uv.lock` matches `pyproject.toml` (`uv sync --locked`).

[0.1.0]: https://github.com/Furkiozknn/nvidia-nim-mcp/releases/tag/v0.1.0
