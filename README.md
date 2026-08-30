# nvidia-nim

An MCP (Model Context Protocol) server that gives Claude Code access to [NVIDIA NIM](https://build.nvidia.com/)'s free-tier models — image generation, translation, LLM chat, vision, content safety, and embeddings — through one consistent set of tools.

Every capability tries multiple models in order, so one being slow, rate-limited, or retired doesn't take the tool down. The caller never needs to know which model actually answered.

## Tools

| Tool | Does |
|---|---|
| `generate_image` | Text-to-image via FLUX models, saved locally |
| `translate_text` | Translation with automatic model fallback |
| `ask_llm` | Ask a non-Anthropic model for a second opinion |
| `describe_image` | Vision-language description of a local image |
| `check_content_safety` | Safe/unsafe verdict on a piece of text |
| `create_embedding` | Semantic embedding vector for search/RAG |

## Fallback chain

Each capability tries NVIDIA's models first, then falls back to any other free-tier provider whose API key is present in `.env` — currently Groq, Mistral, and Gemini. Adding a key later joins the chain automatically, no code changes needed.

## Setup

```bash
uv sync
```

Create a `.env` file in the project root with at least:

```
NVIDIA_API_KEY=your-key-here
```

Get a free key at [build.nvidia.com](https://build.nvidia.com/). Optionally add `GROQ_API_KEY`, `MISTRAL_API_KEY`, `GEMINI_API_KEY`, or `CEREBRAS_API_KEY` to extend the fallback chain — each is skipped silently if unset.

Register it as an MCP server (project or user scope):

```bash
claude mcp add --transport stdio nvidia-nim -- uv run --project /path/to/this/repo nvidia_image.py
```

## License

MIT
