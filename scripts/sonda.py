#!/usr/bin/env python3
"""Start an MCP server over stdio and check it answers: initialize, tools/list,
and optionally call tools. Answers "did my server start, and what does it offer?"
without an MCP client.

    python scripts/sonda.py [--no-keys] [--server-log] [--call TOOL 'JSON'] ... -- COMMAND [ARGS...]

    python scripts/sonda.py -- nvidia-nim-mcp
    python scripts/sonda.py --no-keys --call ask_llm '{"question": "hi"}' -- nvidia-nim-mcp

--no-keys        remove NVIDIA/GROQ/MISTRAL/GEMINI/CEREBRAS keys from the server's environment,
                 so nothing is sent to (or billed by) a provider.
--server-log     show the server's own stderr (start-up line, logging); hidden by default.
--call TOOL JSON call a tool with these arguments and print the result (repeatable).

Exit code: 0 the server answered; 1 a tool call returned an error; 2 usage error;
3 the server did not start or did not answer.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time

KEYS = ("NVIDIA_API_KEY", "GROQ_API_KEY", "MISTRAL_API_KEY", "GEMINI_API_KEY", "CEREBRAS_API_KEY")


def parse(argv: list[str]):
    if "--" not in argv:
        print(__doc__, file=sys.stderr)
        raise SystemExit(2)
    split = argv.index("--")
    opts, command = argv[:split], argv[split + 1:]
    if not command:
        print("sonda: nothing after `--`; give the server command", file=sys.stderr)
        raise SystemExit(2)
    calls, no_keys, server_log = [], False, False
    i = 0
    while i < len(opts):
        if opts[i] == "--call" and i + 2 < len(opts):
            try:
                calls.append((opts[i + 1], json.loads(opts[i + 2])))
            except json.JSONDecodeError as exc:
                print(f"sonda: the arguments of --call {opts[i + 1]} are not JSON ({exc})", file=sys.stderr)
                raise SystemExit(2)
            i += 3
        elif opts[i] == "--no-keys":
            no_keys, i = True, i + 1
        elif opts[i] == "--server-log":
            server_log, i = True, i + 1
        else:
            print(f"sonda: unknown or incomplete option {opts[i]!r}\n", file=sys.stderr)
            print(__doc__, file=sys.stderr)
            raise SystemExit(2)
    return calls, no_keys, server_log, command


async def run(calls, no_keys, server_log, command) -> int:
    from mcp.client import Client
    from mcp.client.stdio import StdioServerParameters, stdio_client

    env = {k: v for k, v in os.environ.items() if not (no_keys and k in KEYS)}
    env["PYTHONIOENCODING"] = "utf-8"
    params = StdioServerParameters(command=command[0], args=command[1:], env=env)
    sink = sys.stderr if server_log else open(os.devnull, "w")
    started, status = time.time(), 0
    try:
        async with Client(stdio_client(params, errlog=sink), read_timeout_seconds=180) as client:
            info = client.server_info
            name = f"{info.name} {info.version}".strip() if info else "(server sent no name)"
            print(f"initialize ok: {name}, {time.time() - started:.1f} s")
            tools = (await client.list_tools()).tools
            print(f"tools/list: {len(tools)} tools")
            for tool in tools:
                text = (tool.description or "").strip()
                print(f"  {tool.name:<22} {text.splitlines()[0] if text else '(no description)'}")
            for tool, arguments in calls:
                began = time.time()
                result = await client.call_tool(tool, arguments)
                body = "\n".join(getattr(c, "text", str(c)) for c in result.content)
                print(f"\ncall {tool} {json.dumps(arguments)}  ({time.time() - began:.1f} s"
                      f"{', ERROR' if result.is_error else ''})")
                print("  " + body.replace("\n", "\n  "))
                status = status or (1 if result.is_error else 0)
    except FileNotFoundError:
        print(f"sonda: cannot start {command[0]!r}: not found on PATH", file=sys.stderr)
        return 3
    except Exception as exc:  # the server exited, spoke something else, or timed out
        print(f"sonda: the server did not answer the MCP handshake ({type(exc).__name__}: {exc})", file=sys.stderr)
        return 3
    return status


def main() -> int:
    try:
        import mcp.client  # noqa: F401
    except ImportError:
        print("sonda: needs the `mcp` package: run it with `uv run python scripts/sonda.py ...`", file=sys.stderr)
        return 2
    return asyncio.run(run(*parse(sys.argv[1:])))


if __name__ == "__main__":
    sys.exit(main())
