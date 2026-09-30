"""``python -m memory_condense.context_proxy`` -- run the local context proxy."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .policy import PassthroughPolicy, RecencyWindowPolicy
from .server import ProxyConfig, Upstream, make_server

PRESETS = {
    # client            upstream the proxy forwards to
    "claude": "https://api.anthropic.com",
    "codex-chatgpt": "https://chatgpt.com/backend-api/codex",
    "codex-api": "https://api.openai.com/v1",
}

DEFAULT_PORTS = {"claude": 8787, "codex-chatgpt": 8788, "codex-api": 8788}


def client_instructions(preset: str, port: int) -> str:
    url = f"http://127.0.0.1:{port}"
    if preset == "claude":
        return (
            f"Claude Code:  set ANTHROPIC_BASE_URL={url}\n"
            "  Leave ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN unset to keep your claude.ai login;\n"
            "  the OAuth capability rides in anthropic-beta, which this proxy forwards verbatim.\n"
            "  Optional: CLAUDE_CODE_ATTRIBUTION_HEADER=0 (the system array is forwarded untouched, so not required)."
        )
    auth = "true" if preset == "codex-chatgpt" else "false"
    return (
        "Codex: add to ~/.codex/config.toml (model_provider is a top-level key, above any table)\n"
        '  model_provider = "local_proxy"\n'
        "\n"
        "  [model_providers.local_proxy]\n"
        '  name = "local context proxy"\n'
        f'  base_url = "{url}"\n'
        '  wire_api = "responses"\n'
        f"  requires_openai_auth = {auth}\n"
        + ('  env_key = "OPENAI_API_KEY"\n' if preset == "codex-api" else "")
        + "  Note: with OPENAI_API_KEY exported Codex ignores custom providers under ChatGPT login."
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="memory_condense.context_proxy")
    p.add_argument("--preset", choices=sorted(PRESETS), default="claude")
    p.add_argument("--upstream", help="override the preset's upstream URL")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int)
    p.add_argument("--policy", choices=["passthrough", "recency"], default="passthrough",
                   help="passthrough = forward unchanged (validate plumbing, measure); recency = condense")
    p.add_argument("--budget", type=int, default=60_000, help="token budget for the recency policy")
    p.add_argument("--ledger", type=Path, help="append one JSON line per request here")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--print-env", action="store_true", help="print client configuration and exit")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    port = args.port or DEFAULT_PORTS[args.preset]
    if args.print_env:
        print(client_instructions(args.preset, port))
        return 0
    policy = RecencyWindowPolicy(budget_tokens=args.budget) if args.policy == "recency" else PassthroughPolicy()
    upstream = Upstream.parse(args.upstream or PRESETS[args.preset])
    locator = None
    if args.policy != "passthrough":
        from .bridge import SessionLocator

        locator = SessionLocator()
    config = ProxyConfig(upstream=upstream, policy=policy, ledger_path=args.ledger, quiet=args.quiet, locator=locator)
    server = make_server(args.host, port, config)
    print(
        f"context proxy on http://{args.host}:{port} -> {upstream.scheme}://{upstream.host}{upstream.base_path}"
        f"  policy={policy.name}" + (f" budget={args.budget}" if args.policy == "recency" else ""),
        file=sys.stderr,
    )
    print(client_instructions(args.preset, port), file=sys.stderr)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0
