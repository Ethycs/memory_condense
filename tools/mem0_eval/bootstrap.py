"""Standard-library bootstrap for the isolated Mem0 Pixi environment."""

from __future__ import annotations

import argparse
import hashlib
import os
import runpy
import socket
import sys
from pathlib import Path
from typing import Sequence


def _tree_sha256(package: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(package.rglob("*.py"), key=lambda item: item.as_posix()):
        relative = path.relative_to(package).as_posix().encode("utf-8")
        payload = path.read_bytes()
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _deny_network() -> None:
    def blocked(*_args, **_kwargs):
        raise RuntimeError("network access is disabled for provider-free preflight")

    socket.create_connection = blocked  # type: ignore[assignment]
    socket.socket.connect = blocked  # type: ignore[method-assign]
    socket.socket.connect_ex = blocked  # type: ignore[method-assign]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify frozen source, then launch an isolated Mem0 tool"
    )
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--expected-tool-sha256")
    parser.add_argument("--allow-network", action="store_true")
    parser.add_argument("--module", required=True)
    parser.add_argument("args", nargs=argparse.REMAINDER)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    repo = Path(args.repository_root).resolve()
    source_package = repo / "src" / "memory_condense"
    tool_package = repo / "tools" / "mem0_eval"
    if not source_package.is_dir() or not tool_package.is_dir():
        raise RuntimeError("repository does not contain the expected source/tool trees")
    source_digest = _tree_sha256(source_package)
    if source_digest != args.expected_source_sha256:
        raise RuntimeError(
            "frozen memory-condense source mismatch: "
            f"{source_digest} != {args.expected_source_sha256}"
        )
    if args.expected_tool_sha256:
        tool_digest = _tree_sha256(tool_package)
        if tool_digest != args.expected_tool_sha256:
            raise RuntimeError(
                "frozen Mem0 tool mismatch: "
                f"{tool_digest} != {args.expected_tool_sha256}"
            )

    os.environ["MEM0_TELEMETRY"] = "false"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "true"
    if not args.allow_network:
        _deny_network()
    sys.path.insert(0, str(repo / "src"))
    sys.path.insert(0, str(repo))
    forwarded = list(args.args)
    if forwarded[:1] == ["--"]:
        forwarded = forwarded[1:]
    sys.argv = [args.module, *forwarded]
    runpy.run_module(args.module, run_name="__main__", alter_sys=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
