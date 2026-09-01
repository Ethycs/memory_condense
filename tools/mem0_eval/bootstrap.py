"""Standard-library bootstrap for the isolated Mem0 Pixi environment."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import os
import re
import runpy
import socket
import sys
import tempfile
from pathlib import Path
from typing import Sequence


_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_CL100K_CACHE_KEY = "9b5ad71b2ce5302211f9c61530b329a4922fc6a4"
_CL100K_CACHE_SHA256 = (
    "223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7"
)


def _tree_sha256(package: Path) -> str:
    digest = hashlib.sha256()
    sources: list[Path] = []
    excluded = {".pixi", ".venv", "__pycache__"}
    for current, directories, files in os.walk(package):
        directories[:] = sorted(
            name for name in directories if name not in excluded
        )
        sources.extend(
            Path(current) / name for name in files if name.endswith(".py")
        )
    for path in sorted(sources, key=lambda item: item.as_posix()):
        relative_path = path.relative_to(package)
        relative = relative_path.as_posix().encode("utf-8")
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


def _bind_verified_tiktoken_cache() -> Path:
    """Bind LiteLLM to one hash-verified local cl100k asset.

    LiteLLM 1.96.2 replaces ``TIKTOKEN_CACHE_DIR`` with its bundled cache
    unless ``CUSTOM_TIKTOKEN_CACHE_DIR`` is set.  That bundle does not contain
    cl100k, so a frozen-v3 import otherwise attempts a download before the
    comparison tool can run.  The standard tiktoken cache is an environment
    asset, not tool source; verify its exact bytes before exposing its path.
    """

    configured = os.environ.get("MEM0_TIKTOKEN_CACHE_DIR")
    cache_dir = Path(
        configured
        if configured is not None
        else Path(tempfile.gettempdir()) / "data-gym-cache"
    ).resolve(strict=True)
    if not cache_dir.is_dir() or cache_dir.is_symlink():
        raise RuntimeError("Mem0 tiktoken cache must be a real directory")
    asset = cache_dir / _CL100K_CACHE_KEY
    if not asset.is_file() or asset.is_symlink():
        raise RuntimeError("hash-verified cl100k tokenizer asset is absent")
    observed = hashlib.sha256(asset.read_bytes()).hexdigest()
    if observed != _CL100K_CACHE_SHA256:
        raise RuntimeError(
            "cl100k tokenizer asset mismatch: "
            f"{observed} != {_CL100K_CACHE_SHA256}"
        )
    os.environ["CUSTOM_TIKTOKEN_CACHE_DIR"] = str(cache_dir)
    os.environ["TIKTOKEN_CACHE_DIR"] = str(cache_dir)
    return cache_dir


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Verify independently frozen source and Mem0-tool trees, then "
            "launch an isolated Mem0 tool"
        )
    )
    parser.add_argument(
        "--source-root",
        required=True,
        help="exact frozen src/memory_condense package directory",
    )
    parser.add_argument(
        "--tool-root",
        required=True,
        help="exact independently frozen tools/mem0_eval package directory",
    )
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--expected-tool-sha256", required=True)
    parser.add_argument("--allow-network", action="store_true")
    parser.add_argument("--module", required=True)
    parser.add_argument("args", nargs=argparse.REMAINDER)
    return parser


def _required_digest(value: str, label: str) -> str:
    digest = str(value).strip().casefold()
    if _SHA256_RE.fullmatch(digest) is None:
        raise RuntimeError(f"{label} must be a lowercase SHA-256 digest")
    return digest


def _package_import_root(
    package: Path,
    *,
    package_name: str,
    parent_name: str,
) -> Path:
    if package.name != package_name:
        raise RuntimeError(
            f"frozen package root must end in {package_name!r}: {package}"
        )
    if package.parent.name != parent_name:
        raise RuntimeError(
            f"frozen {package.name} root must descend directly from {parent_name!r}"
        )
    return package.parent if parent_name == "src" else package.parent.parent


def _trees_overlap(left: Path, right: Path) -> bool:
    return left == right or left in right.parents or right in left.parents


def _validate_tool_module(module: str, tool_package: Path) -> None:
    prefix = "tools.mem0_eval."
    if not module.startswith(prefix):
        raise RuntimeError("bootstrap module is outside the frozen Mem0 tool boundary")
    relative = module.removeprefix(prefix).split(".")
    if not relative or any(not part.isidentifier() for part in relative):
        raise RuntimeError("bootstrap module has an invalid frozen-tool path")
    module_path = tool_package.joinpath(*relative)
    if not module_path.with_suffix(".py").is_file() and not (
        module_path / "__init__.py"
    ).is_file():
        raise RuntimeError("bootstrap module is absent from the frozen Mem0 tool tree")


def _verify_bootstrap_origin(tool_package: Path) -> None:
    expected = (tool_package / "bootstrap.py").resolve(strict=True)
    if Path(__file__).resolve(strict=True) != expected:
        raise RuntimeError("bootstrap executable is outside the frozen Mem0 tool tree")


def _verify_isolated_runtime() -> None:
    if not sys.flags.isolated:
        raise RuntimeError("Mem0 bootstrap must be launched with Python -I")
    forbidden = tuple(
        name
        for name in sys.modules
        if name == "tools"
        or name == "memory_condense"
        or name.startswith("memory_condense.")
        or name == "tools.mem0_eval"
        or name.startswith("tools.mem0_eval.")
    )
    if forbidden:
        raise RuntimeError(
            "source or Mem0 tool modules were imported before frozen-tree verification"
        )


def _verify_import_resolution(
    source_package: Path,
    tool_package: Path,
) -> None:
    expected = {
        "memory_condense": source_package / "__init__.py",
        "tools.mem0_eval": tool_package / "__init__.py",
    }
    for module, expected_origin in expected.items():
        spec = importlib.util.find_spec(module)
        origin = None if spec is None or spec.origin is None else Path(spec.origin)
        if origin is None or origin.resolve(strict=True) != expected_origin.resolve(
            strict=True
        ):
            raise RuntimeError(
                f"{module} does not resolve inside its verified frozen tree"
            )


def _verify_tree(
    package: Path,
    *,
    expected_sha256: str,
    label: str,
) -> None:
    observed = _tree_sha256(package)
    if observed != expected_sha256:
        raise RuntimeError(
            f"frozen {label} mismatch: {observed} != {expected_sha256}"
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    source_package = Path(args.source_root).resolve(strict=True)
    tool_package = Path(args.tool_root).resolve(strict=True)
    if not source_package.is_dir() or not tool_package.is_dir():
        raise RuntimeError("source-root and tool-root must both be directories")
    if _trees_overlap(source_package, tool_package):
        raise RuntimeError("frozen source-root and tool-root must not overlap")
    source_import_root = _package_import_root(
        source_package,
        package_name="memory_condense",
        parent_name="src",
    )
    tool_import_root = _package_import_root(
        tool_package,
        package_name="mem0_eval",
        parent_name="tools",
    )
    expected_source = _required_digest(
        args.expected_source_sha256,
        "expected source SHA-256",
    )
    expected_tool = _required_digest(
        args.expected_tool_sha256,
        "expected tool SHA-256",
    )
    _verify_tree(
        source_package,
        expected_sha256=expected_source,
        label="memory-condense source",
    )
    _verify_tree(
        tool_package,
        expected_sha256=expected_tool,
        label="Mem0 tool",
    )
    _verify_bootstrap_origin(tool_package)
    _verify_isolated_runtime()
    _validate_tool_module(args.module, tool_package)

    os.environ["MEM0_TELEMETRY"] = "false"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "true"
    _bind_verified_tiktoken_cache()
    if not args.allow_network:
        _deny_network()
        os.environ["MEM0_VERIFIED_BOOTSTRAP_NETWORK_DENIED"] = "1"
    else:
        os.environ["MEM0_VERIFIED_BOOTSTRAP_NETWORK_DENIED"] = "0"
    # The launched module can bind its artifact to this already-verified
    # bootstrap authority without trusting a caller-selected import path.
    os.environ["MEM0_VERIFIED_BOOTSTRAP_SOURCE_SHA256"] = expected_source
    os.environ["MEM0_VERIFIED_BOOTSTRAP_TOOL_SHA256"] = expected_tool
    sys.path.insert(0, str(tool_import_root))
    sys.path.insert(0, str(source_import_root))
    _verify_import_resolution(source_package, tool_package)
    forwarded = list(args.args)
    if forwarded[:1] == ["--"]:
        forwarded = forwarded[1:]
    sys.argv = [args.module, *forwarded]
    launch_error: BaseException | None = None
    try:
        runpy.run_module(args.module, run_name="__main__", alter_sys=False)
    except BaseException as exc:
        launch_error = exc
    try:
        _verify_tree(
            source_package,
            expected_sha256=expected_source,
            label="memory-condense source changed during launch",
        )
        _verify_tree(
            tool_package,
            expected_sha256=expected_tool,
            label="Mem0 tool changed during launch",
        )
    except BaseException as recheck_error:
        if launch_error is not None:
            recheck_error.add_note(
                "the launched module also failed with "
                f"{type(launch_error).__name__}"
            )
        raise
    if launch_error is not None:
        raise launch_error
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
