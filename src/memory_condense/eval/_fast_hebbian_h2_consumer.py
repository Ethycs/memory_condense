"""Closed static local-import source identity for the Hebbian H2 consumer.

The scope is the recursively resolved ``memory_condense`` Python import graph
rooted at the H2 implementation, including every package initializer on those
module paths. Unimported sibling modules are deliberately outside the scope.
Third-party packages and non-literal dynamic imports remain environment/runtime
dependencies rather than claims made by this source manifest.
"""

from __future__ import annotations

import ast
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from memory_condense.domain.sealed import SealedIdentity
from memory_condense.eval._fast_hebbian_h2_io import (
    FastHebbianH2ValidationError,
)


FAST_HEBBIAN_H2_CONSUMER_SOURCE_FORMAT = (
    "memory-condense-fast-hebbian-h2-static-local-import-source-manifest-v2"
)
FAST_HEBBIAN_H2_CONSUMER_SOURCE_DOMAIN = (
    "memory-condense-fast-hebbian-h2-static-local-import-source-v2"
)
FAST_HEBBIAN_H2_CONSUMER_SOURCE_ALGORITHM = (
    "sha256-domain-null-prefix-uint64be-framed-sorted-path-utf8-and-bytes-v2"
)
FAST_HEBBIAN_H2_CONSUMER_SOURCE_SCOPE = (
    "closed-static-memory-condense-python-import-closure-with-package-inits-v1"
)
FAST_HEBBIAN_H2_CONSUMER_ROOT_MODULES = (
    "memory_condense.eval.fast_hebbian_h2",
)

_DIGEST_CHARS = frozenset("0123456789abcdef")
_DOMAIN = (FAST_HEBBIAN_H2_CONSUMER_SOURCE_DOMAIN + "\0").encode("ascii")
_LOCAL_PACKAGE = "memory_condense"


def _text(value: object, label: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise FastHebbianH2ValidationError(
            f"{label} must be an exact non-empty string"
        )
    return value


def _digest(value: object, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or not set(value).issubset(_DIGEST_CHARS)
    ):
        raise FastHebbianH2ValidationError(
            f"{label} must be a lowercase SHA-256 digest"
        )
    return value


def _frame(hasher: Any, value: bytes) -> None:
    hasher.update(len(value).to_bytes(8, "big"))
    hasher.update(value)


def _source_root(source_root: str | Path | None) -> Path:
    candidate = (
        Path(__file__).resolve().parents[3]
        if source_root is None
        else Path(source_root)
    )
    if candidate.is_symlink():
        raise FastHebbianH2ValidationError(
            "H2 consumer source root must not be a symlink"
        )
    try:
        root = candidate.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise FastHebbianH2ValidationError(
            "H2 consumer source root does not exist"
        ) from exc
    if not root.is_dir() or root.is_symlink():
        raise FastHebbianH2ValidationError(
            "H2 consumer source root must be a regular directory"
        )
    return root


def _module_path(root: Path, module: str) -> tuple[Path, bool] | None:
    if module != _LOCAL_PACKAGE and not module.startswith(_LOCAL_PACKAGE + "."):
        return None
    parts = module.split(".")
    package = root.joinpath("src", *parts, "__init__.py")
    source = root.joinpath("src", *parts).with_suffix(".py")
    if package.is_file() or package.is_symlink():
        return package, True
    if source.is_file() or source.is_symlink():
        return source, False
    return None


def _absolute_import_from(
    node: ast.ImportFrom,
    *,
    current_module: str,
    current_is_package: bool,
) -> str | None:
    if node.level == 0:
        return node.module
    package = (
        current_module
        if current_is_package
        else current_module.rpartition(".")[0]
    )
    parts = package.split(".") if package else []
    if node.level > len(parts):
        raise FastHebbianH2ValidationError(
            f"relative import escapes package in {current_module}"
        )
    base = parts[: len(parts) - (node.level - 1)]
    if node.module:
        base.extend(node.module.split("."))
    return ".".join(base)


def _package_modules(module: str, *, is_package: bool) -> tuple[str, ...]:
    parts = module.split(".")
    stop = len(parts) + int(is_package) - 1
    return tuple(".".join(parts[:index]) for index in range(1, stop + 1))


def _local_imports(
    tree: ast.AST,
    *,
    current_module: str,
    current_is_package: bool,
    root: Path,
) -> tuple[str, ...]:
    discovered: set[str] = set()

    def add_if_module(module: str | None) -> None:
        if module and _module_path(root, module) is not None:
            discovered.add(module)

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == _LOCAL_PACKAGE or alias.name.startswith(
                    _LOCAL_PACKAGE + "."
                ):
                    if _module_path(root, alias.name) is None:
                        raise FastHebbianH2ValidationError(
                            f"local import is absent: {alias.name}"
                        )
                    discovered.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            target = _absolute_import_from(
                node,
                current_module=current_module,
                current_is_package=current_is_package,
            )
            if target == _LOCAL_PACKAGE or (
                target is not None and target.startswith(_LOCAL_PACKAGE + ".")
            ):
                if _module_path(root, target) is None:
                    raise FastHebbianH2ValidationError(
                        f"local import is absent: {target}"
                    )
                discovered.add(target)
                for alias in node.names:
                    if alias.name != "*":
                        add_if_module(f"{target}.{alias.name}")
        elif isinstance(node, ast.Call) and node.args:
            function = node.func
            function_name = (
                function.id
                if isinstance(function, ast.Name)
                else function.attr if isinstance(function, ast.Attribute) else ""
            )
            first = node.args[0]
            if function_name in {"__import__", "import_module", "_import_module"} and (
                isinstance(first, ast.Constant) and type(first.value) is str
            ):
                name = first.value
                if name == _LOCAL_PACKAGE or name.startswith(_LOCAL_PACKAGE + "."):
                    if _module_path(root, name) is None:
                        raise FastHebbianH2ValidationError(
                            f"literal dynamic local import is absent: {name}"
                        )
                    discovered.add(name)
    return tuple(sorted(discovered))


def _source_snapshots(root: Path) -> dict[str, bytes]:
    pending = list(FAST_HEBBIAN_H2_CONSUMER_ROOT_MODULES)
    visited: set[str] = set()
    snapshots: dict[str, bytes] = {}
    while pending:
        module = pending.pop()
        if module in visited:
            continue
        coordinate = _module_path(root, module)
        if coordinate is None:
            raise FastHebbianH2ValidationError(
                f"H2 consumer root/import module is absent: {module}"
            )
        path, is_package = coordinate
        if path.is_symlink():
            raise FastHebbianH2ValidationError(
                f"H2 consumer source dependency is a symlink: {path}"
            )
        try:
            resolved = path.resolve(strict=True)
            relative = resolved.relative_to(root).as_posix()
        except (OSError, RuntimeError, ValueError) as exc:
            raise FastHebbianH2ValidationError(
                f"H2 consumer source dependency escaped its root: {path}"
            ) from exc
        if not resolved.is_file() or resolved.is_symlink():
            raise FastHebbianH2ValidationError(
                f"H2 consumer source dependency is not regular: {relative}"
            )
        raw = resolved.read_bytes()
        if not raw:
            raise FastHebbianH2ValidationError(
                f"H2 consumer source dependency is empty: {relative}"
            )
        try:
            tree = ast.parse(raw.decode("utf-8-sig"), filename=relative)
        except (UnicodeDecodeError, SyntaxError) as exc:
            raise FastHebbianH2ValidationError(
                f"H2 consumer source is not valid UTF-8 Python: {relative}"
            ) from exc
        visited.add(module)
        prior = snapshots.setdefault(relative, raw)
        if prior != raw:
            raise FastHebbianH2ValidationError(
                f"H2 source modules alias different bytes: {relative}"
            )
        pending.extend(_package_modules(module, is_package=is_package))
        pending.extend(
            _local_imports(
                tree,
                current_module=module,
                current_is_package=is_package,
                root=root,
            )
        )
    for relative, raw in snapshots.items():
        if root.joinpath(*relative.split("/")).read_bytes() != raw:
            raise FastHebbianH2ValidationError(
                f"H2 consumer source changed during reconstruction: {relative}"
            )
    return dict(sorted(snapshots.items()))


@dataclass(frozen=True, slots=True)
class FastHebbianH2ConsumerSourceFile:
    """One repository-relative file in the closed static import scope."""

    path: str
    size_bytes: int
    file_sha256: str

    def __post_init__(self) -> None:
        path = _text(self.path, "consumer source path")
        if "\\" in path or path.startswith("/") or ".." in Path(path).parts:
            raise FastHebbianH2ValidationError(
                "consumer source path must be repository-relative POSIX"
            )
        if type(self.size_bytes) is not int or self.size_bytes < 1:
            raise FastHebbianH2ValidationError(
                "consumer source size must be a positive integer"
            )
        _digest(self.file_sha256, "consumer source file_sha256")

    def identity_payload(self) -> dict[str, object]:
        return {
            "path": self.path,
            "size_bytes": self.size_bytes,
            "file_sha256": self.file_sha256,
        }


@dataclass(frozen=True, slots=True)
class FastHebbianH2ConsumerSourceManifest(SealedIdentity):
    """Text-free projection of the closed static local-import source scope."""

    _SEAL_FIELD = "manifest_sha256"
    _SEAL_MISMATCH = "H2 consumer source manifest seal changed"

    format: str
    domain: str
    algorithm: str
    scope: str
    root_modules: tuple[str, ...]
    files: tuple[FastHebbianH2ConsumerSourceFile, ...]
    source_sha256: str
    manifest_sha256: str = ""

    def __post_init__(self) -> None:
        if self.format != FAST_HEBBIAN_H2_CONSUMER_SOURCE_FORMAT:
            raise FastHebbianH2ValidationError(
                "unsupported H2 consumer source manifest format"
            )
        if self.domain != FAST_HEBBIAN_H2_CONSUMER_SOURCE_DOMAIN:
            raise FastHebbianH2ValidationError(
                "unsupported H2 consumer source hashing domain"
            )
        if self.algorithm != FAST_HEBBIAN_H2_CONSUMER_SOURCE_ALGORITHM:
            raise FastHebbianH2ValidationError(
                "unsupported H2 consumer source hashing algorithm"
            )
        if self.scope != FAST_HEBBIAN_H2_CONSUMER_SOURCE_SCOPE:
            raise FastHebbianH2ValidationError(
                "unsupported H2 consumer source closure scope"
            )
        if self.root_modules != FAST_HEBBIAN_H2_CONSUMER_ROOT_MODULES:
            raise FastHebbianH2ValidationError(
                "H2 consumer source root modules changed"
            )
        if type(self.files) is not tuple or not self.files or any(
            type(row) is not FastHebbianH2ConsumerSourceFile
            for row in self.files
        ):
            raise FastHebbianH2ValidationError(
                "H2 consumer source files changed type"
            )
        paths = tuple(row.path for row in self.files)
        if paths != tuple(sorted(paths)) or len(paths) != len(set(paths)):
            raise FastHebbianH2ValidationError(
                "H2 consumer source closure is not sorted and unique"
            )
        for module in self.root_modules:
            expected = "src/" + module.replace(".", "/") + ".py"
            if expected not in paths:
                raise FastHebbianH2ValidationError(
                    f"H2 consumer source closure omitted root: {module}"
                )
        _digest(self.source_sha256, "H2 consumer source_sha256")
        self._seal()


def build_fast_hebbian_h2_consumer_source_manifest(
    source_root: str | Path | None = None,
) -> FastHebbianH2ConsumerSourceManifest:
    """Resolve and hash H2's static transitive local Python import closure."""

    snapshots = _source_snapshots(_source_root(source_root))
    hasher = hashlib.sha256()
    hasher.update(_DOMAIN)
    rows: list[FastHebbianH2ConsumerSourceFile] = []
    for relative, raw in snapshots.items():
        _frame(hasher, relative.encode("utf-8"))
        _frame(hasher, raw)
        rows.append(
            FastHebbianH2ConsumerSourceFile(
                path=relative,
                size_bytes=len(raw),
                file_sha256=hashlib.sha256(raw).hexdigest(),
            )
        )
    return FastHebbianH2ConsumerSourceManifest(
        format=FAST_HEBBIAN_H2_CONSUMER_SOURCE_FORMAT,
        domain=FAST_HEBBIAN_H2_CONSUMER_SOURCE_DOMAIN,
        algorithm=FAST_HEBBIAN_H2_CONSUMER_SOURCE_ALGORITHM,
        scope=FAST_HEBBIAN_H2_CONSUMER_SOURCE_SCOPE,
        root_modules=FAST_HEBBIAN_H2_CONSUMER_ROOT_MODULES,
        files=tuple(rows),
        source_sha256=hasher.hexdigest(),
    )


__all__ = [
    "FAST_HEBBIAN_H2_CONSUMER_ROOT_MODULES",
    "FAST_HEBBIAN_H2_CONSUMER_SOURCE_ALGORITHM",
    "FAST_HEBBIAN_H2_CONSUMER_SOURCE_DOMAIN",
    "FAST_HEBBIAN_H2_CONSUMER_SOURCE_FORMAT",
    "FAST_HEBBIAN_H2_CONSUMER_SOURCE_SCOPE",
    "FastHebbianH2ConsumerSourceFile",
    "FastHebbianH2ConsumerSourceManifest",
    "build_fast_hebbian_h2_consumer_source_manifest",
]
