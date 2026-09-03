#!/usr/bin/env python3
"""Attest the complete, provider-free policy-v5-r3 confirmation executor.

Version 1 authenticates the frozen policy and the initial execution boundary,
but deliberately refuses to claim end-to-end readiness.  This successor may
be published only from a clean, committed tree after the complete offline
apparatus suite has passed.  Its release scope is intentionally narrow: a
consumer may open the sanitized confirmation treatment, but may neither open
gold nor call a provider.  Those actions require later, stage-local releases.

The module is stdlib-only apart from importing the stdlib-only v1 attester.
It never imports the executor, reads a treatment, reads gold, or constructs a
provider client.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from tools import attest_confirmation_executor_v1 as boundary


FORMAT = "memory-condense-confirmation-executor-attestation-v2"
STATUS = "end_to_end_executor_ready_provider_free"
OFFLINE_TEST_RECEIPT_FORMAT = (
    "memory-condense-confirmation-executor-offline-test-receipt-v1"
)
OFFLINE_TEST_STATUS = "complete_pass_provider_free"
PRODUCTION_ENTRYPOINT = "tools/run_confirmation_policy_v5_r3.py"
MINIMUM_OFFLINE_TEST_COUNT = 170

# This is the deliberately explicit confirmation surface.  Validation-era
# implementation modules used by the frozen policy remain in the set where a
# confirmation adapter delegates to them; this binds those bytes without
# making validation artifacts runtime inputs.
DEFAULT_EXECUTOR_FILES = (
    "tools/attest_confirmation_executor_v1.py",
    "tools/attest_confirmation_executor_v2.py",
    "tools/confirmation_canonical.py",
    "tools/confirmation_contracts.py",
    "tools/confirmation_treatment.py",
    "tools/export_confirmation_treatment_v5_r3.py",
    "tools/plan_confirmation_treatment_pipeline.py",
    "tools/confirmation_namespace_store_adapter.py",
    "tools/confirmation_staged_cumulative_coordinator.py",
    "tools/confirmation_cumulative_retrieval.py",
    "tools/confirmation_production_runtime.py",
    "tools/confirmation_production_phase_adapters.py",
    "tools/confirmation_production_final_adapters.py",
    "tools/confirmation_s0_prompt_preflight.py",
    "tools/confirmation_terra_completion_lifecycle.py",
    "tools/confirmation_protected_s0_plane.py",
    "tools/confirmation_query_payload_parent.py",
    "tools/confirmation_query_expansion_adapter.py",
    "tools/confirmation_evidence_map_parent.py",
    "tools/confirmation_source_streams.py",
    "tools/confirmation_adaptive_source_map.py",
    "tools/confirmation_adaptive_tail.py",
    "tools/confirmation_typed_final.py",
    "tools/confirmation_specialist_v3.py",
    "tools/confirmation_semantic_planes.py",
    "tools/confirmation_terminal_policy_boundary.py",
    "tools/materialize_confirmation_numeric_v5_overlay.py",
    "tools/materialize_confirmation_prediction_plane.py",
    "tools/confirmation_gold_judge_scaffold.py",
    "tools/confirmation_sol_judge_lifecycle.py",
    "tools/matched_eval/confirmation_numeric_policy.py",
    "tools/matched_eval/confirmation_semantic_helpers.py",
    "tools/matched_eval/confirmation_specialist_core.py",
    "tools/matched_eval/confirmation_specialist_reconciliation.py",
    "tools/matched_eval/qa_prompt_policy.py",
    "tools/matched_eval/query_payload_live.py",
    PRODUCTION_ENTRYPOINT,
)

DEFAULT_OFFLINE_TEST_FILES = (
    "tests/test_plan_confirmation_treatment_pipeline.py",
    "tests/test_attest_confirmation_executor_v1.py",
    "tests/test_attest_confirmation_executor_v2.py",
    "tests/test_confirmation_namespace_store_adapter.py",
    "tests/test_confirmation_runtime_policy_export.py",
    "tests/test_confirmation_s0_prompt_preflight.py",
    "tests/test_confirmation_gold_judge_scaffold.py",
    "tests/test_confirmation_cumulative_retrieval.py",
    "tests/test_confirmation_protected_s0_plane.py",
    "tests/test_confirmation_query_payload_parent.py",
    "tests/test_confirmation_query_expansion_adapter.py",
    "tests/test_confirmation_evidence_map_parent.py",
    "tests/test_confirmation_source_streams.py",
    "tests/test_confirmation_adaptive_source_map.py",
    "tests/test_confirmation_adaptive_tail.py",
    "tests/test_confirmation_typed_final.py",
    "tests/test_confirmation_specialist_v3.py",
    "tests/test_confirmation_semantic_planes.py",
    "tests/test_confirmation_terminal_policy_boundary.py",
    "tests/test_confirmation_terminal_v5_plan_adapter.py",
    "tests/test_confirmation_terra_completion_lifecycle.py",
    "tests/test_materialize_confirmation_numeric_v5_overlay.py",
    "tests/test_materialize_confirmation_prediction_plane.py",
    "tests/test_confirmation_sol_judge_lifecycle.py",
    "tests/test_confirmation_production_phase_adapters.py",
    "tests/test_confirmation_production_final_adapters.py",
    "tests/test_confirmation_production_runtime.py",
    "tests/test_run_confirmation_policy_v5_r3.py",
)

# The firebreak is a reachability claim from the one production prediction
# entrypoint, not a claim that post-prediction evaluator files are absent from
# the separately hashed apparatus inventory.
PREDICTION_FIREBREAK_FILES = frozenset({PRODUCTION_ENTRYPOINT})
FORBIDDEN_PREDICTION_IMPORTS = frozenset(
    {
        "tools.confirmation_gold_judge_scaffold",
        "tools.confirmation_sol_judge_lifecycle",
    }
)
FORBIDDEN_PREDICTION_FILES = frozenset(
    {
        "src/memory_condense/eval/_binary_judge_protocol.py",
        "src/memory_condense/eval/benchmark.py",
        "src/memory_condense/eval/locked_split.py",
        "src/memory_condense/ingest/loader.py",
        "tools/v4_population_firebreak/__init__.py",
        "tools/v4_population_firebreak/analysis.py",
        "tools/v4_population_firebreak/population.py",
        "tools/v4_population_firebreak/scoring.py",
        "tools/v4_population_firebreak/verifier.py",
    }
)

_FORBIDDEN_IMPORTED_SYMBOLS = frozenset(
    {
        "_binary_judge",
        "build_judge_prompt",
        "export_confirmation_treatment_input",
        "load_benchmark",
        "load_split_manifest",
        "parse_locomo",
        "parse_longmemeval",
        "reconstruct_population",
        "select_locked_split",
    }
)
_SENSITIVE_LOADER_SYMBOL_RE = re.compile(
    r"^(?:build|evaluate|load|open|parse|read|reconstruct|score|select)_"
    r"(?:.*_)?(?:dataset|gold|judge|reference|split|verdict)(?:_.*)?$"
)

PRODUCTION_SPEC = replace(
    boundary.PRODUCTION_SPEC,
    required_executor_files=DEFAULT_EXECUTOR_FILES,
)

_PASSED_RE = re.compile(r"(?m)(?:^|\s)(\d+) passed(?:\s|$)")
_BAD_OUTCOME_RE = re.compile(
    r"(?i)\b(?:failed|error|errors|skipped|xfailed|xpassed|deselected)\b"
)


class ConfirmationExecutorReadinessError(
    boundary.ConfirmationExecutorAttestationError
):
    """The full confirmation apparatus is not ready to open treatment."""


def _require(ok: object, message: str) -> None:
    if not ok:
        raise ConfirmationExecutorReadinessError(message)


def _sha256_file(path: Path, label: str) -> str:
    return boundary._sha256(boundary._regular_bytes(path, label))  # noqa: SLF001


def _repository_relative(root: Path, value: str | Path) -> str:
    return boundary._repository_relative(root, value)  # noqa: SLF001


def _git_state(root: Path) -> tuple[str, str]:
    return boundary._git_state(root, boundary._git_output)  # noqa: SLF001


def _module_name(relative: str) -> tuple[str, str]:
    path = Path(relative)
    parts = list(path.with_suffix("").parts)
    if parts[:1] == ["src"]:
        parts = parts[1:]
    if parts[-1:] == ["__init__"]:
        parts = parts[:-1]
        module = ".".join(parts)
        return module, module
    module = ".".join(parts)
    return module, ".".join(parts[:-1])


def _resolve_local_module(root: Path, module: str) -> str | None:
    if module == "tools" or module.startswith("tools."):
        base = root / Path(*module.split("."))
    elif module == "memory_condense" or module.startswith("memory_condense."):
        base = root / "src" / Path(*module.split("."))
    else:
        return None
    source = base.with_suffix(".py")
    package = base / "__init__.py"
    if source.is_file():
        return source.relative_to(root).as_posix()
    if package.is_file():
        return package.relative_to(root).as_posix()
    return None


def _package_initializer_files(root: Path, relative: str) -> tuple[str, ...]:
    """Return every initializer Python executes before ``relative``."""

    path = Path(relative)
    parts = list(path.with_suffix("").parts)
    source_prefix = parts[:1] == ["src"]
    if source_prefix:
        parts = parts[1:]
    if not parts:
        return ()
    parent_count = len(parts) - 1
    initializers: list[str] = []
    for count in range(1, parent_count + 1):
        prefix = (["src"] if source_prefix else []) + parts[:count]
        candidate = root.joinpath(*prefix, "__init__.py")
        if candidate.is_file():
            initializers.append(candidate.relative_to(root).as_posix())
    return tuple(initializers)


def _relative_module_name(module: str, package: str) -> str | None:
    if not module.startswith("."):
        return module
    level = len(module) - len(module.lstrip("."))
    suffix = module[level:]
    package_parts = package.split(".") if package else []
    retain = len(package_parts) - (level - 1)
    if retain < 0:
        return None
    parts = package_parts[:retain]
    if suffix:
        parts.extend(suffix.split("."))
    return ".".join(parts) or None


def _static_dynamic_target(
    node: ast.AST,
    *,
    module: str,
    package: str,
) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return _relative_module_name(node.value, package)
    if not isinstance(node, ast.JoinedStr):
        return None
    parts: list[str] = []
    for value in node.values:
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            parts.append(value.value)
        elif (
            isinstance(value, ast.FormattedValue)
            and isinstance(value.value, ast.Name)
            and value.value.id in {"__name__", "__package__"}
        ):
            parts.append(module if value.value.id == "__name__" else package)
        else:
            return None
    return _relative_module_name("".join(parts), package)


def _enclosing_function(
    node: ast.AST, parents: Mapping[ast.AST, ast.AST]
) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    current = parents.get(node)
    while current is not None:
        if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return current
        current = parents.get(current)
    return None


def _closed_lazy_facade_import(
    tree: ast.Module,
    call: ast.Call,
    parents: Mapping[ast.AST, ast.AST],
) -> bool:
    """Recognize the repository's inert, finite-map ``__getattr__`` facade."""

    function = _enclosing_function(call, parents)
    if function is None or function.name != "__getattr__" or not call.args:
        return False
    target = call.args[0]
    if not isinstance(target, ast.Name):
        return False
    export_maps: set[str] = set()
    for statement in tree.body:
        name: str | None = None
        value: ast.AST | None = None
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
            if isinstance(statement.targets[0], ast.Name):
                name = statement.targets[0].id
                value = statement.value
        elif isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name):
            name = statement.target.id
            value = statement.value
        if not name or not isinstance(value, ast.Dict):
            continue
        closed = True
        for row in value.values:
            if not isinstance(row, (ast.Tuple, ast.List)) or not row.elts:
                closed = False
                break
            if not isinstance(row.elts[0], ast.Constant) or not isinstance(
                row.elts[0].value, str
            ):
                closed = False
                break
        if closed:
            export_maps.add(name)
    if not export_maps:
        return False
    for child in ast.walk(function):
        if not isinstance(child, (ast.Assign, ast.AnnAssign)):
            continue
        targets = child.targets if isinstance(child, ast.Assign) else [child.target]
        value = child.value
        if not isinstance(value, ast.Subscript) or not isinstance(value.value, ast.Name):
            continue
        if value.value.id not in export_maps:
            continue
        for assignment_target in targets:
            if isinstance(assignment_target, (ast.Tuple, ast.List)) and any(
                isinstance(item, ast.Name) and item.id == target.id
                for item in assignment_target.elts
            ):
                return True
    return False


def _source_import_facts(
    root: Path, relative: str
) -> tuple[set[str], set[str], tuple[str, ...]]:
    raw = boundary._regular_bytes(root / relative, "executor source")  # noqa: SLF001
    try:
        tree = ast.parse(raw.decode("utf-8"), filename=relative)
    except (UnicodeDecodeError, SyntaxError) as exc:
        raise ConfirmationExecutorReadinessError(
            f"cannot parse executor source: {relative}"
        ) from exc
    module, package = _module_name(relative)
    candidates: set[str] = set()
    imported_symbols: set[str] = set()
    importlib_modules: set[str] = {"importlib"}
    import_module_functions: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                candidates.add(alias.name)
                if alias.name == "importlib":
                    importlib_modules.add(alias.asname or alias.name)
            continue
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.level:
            package_parts = package.split(".") if package else []
            retain = len(package_parts) - (node.level - 1)
            if retain < 0:
                continue
            base_parts = package_parts[:retain]
            if node.module:
                base_parts.extend(node.module.split("."))
            base = ".".join(base_parts)
        else:
            base = node.module or ""
        if base:
            candidates.add(base)
        for alias in node.names:
            imported_symbols.add(alias.name)
            candidates.add(f"{base}.{alias.name}" if base else alias.name)
            if base == "importlib" and alias.name == "import_module":
                import_module_functions.add(alias.asname or alias.name)

    parents = {
        child: parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }
    unresolved_dynamic: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        is_dynamic = (
            isinstance(node.func, ast.Name)
            and node.func.id in ({"__import__"} | import_module_functions)
        ) or (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "import_module"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id in importlib_modules
        )
        if not is_dynamic:
            continue
        target = _static_dynamic_target(
            node.args[0], module=module, package=package
        )
        if target is not None:
            candidates.add(target)
        elif not _closed_lazy_facade_import(tree, node, parents):
            unresolved_dynamic.append(f"{relative}:{getattr(node, 'lineno', 0)}")
    return candidates, imported_symbols, tuple(sorted(unresolved_dynamic))


def _local_imports(root: Path, relative: str) -> tuple[str, ...]:
    candidates, _symbols, _dynamic = _source_import_facts(root, relative)
    resolved = {
        path
        for module in candidates
        if (path := _resolve_local_module(root, module)) is not None
    }
    return tuple(sorted(resolved))


def resolve_transitive_executor_files(
    repository_root: str | Path,
    roots: Sequence[str] = DEFAULT_EXECUTOR_FILES,
) -> tuple[str, ...]:
    """Return the deterministic local Python import closure of the roots."""

    root = Path(repository_root).resolve()
    pending = [
        _repository_relative(root, relative)
        for relative in roots
    ]
    resolved: set[str] = set()
    while pending:
        relative = pending.pop()
        if relative in resolved:
            continue
        _require((root / relative).is_file(), f"executor root is absent: {relative}")
        resolved.add(relative)
        pending.extend(
            dependency
            for dependency in _package_initializer_files(root, relative)
            if dependency not in resolved
        )
        pending.extend(
            dependency
            for dependency in _local_imports(root, relative)
            if dependency not in resolved
        )
    return tuple(sorted(resolved))


def _test_receipt_body(
    *,
    head_commit_sha1: str,
    git_tree_sha1: str,
    test_files: Sequence[str],
    passed_count: int,
) -> dict[str, Any]:
    body = {
        "format": OFFLINE_TEST_RECEIPT_FORMAT,
        "status": OFFLINE_TEST_STATUS,
        "executor_git": {
            "head_commit_sha1": head_commit_sha1,
            "git_tree_sha1": git_tree_sha1,
            "worktree_clean_before_and_after_tests": True,
        },
        "pytest": {
            "test_files": list(test_files),
            "passed_count": passed_count,
            "exit_code": 0,
            "warnings_disabled": True,
            "cache_provider_disabled": True,
        },
        "provider_accounting": {
            "physical_provider_calls": 0,
            "authorized_terra_calls": 0,
            "authorized_sol_calls": 0,
        },
    }
    return {**body, "receipt_identity_sha256": boundary.identity_sha256(body)}


def run_and_publish_offline_test_receipt(
    *,
    repository_root: str | Path,
    output_path: str | Path,
    test_files: Sequence[str] = DEFAULT_OFFLINE_TEST_FILES,
    python_executable: str | Path = sys.executable,
) -> tuple[boundary.SealedJson, bool]:
    """Run the fixed provider-free suite and seal its deterministic result."""

    root = Path(repository_root).resolve()
    _require(root.is_dir(), "repository root is not a directory")
    normalized = tuple(_repository_relative(root, value) for value in test_files)
    _require(normalized == tuple(test_files), "offline test paths are not canonical")
    _require(len(normalized) == len(set(normalized)), "offline test list has duplicates")
    _require(
        all((root / value).is_file() for value in normalized),
        "offline test suite is incomplete",
    )
    before_head, before_tree = _git_state(root)
    command = [
        str(python_executable),
        "-m",
        "pytest",
        "-q",
        "--disable-warnings",
        "-p",
        "no:cacheprovider",
        "--basetemp",
        ".test-tmp-confirmation-attestation-v2",
        *normalized,
    ]
    completed = subprocess.run(
        command,
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    combined = f"{completed.stdout}\n{completed.stderr}"
    matches = _PASSED_RE.findall(combined)
    _require(completed.returncode == 0, "offline confirmation suite failed")
    _require(len(matches) == 1, "could not authenticate pytest pass count")
    _require(not _BAD_OUTCOME_RE.search(combined), "offline suite was not an exact pass")
    passed_count = int(matches[0])
    _require(
        passed_count >= MINIMUM_OFFLINE_TEST_COUNT,
        "offline confirmation suite coverage fell below the frozen floor",
    )
    after_head, after_tree = _git_state(root)
    _require(
        (after_head, after_tree) == (before_head, before_tree),
        "executor Git state changed while the offline suite ran",
    )
    payload = _test_receipt_body(
        head_commit_sha1=after_head,
        git_tree_sha1=after_tree,
        test_files=normalized,
        passed_count=passed_count,
    )
    return boundary.publish_sealed_json(output_path, payload)


def _verified_test_receipt(
    *,
    repository_root: Path,
    receipt_path: str | Path,
    expected_head: str,
    expected_tree: str,
    expected_test_files: Sequence[str],
) -> tuple[boundary.SealedJson, int, str]:
    relative = _repository_relative(repository_root, receipt_path)
    artifact = boundary.read_sealed_json(repository_root / relative, label="offline test receipt")
    payload = artifact.payload
    _require(
        set(payload)
        == {
            "format",
            "status",
            "executor_git",
            "pytest",
            "provider_accounting",
            "receipt_identity_sha256",
        },
        "offline test receipt schema changed",
    )
    body = {key: value for key, value in payload.items() if key != "receipt_identity_sha256"}
    _require(
        payload["receipt_identity_sha256"] == boundary.identity_sha256(body),
        "offline test receipt self-seal differs",
    )
    _require(
        payload["format"] == OFFLINE_TEST_RECEIPT_FORMAT
        and payload["status"] == OFFLINE_TEST_STATUS,
        "offline test receipt is not a complete pass",
    )
    git = boundary._mapping(payload["executor_git"], "offline test Git state")  # noqa: SLF001
    _require(
        git.get("head_commit_sha1") == expected_head
        and git.get("git_tree_sha1") == expected_tree
        and git.get("worktree_clean_before_and_after_tests") is True,
        "offline tests bind another executor Git state",
    )
    pytest_row = boundary._mapping(payload["pytest"], "offline pytest receipt")  # noqa: SLF001
    _require(
        tuple(pytest_row.get("test_files", ())) == tuple(expected_test_files),
        "offline test suite differs from the required suite",
    )
    count = pytest_row.get("passed_count")
    _require(
        type(count) is int and count >= MINIMUM_OFFLINE_TEST_COUNT,
        "offline test pass count is below the frozen floor",
    )
    _require(
        pytest_row.get("exit_code") == 0
        and pytest_row.get("warnings_disabled") is True
        and pytest_row.get("cache_provider_disabled") is True,
        "offline pytest execution controls changed",
    )
    accounting = boundary._mapping(  # noqa: SLF001
        payload["provider_accounting"], "offline provider accounting"
    )
    _require(
        accounting
        == {
            "physical_provider_calls": 0,
            "authorized_terra_calls": 0,
            "authorized_sol_calls": 0,
        },
        "offline suite touched or authorized a provider",
    )
    return artifact, count, relative


def _verify_prediction_firebreak(
    root: Path,
    executor_files: Sequence[str],
    prediction_roots: Sequence[str] = tuple(sorted(PREDICTION_FIREBREAK_FILES)),
) -> None:
    prediction_files = resolve_transitive_executor_files(
        root, prediction_roots
    )
    _require(
        set(prediction_files) <= set(executor_files),
        "prediction dependency escaped the attested executor inventory",
    )
    for relative in prediction_files:
        normalized = relative.casefold()
        stem = Path(relative).stem.casefold()
        forbidden_file = (
            relative in FORBIDDEN_PREDICTION_FILES
            or "judge" in stem
            or "judging" in stem
            or ("gold" in stem and "gold_blind" not in stem)
            or normalized.startswith("tools/v4_population_firebreak/")
        )
        _require(
            not forbidden_file,
            f"prediction-stage judge/gold/data capability is reachable: {relative}",
        )
        imports, symbols, unresolved_dynamic = _source_import_facts(root, relative)
        bad_imports = sorted(
            module
            for module in imports
            if module in FORBIDDEN_PREDICTION_IMPORTS
            or module.startswith("tools.v4_population_firebreak")
            or module
            in {
                "memory_condense.eval._binary_judge_protocol",
                "memory_condense.eval.benchmark",
                "memory_condense.eval.locked_split",
                "memory_condense.ingest.loader",
            }
            or "judge" in module.rsplit(".", 1)[-1].casefold()
            or "judging" in module.rsplit(".", 1)[-1].casefold()
        )
        _require(
            not bad_imports,
            f"prediction-stage module imports judge/gold/data boundary: {relative}",
        )
        bad_symbols = sorted(
            symbol
            for symbol in symbols
            if (
                symbol.casefold() in _FORBIDDEN_IMPORTED_SYMBOLS
                or (
                    "gold_blind" not in symbol.casefold()
                    and _SENSITIVE_LOADER_SYMBOL_RE.fullmatch(symbol.casefold())
                    is not None
                )
            )
        )
        _require(
            not bad_symbols,
            f"prediction-stage module imports judge/gold/data callable: {relative}",
        )
        _require(
            not unresolved_dynamic,
            "prediction-stage module has unresolved dynamic import: "
            + ", ".join(unresolved_dynamic),
        )


def _apparatus_projection(
    *,
    root: Path,
    boundary_payload: Mapping[str, Any],
    receipt: boundary.SealedJson,
    receipt_count: int,
    receipt_relative: str,
) -> dict[str, Any]:
    inventory = boundary._mapping(  # noqa: SLF001
        boundary_payload["executor_inventory"], "executor inventory"
    )
    executor_files = list(boundary._list(inventory["executor_files"], "executor files"))  # noqa: SLF001
    dependency_locks = list(  # noqa: SLF001
        boundary._list(inventory["dependency_locks"], "dependency locks")
    )
    entrypoint = next(
        (row for row in executor_files if row.get("path") == PRODUCTION_ENTRYPOINT),
        None,
    )
    _require(entrypoint is not None, "production entrypoint is absent from inventory")
    entrypoint_sha = _sha256_file(root / PRODUCTION_ENTRYPOINT, "production entrypoint")
    _require(entrypoint.get("sha256") == entrypoint_sha, "production entrypoint changed")
    return {
        "executor_files": executor_files,
        "executor_file_set_sha256": boundary.identity_sha256(executor_files),
        "dependency_locks": dependency_locks,
        "dependency_lock_sha256": boundary.identity_sha256(dependency_locks),
        "offline_test_receipt_path": receipt_relative,
        "offline_test_receipt_sha256": receipt.sha256,
        "offline_test_receipt_sidecar_sha256": receipt.sidecar_sha256,
        "offline_test_count": receipt_count,
        "production_entrypoint_path": PRODUCTION_ENTRYPOINT,
        "production_entrypoint_sha256": entrypoint_sha,
    }


BoundaryCompiler = Callable[..., Mapping[str, Any]]


def compile_confirmation_executor_attestation_v2(
    *,
    repository_root: str | Path,
    offline_test_receipt_path: str | Path,
    executor_files: Sequence[str] = DEFAULT_EXECUTOR_FILES,
    offline_test_files: Sequence[str] = DEFAULT_OFFLINE_TEST_FILES,
    prediction_firebreak_files: Sequence[str] = tuple(
        sorted(PREDICTION_FIREBREAK_FILES)
    ),
    boundary_compiler: BoundaryCompiler = boundary.compile_confirmation_executor_attestation,
) -> dict[str, Any]:
    """Compile readiness without opening treatment or enabling providers."""

    root = Path(repository_root).resolve()
    resolved_executor_files = resolve_transitive_executor_files(root, executor_files)
    base = boundary_compiler(
        repository_root=root,
        executor_files=resolved_executor_files,
        spec=PRODUCTION_SPEC,
    )
    git = boundary._mapping(base["executor_git"], "executor Git state")  # noqa: SLF001
    expected_head = str(git["head_commit_sha1"])
    expected_tree = str(git["git_tree_sha1"])
    receipt, count, receipt_relative = _verified_test_receipt(
        repository_root=root,
        receipt_path=offline_test_receipt_path,
        expected_head=expected_head,
        expected_tree=expected_tree,
        expected_test_files=offline_test_files,
    )
    _verify_prediction_firebreak(
        root, resolved_executor_files, prediction_firebreak_files
    )
    apparatus = _apparatus_projection(
        root=root,
        boundary_payload=base,
        receipt=receipt,
        receipt_count=count,
        receipt_relative=receipt_relative,
    )
    body = {
        "format": FORMAT,
        "status": STATUS,
        "frozen_policy": dict(boundary._mapping(base["frozen_policy"], "frozen policy")),  # noqa: SLF001
        "executor_git": dict(git),
        "apparatus": apparatus,
        "safety": {
            "confirmation_data_opened": False,
            "gold_or_reference_opened": False,
            "provider_execution_available": False,
            "provider_authorization_released": False,
            "end_to_end_readiness_claimed": True,
            "readiness_release_available": True,
            "remaining_executable_parent_stages_in_order": [],
        },
        "provider_accounting": {
            "physical_provider_calls": 0,
            "authorized_terra_calls": 0,
            "authorized_sol_calls": 0,
        },
        "release_scope": {
            "may_open_sanitized_treatment": True,
            "may_open_gold": False,
            "may_call_provider": False,
        },
        "boundary_attestation_v1_identity_sha256": base[
            "attestation_identity_sha256"
        ],
    }
    return {**body, "attestation_identity_sha256": boundary.identity_sha256(body)}


def publish_confirmation_executor_attestation_v2(
    output_path: str | Path, **kwargs: Any
) -> tuple[boundary.SealedJson, bool]:
    return boundary.publish_sealed_json(
        output_path,
        compile_confirmation_executor_attestation_v2(**kwargs),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    tests = subparsers.add_parser("test-receipt", help="run and seal offline suite")
    tests.add_argument("--repository-root", type=Path, default=boundary.REPOSITORY_ROOT)
    tests.add_argument("--output", type=Path, required=True)
    attest = subparsers.add_parser("attest", help="publish readiness attestation")
    attest.add_argument("--repository-root", type=Path, default=boundary.REPOSITORY_ROOT)
    attest.add_argument("--offline-test-receipt", type=Path, required=True)
    attest.add_argument("--output", type=Path, required=True)
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.command == "test-receipt":
        artifact, created = run_and_publish_offline_test_receipt(
            repository_root=args.repository_root,
            output_path=args.output,
        )
    else:
        artifact, created = publish_confirmation_executor_attestation_v2(
            args.output,
            repository_root=args.repository_root,
            offline_test_receipt_path=args.offline_test_receipt,
        )
    return {"path": str(artifact.path), "sha256": artifact.sha256, "created": created}


def main(argv: Sequence[str] | None = None) -> int:
    try:
        print(json.dumps(run(build_parser().parse_args(argv)), sort_keys=True))
    except (ConfirmationExecutorReadinessError, boundary.ConfirmationExecutorAttestationError) as exc:
        print(f"confirmation executor readiness failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_EXECUTOR_FILES",
    "DEFAULT_OFFLINE_TEST_FILES",
    "FORMAT",
    "MINIMUM_OFFLINE_TEST_COUNT",
    "OFFLINE_TEST_RECEIPT_FORMAT",
    "OFFLINE_TEST_STATUS",
    "PRODUCTION_ENTRYPOINT",
    "STATUS",
    "ConfirmationExecutorReadinessError",
    "build_parser",
    "compile_confirmation_executor_attestation_v2",
    "main",
    "publish_confirmation_executor_attestation_v2",
    "run",
    "run_and_publish_offline_test_receipt",
]
