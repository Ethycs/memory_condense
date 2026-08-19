"""Read and identify the frozen implementation without importing it."""

from __future__ import annotations

import ast
import hashlib
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .canonical import AuditError, bytes_sha256


_FULL_GIT_SHA1 = re.compile(r"[0-9a-f]{40}")


@dataclass(frozen=True, slots=True)
class FrozenSource:
    repository_root: Path
    source_commit: str
    implementation_sha256: str
    environment_lock_sha256: str
    environment_lock: bytes
    qa_system_prompt: str
    qa_user_template: str
    qa_no_context: str
    judge_system_prompt: str
    judge_user_template: str
    prompt_proxy_schema: str
    prompt_encoding: str
    framing_per_message: int
    framing_fixed: int
    benchmark_source_sha256: str
    tokenizer_source_sha256: str
    context_packer_source_sha256: str
    lexical_source_sha256: str
    database_source_sha256: str
    database_schema_sql: str
    lexical_token_pattern: str
    lexical_min_token_len: int
    lexical_stopwords: frozenset[str]
    max_expansion_tokens: int

    def blob(self, path: str) -> bytes:
        return _git(
            self.repository_root,
            "cat-file",
            "blob",
            f"{self.source_commit}:{path}",
        )


def _git(repository: Path, *arguments: str) -> bytes:
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=repository,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except OSError as exc:
        raise AuditError(f"cannot execute git: {exc}") from exc
    if completed.returncode != 0:
        reason = completed.stderr.decode("utf-8", errors="replace").strip()
        raise AuditError(f"git {' '.join(arguments[:2])} failed: {reason}")
    return completed.stdout


def _literal_assignments(source: bytes, names: set[str], label: str) -> dict[str, Any]:
    try:
        module = ast.parse(source.decode("utf-8"), filename=label)
    except (UnicodeDecodeError, SyntaxError) as exc:
        raise AuditError(f"cannot parse frozen source {label}: {exc}") from exc
    values: dict[str, Any] = {}
    for node in module.body:
        targets: list[ast.expr] = []
        value: ast.expr | None = None
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
            value = node.value
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
            value = node.value
        if value is None:
            continue
        for target in targets:
            if not isinstance(target, ast.Name) or target.id not in names:
                continue
            try:
                values[target.id] = ast.literal_eval(value)
            except (ValueError, TypeError) as exc:
                raise AuditError(
                    f"frozen constant {label}:{target.id} is not literal"
                ) from exc
    missing = names - set(values)
    if missing:
        raise AuditError(f"frozen source {label} lacks constants: {sorted(missing)}")
    return values


def _lexical_constants(source: bytes, label: str) -> tuple[str, int, frozenset[str]]:
    """Extract the frozen pure-Python lexical tokenizer contract."""

    try:
        module = ast.parse(source.decode("utf-8"), filename=label)
    except (UnicodeDecodeError, SyntaxError) as exc:
        raise AuditError(f"cannot parse frozen source {label}: {exc}") from exc
    pattern: str | None = None
    minimum: int | None = None
    stopwords: frozenset[str] | None = None
    for node in module.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        target = node.targets[0] if isinstance(node, ast.Assign) else node.target
        value = node.value
        if not isinstance(target, ast.Name) or value is None:
            continue
        if target.id == "MIN_TOKEN_LEN":
            raw = ast.literal_eval(value)
            if isinstance(raw, bool) or not isinstance(raw, int):
                raise AuditError("frozen MIN_TOKEN_LEN is not an integer")
            minimum = raw
        elif target.id == "_TOKEN_RE":
            if not (
                isinstance(value, ast.Call)
                and isinstance(value.func, ast.Attribute)
                and value.func.attr == "compile"
                and value.args
            ):
                raise AuditError("frozen _TOKEN_RE is not a literal re.compile")
            raw = ast.literal_eval(value.args[0])
            if not isinstance(raw, str):
                raise AuditError("frozen _TOKEN_RE pattern is not a string")
            pattern = raw
        elif target.id == "STOPWORDS":
            if not (
                isinstance(value, ast.Call)
                and isinstance(value.func, ast.Name)
                and value.func.id == "frozenset"
                and len(value.args) == 1
            ):
                raise AuditError("frozen STOPWORDS is not a literal frozenset")
            raw = ast.literal_eval(value.args[0])
            if not isinstance(raw, set) or any(not isinstance(item, str) for item in raw):
                raise AuditError("frozen STOPWORDS contains non-string values")
            stopwords = frozenset(raw)
    if pattern is None or minimum is None or stopwords is None:
        raise AuditError("frozen lexical tokenizer constants are incomplete")
    return pattern, minimum, stopwords


def _context_budget_default(source: bytes, label: str, field: str) -> int:
    try:
        module = ast.parse(source.decode("utf-8"), filename=label)
    except (UnicodeDecodeError, SyntaxError) as exc:
        raise AuditError(f"cannot parse frozen source {label}: {exc}") from exc
    for node in module.body:
        if not isinstance(node, ast.ClassDef) or node.name != "ContextBudget":
            continue
        for child in node.body:
            if (
                isinstance(child, ast.AnnAssign)
                and isinstance(child.target, ast.Name)
                and child.target.id == field
                and child.value is not None
            ):
                raw = ast.literal_eval(child.value)
                if isinstance(raw, bool) or not isinstance(raw, int) or raw < 1:
                    raise AuditError(f"frozen ContextBudget.{field} is invalid")
                return raw
    raise AuditError(f"frozen ContextBudget lacks {field}")


def _database_schema_sql(source: bytes, label: str) -> str:
    """Reconstruct schema-v9 DDL from literal frozen source fragments."""

    try:
        module = ast.parse(source.decode("utf-8"), filename=label)
    except (UnicodeDecodeError, SyntaxError) as exc:
        raise AuditError(f"cannot parse frozen source {label}: {exc}") from exc
    fragments: dict[str, str] = {}
    version: int | None = None
    base_schema: str | None = None
    wanted = {
        "_V3_INDEXES",
        "_V5_ASSOCIATION_SCHEMA",
        "_V7_HEBBIAN_SCHEMA",
        "_V8_CONSOLIDATION_SCHEMA",
        "_V9_CAUSAL_BINDING_SCHEMA",
    }
    for node in module.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        value = node.value
        for target in targets:
            if not isinstance(target, ast.Name) or value is None:
                continue
            if target.id == "CURRENT_SCHEMA_VERSION":
                raw = ast.literal_eval(value)
                if isinstance(raw, bool) or not isinstance(raw, int):
                    raise AuditError("frozen database schema version is not an integer")
                version = raw
            elif target.id in wanted:
                raw = ast.literal_eval(value)
                if not isinstance(raw, str):
                    raise AuditError(f"frozen database fragment {target.id} is not text")
                fragments[target.id] = raw
            elif target.id == "_SCHEMA_SQL" and base_schema is None:
                try:
                    raw = ast.literal_eval(value)
                except (ValueError, TypeError):
                    continue
                if isinstance(raw, str):
                    base_schema = raw
    if version != 9 or base_schema is None or set(fragments) != wanted:
        raise AuditError("frozen database schema-v9 literals are incomplete")
    return (
        base_schema
        + fragments["_V3_INDEXES"]
        + fragments["_V5_ASSOCIATION_SCHEMA"]
        + fragments["_V7_HEBBIAN_SCHEMA"]
        + fragments["_V8_CONSOLIDATION_SCHEMA"]
        + fragments["_V9_CAUSAL_BINDING_SCHEMA"]
        + "\nINSERT OR REPLACE INTO meta (key, value)"
        + f" VALUES ('schema_version', '{version}');\n"
    )


def load_frozen_source(repository_root: str | Path, source_commit: str) -> FrozenSource:
    root = Path(repository_root).resolve()
    if not (root / ".git").exists():
        raise AuditError(f"repository root has no .git directory: {root}")
    if not _FULL_GIT_SHA1.fullmatch(source_commit):
        raise AuditError("source_commit must be an exact lowercase 40-hex commit ID")
    resolved = _git(root, "rev-parse", "--verify", f"{source_commit}^{{commit}}")
    resolved_text = resolved.decode("ascii", errors="strict").strip()
    if resolved_text != source_commit:
        raise AuditError("source_commit did not resolve to the exact requested commit")

    prefix = "src/memory_condense/"
    raw_paths = _git(
        root,
        "ls-tree",
        "-r",
        "-z",
        "--name-only",
        source_commit,
        "--",
        "src/memory_condense",
    )
    paths = [
        value.decode("utf-8")
        for value in raw_paths.split(b"\0")
        if value and value.decode("utf-8").endswith(".py")
    ]
    if not paths or any(not path.startswith(prefix) for path in paths):
        raise AuditError("frozen commit has no complete src/memory_condense package")
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda value: value[len(prefix) :]):
        relative = path[len(prefix) :].encode("utf-8")
        payload = _git(root, "cat-file", "blob", f"{source_commit}:{path}")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)

    benchmark_path = "src/memory_condense/eval/benchmark.py"
    tokenizer_path = "src/memory_condense/_tokenizer.py"
    context_path = "src/memory_condense/context_packer.py"
    lexical_path = "src/memory_condense/lexical.py"
    database_path = "src/memory_condense/db.py"
    benchmark_source = _git(root, "cat-file", "blob", f"{source_commit}:{benchmark_path}")
    tokenizer_source = _git(root, "cat-file", "blob", f"{source_commit}:{tokenizer_path}")
    context_source = _git(root, "cat-file", "blob", f"{source_commit}:{context_path}")
    lexical_source = _git(root, "cat-file", "blob", f"{source_commit}:{lexical_path}")
    database_source = _git(root, "cat-file", "blob", f"{source_commit}:{database_path}")
    prompts = _literal_assignments(
        benchmark_source,
        {
            "QA_SYSTEM_PROMPT",
            "QA_USER_TEMPLATE",
            "QA_NO_CONTEXT",
            "JUDGE_SYSTEM_PROMPT",
            "JUDGE_USER_TEMPLATE",
        },
        benchmark_path,
    )
    tokenizer = _literal_assignments(
        tokenizer_source,
        {
            "PROMPT_TOKEN_PROXY_SCHEMA",
            "DEFAULT_ENCODING",
            "CHAT_FRAMING_TOKENS_PER_MESSAGE",
            "CHAT_FRAMING_TOKENS_FIXED",
        },
        tokenizer_path,
    )
    lexical_pattern, lexical_minimum, lexical_stopwords = _lexical_constants(
        lexical_source,
        lexical_path,
    )
    max_expansion_tokens = _context_budget_default(
        context_source,
        context_path,
        "max_expansion_tokens",
    )
    database_schema_sql = _database_schema_sql(database_source, database_path)
    if not all(isinstance(prompts[name], str) for name in prompts):
        raise AuditError("frozen QA prompt constants must be strings")
    if not isinstance(tokenizer["PROMPT_TOKEN_PROXY_SCHEMA"], str) or not isinstance(
        tokenizer["DEFAULT_ENCODING"], str
    ):
        raise AuditError("frozen tokenizer identity constants must be strings")
    framing_per = tokenizer["CHAT_FRAMING_TOKENS_PER_MESSAGE"]
    framing_fixed = tokenizer["CHAT_FRAMING_TOKENS_FIXED"]
    if isinstance(framing_per, bool) or not isinstance(framing_per, int):
        raise AuditError("frozen per-message framing reserve must be an integer")
    if isinstance(framing_fixed, bool) or not isinstance(framing_fixed, int):
        raise AuditError("frozen fixed framing reserve must be an integer")

    environment = _git(root, "cat-file", "blob", f"{source_commit}:pixi.lock")
    return FrozenSource(
        repository_root=root,
        source_commit=source_commit,
        implementation_sha256=digest.hexdigest(),
        environment_lock_sha256=bytes_sha256(environment),
        environment_lock=environment,
        qa_system_prompt=str(prompts["QA_SYSTEM_PROMPT"]),
        qa_user_template=str(prompts["QA_USER_TEMPLATE"]),
        qa_no_context=str(prompts["QA_NO_CONTEXT"]),
        judge_system_prompt=str(prompts["JUDGE_SYSTEM_PROMPT"]),
        judge_user_template=str(prompts["JUDGE_USER_TEMPLATE"]),
        prompt_proxy_schema=str(tokenizer["PROMPT_TOKEN_PROXY_SCHEMA"]),
        prompt_encoding=str(tokenizer["DEFAULT_ENCODING"]),
        framing_per_message=framing_per,
        framing_fixed=framing_fixed,
        benchmark_source_sha256=bytes_sha256(benchmark_source),
        tokenizer_source_sha256=bytes_sha256(tokenizer_source),
        context_packer_source_sha256=bytes_sha256(context_source),
        lexical_source_sha256=bytes_sha256(lexical_source),
        database_source_sha256=bytes_sha256(database_source),
        database_schema_sql=database_schema_sql,
        lexical_token_pattern=lexical_pattern,
        lexical_min_token_len=lexical_minimum,
        lexical_stopwords=lexical_stopwords,
        max_expansion_tokens=max_expansion_tokens,
    )


def verify_repository_blob(source: FrozenSource, path: str | Path, label: str) -> str:
    """Require a repository input to equal the blob at the frozen commit."""

    target = Path(path).resolve()
    try:
        relative = target.relative_to(source.repository_root).as_posix()
    except ValueError as exc:
        raise AuditError(f"{label} must be inside the repository root") from exc
    try:
        current = target.read_bytes()
    except OSError as exc:
        raise AuditError(f"cannot read {label} {target}: {exc}") from exc
    frozen = source.blob(relative)
    if current != frozen:
        raise AuditError(f"{label} differs from {source.source_commit}:{relative}")
    return relative
