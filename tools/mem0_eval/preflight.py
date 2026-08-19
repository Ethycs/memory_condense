"""Provider-free preflight for the locked 1M Mem0 comparison campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from memory_condense.eval.reproducibility import (
    environment_lock_sha256,
    file_sha256,
    implementation_sha256,
    project_root,
)

from .protocol import (
    RawStressShard,
    build_raw_stress_shards,
    load_locked_raw_population,
    shard_receipt,
    validate_raw_stress_shard,
)


@dataclass(frozen=True, slots=True)
class SourceValidationPlan:
    dataset_sha256: str
    split_manifest_sha256: str
    policy_manifest_sha256: str
    implementation_sha256: str
    environment_lock_sha256: str
    sample_offsets: tuple[int, ...]
    target_tokens: int
    questions_per_shard: int
    evaluation_identity: Mapping[str, Any]


def _required_int(value: Any, label: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _required_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value.strip()


def load_source_validation_plan(
    *,
    benchmark_file: str | Path,
    split_manifest: str | Path,
    policy_manifest: str | Path,
    repository_root: str | Path | None = None,
) -> SourceValidationPlan:
    """Verify only the source artifacts needed by the Mem0 comparison.

    This intentionally avoids importing the main campaign/benchmark runner,
    keeping the future isolated Mem0 environment free of PyTorch, HNSW, and
    sentence-transformer dependencies.
    """

    repo = (
        Path(repository_root).resolve()
        if repository_root is not None
        else project_root().resolve()
    )
    dataset_path = Path(benchmark_file).resolve()
    split_path = Path(split_manifest).resolve()
    policy_path = Path(policy_manifest).resolve()
    policy_bytes = policy_path.read_bytes()
    try:
        policy = json.loads(policy_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot parse source validation policy: {exc}") from exc
    if not isinstance(policy, dict):
        raise ValueError("source validation policy must be a JSON object")
    required = {
        "format": "memory-condense-retrieval-policy-v1",
        "status": "validation_frozen",
        "split": "validation",
        "claim_profile": "longmemeval-s-1m-100q-95-v1",
    }
    for field, expected in required.items():
        if policy.get(field) != expected:
            raise ValueError(f"source validation policy {field} mismatch")

    dataset_digest = file_sha256(dataset_path)
    split_digest = file_sha256(split_path)
    code_digest = implementation_sha256(repo / "src" / "memory_condense")
    lock_digest = environment_lock_sha256(repo)
    expected_hashes = {
        "dataset_sha256": dataset_digest,
        "split_manifest_sha256": split_digest,
        "implementation_sha256": code_digest,
        "environment_lock_sha256": lock_digest,
    }
    for field, actual in expected_hashes.items():
        if policy.get(field) != actual:
            raise ValueError(f"source validation policy {field} mismatch")
    if policy.get("split_manifest") != split_path.name:
        raise ValueError("source validation split-manifest basename mismatch")

    selection_value = policy.get("selection_artifact")
    selection_digest = policy.get("selection_artifact_sha256")
    if policy.get("selection_artifact_required") is not True:
        raise ValueError("source validation policy must require its selection artifact")
    if not isinstance(selection_value, str) or not selection_value.strip():
        raise ValueError("source validation policy has no selection artifact")
    relative = Path(selection_value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("source validation selection path is unsafe")
    selection_path = (repo / relative).resolve()
    try:
        selection_path.relative_to(repo)
    except ValueError as exc:
        raise ValueError("source validation selection path escapes the repository") from exc
    if not selection_path.is_file() or file_sha256(selection_path) != selection_digest:
        raise ValueError("source validation selection artifact mismatch")

    evaluation = policy.get("evaluation")
    if not isinstance(evaluation, dict):
        raise ValueError("source validation policy has no evaluation object")
    target_tokens = _required_int(
        evaluation.get("stress_context_tokens"),
        "stress_context_tokens",
        minimum=1,
    )
    questions_per_shard = _required_int(
        evaluation.get("stress_questions"),
        "stress_questions",
        minimum=1,
    )
    raw_offsets = evaluation.get("sample_offsets")
    if (
        not isinstance(raw_offsets, list)
        or not raw_offsets
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in raw_offsets
        )
        or len(raw_offsets) != len(set(raw_offsets))
    ):
        raise ValueError("source validation sample offsets are invalid")
    if evaluation.get("stress_question_offset") != 0:
        raise ValueError("source validation stress question offset must be zero")
    if evaluation.get("min_target_questions") != len(raw_offsets) * questions_per_shard:
        raise ValueError("source validation population size is inconsistent")
    if evaluation.get("accuracy_target") != 0.95:
        raise ValueError("source validation accuracy target must be 0.95")
    if evaluation.get("benchmark_format") != "longmemeval":
        raise ValueError("source validation benchmark format must be longmemeval")
    if evaluation.get("use_judge") is not True:
        raise ValueError("source validation must use the frozen judge")
    if evaluation.get("provider_retries") != 0:
        raise ValueError("source validation provider retries must be zero")
    if evaluation.get("max_provider_calls") != 2 * questions_per_shard:
        raise ValueError("source validation provider-call authorization is inconsistent")
    if evaluation.get("max_samples") != 1:
        raise ValueError("source validation must process one stress sample per shard")
    if evaluation.get("recent_window") != 4:
        raise ValueError("source validation recent-window identity mismatch")
    prompt_cap = _required_int(
        evaluation.get("max_prompt_tokens"),
        "max_prompt_tokens",
        minimum=1,
    )
    output_reserve = _required_int(
        evaluation.get("responder_output_token_reserve"),
        "responder_output_token_reserve",
        minimum=1,
    )
    if output_reserve >= prompt_cap:
        raise ValueError("source validation output reserve must be below its prompt cap")
    if evaluation.get("prompt_cap_semantics") != (
        "local_prompt_token_proxy_with_provider_usage_postcheck_v1"
    ):
        raise ValueError("source validation prompt-cap semantics mismatch")
    proxy_identity = evaluation.get("prompt_token_proxy_identity")
    if not isinstance(proxy_identity, dict) or proxy_identity != {
        "schema": "memory-condense-prompt-token-proxy-v1",
        "implementation": "tiktoken",
        "implementation_version": "0.13.0",
        "encoding": "cl100k_base",
        "vocabulary_sha256": (
            "8cd4fc3b76f9fdaf9df7d14f20a41eda79ce45b3e9c5ae8f68b0a41a59c3a9c9"
        ),
        "chat_framing_tokens_per_message": 8,
        "chat_framing_tokens_fixed": 8,
    }:
        raise ValueError("source validation prompt-token proxy identity mismatch")
    evaluation_identity = {
        "responder_model": _required_text(
            evaluation.get("responder_model"), "responder_model"
        ),
        "judge_model": _required_text(evaluation.get("judge_model"), "judge_model"),
        "use_judge": True,
        "provider_retries": 0,
        "max_provider_calls_per_shard": 2 * questions_per_shard,
        "max_prompt_tokens": prompt_cap,
        "prompt_cap_semantics": evaluation["prompt_cap_semantics"],
        "prompt_token_proxy_identity": dict(proxy_identity),
        "responder_output_token_reserve": output_reserve,
        "recent_window": 4,
        "accuracy_target": 0.95,
        "min_target_questions": len(raw_offsets) * questions_per_shard,
        "stress_context_tokens": target_tokens,
        "stress_questions": questions_per_shard,
        "stress_question_offset": 0,
        "max_samples": 1,
        "sample_offsets": list(raw_offsets),
    }
    return SourceValidationPlan(
        dataset_sha256=dataset_digest,
        split_manifest_sha256=split_digest,
        policy_manifest_sha256=hashlib.sha256(policy_bytes).hexdigest(),
        implementation_sha256=code_digest,
        environment_lock_sha256=lock_digest,
        sample_offsets=tuple(raw_offsets),
        target_tokens=target_tokens,
        questions_per_shard=questions_per_shard,
        evaluation_identity=evaluation_identity,
    )


def tool_implementation_sha256(root: str | Path | None = None) -> str:
    """Hash the comparison tool independently of frozen v3 package code."""

    package = Path(root).resolve() if root is not None else Path(__file__).parent
    digest = hashlib.sha256()
    for path in sorted(package.rglob("*.py"), key=lambda item: item.as_posix()):
        relative = path.relative_to(package).as_posix().encode("utf-8")
        payload = path.read_bytes()
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _is_equal_to_or_within(path: Path, parent: Path) -> bool:
    return path == parent or parent in path.parents


def _validated_output_target(
    output: str | Path,
    *,
    benchmark_file: str | Path,
    split_manifest: str | Path,
    policy_manifest: str | Path,
    repository_root: str | Path | None,
) -> Path:
    """Reject output aliases that could invalidate a preflight identity."""

    target = Path(output).resolve(strict=False)
    repository = (
        Path(repository_root).resolve(strict=False)
        if repository_root is not None
        else project_root().resolve()
    )
    protected = (
        (Path(benchmark_file).resolve(strict=False), "benchmark input"),
        (Path(split_manifest).resolve(strict=False), "split-manifest input"),
        (Path(policy_manifest).resolve(strict=False), "policy-manifest input"),
        (
            (repository / "pixi.lock").resolve(strict=False),
            "source environment-lock input",
        ),
        (
            (repository / "src" / "memory_condense").resolve(strict=False),
            "source implementation root",
        ),
        (Path(__file__).resolve().parent, "Mem0 tool implementation root"),
    )
    for protected_path, label in protected:
        if _is_equal_to_or_within(target, protected_path):
            raise ValueError(
                f"preflight output {target} equals or descends from protected "
                f"{label} {protected_path}"
            )
    return target


def _atomic_create_bytes(path: Path, payload: bytes) -> None:
    """Atomically create one flushed file without replacing an existing one."""

    target = path.resolve(strict=False)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(f"refusing to replace existing preflight output {target}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".staging", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def build_preflight_receipt(
    *,
    benchmark_file: str | Path,
    split_manifest: str | Path,
    policy_manifest: str | Path,
    repository_root: str | Path | None = None,
) -> dict[str, Any]:
    """Reconstruct and cross-check all ten shards without loading Mem0."""

    plan = load_source_validation_plan(
        benchmark_file=benchmark_file,
        split_manifest=split_manifest,
        policy_manifest=policy_manifest,
        repository_root=repository_root,
    )
    target_tokens = plan.target_tokens
    questions_per_shard = plan.questions_per_shard
    rows: list[dict[str, Any]] = []
    total_raw_pairs = 0
    total_skipped = 0
    total_adds = 0
    all_question_ids: list[str] = []
    shards = build_raw_stress_shards(
        benchmark_file=benchmark_file,
        split_manifest=split_manifest,
        sample_offsets=plan.sample_offsets,
        target_tokens=target_tokens,
        max_questions=questions_per_shard,
    )
    for offset, shard in zip(plan.sample_offsets, shards, strict=True):
        validate_raw_stress_shard(shard)
        if len(shard.history_sample_ids) != questions_per_shard:
            raise ValueError(
                f"Mem0 shard {offset} admitted {len(shard.history_sample_ids)} "
                f"histories; expected {questions_per_shard}"
            )
        row = shard_receipt(shard)
        row["history_sample_ids_sha256"] = _sha256_json(
            list(shard.history_sample_ids)
        )
        row["question_ids_sha256"] = _sha256_json(list(shard.question_ids))
        rows.append(row)
        total_raw_pairs += shard.add_counts.raw_pairs
        total_skipped += shard.add_counts.skipped_empty_pairs
        total_adds += shard.add_counts.add_requests
        all_question_ids.extend(shard.question_ids)

    if len(all_question_ids) != len(set(all_question_ids)):
        raise ValueError("Mem0 comparison shards repeat question IDs")
    population = load_locked_raw_population(
        benchmark_file=benchmark_file,
        split_manifest=split_manifest,
    )
    validation_question_ids = {
        question.question_id
        for sample in population.validation
        for question in sample.questions
    }
    if set(all_question_ids) != validation_question_ids:
        raise ValueError("Mem0 comparison shards do not cover the frozen population")

    # The receipt is a freeze input.  Re-read and re-hash every bound source
    # after reconstruction so a concurrent edit cannot relabel different
    # dataset/policy/code bytes with the initial identities.
    final_plan = load_source_validation_plan(
        benchmark_file=benchmark_file,
        split_manifest=split_manifest,
        policy_manifest=policy_manifest,
        repository_root=repository_root,
    )
    if final_plan != plan:
        raise ValueError("source validation inputs changed during Mem0 preflight")

    return {
        "format": "memory-condense-mem0-comparison-preflight-v1",
        "status": "provider_free_ready",
        "dataset_sha256": plan.dataset_sha256,
        "split_manifest_sha256": plan.split_manifest_sha256,
        "source_validation_policy_sha256": plan.policy_manifest_sha256,
        "source_implementation_sha256": plan.implementation_sha256,
        "source_environment_lock_sha256": plan.environment_lock_sha256,
        "mem0_tool_implementation_sha256": tool_implementation_sha256(),
        "input_order_protocol": (
            "locked-record-order+official-within-record-date-sort+"
            "consecutive-1-or-2-turn-slices-v1"
        ),
        "source_session_date_exposure": "diagnostics_only_not_model_input",
        "retrieved_created_at_exposure": "answer_prompt_date_headings",
        "provenance": "request_window_non_evidence",
        "supports_exact_source_provenance": False,
        "source_evaluation_identity": dict(plan.evaluation_identity),
        "sample_offsets": list(plan.sample_offsets),
        "shards": rows,
        "totals": {
            "shards": len(rows),
            "questions": len(all_question_ids),
            "question_ids_sha256": _sha256_json(sorted(all_question_ids)),
            "raw_pairs": total_raw_pairs,
            "skipped_empty_pairs": total_skipped,
            "mem0_add_operations": total_adds,
            "expected_logical_extraction_calls": total_adds,
            "logical_extraction_call_boundary": "Memory.llm.generate_response",
            "logical_extraction_calls_per_add": 1,
            "mem0_search_operations": len(all_question_ids),
            "responder_calls": len(all_question_ids),
            "judge_calls": len(all_question_ids),
            "answer_judge_provider_calls": 2 * len(all_question_ids),
            "underlying_mem0_provider_calls": None,
            "underlying_mem0_provider_usage_status": (
                "unavailable_from_mem0_oss_public_api"
            ),
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reconstruct the locked Mem0 comparison without model calls"
    )
    parser.add_argument("--benchmark-file", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--policy-manifest", required=True)
    parser.add_argument("--repository-root")
    parser.add_argument("--output")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output = None
    if args.output:
        output = _validated_output_target(
            args.output,
            benchmark_file=args.benchmark_file,
            split_manifest=args.split_manifest,
            policy_manifest=args.policy_manifest,
            repository_root=args.repository_root,
        )
    receipt = build_preflight_receipt(
        benchmark_file=args.benchmark_file,
        split_manifest=args.split_manifest,
        policy_manifest=args.policy_manifest,
        repository_root=args.repository_root,
    )
    rendered = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    if output is not None:
        _atomic_create_bytes(output, rendered.encode("utf-8"))
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
