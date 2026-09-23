"""Prove and time binary ranked-prefix packing over the sealed v7 full100.

This provider-free successor changes only the packing algorithm.  It opens the
sealed adaptive-v7 selection, rebuilds every selected evidence prefix with the
production binary packer and the existing v7 prompt renderer/token counters,
and requires exact equality with all 100 sealed provider payloads.  Timing is
kept out of the deterministic run receipt and records only the binary packer
call; loading, validation, serialization, hashing, and provider work are
excluded.
"""

from __future__ import annotations

import argparse
import hashlib
import statistics
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from memory_condense.domain.discourse import identity_sha256, quote_sha256
from memory_condense.domain.integrity import file_sha256
from memory_condense.search.packing.ranked_prefix_prompt import (
    AUDIT_FORMAT,
    PACKER_ID,
    RankedPrefixPromptPack,
    pack_ranked_prefix_prompt,
)

try:
    from tools import assay_hot_retrieval_1m as hot
    from tools import assay_hot_retrieval_adaptive_full100 as v7
    from tools import assay_hot_retrieval_full100 as full100
except ModuleNotFoundError:  # Direct ``python tools/...py`` execution.
    import assay_hot_retrieval_1m as hot
    import assay_hot_retrieval_adaptive_full100 as v7
    import assay_hot_retrieval_full100 as full100


RUN_FORMAT = "memory-condense-hot-retrieval-binary-packing-full100-run-v1"
RUNTIME_FORMAT = (
    "memory-condense-hot-retrieval-binary-packing-full100-runtime-v1"
)
REPLAY_FORMAT = (
    "memory-condense-hot-retrieval-binary-packing-full100-replay-v1"
)
POLICY_ID = "hot-raw-chunk-v8-binary-ranked-prefix-equivalence"
EXPECTED_V7_SELECTION_SHA256 = (
    "867a4439af1c369c3f702491045392b8c64c5e2b3b4216c93fd973ada1b6df20"
)
EXPECTED_POPULATION_SHA256 = v7.EXPECTED_POPULATION_SHA256
EXPECTED_QUESTION_COUNT = v7.EXPECTED_QUESTION_COUNT
MAX_CONTEXT_TOKENS = v7.MAX_CONTEXT_TOKENS
MAX_PROMPT_TOKENS = v7.MAX_PROMPT_TOKENS
OUTPUT_TOKEN_RESERVE = hot.RESPONDER_OUTPUT_TOKEN_RESERVE

DEFAULT_V7_ROOT = v7.DEFAULT_OUTPUT_ROOT
DEFAULT_OUTPUT_ROOT = Path(
    "eval_results/longmemeval-1m-hot-retrieval-binary-packing-"
    "full100-validation-20260905"
)
RUN_NAME = "run.json"
RUNTIME_NAME = "runtime.json"
REPLAY_NAME = "replay.json"
_EQUIVALENCE_FIELDS = (
    "packed_count",
    "ranked_prefix",
    "context_tokens",
    "prompt_tokens",
    "prompt_workspace_tokens",
    "provider_messages",
    "provider_payload_bytes",
    "provider_payload_sha256",
)


def _implementation_identity() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    paths = (
        "tools/assay_hot_retrieval_binary_packing_full100.py",
        "tools/assay_hot_retrieval_adaptive_full100.py",
        "tools/assay_hot_retrieval_full100.py",
        "tools/assay_hot_retrieval_1m.py",
        "src/memory_condense/search/packing/ranked_prefix_prompt.py",
        "src/memory_condense/eval/_retrieval_qa_prompt.py",
        "src/memory_condense/domain/_tokenizer.py",
        "src/memory_condense/domain/discourse.py",
        "src/memory_condense/domain/integrity.py",
    )
    files = {path: file_sha256(root / path) for path in paths}
    return {
        "format": (
            "memory-condense-hot-retrieval-binary-packing-"
            "full100-implementation-v1"
        ),
        "files": files,
        "sha256": identity_sha256(
            [{"path": path, "sha256": digest} for path, digest in files.items()]
        ),
    }


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _relative_to_repository(path: Path, *, label: str) -> str:
    try:
        relative = path.resolve().relative_to(_repository_root())
    except ValueError as exc:
        raise ValueError(f"{label} must be inside the repository") from exc
    return relative.as_posix()


def _exact_nonnegative_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    if value < 0:
        raise ValueError(f"{label} must be non-negative")
    return value


def _load_v7_bundle(
    v7_root: Path,
) -> tuple[dict[str, Any], str, dict[str, Any], str]:
    selection, selection_sha = v7._load_selection(v7_root)  # noqa: SLF001
    if selection_sha != EXPECTED_V7_SELECTION_SHA256:
        raise ValueError(
            f"sealed v7 selection changed ({selection_sha} != "
            f"{EXPECTED_V7_SELECTION_SHA256})"
        )
    runtime, runtime_sha = hot._read_json_artifact(  # noqa: SLF001
        v7_root / v7.RUNTIME_NAME
    )
    samples = runtime.get("samples")
    if (
        runtime.get("format") != v7.RUNTIME_FORMAT
        or runtime.get("status")
        != "provider_free_adaptive_increment_over_sealed_parent"
        or runtime.get("selection_sha256") != selection_sha
        or runtime.get("population_identity_sha256")
        != EXPECTED_POPULATION_SHA256
        or runtime.get("provider_calls") != 0
        or runtime.get("qwen_calls") != 0
        or not isinstance(samples, list)
        or len(samples) != EXPECTED_QUESTION_COUNT
        or [row.get("ordinal") for row in samples]
        != list(range(EXPECTED_QUESTION_COUNT))
    ):
        raise ValueError("sealed v7 runtime receipt changed")
    for selected, sample in zip(
        selection["questions"], samples, strict=True
    ):
        timings = sample.get("timings_ns")
        if (
            sample.get("question_id") != selected.get("question_id")
            or sample.get("shard_offset") != selected.get("shard_offset")
            or not isinstance(timings, Mapping)
        ):
            raise ValueError("sealed v7 runtime sample binding changed")
        _exact_nonnegative_int(
            timings.get("pack_and_prompt_render_count_ns"),
            "sealed v7 linear pack timing",
        )
    return selection, selection_sha, runtime, runtime_sha


def _load_prompt_map(
    selection: Mapping[str, Any],
) -> tuple[dict[int, str], str]:
    bindings = selection.get("bindings")
    if not isinstance(bindings, Mapping):
        raise ValueError("sealed v7 selection omitted bindings")
    parent_root = v7._bound_parent_root(bindings)  # noqa: SLF001
    probes, probes_sha = full100._load_probes(parent_root)  # noqa: SLF001
    rows = probes.get("questions")
    if (
        not isinstance(rows, list)
        or len(rows) != EXPECTED_QUESTION_COUNT
        or [row.get("ordinal") for row in rows]
        != list(range(EXPECTED_QUESTION_COUNT))
    ):
        raise ValueError("sealed prompt population changed")
    prompts: dict[int, str] = {}
    for selected, probe in zip(selection["questions"], rows, strict=True):
        ordinal = int(selected["ordinal"])
        prompt = probe.get("prompt_question")
        if (
            not isinstance(prompt, str)
            or not prompt
            or probe.get("question_id") != selected.get("question_id")
            or quote_sha256(prompt) != selected.get("prompt_question_sha256")
        ):
            raise ValueError("sealed prompt/question binding changed")
        prompts[ordinal] = prompt
    return prompts, probes_sha


def _expected_payload_bytes(arm: Mapping[str, Any]) -> bytes:
    messages = arm.get("provider_messages")
    if not isinstance(messages, list):
        raise ValueError("sealed v7 arm omitted provider messages")
    payload = hot._canonical_json_bytes({"messages": messages})  # noqa: SLF001
    if (
        len(payload) != arm.get("provider_payload_utf8_bytes")
        or hashlib.sha256(payload).hexdigest()
        != arm.get("provider_payload_sha256")
    ):
        raise ValueError("sealed v7 provider payload bytes changed")
    return payload


def _repack_question(
    selected: Mapping[str, Any],
    *,
    prompt_question: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    arms = selected.get("arms")
    arm = (
        arms.get("a3_protected_union")
        if isinstance(arms, Mapping)
        else None
    )
    if not isinstance(arm, Mapping):
        raise ValueError("sealed v7 row omitted its provider arm")
    selected_rows = arm.get("selected_evidence")
    packed_rows = arm.get("packed_evidence")
    if (
        not isinstance(selected_rows, list)
        or not isinstance(packed_rows, list)
        or not all(isinstance(row, Mapping) for row in selected_rows)
        or not all(isinstance(row, Mapping) for row in packed_rows)
    ):
        raise ValueError("sealed v7 arm omitted ranked rendered evidence")
    rendered_texts = tuple(str(row.get("rendered_text", "")) for row in selected_rows)
    if any(not text for text in rendered_texts):
        raise ValueError("sealed v7 selected evidence omitted rendered text")

    started = time.perf_counter_ns()
    packed: RankedPrefixPromptPack[str, list[dict[str, str]]] = (
        pack_ranked_prefix_prompt(
            rendered_texts,
            count_context_tokens=hot._context_token_proxy,  # noqa: SLF001
            render_prompt=lambda prefix: hot.build_qa_prompt(
                prompt_question, list(prefix)
            ),
            count_prompt_tokens=hot.count_chat_prompt_token_proxy,
            max_context_tokens=MAX_CONTEXT_TOKENS,
            max_prompt_tokens=MAX_PROMPT_TOKENS,
            output_token_reserve=OUTPUT_TOKEN_RESERVE,
        )
    )
    pack_only_ns = time.perf_counter_ns() - started

    expected_bytes = _expected_payload_bytes(arm)
    actual_bytes = hot._canonical_json_bytes(  # noqa: SLF001
        {"messages": packed.rendered_prompt}
    )
    expected_packed_count = len(packed_rows)
    expected_packed_texts = tuple(
        str(row.get("rendered_text", "")) for row in packed_rows
    )
    equivalence = {
        "packed_count": packed.packed_count == expected_packed_count,
        "ranked_prefix": packed.packed_items == expected_packed_texts,
        "context_tokens": (
            packed.context_token_count == arm.get("context_token_proxy")
        ),
        "prompt_tokens": (
            packed.prompt_token_count == arm.get("prompt_token_proxy")
        ),
        "prompt_workspace_tokens": (
            packed.prompt_workspace_token_count
            == arm.get("prompt_workspace_token_proxy")
        ),
        "provider_messages": packed.rendered_prompt == arm.get("provider_messages"),
        "provider_payload_bytes": actual_bytes == expected_bytes,
        "provider_payload_sha256": (
            hashlib.sha256(actual_bytes).hexdigest()
            == arm.get("provider_payload_sha256")
        ),
    }
    failed = [name for name, matches in equivalence.items() if not matches]
    if failed:
        raise RuntimeError(
            f"binary packing differs from sealed v7 at ordinal "
            f"{selected.get('ordinal')}: {', '.join(failed)}"
        )
    selected_ids = [str(row.get("chunk_id", "")) for row in selected_rows]
    packed_ids = [str(row.get("chunk_id", "")) for row in packed_rows]
    semantic = {
        "ordinal": int(selected["ordinal"]),
        "shard_offset": int(selected["shard_offset"]),
        "local_ordinal": int(selected["local_ordinal"]),
        "question_id": str(selected["question_id"]),
        "probe_sha256": str(selected["probe_sha256"]),
        "prompt_question_sha256": str(selected["prompt_question_sha256"]),
        "selected_count": len(selected_rows),
        "packed_count": packed.packed_count,
        "dropped_count": len(selected_rows) - packed.packed_count,
        "selected_chunk_sequence_sha256": identity_sha256(selected_ids),
        "packed_chunk_sequence_sha256": identity_sha256(packed_ids),
        "context_token_proxy": packed.context_token_count,
        "prompt_token_proxy": packed.prompt_token_count,
        "prompt_workspace_token_proxy": packed.prompt_workspace_token_count,
        "provider_payload_utf8_bytes": len(actual_bytes),
        "provider_payload_sha256": hashlib.sha256(actual_bytes).hexdigest(),
        "equivalence": equivalence,
        "all_equivalent": True,
        "packer_audit": packed.audit.projection(),
    }
    timing = {
        "ordinal": semantic["ordinal"],
        "question_id": semantic["question_id"],
        "shard_offset": semantic["shard_offset"],
        "binary_pack_only_ns": pack_only_ns,
    }
    return semantic, timing


def _collect(
    selection: Mapping[str, Any],
    runtime: Mapping[str, Any],
    prompt_by_ordinal: Mapping[int, str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    baseline_by_ordinal = {
        int(row["ordinal"]): row
        for row in runtime["samples"]
    }
    rows: list[dict[str, Any]] = []
    timings: list[dict[str, Any]] = []
    for selected in selection["questions"]:
        ordinal = int(selected["ordinal"])
        prompt = prompt_by_ordinal.get(ordinal)
        if not isinstance(prompt, str) or not prompt:
            raise ValueError(f"ordinal {ordinal} omitted its sealed prompt")
        semantic, timing = _repack_question(
            selected,
            prompt_question=prompt,
        )
        baseline = baseline_by_ordinal.get(ordinal)
        baseline_timings = (
            baseline.get("timings_ns")
            if isinstance(baseline, Mapping)
            else None
        )
        if not isinstance(baseline_timings, Mapping):
            raise ValueError(f"ordinal {ordinal} omitted its v7 baseline")
        timing["sealed_v7_linear_pack_only_ns"] = _exact_nonnegative_int(
            baseline_timings.get("pack_and_prompt_render_count_ns"),
            "sealed v7 linear pack timing",
        )
        rows.append(semantic)
        timings.append(timing)
    rows.sort(key=lambda row: int(row["ordinal"]))
    timings.sort(key=lambda row: int(row["ordinal"]))
    if [row["ordinal"] for row in rows] != list(range(EXPECTED_QUESTION_COUNT)):
        raise RuntimeError("binary packing population changed")
    return rows, timings


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(rows) != EXPECTED_QUESTION_COUNT:
        raise ValueError("binary packing aggregate requires the locked full100")
    audits = [row["packer_audit"] for row in rows]
    return {
        "question_count": len(rows),
        "byte_equivalent_question_count": sum(
            row.get("all_equivalent") is True for row in rows
        ),
        "payload_change_count": sum(
            row.get("all_equivalent") is not True for row in rows
        ),
        "selected_count": sum(int(row["selected_count"]) for row in rows),
        "packed_count": sum(int(row["packed_count"]) for row in rows),
        "dropped_count": sum(int(row["dropped_count"]) for row in rows),
        "complete_prefix_fast_path_count": sum(
            audit.get("complete_prefix_fast_path") is True for audit in audits
        ),
        "binary_fallback_count": sum(
            audit.get("complete_prefix_fast_path") is False for audit in audits
        ),
        "context_count_call_count": sum(
            int(audit["context_count_call_count"]) for audit in audits
        ),
        "prompt_render_call_count": sum(
            int(audit["prompt_render_call_count"]) for audit in audits
        ),
        "prompt_count_call_count": sum(
            int(audit["prompt_count_call_count"]) for audit in audits
        ),
        "all_payloads_byte_equivalent": all(
            row.get("all_equivalent") is True for row in rows
        ),
    }


def _controls() -> dict[str, Any]:
    return {
        "policy_id": POLICY_ID,
        "packer_id": PACKER_ID,
        "packer_audit_format": AUDIT_FORMAT,
        "selection_rule": "longest_fitting_ranked_prefix",
        "search": "complete_prefix_fast_path_then_binary_search",
        "monotonicity_contract": "nondecreasing_append_only_prefix_costs",
        "equivalence_oracle": "sealed_v7_linear_ranked_prefix_payload",
        "max_context_token_proxy": MAX_CONTEXT_TOKENS,
        "max_prompt_workspace_token_proxy": MAX_PROMPT_TOKENS,
        "responder_output_token_reserve": OUTPUT_TOKEN_RESERVE,
    }


def _run_body(
    *,
    v7_root: Path,
    selection_sha: str,
    v7_runtime_sha: str,
    probes_sha: str,
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    materialized_rows = [dict(row) for row in rows]
    return {
        "format": RUN_FORMAT,
        "status": "sealed_provider_free_binary_packing_v7_equivalence",
        "bindings": {
            "v7_output_relative_path": _relative_to_repository(
                v7_root, label="v7 output root"
            ),
            "v7_selection_sha256": selection_sha,
            "v7_runtime_sha256": v7_runtime_sha,
            "prompt_probes_sha256": probes_sha,
            "population_identity_sha256": EXPECTED_POPULATION_SHA256,
        },
        "controls": _controls(),
        "implementation": _implementation_identity(),
        "question_population_sha256": identity_sha256(materialized_rows),
        "questions": materialized_rows,
        "aggregate": _aggregate(materialized_rows),
        "gold_fields_present": False,
        "retained_request_token_state_bytes": 0,
        "qwen_calls": 0,
        "responder_calls": 0,
        "judge_calls": 0,
        "provider_calls": 0,
    }


def _timing_summary(values: Sequence[int]) -> dict[str, int | float]:
    ordered = sorted(_exact_nonnegative_int(value, "timing") for value in values)
    if not ordered:
        raise ValueError("cannot summarize empty timings")
    return {
        "count": len(ordered),
        "min": ordered[0],
        "p50": statistics.median(ordered),
        "mean": statistics.fmean(ordered),
        "p95": ordered[max(0, (95 * len(ordered) + 99) // 100 - 1)],
        "max": ordered[-1],
        "total": sum(ordered),
    }


def _speedup(baseline: int | float, treatment: int | float) -> float | None:
    return None if treatment <= 0 else float(baseline) / float(treatment)


def _runtime_body(
    *,
    run_sha: str,
    selection_sha: str,
    v7_runtime_sha: str,
    elapsed_ns: int,
    timings: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    samples = [dict(row) for row in timings]
    binary = _timing_summary(
        [int(row["binary_pack_only_ns"]) for row in samples]
    )
    baseline = _timing_summary(
        [int(row["sealed_v7_linear_pack_only_ns"]) for row in samples]
    )
    return {
        "format": RUNTIME_FORMAT,
        "status": "sealed_binary_pack_only_runtime_over_v7_payloads",
        "run_sha256": run_sha,
        "v7_selection_sha256": selection_sha,
        "v7_runtime_sha256": v7_runtime_sha,
        "population_identity_sha256": EXPECTED_POPULATION_SHA256,
        "implementation_sha256": _implementation_identity()["sha256"],
        "elapsed_ns": elapsed_ns,
        "samples": samples,
        "binary_pack_only_ns": binary,
        "sealed_v7_linear_pack_only_ns": baseline,
        "speedup": {
            "total_ratio": _speedup(baseline["total"], binary["total"]),
            "mean_ratio": _speedup(baseline["mean"], binary["mean"]),
            "p50_ratio": _speedup(baseline["p50"], binary["p50"]),
            "p95_ratio": _speedup(baseline["p95"], binary["p95"]),
        },
        "timing_scope": {
            "binary": "pack_ranked_prefix_prompt_call_only",
            "baseline": (
                "sealed_v7_pack_and_prompt_render_count_ns_"
                "non_contemporaneous_same_machine_artifact"
            ),
            "excluded": [
                "artifact_loading",
                "v7_validation",
                "payload_serialization",
                "payload_hashing",
                "provider_rtt",
                "provider_prefill",
                "provider_decode",
            ],
        },
        "qwen_calls": 0,
        "provider_calls": 0,
    }


def _validate_question_rows(
    rows: object,
    selection: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if not isinstance(rows, list) or len(rows) != EXPECTED_QUESTION_COUNT:
        raise ValueError("binary packing run changed its question population")
    materialized: list[dict[str, Any]] = []
    for row, selected in zip(rows, selection["questions"], strict=True):
        if not isinstance(row, dict):
            raise ValueError("binary packing question row must be an object")
        arm = selected["arms"]["a3_protected_union"]
        audit = row.get("packer_audit")
        equivalence = row.get("equivalence")
        if (
            row.get("ordinal") != selected.get("ordinal")
            or row.get("question_id") != selected.get("question_id")
            or row.get("probe_sha256") != selected.get("probe_sha256")
            or row.get("prompt_question_sha256")
            != selected.get("prompt_question_sha256")
            or row.get("selected_count") != len(arm["selected_evidence"])
            or row.get("packed_count") != len(arm["packed_evidence"])
            or row.get("dropped_count") != len(arm["dropped_chunk_ids"])
            or row.get("context_token_proxy")
            != arm.get("context_token_proxy")
            or row.get("prompt_token_proxy") != arm.get("prompt_token_proxy")
            or row.get("prompt_workspace_token_proxy")
            != arm.get("prompt_workspace_token_proxy")
            or row.get("provider_payload_sha256")
            != arm.get("provider_payload_sha256")
            or row.get("provider_payload_utf8_bytes")
            != arm.get("provider_payload_utf8_bytes")
            or row.get("selected_chunk_sequence_sha256")
            != identity_sha256(
                [str(item["chunk_id"]) for item in arm["selected_evidence"]]
            )
            or row.get("packed_chunk_sequence_sha256")
            != identity_sha256(
                [str(item["chunk_id"]) for item in arm["packed_evidence"]]
            )
            or row.get("all_equivalent") is not True
            or not isinstance(equivalence, Mapping)
            or dict(equivalence)
            != {field: True for field in _EQUIVALENCE_FIELDS}
            or not isinstance(audit, Mapping)
            or audit.get("audit_format") != AUDIT_FORMAT
            or audit.get("packer_id") != PACKER_ID
            or audit.get("candidate_count") != len(arm["selected_evidence"])
            or audit.get("packed_count") != len(arm["packed_evidence"])
            or audit.get("dropped_count") != len(arm["dropped_chunk_ids"])
            or audit.get("maximal_prefix_boundary_validated") is not True
            or audit.get("sampled_monotonicity_validated") is not True
        ):
            raise ValueError("binary packing equivalence row changed")
        materialized.append(row)
    full100._assert_gold_free_rows(materialized)  # noqa: SLF001
    return materialized


def _validate_run_body(
    body: Mapping[str, Any],
    *,
    v7_root: Path,
    selection: Mapping[str, Any],
    selection_sha: str,
    v7_runtime_sha: str,
    probes_sha: str,
) -> None:
    rows = _validate_question_rows(body.get("questions"), selection)
    expected = _run_body(
        v7_root=v7_root,
        selection_sha=selection_sha,
        v7_runtime_sha=v7_runtime_sha,
        probes_sha=probes_sha,
        rows=rows,
    )
    if dict(body) != expected:
        raise ValueError("binary packing run receipt changed")


def _load_run(
    *,
    v7_root: Path,
    output_root: Path,
    selection: Mapping[str, Any],
    selection_sha: str,
    v7_runtime_sha: str,
    probes_sha: str,
) -> tuple[dict[str, Any], str]:
    body, digest = hot._read_json_artifact(  # noqa: SLF001
        output_root / RUN_NAME
    )
    _validate_run_body(
        body,
        v7_root=v7_root,
        selection=selection,
        selection_sha=selection_sha,
        v7_runtime_sha=v7_runtime_sha,
        probes_sha=probes_sha,
    )
    return body, digest


def _validate_runtime(
    *,
    output_root: Path,
    run_sha: str,
    selection_sha: str,
    v7_runtime_sha: str,
) -> tuple[dict[str, Any], str]:
    body, digest = hot._read_json_artifact(  # noqa: SLF001
        output_root / RUNTIME_NAME
    )
    samples = body.get("samples")
    elapsed_ns = _exact_nonnegative_int(body.get("elapsed_ns"), "elapsed_ns")
    if (
        body.get("format") != RUNTIME_FORMAT
        or body.get("status")
        != "sealed_binary_pack_only_runtime_over_v7_payloads"
        or body.get("run_sha256") != run_sha
        or body.get("v7_selection_sha256") != selection_sha
        or body.get("v7_runtime_sha256") != v7_runtime_sha
        or body.get("population_identity_sha256")
        != EXPECTED_POPULATION_SHA256
        or body.get("implementation_sha256")
        != _implementation_identity()["sha256"]
        or body.get("provider_calls") != 0
        or body.get("qwen_calls") != 0
        or not isinstance(samples, list)
        or len(samples) != EXPECTED_QUESTION_COUNT
        or [row.get("ordinal") for row in samples]
        != list(range(EXPECTED_QUESTION_COUNT))
    ):
        raise ValueError("binary packing runtime receipt changed")
    binary_values: list[int] = []
    baseline_values: list[int] = []
    for row in samples:
        binary_values.append(
            _exact_nonnegative_int(
                row.get("binary_pack_only_ns"), "binary pack timing"
            )
        )
        baseline_values.append(
            _exact_nonnegative_int(
                row.get("sealed_v7_linear_pack_only_ns"),
                "sealed v7 pack timing",
            )
        )
    if (
        body.get("binary_pack_only_ns") != _timing_summary(binary_values)
        or body.get("sealed_v7_linear_pack_only_ns")
        != _timing_summary(baseline_values)
    ):
        raise ValueError("binary packing runtime summary changed")
    expected = _runtime_body(
        run_sha=run_sha,
        selection_sha=selection_sha,
        v7_runtime_sha=v7_runtime_sha,
        elapsed_ns=elapsed_ns,
        timings=samples,
    )
    if body != expected:
        raise ValueError("binary packing runtime fields changed")
    return body, digest


def run(*, v7_root: Path, output_root: Path) -> str:
    selection, selection_sha, runtime, v7_runtime_sha = _load_v7_bundle(v7_root)
    prompts, probes_sha = _load_prompt_map(selection)
    path = output_root / RUN_NAME
    if path.exists():
        _body, run_sha = _load_run(
            v7_root=v7_root,
            output_root=output_root,
            selection=selection,
            selection_sha=selection_sha,
            v7_runtime_sha=v7_runtime_sha,
            probes_sha=probes_sha,
        )
        _validate_runtime(
            output_root=output_root,
            run_sha=run_sha,
            selection_sha=selection_sha,
            v7_runtime_sha=v7_runtime_sha,
        )
        print(f"Binary packing run verified: {path} ({run_sha})", flush=True)
        return run_sha

    started = time.perf_counter_ns()
    rows, timings = _collect(selection, runtime, prompts)
    elapsed_ns = time.perf_counter_ns() - started
    body = _run_body(
        v7_root=v7_root,
        selection_sha=selection_sha,
        v7_runtime_sha=v7_runtime_sha,
        probes_sha=probes_sha,
        rows=rows,
    )
    if (
        body["aggregate"]["byte_equivalent_question_count"]
        != EXPECTED_QUESTION_COUNT
    ):
        raise RuntimeError("binary packing failed the full100 equivalence gate")
    run_sha = hot._atomic_write_json(path, body)  # noqa: SLF001
    runtime_body = _runtime_body(
        run_sha=run_sha,
        selection_sha=selection_sha,
        v7_runtime_sha=v7_runtime_sha,
        elapsed_ns=elapsed_ns,
        timings=timings,
    )
    runtime_sha = hot._atomic_write_json(  # noqa: SLF001
        output_root / RUNTIME_NAME, runtime_body
    )
    print(
        f"Binary packing run published: {run_sha}; runtime={runtime_sha}; "
        "byte_equivalent=100/100",
        flush=True,
    )
    return run_sha


def replay(*, v7_root: Path, output_root: Path) -> str:
    selection, selection_sha, runtime, v7_runtime_sha = _load_v7_bundle(v7_root)
    prompts, probes_sha = _load_prompt_map(selection)
    expected, run_sha = _load_run(
        v7_root=v7_root,
        output_root=output_root,
        selection=selection,
        selection_sha=selection_sha,
        v7_runtime_sha=v7_runtime_sha,
        probes_sha=probes_sha,
    )
    _validate_runtime(
        output_root=output_root,
        run_sha=run_sha,
        selection_sha=selection_sha,
        v7_runtime_sha=v7_runtime_sha,
    )
    rows, _timings = _collect(selection, runtime, prompts)
    replayed = _run_body(
        v7_root=v7_root,
        selection_sha=selection_sha,
        v7_runtime_sha=v7_runtime_sha,
        probes_sha=probes_sha,
        rows=rows,
    )
    expected_bytes = hot._canonical_json_bytes(expected)  # noqa: SLF001
    replayed_bytes = hot._canonical_json_bytes(replayed)  # noqa: SLF001
    if replayed_bytes != expected_bytes:
        raise RuntimeError("binary packing replay differs from the sealed run")
    body = {
        "format": REPLAY_FORMAT,
        "status": "byte_identical_provider_free_binary_packing_replay",
        "run_sha256": run_sha,
        "v7_selection_sha256": selection_sha,
        "v7_runtime_sha256": v7_runtime_sha,
        "population_identity_sha256": EXPECTED_POPULATION_SHA256,
        "implementation_sha256": _implementation_identity()["sha256"],
        "run_payload_sha256": hashlib.sha256(expected_bytes).hexdigest(),
        "replayed_payload_sha256": hashlib.sha256(replayed_bytes).hexdigest(),
        "run_payload_utf8_bytes": len(expected_bytes),
        "replayed_payload_utf8_bytes": len(replayed_bytes),
        "byte_identical": True,
        "question_count": EXPECTED_QUESTION_COUNT,
        "gold_fields_present": False,
        "qwen_calls": 0,
        "provider_calls": 0,
    }
    replay_sha = hot._atomic_write_json(  # noqa: SLF001
        output_root / REPLAY_NAME, body
    )
    print(
        f"Binary packing replay published: {replay_sha}; byte_identical=true",
        flush=True,
    )
    return replay_sha


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v7-root", type=Path, default=DEFAULT_V7_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("run")
    commands.add_parser("replay")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    v7_root = args.v7_root.resolve()
    output_root = args.output_root.resolve()
    if args.command == "run":
        run(v7_root=v7_root, output_root=output_root)
    elif args.command == "replay":
        replay(v7_root=v7_root, output_root=output_root)
    else:  # pragma: no cover
        raise AssertionError(f"unhandled command: {args.command}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
