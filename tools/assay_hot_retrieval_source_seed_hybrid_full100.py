"""Provider-free source-seed hybrid over sealed v2 projection and v7 raw arms.

The assay composes two already selected evidence planes without rerunning
retrieval or assertion extraction.  It first gives the sealed v2 assertion
projection its own 1,400-token ranked-prefix budget.  The production hybrid
composer then orders one exact raw seed per v7 source, the budgeted projection,
and the remaining v7 raw rows; exact chunk-ID deduplication occurs only after
that ordering.  A failed all-source-seed gate reuses the exact sealed v7 prompt.

Run and replay never open benchmark gold, provider responses, or the v2 score.
The compact selection reconstructs exact provider messages from hash-pinned
parents.  Only ``score`` may join benchmark gold, after a byte-identical replay.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:
    repository_root = str(Path(__file__).resolve().parents[1])
    if repository_root not in sys.path:
        sys.path.insert(0, repository_root)

from memory_condense.domain.discourse import identity_sha256, quote_sha256
from memory_condense.domain.integrity import file_sha256
from memory_condense.search.packing.ranked_prefix_prompt import (
    AUDIT_FORMAT as PREFIX_AUDIT_FORMAT,
    PACKER_ID as PREFIX_PACKER_ID,
    RankedPrefixPromptPack,
    pack_ranked_prefix_prompt,
)
from memory_condense.search.source_preserving_hybrid import (
    AUDIT_FORMAT as HYBRID_AUDIT_FORMAT,
    POLICY_ID as HYBRID_CORE_POLICY_ID,
    RESULT_FORMAT as HYBRID_RESULT_FORMAT,
    ExactRawPromptFallback,
    HybridEvidence,
    HybridPackMode,
    SourcePreservingHybridPack,
    pack_source_preserving_hybrid,
)
from tools import assay_hot_retrieval_1m as hot
from tools import assay_hot_retrieval_assertion_projection_full100 as v2
from tools import assay_hot_retrieval_full100 as full100


EXPECTED_V2_SELECTION_SHA256 = (
    "cdb0c591912892fe7f2296f2a17e5bd8000d97d10da8c48da4bce26dab007e57"
)
EXPECTED_V2_RUNTIME_SHA256 = (
    "e75dad54d65d292890f2a6577ca12dc71d3e021611b818ac2ce1a791d523592b"
)
EXPECTED_V2_RUN_MANIFEST_SHA256 = (
    "c5772024b6adb8c56b17e98bc2c2d8ae7f10b904c05010cbdaa66d2dde1ccc69"
)
EXPECTED_V2_REPLAY_SHA256 = (
    "c61037081506aa1613142dd12cf78a9a607e6cbca8d39bfd3b6f425b67664afd"
)
EXPECTED_V7_SELECTION_SHA256 = (
    "867a4439af1c369c3f702491045392b8c64c5e2b3b4216c93fd973ada1b6df20"
)
EXPECTED_POPULATION_SHA256 = v2.EXPECTED_POPULATION_SHA256
EXPECTED_QUESTION_COUNT = v2.EXPECTED_QUESTION_COUNT

PROJECTION_CONTEXT_TOKEN_CAP = 1_400
MAX_CONTEXT_TOKENS = v2.MAX_CONTEXT_TOKENS
MAX_PROMPT_TOKENS = v2.MAX_PROMPT_TOKENS
OUTPUT_TOKEN_RESERVE = v2.OUTPUT_TOKEN_RESERVE

POLICY_ID = "sealed-v2-source-seed-hybrid-v3"
SELECTION_FORMAT = "memory-condense-source-seed-hybrid-full100-selection-v1"
RUNTIME_FORMAT = "memory-condense-source-seed-hybrid-full100-runtime-v1"
RUN_MANIFEST_FORMAT = (
    "memory-condense-source-seed-hybrid-full100-run-manifest-v1"
)
REPLAY_FORMAT = "memory-condense-source-seed-hybrid-full100-replay-v1"
SCORE_FORMAT = "memory-condense-source-seed-hybrid-full100-score-v1"
PROJECTION_PREFIX_FORMAT = (
    "memory-condense-source-seed-hybrid-projection-prefix-v1"
)
HYBRID_RECEIPT_FORMAT = (
    "memory-condense-source-seed-hybrid-core-receipt-v1"
)
PROVIDER_PACKET_FORMAT = (
    "memory-condense-source-seed-hybrid-provider-packet-v1"
)

DEFAULT_V2_ROOT = v2.DEFAULT_OUTPUT_ROOT
DEFAULT_V7_ROOT = v2.DEFAULT_V7_ROOT


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _primary_checkout_root() -> Path:
    root = _repository_root()
    if root.parent.name == ".worktrees":
        return root.parent.parent
    return root


DEFAULT_SOURCE_ROOT = _primary_checkout_root() / (
    "eval_results/longmemeval-1m-recall-guarded-cumulative-"
    "validation-20260822"
)
DEFAULT_OUTPUT_ROOT = Path(
    "eval_results/longmemeval-1m-hot-retrieval-source-seed-hybrid-v3-"
    "full100-validation-20260906"
)

SELECTION_NAME = "selection.json"
RUNTIME_NAME = "runtime.json"
RUN_MANIFEST_NAME = "run_manifest.json"
REPLAY_NAME = "replay.json"
SCORE_NAME = "scores.json"


@dataclass(frozen=True, slots=True)
class _ParentBundle:
    v2_selection: dict[str, Any]
    v2_selection_sha256: str
    v2_runtime_sha256: str
    v2_run_manifest_sha256: str
    v2_replay_sha256: str
    v7_selection: dict[str, Any]
    v7_selection_sha256: str
    parent_root: Path
    probes: dict[str, Any]
    probes_sha256: str
    catalog: dict[str, Any]
    catalog_sha256: str


def _relative_to_repository(path: Path, *, label: str) -> str:
    return v2._relative_to_repository(path, label=label)  # noqa: SLF001


def _load_parent_bundle(
    *,
    v2_root: Path,
    v7_root: Path,
    source_root: Path,
) -> _ParentBundle:
    """Validate all sealed parents without ever opening the v2 score."""

    (
        selection,
        selection_sha,
        v7_selection,
        v7_sha,
        parent_root,
        probes,
        probes_sha,
        catalog,
        catalog_sha,
    ) = v2._load_selection(  # noqa: SLF001
        v7_root=v7_root,
        source_root=source_root,
        output_root=v2_root,
    )
    if selection_sha != EXPECTED_V2_SELECTION_SHA256:
        raise ValueError("sealed v2 selection changed")
    if v7_sha != EXPECTED_V7_SELECTION_SHA256:
        raise ValueError("sealed v7 selection changed")
    _runtime, runtime_sha = v2._validate_runtime(  # noqa: SLF001
        output_root=v2_root,
        selection_sha=selection_sha,
        v7_selection_sha=v7_sha,
    )
    if runtime_sha != EXPECTED_V2_RUNTIME_SHA256:
        raise ValueError("sealed v2 runtime changed")
    _manifest, manifest_sha = v2._validate_run_manifest(  # noqa: SLF001
        output_root=v2_root,
        selection_sha=selection_sha,
        runtime_sha=runtime_sha,
        v7_selection_sha=v7_sha,
    )
    if manifest_sha != EXPECTED_V2_RUN_MANIFEST_SHA256:
        raise ValueError("sealed v2 run manifest changed")
    _replay, replay_sha = v2._validate_replay(  # noqa: SLF001
        output_root=v2_root,
        selection=selection,
        selection_sha=selection_sha,
        v7_selection_sha=v7_sha,
    )
    if replay_sha != EXPECTED_V2_REPLAY_SHA256:
        raise ValueError("sealed v2 replay changed")
    return _ParentBundle(
        v2_selection=selection,
        v2_selection_sha256=selection_sha,
        v2_runtime_sha256=runtime_sha,
        v2_run_manifest_sha256=manifest_sha,
        v2_replay_sha256=replay_sha,
        v7_selection=v7_selection,
        v7_selection_sha256=v7_sha,
        parent_root=parent_root,
        probes=probes,
        probes_sha256=probes_sha,
        catalog=catalog,
        catalog_sha256=catalog_sha,
    )


def _parent_fingerprint(parent: _ParentBundle) -> tuple[str, ...]:
    return (
        parent.v2_selection_sha256,
        parent.v2_runtime_sha256,
        parent.v2_run_manifest_sha256,
        parent.v2_replay_sha256,
        parent.v7_selection_sha256,
        parent.probes_sha256,
        parent.catalog_sha256,
    )


def _implementation_identity() -> dict[str, Any]:
    root = _repository_root()
    paths = (
        "tools/assay_hot_retrieval_source_seed_hybrid_full100.py",
        "tools/assay_hot_retrieval_assertion_projection_full100.py",
        "src/memory_condense/search/source_preserving_hybrid.py",
        "src/memory_condense/search/packing/ranked_prefix_prompt.py",
    )
    files = {path: file_sha256(root / path) for path in paths}
    parent = v2._implementation_identity()  # noqa: SLF001
    material = {
        "files": files,
        "sealed_v2_implementation_sha256": parent["sha256"],
    }
    return {
        "format": "memory-condense-source-seed-hybrid-implementation-v1",
        **material,
        "sha256": identity_sha256(material),
    }


def _seal_compact(value: Mapping[str, Any]) -> dict[str, Any]:
    sealed = dict(value)
    sealed["receipt_sha256"] = identity_sha256(sealed)
    return sealed


def _validate_compact(value: object, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    body = dict(value)
    receipt = body.pop("receipt_sha256", None)
    if receipt != identity_sha256(body):
        raise ValueError(f"{label} receipt changed")
    return dict(value)


def _rendered_texts(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    return [str(row["rendered_text"]) for row in rows]


def _hybrid_rendered_texts(
    rows: Sequence[HybridEvidence[Mapping[str, Any], Mapping[str, Any]]],
) -> list[str]:
    return [str(row.value["rendered_text"]) for row in rows]


def _prompt_payload(messages: object) -> bytes:
    return hot._canonical_json_bytes({"messages": messages})  # noqa: SLF001


def _prompt_sha256(messages: object) -> str:
    return hashlib.sha256(_prompt_payload(messages)).hexdigest()


def _resolve_parent_arms(
    *,
    v2_row: Mapping[str, Any],
    v7_row: Mapping[str, Any],
    dated_question: str,
    v7_selection_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    active_sources = v2._active_source_ids(v7_row)  # noqa: SLF001
    projection = v2._resolve_projection_arm(  # noqa: SLF001
        v2_row["arms"]["assertion_projection"],
        compact_projection=v2_row["assertion_projection"],
        active_source_ids=active_sources,
        dated_question=dated_question,
    )
    fallback = v2._resolve_fallback_reference(  # noqa: SLF001
        v2_row["arms"]["v7_fallback_ref"],
        v7_row=v7_row,
        v7_selection_sha=v7_selection_sha256,
    )
    hot._validate_arm_payload(  # noqa: SLF001
        projection,
        prompt_question=dated_question,
        max_context_tokens=MAX_CONTEXT_TOKENS,
        max_prompt_tokens=MAX_PROMPT_TOKENS,
    )
    hot._validate_arm_payload(  # noqa: SLF001
        fallback,
        prompt_question=dated_question,
        max_context_tokens=MAX_CONTEXT_TOKENS,
        max_prompt_tokens=MAX_PROMPT_TOKENS,
    )
    return projection, fallback


def _pack_projection_prefix(
    projection_arm: Mapping[str, Any],
    *,
    dated_question: str,
) -> RankedPrefixPromptPack[Mapping[str, Any], list[dict[str, str]]]:
    candidates = tuple(projection_arm["selected_evidence"])
    return pack_ranked_prefix_prompt(
        candidates,
        count_context_tokens=lambda rows: hot._context_token_proxy(  # noqa: SLF001
            _rendered_texts(rows)
        ),
        render_prompt=lambda rows: hot.build_qa_prompt(
            dated_question,
            _rendered_texts(rows),
        ),
        count_prompt_tokens=hot.count_chat_prompt_token_proxy,
        max_context_tokens=PROJECTION_CONTEXT_TOKEN_CAP,
        max_prompt_tokens=MAX_PROMPT_TOKENS,
        output_token_reserve=OUTPUT_TOKEN_RESERVE,
    )


def _projection_prefix_receipt(
    pack: RankedPrefixPromptPack[Mapping[str, Any], object],
    *,
    input_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    inputs = [str(row["chunk_id"]) for row in input_rows]
    selected = [str(row["chunk_id"]) for row in pack.packed_items]
    dropped = [str(row["chunk_id"]) for row in pack.dropped_items]
    audit = pack.audit.projection()
    value = {
        "format": PROJECTION_PREFIX_FORMAT,
        "policy": {
            "packer_id": PREFIX_PACKER_ID,
            "max_context_tokens": PROJECTION_CONTEXT_TOKEN_CAP,
            "max_prompt_tokens": MAX_PROMPT_TOKENS,
            "output_token_reserve": OUTPUT_TOKEN_RESERVE,
            "cross_arm_deduplication": "after_projection_prefix_selection",
        },
        "input_count": len(inputs),
        "input_chunk_ids_sha256": identity_sha256(inputs),
        "selected_count": len(selected),
        "selected_chunk_ids_sha256": identity_sha256(selected),
        "dropped_count": len(dropped),
        "dropped_chunk_ids_sha256": identity_sha256(dropped),
        "context_token_count": pack.context_token_count,
        "prompt_token_count": pack.prompt_token_count,
        "prompt_workspace_token_count": pack.prompt_workspace_token_count,
        "packing_audit": audit,
        "packing_audit_sha256": identity_sha256(audit),
    }
    return _seal_compact(value)


def _raw_fallback_contract(
    fallback_arm: Mapping[str, Any],
) -> ExactRawPromptFallback[list[dict[str, str]]]:
    return ExactRawPromptFallback(
        raw_chunk_ids=tuple(str(value) for value in fallback_arm["packed_chunk_ids"]),
        rendered_prompt=copy.deepcopy(fallback_arm["provider_messages"]),
        context_token_count=int(fallback_arm["context_token_proxy"]),
        prompt_token_count=int(fallback_arm["prompt_token_proxy"]),
        output_token_reserve=OUTPUT_TOKEN_RESERVE,
        prompt_workspace_token_count=int(
            fallback_arm["prompt_workspace_token_proxy"]
        ),
        prompt_sha256=str(fallback_arm["provider_payload_sha256"]),
    )


def _pack_hybrid(
    *,
    fallback_arm: Mapping[str, Any],
    projected_prefix: Sequence[Mapping[str, Any]],
    dated_question: str,
) -> SourcePreservingHybridPack[
    Mapping[str, Any], Mapping[str, Any], list[dict[str, str]]
]:
    raws = tuple(fallback_arm["packed_evidence"])
    projections = tuple(projected_prefix)
    return pack_source_preserving_hybrid(
        raws,
        projections,
        raw_chunk_id=lambda row: str(row["chunk_id"]),
        raw_source_id=lambda row: str(row["source_id"]),
        projected_chunk_id=lambda row: str(row["chunk_id"]),
        count_context_tokens=lambda rows: hot._context_token_proxy(  # noqa: SLF001
            _hybrid_rendered_texts(rows)
        ),
        render_prompt=lambda rows: hot.build_qa_prompt(
            dated_question,
            _hybrid_rendered_texts(rows),
        ),
        count_prompt_tokens=hot.count_chat_prompt_token_proxy,
        prompt_sha256=_prompt_sha256,
        max_context_tokens=MAX_CONTEXT_TOKENS,
        max_prompt_tokens=MAX_PROMPT_TOKENS,
        output_token_reserve=OUTPUT_TOKEN_RESERVE,
        raw_fallback=_raw_fallback_contract(fallback_arm),
    )


def _hybrid_receipt(
    result: SourcePreservingHybridPack[Any, Any, Any],
) -> dict[str, Any]:
    audit = result.audit.projection()
    projected_ids = list(result.audit.projected_selected_chunk_ids)
    raw_ids = list(result.audit.raw_selected_chunk_ids)
    source_seeds = [row.projection() for row in result.audit.source_seeds]
    before = [row.projection() for row in result.audit.ranked_before_dedup]
    after = [row.projection() for row in result.audit.ranked_after_dedup]
    duplicates = [row.projection() for row in result.audit.exact_duplicates]
    packed = [row.projection() for row in result.packed_items]
    dropped = [row.projection() for row in result.dropped_items]
    outer_audit = (
        None
        if result.audit.packing_audit is None
        else result.audit.packing_audit.projection()
    )
    value = {
        "format": HYBRID_RECEIPT_FORMAT,
        "core_policy_id": HYBRID_CORE_POLICY_ID,
        "core_result_format": HYBRID_RESULT_FORMAT,
        "core_audit_format": HYBRID_AUDIT_FORMAT,
        "mode": result.mode.value,
        "core_result_receipt_sha256": result.receipt_sha256,
        "core_result_sha256": identity_sha256(result.projection()),
        "core_audit_receipt_sha256": result.audit.receipt_sha256,
        "core_audit_sha256": identity_sha256(audit),
        "raw_selected_count": len(raw_ids),
        "raw_selected_chunk_ids_sha256": identity_sha256(raw_ids),
        "projection_selected_count": len(projected_ids),
        "projection_selected_chunk_ids_sha256": identity_sha256(projected_ids),
        "source_seed_count": len(source_seeds),
        "source_seeds_sha256": identity_sha256(source_seeds),
        "ranked_before_dedup_count": len(before),
        "ranked_before_dedup_sha256": identity_sha256(before),
        "ranked_after_dedup_count": len(after),
        "ranked_after_dedup_sha256": identity_sha256(after),
        "exact_duplicate_count": len(duplicates),
        "exact_duplicates_sha256": identity_sha256(duplicates),
        "outer_packing_audit": outer_audit,
        "outer_packing_audit_sha256": (
            None if outer_audit is None else identity_sha256(outer_audit)
        ),
        "packing_status": result.audit.packing_status,
        "seed_gate": result.audit.seed_gate.projection(),
        "packed_count": len(packed),
        "packed_items_sha256": identity_sha256(packed),
        "dropped_count": len(dropped),
        "dropped_items_sha256": identity_sha256(dropped),
        "context_token_count": result.context_token_count,
        "prompt_token_count": result.prompt_token_count,
        "prompt_workspace_token_count": result.prompt_workspace_token_count,
        "effective_prompt_sha256": result.effective_prompt_sha256,
        "raw_fallback_reused": result.raw_fallback_reused,
        "raw_fallback_receipt_sha256": (
            result.audit.raw_fallback_receipt_sha256
        ),
        "raw_fallback_prompt_sha256": (
            result.audit.raw_fallback_prompt_sha256
        ),
    }
    return _seal_compact(value)


def _effective_arm(
    result: SourcePreservingHybridPack[Any, Any, Any],
    *,
    fallback_arm: Mapping[str, Any],
    dated_question: str,
) -> dict[str, Any]:
    if result.mode is HybridPackMode.RAW_FALLBACK:
        arm = copy.deepcopy(dict(fallback_arm))
        if (
            result.rendered_prompt != arm["provider_messages"]
            or result.effective_prompt_sha256 != arm["provider_payload_sha256"]
        ):
            raise RuntimeError("core fallback differs from sealed v7 provider prompt")
    else:
        selected = [
            copy.deepcopy(dict(row.value))
            for row in result.audit.ranked_after_dedup
        ]
        packed = [copy.deepcopy(dict(row.value)) for row in result.packed_items]
        selected_ids = [str(row["chunk_id"]) for row in selected]
        packed_ids = [str(row["chunk_id"]) for row in packed]
        messages = result.rendered_prompt
        serialized = _prompt_payload(messages)
        arm = {
            "selected_evidence": selected,
            "packed_evidence": packed,
            "selected_chunk_ids": selected_ids,
            "packed_chunk_ids": packed_ids,
            "dropped_chunk_ids": selected_ids[len(packed) :],
            "context_token_proxy": result.context_token_count,
            "prompt_token_proxy": result.prompt_token_count,
            "prompt_workspace_token_proxy": result.prompt_workspace_token_count,
            "provider_messages": messages,
            "provider_payload_sha256": hashlib.sha256(serialized).hexdigest(),
            "provider_payload_utf8_bytes": len(serialized),
            "raw_evidence_only": True,
        }
    hot._validate_arm_payload(  # noqa: SLF001
        arm,
        prompt_question=dated_question,
        max_context_tokens=MAX_CONTEXT_TOKENS,
        max_prompt_tokens=MAX_PROMPT_TOKENS,
    )
    if (
        arm["provider_payload_sha256"] != result.effective_prompt_sha256
        or arm["context_token_proxy"] != result.context_token_count
        or arm["prompt_token_proxy"] != result.prompt_token_count
        or arm["prompt_workspace_token_proxy"]
        != result.prompt_workspace_token_count
    ):
        raise RuntimeError("effective provider arm changed core accounting")
    return arm


def _provider_packet_receipt(
    arm: Mapping[str, Any],
    *,
    mode: str,
    v2_question_sha256: str,
    v2_row: Mapping[str, Any],
    projection_prefix_receipt: Mapping[str, Any],
    hybrid_receipt: Mapping[str, Any],
    parent: _ParentBundle,
) -> dict[str, Any]:
    selected_ids = [str(value) for value in arm["selected_chunk_ids"]]
    packed_ids = [str(value) for value in arm["packed_chunk_ids"]]
    dropped_ids = [str(value) for value in arm["dropped_chunk_ids"]]
    value = {
        "format": PROVIDER_PACKET_FORMAT,
        "route": mode,
        "selected_count": len(selected_ids),
        "selected_chunk_ids_sha256": identity_sha256(selected_ids),
        "packed_count": len(packed_ids),
        "packed_chunk_ids_sha256": identity_sha256(packed_ids),
        "dropped_count": len(dropped_ids),
        "dropped_chunk_ids_sha256": identity_sha256(dropped_ids),
        "context_token_proxy": int(arm["context_token_proxy"]),
        "prompt_token_proxy": int(arm["prompt_token_proxy"]),
        "prompt_workspace_token_proxy": int(
            arm["prompt_workspace_token_proxy"]
        ),
        "provider_payload_sha256": str(arm["provider_payload_sha256"]),
        "provider_payload_utf8_bytes": int(arm["provider_payload_utf8_bytes"]),
        "raw_evidence_only": arm["raw_evidence_only"],
        "parent_receipts": {
            "v2_selection_sha256": parent.v2_selection_sha256,
            "v2_question_sha256": v2_question_sha256,
            "v2_projection_packet_receipt_sha256": v2_row["arms"][
                "assertion_projection"
            ]["compact_receipt_sha256"],
            "v7_fallback_reference_receipt_sha256": v2_row["arms"][
                "v7_fallback_ref"
            ]["receipt_sha256"],
            "projection_prefix_receipt_sha256": projection_prefix_receipt[
                "receipt_sha256"
            ],
            "hybrid_receipt_sha256": hybrid_receipt["receipt_sha256"],
        },
    }
    return _seal_compact(value)


def _compose_question(
    *,
    v2_row: Mapping[str, Any],
    v7_row: Mapping[str, Any],
    dated_question: str,
    parent: _ParentBundle,
) -> tuple[dict[str, Any], dict[str, int], dict[str, Any]]:
    started = time.perf_counter_ns()
    projection_arm, fallback_arm = _resolve_parent_arms(
        v2_row=v2_row,
        v7_row=v7_row,
        dated_question=dated_question,
        v7_selection_sha256=parent.v7_selection_sha256,
    )
    prefix_started = time.perf_counter_ns()
    prefix = _pack_projection_prefix(
        projection_arm,
        dated_question=dated_question,
    )
    prefix_ns = time.perf_counter_ns() - prefix_started
    prefix_receipt = _projection_prefix_receipt(
        prefix,
        input_rows=projection_arm["selected_evidence"],
    )
    hybrid_started = time.perf_counter_ns()
    hybrid = _pack_hybrid(
        fallback_arm=fallback_arm,
        projected_prefix=prefix.packed_items,
        dated_question=dated_question,
    )
    hybrid_ns = time.perf_counter_ns() - hybrid_started
    core_receipt = _hybrid_receipt(hybrid)
    provider_started = time.perf_counter_ns()
    arm = _effective_arm(
        hybrid,
        fallback_arm=fallback_arm,
        dated_question=dated_question,
    )
    v2_question_sha = identity_sha256(v2_row)
    packet = _provider_packet_receipt(
        arm,
        mode=hybrid.mode.value,
        v2_question_sha256=v2_question_sha,
        v2_row=v2_row,
        projection_prefix_receipt=prefix_receipt,
        hybrid_receipt=core_receipt,
        parent=parent,
    )
    provider_ns = time.perf_counter_ns() - provider_started
    identity_keys = (
        "ordinal",
        "shard_offset",
        "local_ordinal",
        "question_id",
        "probe_sha256",
        "retrieval_query_sha256",
        "prompt_question_sha256",
    )
    semantic = {
        key: copy.deepcopy(v2_row[key]) for key in identity_keys
    }
    semantic.update(
        {
            "v2_question_sha256": v2_question_sha,
            "projection_prefix": prefix_receipt,
            "source_seed_hybrid": core_receipt,
            "provider_packet": packet,
            "route_adoption": {
                "status": "decided",
                "selected_arm": (
                    "source_seed_hybrid"
                    if hybrid.mode is HybridPackMode.HYBRID
                    else "v7_exact_raw_fallback"
                ),
                "decision_basis": "all_exact_v7_source_seeds_in_packed_prefix",
                "seed_gate_receipt_sha256": (
                    hybrid.audit.seed_gate.receipt_sha256
                ),
            },
        }
    )
    timing = {
        "ordinal": int(v2_row["ordinal"]),
        "question_id": str(v2_row["question_id"]),
        "shard_offset": int(v2_row["shard_offset"]),
        "projection_prefix_pack_ns": prefix_ns,
        "hybrid_compose_pack_ns": hybrid_ns,
        "provider_packet_ns": provider_ns,
        "question_total_ns": time.perf_counter_ns() - started,
    }
    return semantic, timing, arm


def _probe_by_ordinal(parent: _ParentBundle) -> dict[int, Mapping[str, Any]]:
    rows = parent.probes.get("questions")
    if not isinstance(rows, list) or len(rows) != EXPECTED_QUESTION_COUNT:
        raise ValueError("sealed probe population changed")
    return {int(row["ordinal"]): row for row in rows}


def _collect(
    parent: _ParentBundle,
) -> tuple[list[dict[str, Any]], list[dict[str, int]]]:
    probes = _probe_by_ordinal(parent)
    rows: list[dict[str, Any]] = []
    timings: list[dict[str, int]] = []
    for v2_row, v7_row in zip(
        parent.v2_selection["questions"],
        parent.v7_selection["questions"],
        strict=True,
    ):
        ordinal = int(v2_row["ordinal"])
        probe = probes.get(ordinal)
        if not isinstance(probe, Mapping):
            raise ValueError("sealed v2 row omitted its dated question")
        dated_question = probe.get("prompt_question")
        if (
            not isinstance(dated_question, str)
            or not dated_question
            or probe.get("question_id") != v2_row.get("question_id")
            or quote_sha256(dated_question)
            != v2_row.get("prompt_question_sha256")
        ):
            raise ValueError("sealed dated-question binding changed")
        semantic, timing, _arm = _compose_question(
            v2_row=v2_row,
            v7_row=v7_row,
            dated_question=dated_question,
            parent=parent,
        )
        rows.append(semantic)
        timings.append(timing)
        if len(rows) % 10 == 0:
            print(f"Composed source-seed hybrid: {len(rows)}/100", flush=True)
    rows.sort(key=lambda row: int(row["ordinal"]))
    timings.sort(key=lambda row: int(row["ordinal"]))
    if [row["ordinal"] for row in rows] != list(range(EXPECTED_QUESTION_COUNT)):
        raise RuntimeError("source-seed hybrid question population changed")
    full100._assert_gold_free_rows(rows)  # noqa: SLF001
    return rows, timings


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(rows) != EXPECTED_QUESTION_COUNT:
        raise ValueError("source-seed hybrid requires the locked full100")
    packets = [row["provider_packet"] for row in rows]
    prefixes = [row["projection_prefix"] for row in rows]
    hybrids = [row["source_seed_hybrid"] for row in rows]
    return {
        "question_count": len(rows),
        "hybrid_route_count": sum(
            row["route_adoption"]["selected_arm"] == "source_seed_hybrid"
            for row in rows
        ),
        "raw_fallback_route_count": sum(
            row["route_adoption"]["selected_arm"] == "v7_exact_raw_fallback"
            for row in rows
        ),
        "projection_input_chunk_count": sum(
            int(row["input_count"]) for row in prefixes
        ),
        "projection_prefix_chunk_count": sum(
            int(row["selected_count"]) for row in prefixes
        ),
        "projection_prefix_dropped_chunk_count": sum(
            int(row["dropped_count"]) for row in prefixes
        ),
        "source_seed_count": sum(int(row["source_seed_count"]) for row in hybrids),
        "exact_cross_arm_duplicate_count": sum(
            int(row["exact_duplicate_count"]) for row in hybrids
        ),
        "packed_chunk_count": sum(int(row["packed_count"]) for row in packets),
        "max_packed_chunks": max(int(row["packed_count"]) for row in packets),
        "max_context_token_proxy": max(
            int(row["context_token_proxy"]) for row in packets
        ),
        "max_prompt_workspace_token_proxy": max(
            int(row["prompt_workspace_token_proxy"]) for row in packets
        ),
        "all_routes_decided": all(
            row["route_adoption"]["status"] == "decided" for row in rows
        ),
        "all_provider_packets_raw_evidence_only": all(
            row["raw_evidence_only"] is True for row in packets
        ),
    }


def _controls() -> dict[str, Any]:
    return {
        "policy_id": POLICY_ID,
        "parent_projection": "sealed_assertion_projection_v2",
        "parent_raw": "sealed_adaptive_v7_a3_protected_union",
        "projection_prefix_packer": PREFIX_PACKER_ID,
        "projection_context_token_cap": PROJECTION_CONTEXT_TOKEN_CAP,
        "projection_selection_before_cross_arm_dedup": True,
        "hybrid_core_policy_id": HYBRID_CORE_POLICY_ID,
        "hybrid_order": [
            "first_exact_raw_row_per_source",
            "budgeted_projection_prefix",
            "remaining_raw_rows",
        ],
        "cross_arm_dedup": "exact_chunk_id_after_three_band_selection",
        "source_seed_gate": "all_required_seeds_in_outer_packed_prefix",
        "failed_source_seed_gate": "reuse_exact_sealed_v7_provider_prompt",
        "max_context_token_proxy": MAX_CONTEXT_TOKENS,
        "max_prompt_workspace_token_proxy": MAX_PROMPT_TOKENS,
        "responder_output_token_reserve": OUTPUT_TOKEN_RESERVE,
        "provider_calls": 0,
    }


def _bindings(
    *,
    parent: _ParentBundle,
    v2_root: Path,
    v7_root: Path,
    source_root: Path,
) -> dict[str, Any]:
    return {
        "v2_output_relative_path": _relative_to_repository(
            v2_root, label="v2 output root"
        ),
        "v7_output_relative_path": _relative_to_repository(
            v7_root, label="v7 output root"
        ),
        "source_root_relative_path": _relative_to_repository(
            source_root, label="source root"
        ),
        "v2_selection_sha256": parent.v2_selection_sha256,
        "v2_runtime_sha256": parent.v2_runtime_sha256,
        "v2_run_manifest_sha256": parent.v2_run_manifest_sha256,
        "v2_replay_sha256": parent.v2_replay_sha256,
        "v2_implementation_sha256": parent.v2_selection["implementation"][
            "sha256"
        ],
        "v7_selection_sha256": parent.v7_selection_sha256,
        "probes_sha256": parent.probes_sha256,
        "compiled_catalog_sha256": parent.catalog_sha256,
        "population_identity_sha256": EXPECTED_POPULATION_SHA256,
    }


def _selection_body(
    *,
    parent: _ParentBundle,
    v2_root: Path,
    v7_root: Path,
    source_root: Path,
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    materialized = [dict(row) for row in rows]
    return {
        "format": SELECTION_FORMAT,
        "status": "sealed_gold_blind_source_seed_hybrid_full100",
        "bindings": _bindings(
            parent=parent,
            v2_root=v2_root,
            v7_root=v7_root,
            source_root=source_root,
        ),
        "controls": _controls(),
        "implementation": _implementation_identity(),
        "question_population_sha256": identity_sha256(materialized),
        "questions": materialized,
        "aggregate": _aggregate(materialized),
        "route_adoption": "computable_source_seed_gate",
        "gold_fields_present": False,
        "retained_request_token_state_bytes": 0,
        "qwen_calls": 0,
        "responder_calls": 0,
        "judge_calls": 0,
        "provider_calls": 0,
    }


def _timing_summary(values: Sequence[int]) -> dict[str, int | float]:
    ordered = sorted(int(value) for value in values)
    if not ordered or any(value < 0 for value in ordered):
        raise ValueError("timing samples must be non-negative")
    return {
        "count": len(ordered),
        "min": ordered[0],
        "p50": statistics.median(ordered),
        "mean": statistics.fmean(ordered),
        "p95": ordered[max(0, (95 * len(ordered) + 99) // 100 - 1)],
        "max": ordered[-1],
        "total": sum(ordered),
    }


def _runtime_body(
    *,
    selection_sha256: str,
    parent: _ParentBundle,
    elapsed_ns: int,
    timings: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    samples = [dict(row) for row in timings]
    fields = (
        "projection_prefix_pack_ns",
        "hybrid_compose_pack_ns",
        "provider_packet_ns",
        "question_total_ns",
    )
    return {
        "format": RUNTIME_FORMAT,
        "status": "provider_free_sealed_parent_hybrid_runtime",
        "selection_sha256": selection_sha256,
        "v2_selection_sha256": parent.v2_selection_sha256,
        "v7_selection_sha256": parent.v7_selection_sha256,
        "population_identity_sha256": EXPECTED_POPULATION_SHA256,
        "implementation_sha256": _implementation_identity()["sha256"],
        "elapsed_ns": elapsed_ns,
        "samples": samples,
        "summaries_ns": {
            field: _timing_summary([int(row[field]) for row in samples])
            for field in fields
        },
        "latency_scope": (
            "incremental_sealed_parent_projection_prefix_hybrid_pack_"
            "provider_packet_not_end_to_end"
        ),
        "parent_retrieval_projection_provider_rtt_excluded": True,
        "timings_excluded_from_semantic_identity": True,
        "qwen_calls": 0,
        "provider_calls": 0,
    }


def _run_manifest_body(
    *,
    selection_sha256: str,
    runtime_sha256: str,
    parent: _ParentBundle,
) -> dict[str, Any]:
    return {
        "format": RUN_MANIFEST_FORMAT,
        "status": "complete_provider_free_source_seed_hybrid_run_bundle",
        "selection": {"path": SELECTION_NAME, "sha256": selection_sha256},
        "runtime": {"path": RUNTIME_NAME, "sha256": runtime_sha256},
        "v2_selection_sha256": parent.v2_selection_sha256,
        "v2_replay_sha256": parent.v2_replay_sha256,
        "v7_selection_sha256": parent.v7_selection_sha256,
        "population_identity_sha256": EXPECTED_POPULATION_SHA256,
        "implementation_sha256": _implementation_identity()["sha256"],
        "question_count": EXPECTED_QUESTION_COUNT,
        "complete": True,
        "gold_fields_present": False,
        "qwen_calls": 0,
        "responder_calls": 0,
        "judge_calls": 0,
        "provider_calls": 0,
    }


def _replay_body(
    *,
    selection: Mapping[str, Any],
    selection_sha256: str,
    runtime_sha256: str,
    run_manifest_sha256: str,
    parent: _ParentBundle,
) -> dict[str, Any]:
    payload = hot._canonical_json_bytes(selection)  # noqa: SLF001
    observed = hashlib.sha256(payload).hexdigest()
    if observed != selection_sha256:
        raise ValueError("selection bytes differ from their sealed digest")
    return {
        "format": REPLAY_FORMAT,
        "status": "byte_identical_gold_blind_source_seed_hybrid_replay",
        "selection_sha256": selection_sha256,
        "runtime_sha256": runtime_sha256,
        "run_manifest_sha256": run_manifest_sha256,
        "v2_selection_sha256": parent.v2_selection_sha256,
        "v2_replay_sha256": parent.v2_replay_sha256,
        "v7_selection_sha256": parent.v7_selection_sha256,
        "population_identity_sha256": EXPECTED_POPULATION_SHA256,
        "implementation_sha256": _implementation_identity()["sha256"],
        "selection_payload_sha256": observed,
        "replayed_payload_sha256": observed,
        "selection_payload_utf8_bytes": len(payload),
        "replayed_payload_utf8_bytes": len(payload),
        "byte_identical": True,
        "question_count": EXPECTED_QUESTION_COUNT,
        "route_adoption": "computable_source_seed_gate",
        "gold_fields_present": False,
        "qwen_calls": 0,
        "responder_calls": 0,
        "judge_calls": 0,
        "provider_calls": 0,
    }


def _validate_question_rows(
    rows: object,
    *,
    parent: _ParentBundle,
) -> list[dict[str, Any]]:
    if not isinstance(rows, list) or len(rows) != EXPECTED_QUESTION_COUNT:
        raise ValueError("source-seed hybrid rows changed")
    probes = _probe_by_ordinal(parent)
    materialized: list[dict[str, Any]] = []
    for row, v2_row, v7_row in zip(
        rows,
        parent.v2_selection["questions"],
        parent.v7_selection["questions"],
        strict=True,
    ):
        if not isinstance(row, dict):
            raise ValueError("source-seed hybrid row must be an object")
        probe = probes[int(v2_row["ordinal"])]
        expected, _timing, _arm = _compose_question(
            v2_row=v2_row,
            v7_row=v7_row,
            dated_question=str(probe["prompt_question"]),
            parent=parent,
        )
        if row != expected:
            raise ValueError("source-seed hybrid question receipt changed")
        _validate_compact(row.get("projection_prefix"), label="projection prefix")
        _validate_compact(row.get("source_seed_hybrid"), label="hybrid core")
        _validate_compact(row.get("provider_packet"), label="provider packet")
        materialized.append(row)
    full100._assert_gold_free_rows(materialized)  # noqa: SLF001
    return materialized


def _load_compact_selection(
    *,
    v2_root: Path,
    v7_root: Path,
    source_root: Path,
    output_root: Path,
) -> tuple[dict[str, Any], str, _ParentBundle]:
    parent = _load_parent_bundle(
        v2_root=v2_root,
        v7_root=v7_root,
        source_root=source_root,
    )
    body, digest = hot._read_json_artifact(  # noqa: SLF001
        output_root / SELECTION_NAME
    )
    rows = _validate_question_rows(body.get("questions"), parent=parent)
    expected = _selection_body(
        parent=parent,
        v2_root=v2_root,
        v7_root=v7_root,
        source_root=source_root,
        rows=rows,
    )
    if body != expected:
        raise ValueError("source-seed hybrid selection changed")
    return body, digest, parent


def _bound_repository_path(value: object, *, label: str) -> Path:
    """Resolve one authenticated checkout-scoped artifact binding safely."""

    if not isinstance(value, str) or ":" not in value:
        raise ValueError(f"{label} binding must be checkout-scoped")
    scope, relative_text = value.split(":", 1)
    roots = {
        "current-worktree": _repository_root(),
        "primary-checkout": _primary_checkout_root(),
    }
    root = roots.get(scope)
    relative = Path(relative_text)
    if (
        root is None
        or not relative_text
        or relative.is_absolute()
        or ".." in relative.parts
    ):
        raise ValueError(f"{label} binding escaped its repository checkout")
    target = (root / relative).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError(f"{label} binding escaped its repository checkout")
    return target


def _load_selection(output_root: Path) -> tuple[dict[str, Any], str]:
    """Authenticate compact bytes, then materialize the effective provider arm.

    The returned digest is always the exact on-disk compact selection digest,
    not a hash of the larger in-memory materialization.  This is the provider
    boundary consumed by downstream answer runners.
    """

    compact_probe, probe_sha = hot._read_json_artifact(  # noqa: SLF001
        output_root / SELECTION_NAME
    )
    bindings = compact_probe.get("bindings")
    if (
        compact_probe.get("format") != SELECTION_FORMAT
        or compact_probe.get("status")
        != "sealed_gold_blind_source_seed_hybrid_full100"
        or not isinstance(bindings, Mapping)
        or bindings.get("v2_selection_sha256")
        != EXPECTED_V2_SELECTION_SHA256
        or bindings.get("v2_runtime_sha256") != EXPECTED_V2_RUNTIME_SHA256
        or bindings.get("v2_run_manifest_sha256")
        != EXPECTED_V2_RUN_MANIFEST_SHA256
        or bindings.get("v2_replay_sha256") != EXPECTED_V2_REPLAY_SHA256
        or bindings.get("v7_selection_sha256")
        != EXPECTED_V7_SELECTION_SHA256
        or bindings.get("population_identity_sha256")
        != EXPECTED_POPULATION_SHA256
    ):
        raise ValueError("compact hybrid selection parent bindings changed")
    v2_root = _bound_repository_path(
        bindings.get("v2_output_relative_path"), label="v2 output root"
    )
    v7_root = _bound_repository_path(
        bindings.get("v7_output_relative_path"), label="v7 output root"
    )
    source_root = _bound_repository_path(
        bindings.get("source_root_relative_path"), label="source root"
    )
    compact, digest, parent = _load_compact_selection(
        v2_root=v2_root,
        v7_root=v7_root,
        source_root=source_root,
        output_root=output_root,
    )
    if digest != probe_sha or compact != compact_probe:
        raise RuntimeError("compact hybrid selection changed while loading")
    # This public loader is the downstream provider boundary, not a recovery
    # helper.  A selection-only or selection/runtime crash window must never
    # become provider-eligible.  Replay itself uses the private compact loader
    # so it can create the final prerequisite without circular validation.
    _runtime, runtime_sha = _validate_runtime(
        output_root=output_root,
        selection_sha256=digest,
        parent=parent,
    )
    _manifest, manifest_sha = _validate_run_manifest(
        output_root=output_root,
        selection_sha256=digest,
        runtime_sha256=runtime_sha,
        parent=parent,
    )
    replay_receipt, _replay_sha = _validate_replay(
        output_root=output_root,
        selection=compact,
        selection_sha256=digest,
        runtime_sha256=runtime_sha,
        run_manifest_sha256=manifest_sha,
        parent=parent,
    )
    if replay_receipt.get("byte_identical") is not True:
        raise ValueError("public materialization requires byte-identical replay")
    probes = _probe_by_ordinal(parent)
    materialized = copy.deepcopy(compact)
    for row, compact_row, v2_row, v7_row in zip(
        materialized["questions"],
        compact["questions"],
        parent.v2_selection["questions"],
        parent.v7_selection["questions"],
        strict=True,
    ):
        probe = probes[int(compact_row["ordinal"])]
        expected, _timing, arm = _compose_question(
            v2_row=v2_row,
            v7_row=v7_row,
            dated_question=str(probe["prompt_question"]),
            parent=parent,
        )
        if compact_row != expected:
            raise ValueError("compact row changed during provider materialization")
        packet = compact_row["provider_packet"]
        if (
            arm["provider_payload_sha256"]
            != packet["provider_payload_sha256"]
            or arm["provider_payload_utf8_bytes"]
            != packet["provider_payload_utf8_bytes"]
            or arm["context_token_proxy"] != packet["context_token_proxy"]
            or arm["prompt_token_proxy"] != packet["prompt_token_proxy"]
            or arm["prompt_workspace_token_proxy"]
            != packet["prompt_workspace_token_proxy"]
            or arm["prompt_workspace_token_proxy"] > MAX_PROMPT_TOKENS
            or arm["context_token_proxy"] > MAX_CONTEXT_TOKENS
        ):
            raise ValueError("materialized provider packet differs from its receipt")
        row["arms"] = {"a3_protected_union": arm}
    full100._assert_gold_free_rows(materialized["questions"])  # noqa: SLF001
    return materialized, digest


def _validate_runtime(
    *,
    output_root: Path,
    selection_sha256: str,
    parent: _ParentBundle,
) -> tuple[dict[str, Any], str]:
    body, digest = hot._read_json_artifact(  # noqa: SLF001
        output_root / RUNTIME_NAME
    )
    samples = body.get("samples")
    if (
        not isinstance(samples, list)
        or len(samples) != EXPECTED_QUESTION_COUNT
        or [row.get("ordinal") for row in samples]
        != list(range(EXPECTED_QUESTION_COUNT))
    ):
        raise ValueError("source-seed hybrid runtime samples changed")
    expected = _runtime_body(
        selection_sha256=selection_sha256,
        parent=parent,
        elapsed_ns=int(body["elapsed_ns"]),
        timings=samples,
    )
    if body != expected:
        raise ValueError("source-seed hybrid runtime changed")
    return body, digest


def _validate_run_manifest(
    *,
    output_root: Path,
    selection_sha256: str,
    runtime_sha256: str,
    parent: _ParentBundle,
) -> tuple[dict[str, Any], str]:
    body, digest = hot._read_json_artifact(  # noqa: SLF001
        output_root / RUN_MANIFEST_NAME
    )
    expected = _run_manifest_body(
        selection_sha256=selection_sha256,
        runtime_sha256=runtime_sha256,
        parent=parent,
    )
    if body != expected:
        raise ValueError("source-seed hybrid run manifest changed")
    return body, digest


def _validate_replay(
    *,
    output_root: Path,
    selection: Mapping[str, Any],
    selection_sha256: str,
    runtime_sha256: str,
    run_manifest_sha256: str,
    parent: _ParentBundle,
) -> tuple[dict[str, Any], str]:
    body, digest = hot._read_json_artifact(  # noqa: SLF001
        output_root / REPLAY_NAME
    )
    expected = _replay_body(
        selection=selection,
        selection_sha256=selection_sha256,
        runtime_sha256=runtime_sha256,
        run_manifest_sha256=run_manifest_sha256,
        parent=parent,
    )
    if body != expected:
        raise ValueError("source-seed hybrid replay changed")
    return body, digest


def _known_artifact_paths(output_root: Path) -> tuple[Path, ...]:
    names = (
        SELECTION_NAME,
        RUNTIME_NAME,
        RUN_MANIFEST_NAME,
        REPLAY_NAME,
        SCORE_NAME,
    )
    return tuple(
        path
        for name in names
        for path in (output_root / name, output_root / f"{name}.sha256")
    )


def _assert_fresh_output_root(output_root: Path) -> None:
    present = [path.name for path in _known_artifact_paths(output_root) if path.exists()]
    if present:
        raise FileExistsError(
            "refusing to publish into a non-empty hybrid artifact lifecycle: "
            + ", ".join(present)
        )


def _remove_exact_artifact(path: Path, expected_sha256: str) -> None:
    sidecar = path.with_name(path.name + ".sha256")
    expected_sidecar = f"{expected_sha256}  {path.name}\n".encode("ascii")
    if sidecar.is_file() and sidecar.read_bytes() == expected_sidecar:
        sidecar.unlink()
    if path.is_file() and file_sha256(path) == expected_sha256:
        path.unlink()


def _recover_or_validate_existing_run(
    *,
    v2_root: Path,
    v7_root: Path,
    source_root: Path,
    output_root: Path,
) -> str | None:
    selection, selection_sha, parent = _load_compact_selection(
        v2_root=v2_root,
        v7_root=v7_root,
        source_root=source_root,
        output_root=output_root,
    )
    del selection
    runtime_path = output_root / RUNTIME_NAME
    runtime_sidecar = runtime_path.with_name(runtime_path.name + ".sha256")
    if not runtime_path.exists() and not runtime_sidecar.exists():
        downstream = tuple(_known_artifact_paths(output_root)[4:])
        if any(path.exists() for path in downstream):
            raise ValueError("selection-only crash state has downstream artifacts")
        _remove_exact_artifact(output_root / SELECTION_NAME, selection_sha)
        return None
    if runtime_path.exists() != runtime_sidecar.exists():
        raise ValueError("runtime artifact/sidecar crash state is not provable")
    _runtime, runtime_sha = _validate_runtime(
        output_root=output_root,
        selection_sha256=selection_sha,
        parent=parent,
    )
    manifest_path = output_root / RUN_MANIFEST_NAME
    manifest_sidecar = manifest_path.with_name(manifest_path.name + ".sha256")
    if manifest_sidecar.exists() and not manifest_path.exists():
        raise ValueError("run manifest sidecar is orphaned")
    if manifest_path.exists() and not manifest_sidecar.exists():
        expected = _run_manifest_body(
            selection_sha256=selection_sha,
            runtime_sha256=runtime_sha,
            parent=parent,
        )
        hot._atomic_write_json(manifest_path, expected)  # noqa: SLF001
    if not manifest_path.exists():
        downstream = (
            output_root / REPLAY_NAME,
            output_root / f"{REPLAY_NAME}.sha256",
            output_root / SCORE_NAME,
            output_root / f"{SCORE_NAME}.sha256",
        )
        if any(path.exists() for path in downstream):
            raise ValueError("cannot complete manifest above downstream artifacts")
        hot._atomic_write_json(  # noqa: SLF001
            manifest_path,
            _run_manifest_body(
                selection_sha256=selection_sha,
                runtime_sha256=runtime_sha,
                parent=parent,
            ),
        )
    _validate_run_manifest(
        output_root=output_root,
        selection_sha256=selection_sha,
        runtime_sha256=runtime_sha,
        parent=parent,
    )
    return selection_sha


def run(
    *,
    v2_root: Path,
    v7_root: Path,
    source_root: Path,
    output_root: Path,
) -> str:
    parent = _load_parent_bundle(
        v2_root=v2_root,
        v7_root=v7_root,
        source_root=source_root,
    )
    if (output_root / SELECTION_NAME).exists():
        existing = _recover_or_validate_existing_run(
            v2_root=v2_root,
            v7_root=v7_root,
            source_root=source_root,
            output_root=output_root,
        )
        if existing is not None:
            return existing
    _assert_fresh_output_root(output_root)
    started = time.perf_counter_ns()
    rows, timings = _collect(parent)
    elapsed_ns = time.perf_counter_ns() - started
    body = _selection_body(
        parent=parent,
        v2_root=v2_root,
        v7_root=v7_root,
        source_root=source_root,
        rows=rows,
    )
    aggregate = body["aggregate"]
    if (
        aggregate["question_count"] != EXPECTED_QUESTION_COUNT
        or aggregate["hybrid_route_count"]
        + aggregate["raw_fallback_route_count"]
        != EXPECTED_QUESTION_COUNT
        or aggregate["max_context_token_proxy"] > MAX_CONTEXT_TOKENS
        or aggregate["max_prompt_workspace_token_proxy"] > MAX_PROMPT_TOKENS
        or not aggregate["all_routes_decided"]
        or not aggregate["all_provider_packets_raw_evidence_only"]
    ):
        raise RuntimeError("source-seed hybrid failed its structural gate")
    current_parent = _load_parent_bundle(
        v2_root=v2_root,
        v7_root=v7_root,
        source_root=source_root,
    )
    if _parent_fingerprint(current_parent) != _parent_fingerprint(parent):
        raise RuntimeError("hybrid parents changed during collection")
    _assert_fresh_output_root(output_root)
    selection_sha = hashlib.sha256(
        hot._canonical_json_bytes(body)  # noqa: SLF001
    ).hexdigest()
    runtime = _runtime_body(
        selection_sha256=selection_sha,
        parent=parent,
        elapsed_ns=elapsed_ns,
        timings=timings,
    )
    runtime_sha = hashlib.sha256(
        hot._canonical_json_bytes(runtime)  # noqa: SLF001
    ).hexdigest()
    manifest = _run_manifest_body(
        selection_sha256=selection_sha,
        runtime_sha256=runtime_sha,
        parent=parent,
    )
    manifest_sha = hashlib.sha256(
        hot._canonical_json_bytes(manifest)  # noqa: SLF001
    ).hexdigest()
    selection_path = output_root / SELECTION_NAME
    runtime_path = output_root / RUNTIME_NAME
    manifest_path = output_root / RUN_MANIFEST_NAME
    try:
        if hot._atomic_write_json(selection_path, body) != selection_sha:  # noqa: SLF001
            raise RuntimeError("selection publication digest changed")
        if hot._atomic_write_json(runtime_path, runtime) != runtime_sha:  # noqa: SLF001
            raise RuntimeError("runtime publication digest changed")
        # Completion marker is deliberately published last.
        if hot._atomic_write_json(manifest_path, manifest) != manifest_sha:  # noqa: SLF001
            raise RuntimeError("run manifest publication digest changed")
        _load_compact_selection(
            v2_root=v2_root,
            v7_root=v7_root,
            source_root=source_root,
            output_root=output_root,
        )
        _validate_runtime(
            output_root=output_root,
            selection_sha256=selection_sha,
            parent=parent,
        )
        _validate_run_manifest(
            output_root=output_root,
            selection_sha256=selection_sha,
            runtime_sha256=runtime_sha,
            parent=parent,
        )
    except BaseException:
        _remove_exact_artifact(manifest_path, manifest_sha)
        _remove_exact_artifact(runtime_path, runtime_sha)
        _remove_exact_artifact(selection_path, selection_sha)
        raise
    print(
        f"Source-seed hybrid published: {selection_sha}; runtime={runtime_sha}; "
        f"manifest={manifest_sha}; hybrid={aggregate['hybrid_route_count']}/100",
        flush=True,
    )
    return selection_sha


def replay(
    *,
    v2_root: Path,
    v7_root: Path,
    source_root: Path,
    output_root: Path,
) -> str:
    expected, selection_sha, parent = _load_compact_selection(
        v2_root=v2_root,
        v7_root=v7_root,
        source_root=source_root,
        output_root=output_root,
    )
    _runtime, runtime_sha = _validate_runtime(
        output_root=output_root,
        selection_sha256=selection_sha,
        parent=parent,
    )
    _manifest, manifest_sha = _validate_run_manifest(
        output_root=output_root,
        selection_sha256=selection_sha,
        runtime_sha256=runtime_sha,
        parent=parent,
    )
    rows, _timings = _collect(parent)
    replayed = _selection_body(
        parent=parent,
        v2_root=v2_root,
        v7_root=v7_root,
        source_root=source_root,
        rows=rows,
    )
    if hot._canonical_json_bytes(expected) != hot._canonical_json_bytes(replayed):  # noqa: SLF001
        raise RuntimeError("source-seed hybrid replay differs from selection")
    current, current_sha, current_parent = _load_compact_selection(
        v2_root=v2_root,
        v7_root=v7_root,
        source_root=source_root,
        output_root=output_root,
    )
    if current != expected or current_sha != selection_sha:
        raise RuntimeError("replay selection changed during reconstruction")
    _current_runtime, current_runtime_sha = _validate_runtime(
        output_root=output_root,
        selection_sha256=selection_sha,
        parent=current_parent,
    )
    _current_manifest, current_manifest_sha = _validate_run_manifest(
        output_root=output_root,
        selection_sha256=selection_sha,
        runtime_sha256=runtime_sha,
        parent=current_parent,
    )
    if current_runtime_sha != runtime_sha or current_manifest_sha != manifest_sha:
        raise RuntimeError("replay run bundle changed during reconstruction")
    body = _replay_body(
        selection=expected,
        selection_sha256=selection_sha,
        runtime_sha256=runtime_sha,
        run_manifest_sha256=manifest_sha,
        parent=parent,
    )
    digest = hot._atomic_write_json(output_root / REPLAY_NAME, body)  # noqa: SLF001
    _validate_replay(
        output_root=output_root,
        selection=expected,
        selection_sha256=selection_sha,
        runtime_sha256=runtime_sha,
        run_manifest_sha256=manifest_sha,
        parent=parent,
    )
    print(f"Source-seed hybrid replay: {digest}; byte_identical=true", flush=True)
    return digest


def _score_arm(arm: Mapping[str, Any], question: Any) -> dict[str, Any]:
    return v2._arm_score(arm, question)  # noqa: SLF001


def score(
    *,
    dataset: Path,
    split_manifest: Path,
    v2_root: Path,
    v7_root: Path,
    source_root: Path,
    output_root: Path,
) -> str:
    selection, selection_sha, parent = _load_compact_selection(
        v2_root=v2_root,
        v7_root=v7_root,
        source_root=source_root,
        output_root=output_root,
    )
    _runtime, runtime_sha = _validate_runtime(
        output_root=output_root,
        selection_sha256=selection_sha,
        parent=parent,
    )
    _manifest, manifest_sha = _validate_run_manifest(
        output_root=output_root,
        selection_sha256=selection_sha,
        runtime_sha256=runtime_sha,
        parent=parent,
    )
    _replay, replay_sha = _validate_replay(
        output_root=output_root,
        selection=selection,
        selection_sha256=selection_sha,
        runtime_sha256=runtime_sha,
        run_manifest_sha256=manifest_sha,
        parent=parent,
    )
    samples, _identities, population = full100._load_population(  # noqa: SLF001
        dataset, split_manifest
    )
    questions = full100._flatten_questions(samples)  # noqa: SLF001
    probes = _probe_by_ordinal(parent)
    rows: list[dict[str, Any]] = []
    for selected, v2_row, v7_row, question in zip(
        selection["questions"],
        parent.v2_selection["questions"],
        parent.v7_selection["questions"],
        questions,
        strict=True,
    ):
        if (
            selected.get("question_id") != question.question_id
            or selected.get("retrieval_query_sha256")
            != quote_sha256(question.question)
            or selected.get("prompt_question_sha256")
            != quote_sha256(question.dated_question)
        ):
            raise ValueError("score population differs from sealed selection")
        _semantic, _timing, hybrid_arm = _compose_question(
            v2_row=v2_row,
            v7_row=v7_row,
            dated_question=str(probes[int(selected["ordinal"])]["prompt_question"]),
            parent=parent,
        )
        fallback_arm = v2._resolve_fallback_reference(  # noqa: SLF001
            v2_row["arms"]["v7_fallback_ref"],
            v7_row=v7_row,
            v7_selection_sha=parent.v7_selection_sha256,
        )
        rows.append(
            {
                "ordinal": int(selected["ordinal"]),
                "question_id": question.question_id,
                "effective_hybrid": _score_arm(hybrid_arm, question),
                "v7_fallback": _score_arm(fallback_arm, question),
            }
        )
    aggregate: dict[str, Any] = {}
    for arm_id in ("effective_hybrid", "v7_fallback"):
        arm_rows = [row[arm_id] for row in rows]
        recalls = [
            float(row["gold_source_id_recall"])
            for row in arm_rows
            if row["gold_source_id_recall"] is not None
        ]
        aggregate[arm_id] = {
            "all_gold_source_id_reach_hits": sum(
                row["all_gold_source_ids_reached"] is True for row in arm_rows
            ),
            "literal_answer_hits": sum(
                row["literal_answer"] is True for row in arm_rows
            ),
            "mean_gold_source_id_recall": (
                None if not recalls else statistics.fmean(recalls)
            ),
            "mean_best_f1": statistics.fmean(
                float(row["best_f1"]) for row in arm_rows
            ),
        }
    body = {
        "format": SCORE_FORMAT,
        "status": "post_replay_gold_join_source_seed_hybrid_assay",
        "selection_sha256": selection_sha,
        "runtime_sha256": runtime_sha,
        "run_manifest_sha256": manifest_sha,
        "replay_sha256": replay_sha,
        "v2_selection_sha256": parent.v2_selection_sha256,
        "v7_selection_sha256": parent.v7_selection_sha256,
        "population_identity_sha256": population["population_identity_sha256"],
        "question_count": EXPECTED_QUESTION_COUNT,
        "route_adoption": "computable_source_seed_gate",
        "questions": rows,
        "aggregate": aggregate,
        "gold_fields_present": True,
        "qwen_calls": 0,
        "responder_calls": 0,
        "judge_calls": 0,
        "provider_calls": 0,
    }
    current, current_sha, current_parent = _load_compact_selection(
        v2_root=v2_root,
        v7_root=v7_root,
        source_root=source_root,
        output_root=output_root,
    )
    if current != selection or current_sha != selection_sha:
        raise RuntimeError("score selection changed during gold join")
    _current_runtime, current_runtime_sha = _validate_runtime(
        output_root=output_root,
        selection_sha256=selection_sha,
        parent=current_parent,
    )
    _current_manifest, current_manifest_sha = _validate_run_manifest(
        output_root=output_root,
        selection_sha256=selection_sha,
        runtime_sha256=runtime_sha,
        parent=current_parent,
    )
    _current_replay, current_replay_sha = _validate_replay(
        output_root=output_root,
        selection=selection,
        selection_sha256=selection_sha,
        runtime_sha256=runtime_sha,
        run_manifest_sha256=manifest_sha,
        parent=current_parent,
    )
    if (
        current_runtime_sha != runtime_sha
        or current_manifest_sha != manifest_sha
        or current_replay_sha != replay_sha
    ):
        raise RuntimeError("score prerequisites changed during gold join")
    digest = hot._atomic_write_json(output_root / SCORE_NAME, body)  # noqa: SLF001
    print(
        f"Source-seed hybrid score: {digest}; source reach="
        f"{aggregate['effective_hybrid']['all_gold_source_id_reach_hits']}/100, "
        f"literal={aggregate['effective_hybrid']['literal_answer_hits']}/100; "
        f"v7 source reach={aggregate['v7_fallback']['all_gold_source_id_reach_hits']}/100, "
        f"literal={aggregate['v7_fallback']['literal_answer_hits']}/100",
        flush=True,
    )
    return digest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v2-root", type=Path, default=DEFAULT_V2_ROOT)
    parser.add_argument("--v7-root", type=Path, default=DEFAULT_V7_ROOT)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("run")
    commands.add_parser("replay")
    score_parser = commands.add_parser("score")
    score_parser.add_argument("--dataset", type=Path, required=True)
    score_parser.add_argument("--split-manifest", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    common = {
        "v2_root": args.v2_root.resolve(),
        "v7_root": args.v7_root.resolve(),
        "source_root": args.source_root.resolve(),
        "output_root": args.output_root.resolve(),
    }
    if args.command == "run":
        run(**common)
    elif args.command == "replay":
        replay(**common)
    elif args.command == "score":
        score(
            dataset=args.dataset.resolve(),
            split_manifest=args.split_manifest.resolve(),
            **common,
        )
    else:  # pragma: no cover
        raise AssertionError(f"unhandled command: {args.command}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
