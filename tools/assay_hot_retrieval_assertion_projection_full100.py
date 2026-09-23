"""Provider-free activated-source assertion-projection assay over sealed v7.

The treatment projects only the complete raw rows belonging to opaque source
IDs already activated by v7 packed evidence.  Its provider packet and the
untouched v7 fallback packet remain separate arms.  Run/replay never open gold
or provider outputs; score is the only command allowed to open the benchmark.

The projection-core adapter is wired to the pure
``activated_assertion_projection`` production module.  Published artifacts
bind that implementation and every prerequisite receipt by content hash.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

# Direct ``python tools/...py`` execution otherwise exposes ``tools/`` rather
# than the repository root on ``sys.path``.  Normalize that entry point before
# importing helpers whose own imports are rooted at ``tools``.
if __package__ in {None, ""}:
    repository_root = str(Path(__file__).resolve().parents[1])
    if repository_root not in sys.path:
        sys.path.insert(0, repository_root)

from memory_condense.domain.discourse import identity_sha256, quote_sha256
from memory_condense.domain.integrity import file_sha256
from memory_condense.eval.answer_value_coverage import (
    answer_value_component_coverage,
    best_f1,
    contains_answer,
)
from memory_condense.persistence.db import (
    INDEXED_CHUNK_SQL,
    TURN_SOURCE_ID_SQL,
    Database,
)
from memory_condense.search.activated_assertion_projection import (
    ActivatedAssertionCandidate,
    ActivatedAssertionPolicy,
    ActivatedAssertionProjection,
    AssertionRoleMode,
    QuestionAssertionHint,
    project_activated_assertions,
)
from memory_condense.search.packing.ranked_prefix_prompt import (
    AUDIT_FORMAT as BINARY_PACK_AUDIT_FORMAT,
    PACKER_ID as BINARY_PACKER_ID,
    pack_ranked_prefix_prompt,
)
from tools import assay_hot_retrieval_1m as hot
from tools import assay_hot_retrieval_adaptive_full100 as v7
from tools import assay_hot_retrieval_full100 as full100
from tools.matched_eval.typed_operator_spec import (
    TypedOperatorSpec,
    compile_typed_operator_spec,
)


EXPECTED_V7_SELECTION_SHA256 = (
    "867a4439af1c369c3f702491045392b8c64c5e2b3b4216c93fd973ada1b6df20"
)
EXPECTED_POPULATION_SHA256 = v7.EXPECTED_POPULATION_SHA256
EXPECTED_QUESTION_COUNT = v7.EXPECTED_QUESTION_COUNT
MAX_CONTEXT_TOKENS = v7.MAX_CONTEXT_TOKENS
MAX_PROMPT_TOKENS = v7.MAX_PROMPT_TOKENS
MAX_PROJECTION_CHUNKS = 40
OUTPUT_TOKEN_RESERVE = hot.RESPONDER_OUTPUT_TOKEN_RESERVE

SELECTION_FORMAT = (
    "memory-condense-hot-retrieval-assertion-projection-full100-selection-v2"
)
RUNTIME_FORMAT = (
    "memory-condense-hot-retrieval-assertion-projection-full100-runtime-v2"
)
REPLAY_FORMAT = (
    "memory-condense-hot-retrieval-assertion-projection-full100-replay-v2"
)
SCORE_FORMAT = (
    "memory-condense-hot-retrieval-assertion-projection-full100-score-v2"
)
RUN_MANIFEST_FORMAT = (
    "memory-condense-hot-retrieval-assertion-projection-full100-run-manifest-v2"
)
FALLBACK_REF_FORMAT = "memory-condense-sealed-v7-arm-reference-v2"
COMPACT_PROJECTION_FORMAT = "memory-condense-assertion-projection-receipt-v2"
COMPACT_ARM_FORMAT = "memory-condense-assertion-provider-packet-receipt-v2"
COMPACT_SCAN_FORMAT = "memory-condense-active-source-scan-receipt-v2"
COMPACT_PACKING_AUDIT_FORMAT = "memory-condense-prefix-packing-audit-receipt-v2"
POLICY_ID = "activated-source-assertion-projection-assay-v2"

DEFAULT_V7_ROOT = v7.DEFAULT_OUTPUT_ROOT
DEFAULT_SOURCE_ROOT = v7.DEFAULT_SOURCE_ROOT
DEFAULT_OUTPUT_ROOT = Path(
    "eval_results/longmemeval-1m-hot-retrieval-assertion-projection-v2-"
    "full100-validation-20260906"
)
SELECTION_NAME = "selection.json"
RUNTIME_NAME = "runtime.json"
REPLAY_NAME = "replay.json"
SCORE_NAME = "scores.json"
RUN_MANIFEST_NAME = "run_manifest.json"


def _active_source_ids(v7_row: Mapping[str, Any]) -> tuple[str, ...]:
    """Return opaque sources in first-packed-evidence occurrence order."""

    arms = v7_row.get("arms")
    arm = (
        arms.get("a3_protected_union")
        if isinstance(arms, Mapping)
        else None
    )
    packed = arm.get("packed_evidence") if isinstance(arm, Mapping) else None
    if not isinstance(packed, list) or not packed:
        raise ValueError("v7 row omitted its packed fallback evidence")
    ordered: list[str] = []
    seen: set[str] = set()
    for evidence in packed:
        if not isinstance(evidence, Mapping):
            raise ValueError("v7 packed evidence must be an object")
        source_id = evidence.get("source_id")
        if not isinstance(source_id, str) or not source_id:
            raise ValueError("v7 packed evidence omitted source_id")
        if source_id not in seen:
            seen.add(source_id)
            ordered.append(source_id)
    return tuple(ordered)


def _scan_active_source_rows(
    database: Any,
    *,
    active_source_ids: Sequence[str],
    metadata_by_id: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Batch-read exactly the active source partition and prove completeness."""

    sources = tuple(active_source_ids)
    if (
        not sources
        or any(not isinstance(source_id, str) or not source_id for source_id in sources)
        or len(sources) != len(set(sources))
    ):
        raise ValueError("active source IDs must be non-empty unique strings")
    placeholders = ",".join("?" for _ in sources)
    source_expr = TURN_SOURCE_ID_SQL
    sql = (
        "SELECT c.chunk_id, c.text, c.token_count, c.turn_id, "
        + source_expr
        + ", t.role, t.created_at, t.ordinal, c.start_char "
        "FROM chunks AS c JOIN turns AS t ON t.turn_id = c.turn_id "
        f"WHERE {source_expr} IN ({placeholders}) AND {INDEXED_CHUNK_SQL} "
        "ORDER BY t.ordinal, c.start_char, c.chunk_id"
    )
    cursor = database.execute(sql, sources)
    source_set = set(sources)
    buckets: dict[str, list[dict[str, Any]]] = {source_id: [] for source_id in sources}
    seen_chunks: set[str] = set()
    for raw in cursor:
        if isinstance(raw, (str, bytes, Mapping)) or len(raw) != 9:
            raise ValueError("active-source SQL returned a malformed row")
        chunk_id = str(raw[0])
        source_id = str(raw[4])
        if source_id not in source_set:
            raise ValueError("active-source SQL escaped its exact source set")
        if not chunk_id or chunk_id in seen_chunks:
            raise ValueError("active-source SQL returned a duplicate chunk")
        metadata = metadata_by_id.get(chunk_id)
        if not isinstance(metadata, Mapping):
            raise ValueError("active-source row is absent from compiled metadata")
        structural = {
            "chunk_id": chunk_id,
            "token_count": int(raw[2]),
            "turn_id": str(raw[3]),
            "source_id": source_id,
            "role": str(raw[5]),
            # Preserve the database value for the exact compiled/DB receipt
            # comparison.  The projection adapter normalizes NULL to an empty
            # string only after this structural boundary is proven.
            "created_at": raw[6],
            "ordinal": int(raw[7]),
            "start_char": int(raw[8]),
        }
        if any(metadata.get(key) != value for key, value in structural.items()):
            raise ValueError("active-source row differs from compiled metadata")
        text = str(raw[1])
        buckets[source_id].append(
            {
                **structural,
                "text": text,
                "text_sha256": quote_sha256(text),
            }
        )
        seen_chunks.add(chunk_id)

    expected_ids = {
        str(chunk_id)
        for chunk_id, metadata in metadata_by_id.items()
        if metadata.get("source_id") in source_set
    }
    if seen_chunks != expected_ids:
        raise ValueError("active-source SQL did not exhaust the compiled partition")
    rows = [row for source_id in sources for row in buckets[source_id]]
    scanned_sources = tuple(dict.fromkeys(str(row["source_id"]) for row in rows))
    if set(scanned_sources) != source_set:
        raise ValueError("an activated source has no compiled raw rows")
    audit = {
        "scan_contract": "exact_opaque_active_source_partition_v1",
        "active_source_ids": list(sources),
        "active_source_count": len(sources),
        "sql_parameter_count": len(sources),
        "compiled_partition_chunk_count": len(expected_ids),
        "scanned_chunk_count": len(rows),
        "scanned_source_ids": list(scanned_sources),
        "source_confinement_validated": True,
        "compiled_partition_exhaustive": True,
        "ordered_by": (
            "v7_source_activation_then_turn_ordinal_start_char_chunk_id"
        ),
    }
    return rows, audit


def _fallback_arm(v7_row: Mapping[str, Any]) -> dict[str, Any]:
    """Copy the exact v7 arm and prove its canonical provider bytes."""

    arms = v7_row.get("arms")
    arm = (
        arms.get("a3_protected_union")
        if isinstance(arms, Mapping)
        else None
    )
    if not isinstance(arm, Mapping):
        raise ValueError("v7 row omitted its fallback arm")
    copied = copy.deepcopy(dict(arm))
    payload = hot._canonical_json_bytes(  # noqa: SLF001
        {"messages": copied.get("provider_messages")}
    )
    if (
        hashlib.sha256(payload).hexdigest()
        != copied.get("provider_payload_sha256")
        or len(payload) != copied.get("provider_payload_utf8_bytes")
    ):
        raise ValueError("v7 fallback provider payload changed")
    return copied


def _fallback_reference(
    v7_row: Mapping[str, Any],
    *,
    v7_selection_sha: str,
) -> dict[str, Any]:
    """Return a small content-addressed pointer to the sealed v7 arm."""

    fallback = _fallback_arm(v7_row)
    messages = hot._canonical_json_bytes(  # noqa: SLF001
        {"messages": fallback["provider_messages"]}
    )
    value = {
        "format": FALLBACK_REF_FORMAT,
        "v7_selection_sha256": v7_selection_sha,
        "ordinal": int(v7_row["ordinal"]),
        "question_id": str(v7_row["question_id"]),
        "arm_id": "a3_protected_union",
        "arm_semantic_sha256": identity_sha256(fallback),
        "provider_payload_sha256": hashlib.sha256(messages).hexdigest(),
        "provider_payload_utf8_bytes": len(messages),
    }
    value["receipt_sha256"] = identity_sha256(value)
    return value


def _resolve_fallback_reference(
    reference: Mapping[str, Any],
    *,
    v7_row: Mapping[str, Any],
    v7_selection_sha: str,
) -> dict[str, Any]:
    """Resolve and prove a compact reference against the sealed parent row."""

    expected = _fallback_reference(
        v7_row,
        v7_selection_sha=v7_selection_sha,
    )
    if dict(reference) != expected:
        raise ValueError("sealed v7 fallback reference changed")
    return _fallback_arm(v7_row)


def _projection_core_adapter(
    *,
    dated_question: str,
    active_source_ids: Sequence[str],
    rows: Sequence[Mapping[str, Any]],
) -> tuple[
    ActivatedAssertionProjection,
    TypedOperatorSpec,
    QuestionAssertionHint,
]:
    """Compile question-only controls and invoke the exact pure core API."""

    spec = compile_typed_operator_spec(dated_question)
    obligation_terms = tuple(
        dict.fromkeys(
            term
            for slot in spec.required_slots
            for term in slot.match_terms
        )
    )
    role_mode = {
        "user": AssertionRoleMode.USER,
        "assistant": AssertionRoleMode.ASSISTANT,
    }.get(spec.required_evidence_role)
    hint = QuestionAssertionHint(
        dated_question_sha256=hashlib.sha256(
            dated_question.encode("utf-8")
        ).hexdigest(),
        obligation_terms=obligation_terms,
        role_mode=role_mode,
    )
    policy = ActivatedAssertionPolicy(include_proposed=spec.include_proposed)
    candidates = tuple(
        ActivatedAssertionCandidate(
            chunk_id=str(row["chunk_id"]),
            source_id=str(row["source_id"]),
            role=str(row["role"]),
            created_at=(
                "" if row["created_at"] is None else str(row["created_at"])
            ),
            text=str(row["text"]),
            token_count=int(row["token_count"]),
        )
        for row in rows
    )
    projection = project_activated_assertions(
        dated_question,
        active_source_ids,
        candidates,
        protected_chunk_ids=(),
        policy=policy,
        question_hint=hint,
        count_tokens=hot.count_tokens,
    )
    return projection, spec, hint


def _implementation_identity() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    paths = (
        "tools/assay_hot_retrieval_assertion_projection_full100.py",
        "tools/assay_hot_retrieval_adaptive_full100.py",
        "tools/assay_hot_retrieval_full100.py",
        "tools/assay_hot_retrieval_1m.py",
        "tools/matched_eval/typed_operator_spec.py",
        "tools/_routed_repair_routing.py",
        "tools/matched_eval/contracts.py",
        "src/memory_condense/search/activated_assertion_projection.py",
        "src/memory_condense/search/packing/ranked_prefix_prompt.py",
        "src/memory_condense/search/closure/compiler.py",
        "src/memory_condense/search/closure/semantics.py",
        "src/memory_condense/search/selectors/set_program.py",
        "src/memory_condense/eval/_retrieval_qa_prompt.py",
        "src/memory_condense/eval/answer_value_coverage.py",
        "src/memory_condense/eval/_answer_normalization.py",
        "src/memory_condense/eval/recall_models.py",
        "src/memory_condense/domain/_tokenizer.py",
        "src/memory_condense/domain/_discourse_identity.py",
        "src/memory_condense/domain/discourse.py",
        "src/memory_condense/domain/integrity.py",
        "src/memory_condense/domain/sealed.py",
        "src/memory_condense/domain/text_numbers.py",
        "src/memory_condense/persistence/db.py",
    )
    files = {path: file_sha256(root / path) for path in paths}
    return {
        "format": (
            "memory-condense-hot-retrieval-assertion-projection-"
            "full100-implementation-v2"
        ),
        "files": files,
        "sha256": identity_sha256(
            [{"path": path, "sha256": digest} for path, digest in files.items()]
        ),
    }


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _relative_to_repository(path: Path, *, label: str) -> str:
    checkout = _repository_root()
    roots = [checkout]
    # Linked worktrees keep large immutable receipts in the primary checkout.
    if checkout.parent.name == ".worktrees":
        roots.append(checkout.parent.parent)
    for root in roots:
        try:
            relative = path.resolve().relative_to(root)
        except ValueError:
            continue
        scope = "current-worktree" if root == checkout else "primary-checkout"
        return f"{scope}:{relative.as_posix()}"
    raise ValueError(f"{label} must be inside a repository checkout")


def _load_v7_selection(v7_root: Path) -> tuple[dict[str, Any], str]:
    selection, digest = v7._load_selection(v7_root)  # noqa: SLF001
    if digest != EXPECTED_V7_SELECTION_SHA256:
        raise ValueError(
            f"sealed v7 selection changed ({digest} != "
            f"{EXPECTED_V7_SELECTION_SHA256})"
        )
    return selection, digest


def _parent_material(
    selection: Mapping[str, Any],
) -> tuple[Path, dict[str, Any], str, dict[str, Any], str]:
    bindings = selection.get("bindings")
    if not isinstance(bindings, Mapping):
        raise ValueError("v7 selection omitted parent bindings")
    parent_root = v7._bound_parent_root(bindings)  # noqa: SLF001
    probes, probes_sha = full100._load_probes(parent_root)  # noqa: SLF001
    catalog, catalog_sha = full100._load_catalog(parent_root)  # noqa: SLF001
    return parent_root, probes, probes_sha, catalog, catalog_sha


def _projection_evidence_rows(
    projection: ActivatedAssertionProjection,
    *,
    active_source_ids: Sequence[str],
    scanned_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if projection.receipt_sha256 != identity_sha256(
        projection.projection(include_receipt=False)
    ):
        raise RuntimeError("projection receipt changed before provider packing")
    source_by_handle = {
        f"G{index:06d}": source_id
        for index, source_id in enumerate(active_source_ids, 1)
    }
    expected_bindings = tuple(
        (handle, quote_sha256(source_id))
        for handle, source_id in source_by_handle.items()
    )
    if projection.active_source_bindings != expected_bindings:
        raise RuntimeError("projection changed opaque source bindings")
    scanned_by_id = {str(row["chunk_id"]): row for row in scanned_rows}
    if len(scanned_by_id) != len(scanned_rows):
        raise RuntimeError("scanned projection rows repeat a chunk ID")
    evidence: list[dict[str, Any]] = []
    selected_chunk_ids: set[str] = set()
    for rank, fact in enumerate(projection.selected_facts, 1):
        source_id = source_by_handle.get(fact.source_handle)
        source = scanned_by_id.get(fact.chunk_id)
        source_text = "" if source is None else str(source["text"])
        source_created_at = (
            ""
            if source is None or source["created_at"] is None
            else str(source["created_at"])
        )
        if (
            source_id is None
            or source is None
            or source.get("source_id") != source_id
            or fact.chunk_id in selected_chunk_ids
            or quote_sha256(source_text) != fact.input_text_sha256
            or fact.role != str(source["role"]).casefold()
            or fact.source_created_at != source_created_at
            or fact.input_token_count != int(source["token_count"])
            or hot.count_tokens(source_text) != fact.input_token_count
            or hot.count_tokens(fact.quote) != fact.fact_token_count
            or fact.receipt_sha256
            != identity_sha256(fact.projection(include_receipt=False))
        ):
            raise RuntimeError("selected assertion changed its source row")
        for span in fact.quote_spans:
            if (
                span.start_char < 0
                or span.end_char > len(source_text)
                or source_text[span.start_char : span.end_char] != span.quote
            ):
                raise RuntimeError("selected assertion quote span changed")
        if fact.quote != "\n".join(span.quote for span in fact.quote_spans):
            raise RuntimeError("selected assertion quote assembly changed")
        selected_chunk_ids.add(fact.chunk_id)
        provenance = [fact.source_handle, fact.role]
        if fact.source_created_at:
            provenance.append(f"source-time={fact.source_created_at}")
        rendered = f"[{' | '.join(provenance)}] {fact.quote}"
        evidence.append(
            {
                "evidence_id": fact.chunk_id,
                "chunk_id": fact.chunk_id,
                "turn_id": str(source["turn_id"]),
                "source_id": source_id,
                "source_handle": fact.source_handle,
                "role": fact.role,
                "created_at": fact.source_created_at,
                "created_at_semantics": "source_metadata_only_not_event_time",
                "route": "activated_assertion_projection",
                "score": 1.0 / rank,
                "raw_text": fact.quote,
                "raw_text_sha256": quote_sha256(fact.quote),
                "rendered_text": rendered,
                "rendered_text_sha256": quote_sha256(rendered),
                "assertion_fact_receipt_sha256": fact.receipt_sha256,
            }
        )
    if len(evidence) > MAX_PROJECTION_CHUNKS:
        raise RuntimeError("projection exceeded the hard 40-chunk cap")
    rendered_payload = "\n".join(str(row["rendered_text"]) for row in evidence)
    if hot.count_tokens(rendered_payload) != projection.payload_token_count:
        raise RuntimeError("projection payload rendering changed")
    return evidence


def _projection_arm(
    projection: ActivatedAssertionProjection,
    *,
    dated_question: str,
    active_source_ids: Sequence[str],
    scanned_rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, int]]:
    evidence = _projection_evidence_rows(
        projection,
        active_source_ids=active_source_ids,
        scanned_rows=scanned_rows,
    )
    rendered = tuple(str(row["rendered_text"]) for row in evidence)
    pack_started = time.perf_counter_ns()
    packed = pack_ranked_prefix_prompt(
        rendered,
        count_context_tokens=hot._context_token_proxy,  # noqa: SLF001
        render_prompt=lambda prefix: hot.build_qa_prompt(
            dated_question, list(prefix)
        ),
        count_prompt_tokens=hot.count_chat_prompt_token_proxy,
        max_context_tokens=MAX_CONTEXT_TOKENS,
        max_prompt_tokens=MAX_PROMPT_TOKENS,
        output_token_reserve=OUTPUT_TOKEN_RESERVE,
    )
    pack_ns = time.perf_counter_ns() - pack_started
    serialize_started = time.perf_counter_ns()
    serialized = hot._canonical_json_bytes(  # noqa: SLF001
        {"messages": packed.rendered_prompt}
    )
    serialize_ns = time.perf_counter_ns() - serialize_started
    packed_evidence = evidence[: packed.packed_count]
    arm = {
        "selected_evidence": evidence,
        "packed_evidence": packed_evidence,
        "selected_chunk_ids": [str(row["chunk_id"]) for row in evidence],
        "packed_chunk_ids": [
            str(row["chunk_id"]) for row in packed_evidence
        ],
        "dropped_chunk_ids": [
            str(row["chunk_id"]) for row in evidence[packed.packed_count :]
        ],
        "context_token_proxy": packed.context_token_count,
        "prompt_token_proxy": packed.prompt_token_count,
        "prompt_workspace_token_proxy": packed.prompt_workspace_token_count,
        "provider_messages": packed.rendered_prompt,
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
        len(arm["packed_chunk_ids"]) > MAX_PROJECTION_CHUNKS
        or arm["context_token_proxy"] > MAX_CONTEXT_TOKENS
        or arm["prompt_workspace_token_proxy"] > MAX_PROMPT_TOKENS
    ):
        raise RuntimeError("projection provider packet exceeded a hard cap")
    return (
        arm,
        packed.audit.projection(),
        {"binary_pack_ns": pack_ns, "serialize_ns": serialize_ns},
    )


def _content_receipt(value: Mapping[str, Any]) -> str:
    return identity_sha256(dict(value))


def _seal_compact(value: Mapping[str, Any]) -> dict[str, Any]:
    sealed = dict(value)
    sealed["compact_receipt_sha256"] = _content_receipt(sealed)
    return sealed


def _validate_compact_receipt(value: object, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    body = dict(value)
    receipt = body.pop("compact_receipt_sha256", None)
    if receipt != _content_receipt(body):
        raise ValueError(f"{label} compact receipt changed")
    return dict(value)


def _compact_scan_audit(audit: Mapping[str, Any]) -> dict[str, Any]:
    value = {
        "format": COMPACT_SCAN_FORMAT,
        "scan_contract": audit["scan_contract"],
        "active_source_count": int(audit["active_source_count"]),
        "active_source_ids_sha256": identity_sha256(
            audit["active_source_ids"]
        ),
        "sql_parameter_count": int(audit["sql_parameter_count"]),
        "compiled_partition_chunk_count": int(
            audit["compiled_partition_chunk_count"]
        ),
        "scanned_chunk_count": int(audit["scanned_chunk_count"]),
        "scanned_source_ids_sha256": identity_sha256(
            audit["scanned_source_ids"]
        ),
        "source_confinement_validated": audit[
            "source_confinement_validated"
        ],
        "compiled_partition_exhaustive": audit[
            "compiled_partition_exhaustive"
        ],
        "ordered_by": audit["ordered_by"],
        "full_audit_sha256": identity_sha256(audit),
    }
    return _seal_compact(value)


def _validate_compact_scan_audit(
    value: object,
    *,
    active_source_ids: Sequence[str],
) -> dict[str, Any]:
    compact = _validate_compact_receipt(value, label="active-source scan")
    sources = list(active_source_ids)
    if (
        compact.get("format") != COMPACT_SCAN_FORMAT
        or compact.get("active_source_count") != len(sources)
        or compact.get("sql_parameter_count") != len(sources)
        or compact.get("active_source_ids_sha256") != identity_sha256(sources)
        or compact.get("source_confinement_validated") is not True
        or compact.get("compiled_partition_exhaustive") is not True
        or not isinstance(compact.get("compiled_partition_chunk_count"), int)
        or not isinstance(compact.get("scanned_chunk_count"), int)
        or compact.get("compiled_partition_chunk_count")
        != compact.get("scanned_chunk_count")
    ):
        raise ValueError("active-source compact scan receipt changed")
    return compact


def _compact_projection(
    projection: ActivatedAssertionProjection,
) -> dict[str, Any]:
    full = projection.projection()
    facts = [fact.projection() for fact in projection.selected_facts]
    lane_audits = [audit.projection() for audit in projection.lane_audits]
    candidate_audits = [
        audit.projection() for audit in projection.candidate_audits
    ]
    lane_summaries = []
    for audit, projected in zip(
        projection.lane_audits, lane_audits, strict=True
    ):
        lane_summaries.append(
            {
                "reason": audit.reason.value,
                "budget": audit.budget,
                "ranked_candidate_count": len(
                    audit.ranked_candidate_chunk_ids
                ),
                "selected_before_dedup_count": len(
                    audit.selected_before_dedup_chunk_ids
                ),
                "retained_after_dedup_count": len(
                    audit.retained_after_dedup_chunk_ids
                ),
                "protected_duplicate_count": len(
                    audit.protected_duplicate_chunk_ids
                ),
                "exact_duplicate_count": len(
                    audit.exact_duplicate_chunk_ids
                ),
                "semantic_duplicate_count": len(
                    audit.semantic_duplicate_chunk_ids
                ),
                "refilled_count": len(audit.refilled_chunk_ids),
                "packing_refilled_count": len(
                    audit.packing_refilled_chunk_ids
                ),
                "token_unpacked_count": len(
                    audit.token_unpacked_chunk_ids
                ),
                "packed_count": len(audit.packed_chunk_ids),
                "unfilled_slots": audit.unfilled_slots,
                "full_audit_sha256": identity_sha256(projected),
            }
        )
    value = {
        "format": COMPACT_PROJECTION_FORMAT,
        "projection_receipt_sha256": projection.receipt_sha256,
        "dated_question_sha256": projection.dated_question_sha256,
        "policy": projection.policy.projection(),
        "question_hint_receipt_sha256": (
            projection.question_hint_receipt_sha256
        ),
        "effective_include_proposed": projection.effective_include_proposed,
        "role_route": projection.role_route.projection(),
        "active_source_bindings": full["active_source_bindings"],
        "candidate_population_sha256": projection.candidate_population_sha256,
        "protected_chunk_ids": list(projection.protected_chunk_ids),
        "selected_facts": facts,
        "selected_fact_count": len(facts),
        "selected_facts_sha256": identity_sha256(facts),
        "lane_audit_count": len(lane_audits),
        "lane_audits_sha256": identity_sha256(lane_audits),
        "lane_audit_summaries": lane_summaries,
        "candidate_audit_count": len(candidate_audits),
        "candidate_audits_sha256": identity_sha256(candidate_audits),
        "candidate_included_count": sum(
            audit.included for audit in projection.candidate_audits
        ),
        "payload_token_count": projection.payload_token_count,
        "token_budget_exhausted": projection.token_budget_exhausted,
    }
    return _seal_compact(value)


def _validate_compact_projection(value: object) -> dict[str, Any]:
    compact = _validate_compact_receipt(value, label="assertion projection")
    facts = compact.get("selected_facts")
    lane_summaries = compact.get("lane_audit_summaries")
    bindings = compact.get("active_source_bindings")
    if (
        compact.get("format") != COMPACT_PROJECTION_FORMAT
        or not isinstance(facts, list)
        or compact.get("selected_fact_count") != len(facts)
        or len(facts) > MAX_PROJECTION_CHUNKS
        or compact.get("selected_facts_sha256") != identity_sha256(facts)
        or not isinstance(lane_summaries, list)
        or compact.get("lane_audit_count") != len(lane_summaries)
        or not isinstance(bindings, list)
        or not isinstance(compact.get("candidate_audit_count"), int)
        or not isinstance(compact.get("candidate_included_count"), int)
        or compact.get("candidate_included_count") != len(facts)
        or not isinstance(compact.get("payload_token_count"), int)
        or compact.get("payload_token_count") < 0
        or not isinstance(compact.get("token_budget_exhausted"), bool)
    ):
        raise ValueError("compact assertion projection changed")
    for fact in facts:
        if not isinstance(fact, dict):
            raise ValueError("compact selected fact must be an object")
        fact_body = dict(fact)
        receipt = fact_body.pop("receipt_sha256", None)
        if receipt != identity_sha256(fact_body):
            raise ValueError("compact selected fact receipt changed")
        spans = fact.get("quote_spans")
        if not isinstance(spans, list) or not spans:
            raise ValueError("compact selected fact omitted exact spans")
        for span in spans:
            if not isinstance(span, Mapping):
                raise ValueError("compact fact span must be an object")
            quote = span.get("quote")
            start = span.get("start_char")
            end = span.get("end_char")
            if (
                not isinstance(quote, str)
                or not quote
                or not isinstance(start, int)
                or not isinstance(end, int)
                or not 0 <= start < end
                or end - start != len(quote)
                or span.get("quote_sha256") != quote_sha256(quote)
            ):
                raise ValueError("compact fact span receipt changed")
    return compact


def _compact_packing_audit(audit: Mapping[str, Any]) -> dict[str, Any]:
    probes = audit.get("probes")
    if not isinstance(probes, list):
        raise ValueError("packing audit omitted probes")
    omitted = {"probes", "probed_prefix_counts"}
    value = {
        key: item for key, item in audit.items() if key not in omitted
    }
    value.update(
        {
            "format": COMPACT_PACKING_AUDIT_FORMAT,
            "probe_count": len(probes),
            "probes_sha256": identity_sha256(probes),
            "probed_prefix_counts_sha256": identity_sha256(
                audit.get("probed_prefix_counts")
            ),
            "full_audit_sha256": identity_sha256(audit),
        }
    )
    return _seal_compact(value)


def _validate_compact_packing_audit(value: object) -> dict[str, Any]:
    compact = _validate_compact_receipt(value, label="projection packing audit")
    if (
        compact.get("format") != COMPACT_PACKING_AUDIT_FORMAT
        or compact.get("packer_id") != BINARY_PACKER_ID
        or compact.get("audit_format") != BINARY_PACK_AUDIT_FORMAT
        or compact.get("max_context_tokens") != MAX_CONTEXT_TOKENS
        or compact.get("max_prompt_tokens") != MAX_PROMPT_TOKENS
        or compact.get("output_token_reserve") != OUTPUT_TOKEN_RESERVE
        or compact.get("sampled_monotonicity_validated") is not True
        or compact.get("maximal_prefix_boundary_validated") is not True
        or compact.get("candidate_count")
        != compact.get("packed_count") + compact.get("dropped_count")
        or not isinstance(compact.get("probe_count"), int)
    ):
        raise ValueError("compact projection packing audit changed")
    return compact


def _evidence_from_compact_projection(
    projection: Mapping[str, Any],
    *,
    active_source_ids: Sequence[str],
) -> list[dict[str, Any]]:
    compact = _validate_compact_projection(projection)
    source_by_handle = {
        f"G{index:06d}": source_id
        for index, source_id in enumerate(active_source_ids, 1)
    }
    expected_bindings = [
        {"source_handle": handle, "source_id_sha256": quote_sha256(source_id)}
        for handle, source_id in source_by_handle.items()
    ]
    if compact["active_source_bindings"] != expected_bindings:
        raise ValueError("compact projection source bindings changed")
    evidence = []
    for rank, fact in enumerate(compact["selected_facts"], 1):
        source_id = source_by_handle.get(fact.get("source_handle"))
        if source_id is None:
            raise ValueError("compact fact references an unknown source handle")
        provenance = [str(fact["source_handle"]), str(fact["role"])]
        if fact.get("source_created_at"):
            provenance.append(f"source-time={fact['source_created_at']}")
        rendered = f"[{' | '.join(provenance)}] {fact['quote']}"
        evidence.append(
            {
                "evidence_id": str(fact["chunk_id"]),
                "chunk_id": str(fact["chunk_id"]),
                "source_id": source_id,
                "source_handle": str(fact["source_handle"]),
                "role": str(fact["role"]),
                "created_at": str(fact["source_created_at"]),
                "created_at_semantics": (
                    "source_metadata_only_not_event_time"
                ),
                "route": "activated_assertion_projection",
                "score": 1.0 / rank,
                "raw_text": str(fact["quote"]),
                "raw_text_sha256": quote_sha256(str(fact["quote"])),
                "rendered_text": rendered,
                "rendered_text_sha256": quote_sha256(rendered),
                "assertion_fact_receipt_sha256": fact["receipt_sha256"],
            }
        )
    return evidence


def _compact_projection_arm(
    arm: Mapping[str, Any],
    *,
    compact_projection: Mapping[str, Any],
) -> dict[str, Any]:
    selected_ids = [str(value) for value in arm["selected_chunk_ids"]]
    packed_ids = [str(value) for value in arm["packed_chunk_ids"]]
    value = {
        "format": COMPACT_ARM_FORMAT,
        "projection_compact_receipt_sha256": compact_projection[
            "compact_receipt_sha256"
        ],
        "selected_count": len(selected_ids),
        "packed_count": len(packed_ids),
        "dropped_count": len(arm["dropped_chunk_ids"]),
        "selected_chunk_ids_sha256": identity_sha256(selected_ids),
        "packed_chunk_ids_sha256": identity_sha256(packed_ids),
        "context_token_proxy": int(arm["context_token_proxy"]),
        "prompt_token_proxy": int(arm["prompt_token_proxy"]),
        "prompt_workspace_token_proxy": int(
            arm["prompt_workspace_token_proxy"]
        ),
        "provider_payload_sha256": arm["provider_payload_sha256"],
        "provider_payload_utf8_bytes": arm["provider_payload_utf8_bytes"],
        "raw_evidence_only": arm["raw_evidence_only"],
    }
    return _seal_compact(value)


def _resolve_projection_arm(
    reference: Mapping[str, Any],
    *,
    compact_projection: Mapping[str, Any],
    active_source_ids: Sequence[str],
    dated_question: str,
) -> dict[str, Any]:
    compact_ref = _validate_compact_receipt(
        reference, label="assertion provider packet"
    )
    evidence = _evidence_from_compact_projection(
        compact_projection,
        active_source_ids=active_source_ids,
    )
    packed_count = compact_ref.get("packed_count")
    if (
        compact_ref.get("format") != COMPACT_ARM_FORMAT
        or compact_ref.get("projection_compact_receipt_sha256")
        != compact_projection.get("compact_receipt_sha256")
        or not isinstance(packed_count, int)
        or not 0 <= packed_count <= len(evidence)
    ):
        raise ValueError("compact assertion provider packet changed")
    packed = evidence[:packed_count]
    selected_ids = [str(row["chunk_id"]) for row in evidence]
    packed_ids = selected_ids[:packed_count]
    messages = hot.build_qa_prompt(
        dated_question,
        [str(row["rendered_text"]) for row in packed],
    )
    serialized = hot._canonical_json_bytes({"messages": messages})  # noqa: SLF001
    resolved = {
        "selected_evidence": evidence,
        "packed_evidence": packed,
        "selected_chunk_ids": selected_ids,
        "packed_chunk_ids": packed_ids,
        "dropped_chunk_ids": selected_ids[packed_count:],
        "context_token_proxy": hot._context_token_proxy(  # noqa: SLF001
            [str(row["rendered_text"]) for row in packed]
        ),
        "prompt_token_proxy": hot.count_chat_prompt_token_proxy(messages),
        "prompt_workspace_token_proxy": (
            hot.count_chat_prompt_token_proxy(messages) + OUTPUT_TOKEN_RESERVE
        ),
        "provider_messages": messages,
        "provider_payload_sha256": hashlib.sha256(serialized).hexdigest(),
        "provider_payload_utf8_bytes": len(serialized),
        "raw_evidence_only": True,
    }
    hot._validate_arm_payload(  # noqa: SLF001
        resolved,
        prompt_question=dated_question,
        max_context_tokens=MAX_CONTEXT_TOKENS,
        max_prompt_tokens=MAX_PROMPT_TOKENS,
    )
    expected = _compact_projection_arm(
        resolved,
        compact_projection=compact_projection,
    )
    if compact_ref != expected:
        raise ValueError("assertion provider packet receipt changed")
    return resolved


def _assay_question(
    selected: Mapping[str, Any],
    *,
    v7_selection_sha: str,
    dated_question: str,
    database: Database,
    metadata_by_id: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    started = time.perf_counter_ns()
    active_sources = _active_source_ids(selected)
    scan_started = time.perf_counter_ns()
    scanned, scan_audit = _scan_active_source_rows(
        database,
        active_source_ids=active_sources,
        metadata_by_id=metadata_by_id,
    )
    scan_ns = time.perf_counter_ns() - scan_started
    core_started = time.perf_counter_ns()
    projection, operator, hint = _projection_core_adapter(
        dated_question=dated_question,
        active_source_ids=active_sources,
        rows=scanned,
    )
    core_ns = time.perf_counter_ns() - core_started
    projection_arm, packing_audit, packet_timings = _projection_arm(
        projection,
        dated_question=dated_question,
        active_source_ids=active_sources,
        scanned_rows=scanned,
    )
    compact_projection = _compact_projection(projection)
    compact_projection_arm = _compact_projection_arm(
        projection_arm,
        compact_projection=compact_projection,
    )
    semantic = {
        "ordinal": int(selected["ordinal"]),
        "shard_offset": int(selected["shard_offset"]),
        "local_ordinal": int(selected["local_ordinal"]),
        "question_id": str(selected["question_id"]),
        "probe_sha256": str(selected["probe_sha256"]),
        "retrieval_query_sha256": str(selected["retrieval_query_sha256"]),
        "prompt_question_sha256": str(selected["prompt_question_sha256"]),
        "active_source_scan": _compact_scan_audit(scan_audit),
        "typed_operator_spec": operator.projection(),
        "question_assertion_hint": hint.projection(),
        "assertion_projection": compact_projection,
        "projection_packing_audit": _compact_packing_audit(packing_audit),
        "arms": {
            "assertion_projection": compact_projection_arm,
            "v7_fallback_ref": _fallback_reference(
                selected,
                v7_selection_sha=v7_selection_sha,
            ),
        },
        "route_adoption": {
            "status": "undecided",
            "selected_arm": None,
            "claim": "projection_assay_only_not_an_answer_policy",
        },
    }
    timing = {
        "ordinal": semantic["ordinal"],
        "question_id": semantic["question_id"],
        "shard_offset": semantic["shard_offset"],
        "active_source_scan_ns": scan_ns,
        "assertion_projection_ns": core_ns,
        **packet_timings,
        "question_total_ns": time.perf_counter_ns() - started,
    }
    return semantic, timing


def _collect(
    *,
    selection: Mapping[str, Any],
    v7_selection_sha: str,
    source_root: Path,
    parent_root: Path,
    probes: Mapping[str, Any],
    catalog: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    probe_rows = probes.get("questions")
    source_bindings = probes.get("source_bindings")
    catalog_rows = catalog.get("shards")
    if (
        not isinstance(probe_rows, list)
        or not isinstance(source_bindings, list)
        or not isinstance(catalog_rows, list)
    ):
        raise ValueError("sealed parent material changed")
    probe_by_ordinal = {int(row["ordinal"]): row for row in probe_rows}
    selected_by_offset = {
        offset: [
            row
            for row in selection["questions"]
            if int(row["shard_offset"]) == offset
        ]
        for offset in full100.LOCKED_100Q_OFFSETS
    }
    identity_by_offset = dict(
        zip(
            full100.LOCKED_100Q_OFFSETS,
            probes["population_identity"][
                "ordered_shard_identity_sha256s"
            ],
            strict=True,
        )
    )
    rows: list[dict[str, Any]] = []
    timings: list[dict[str, Any]] = []
    for source_row, catalog_row, offset in zip(
        source_bindings,
        catalog_rows,
        full100.LOCKED_100Q_OFFSETS,
        strict=True,
    ):
        binding = full100._load_source_binding(  # noqa: SLF001
            source_root,
            offset,
            expected_shard_identity=None,
            expected_shard_identity_sha256=str(identity_by_offset[offset]),
            verify_large_files=True,
        )
        if binding.artifact_row() != source_row:
            raise ValueError(f"offset {offset:03d} source receipt changed")
        metadata, compiled_source = v7._load_compiled_metadata(  # noqa: SLF001
            parent_root,
            shard_offset=offset,
            expected_manifest_sha256=str(catalog_row["compiled_sha256"]),
        )
        if compiled_source != binding.artifact_row():
            raise ValueError(
                f"offset {offset:03d} compiled/source receipt changed"
            )
        with Database(binding.database_path, read_only=True) as database:
            for selected in selected_by_offset[offset]:
                ordinal = int(selected["ordinal"])
                probe = probe_by_ordinal.get(ordinal)
                if not isinstance(probe, Mapping):
                    raise ValueError(f"ordinal {ordinal} omitted its sealed probe")
                dated_question = probe.get("prompt_question")
                if (
                    not isinstance(dated_question, str)
                    or not dated_question
                    or probe.get("question_id") != selected.get("question_id")
                    or quote_sha256(dated_question)
                    != selected.get("prompt_question_sha256")
                ):
                    raise ValueError("sealed dated-question binding changed")
                semantic, timing = _assay_question(
                    selected,
                    v7_selection_sha=v7_selection_sha,
                    dated_question=dated_question,
                    database=database,
                    metadata_by_id=metadata,
                )
                rows.append(semantic)
                timings.append(timing)
        print(f"Projected offset-{offset:03d}: 10/10", flush=True)
    rows.sort(key=lambda row: int(row["ordinal"]))
    timings.sort(key=lambda row: int(row["ordinal"]))
    if [row["ordinal"] for row in rows] != list(range(EXPECTED_QUESTION_COUNT)):
        raise RuntimeError("assertion-projection question population changed")
    full100._assert_gold_free_rows(rows)  # noqa: SLF001
    return rows, timings


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(rows) != EXPECTED_QUESTION_COUNT:
        raise ValueError("assertion projection requires the locked full100")
    projection_arms = [row["arms"]["assertion_projection"] for row in rows]
    return {
        "question_count": len(rows),
        "route_adoption_undecided_count": sum(
            row["route_adoption"]["status"] == "undecided" for row in rows
        ),
        "projection_selected_chunk_count": sum(
            int(arm["selected_count"]) for arm in projection_arms
        ),
        "projection_packed_chunk_count": sum(
            int(arm["packed_count"]) for arm in projection_arms
        ),
        "projection_dropped_chunk_count": sum(
            int(arm["dropped_count"]) for arm in projection_arms
        ),
        "projection_max_packed_chunks": max(
            int(arm["packed_count"]) for arm in projection_arms
        ),
        "projection_max_context_token_proxy": max(
            int(arm["context_token_proxy"]) for arm in projection_arms
        ),
        "projection_max_prompt_workspace_token_proxy": max(
            int(arm["prompt_workspace_token_proxy"])
            for arm in projection_arms
        ),
        "fallback_reference_count": sum(
            row["arms"]["v7_fallback_ref"]["format"]
            == FALLBACK_REF_FORMAT
            for row in rows
        ),
        "projection_and_fallback_separate": all(
            set(row["arms"])
            == {"assertion_projection", "v7_fallback_ref"}
            for row in rows
        ),
    }


def _controls() -> dict[str, Any]:
    return {
        "policy_id": POLICY_ID,
        "active_source_derivation": (
            "first_occurrence_exact_source_id_from_v7_packed_evidence"
        ),
        "active_source_scan": "one_parameterized_exact_IN_query_per_question",
        "projection_core": "activated_assertion_projection_v1",
        "operator_source": "dated_question_only_typed_operator_spec_v1",
        "protected_chunk_ids": [],
        "projection_max_chunks": MAX_PROJECTION_CHUNKS,
        "max_context_token_proxy": MAX_CONTEXT_TOKENS,
        "max_prompt_workspace_token_proxy": MAX_PROMPT_TOKENS,
        "responder_output_token_reserve": OUTPUT_TOKEN_RESERVE,
        "binary_packer_id": BINARY_PACKER_ID,
        "binary_packer_audit_format": BINARY_PACK_AUDIT_FORMAT,
        "fallback": "sha_pinned_exact_resolver_to_sealed_adaptive_v7_arm",
        "selection_materialization": (
            "compact_projection_receipts_and_exact_fact_spans_v2"
        ),
        "route_adoption": "undecided",
    }


def _selection_body(
    *,
    v7_root: Path,
    source_root: Path,
    v7_selection: Mapping[str, Any],
    v7_selection_sha: str,
    probes_sha: str,
    catalog_sha: str,
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    materialized = [dict(row) for row in rows]
    return {
        "format": SELECTION_FORMAT,
        "status": "sealed_gold_blind_assertion_projection_full100_assay",
        "bindings": {
            "v7_output_relative_path": _relative_to_repository(
                v7_root, label="v7 output root"
            ),
            "source_root_relative_path": _relative_to_repository(
                source_root, label="source root"
            ),
            "v7_selection_sha256": v7_selection_sha,
            "v7_bindings": v7_selection["bindings"],
            "probes_sha256": probes_sha,
            "compiled_catalog_sha256": catalog_sha,
            "population_identity_sha256": EXPECTED_POPULATION_SHA256,
        },
        "controls": _controls(),
        "implementation": _implementation_identity(),
        "question_population_sha256": identity_sha256(materialized),
        "questions": materialized,
        "aggregate": _aggregate(materialized),
        "route_adoption": "undecided",
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
    selection_sha: str,
    v7_selection_sha: str,
    elapsed_ns: int,
    timings: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    samples = [dict(row) for row in timings]
    fields = (
        "active_source_scan_ns",
        "assertion_projection_ns",
        "binary_pack_ns",
        "serialize_ns",
        "question_total_ns",
    )
    return {
        "format": RUNTIME_FORMAT,
        "status": "provider_free_assertion_projection_runtime",
        "selection_sha256": selection_sha,
        "v7_selection_sha256": v7_selection_sha,
        "population_identity_sha256": EXPECTED_POPULATION_SHA256,
        "implementation_sha256": _implementation_identity()["sha256"],
        "elapsed_ns": elapsed_ns,
        "samples": samples,
        "summaries_ns": {
            field: _timing_summary([int(row[field]) for row in samples])
            for field in fields
        },
        "latency_scope": (
            "incremental_activated_source_scan_projection_pack_serialize_"
            "not_end_to_end"
        ),
        "timings_excluded_from_semantic_identity": True,
        "provider_rtt_prefill_decode_excluded": True,
        "qwen_calls": 0,
        "provider_calls": 0,
    }


def _validate_question_rows(
    rows: object,
    *,
    v7_selection: Mapping[str, Any],
    v7_selection_sha: str,
    probes: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if not isinstance(rows, list) or len(rows) != EXPECTED_QUESTION_COUNT:
        raise ValueError("assertion-projection rows changed")
    probe_by_ordinal = {
        int(row["ordinal"]): row for row in probes["questions"]
    }
    materialized: list[dict[str, Any]] = []
    for row, v7_row in zip(rows, v7_selection["questions"], strict=True):
        if not isinstance(row, dict):
            raise ValueError("assertion-projection row must be an object")
        expected_keys = {
            "ordinal",
            "shard_offset",
            "local_ordinal",
            "question_id",
            "probe_sha256",
            "retrieval_query_sha256",
            "prompt_question_sha256",
            "active_source_scan",
            "typed_operator_spec",
            "question_assertion_hint",
            "assertion_projection",
            "projection_packing_audit",
            "arms",
            "route_adoption",
        }
        if set(row) != expected_keys:
            raise ValueError("compact assertion-projection row shape changed")
        ordinal = int(v7_row["ordinal"])
        probe = probe_by_ordinal.get(ordinal)
        if not isinstance(probe, Mapping):
            raise ValueError("assertion-projection row lost its probe")
        prompt = probe.get("prompt_question")
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("assertion-projection row lost its dated question")
        active_sources = _active_source_ids(v7_row)
        arms = row.get("arms")
        route = row.get("route_adoption")
        if (
            any(
                row.get(key) != v7_row.get(key)
                for key in (
                    "ordinal",
                    "shard_offset",
                    "local_ordinal",
                    "question_id",
                    "probe_sha256",
                    "retrieval_query_sha256",
                    "prompt_question_sha256",
                )
            )
            or not isinstance(arms, Mapping)
            or set(arms) != {"assertion_projection", "v7_fallback_ref"}
            or not isinstance(route, Mapping)
            or route
            != {
                "status": "undecided",
                "selected_arm": None,
                "claim": "projection_assay_only_not_an_answer_policy",
            }
        ):
            raise ValueError("assertion-projection/fallback boundary changed")
        scan = _validate_compact_scan_audit(
            row.get("active_source_scan"),
            active_source_ids=active_sources,
        )
        spec = compile_typed_operator_spec(prompt)
        obligation_terms = tuple(
            dict.fromkeys(
                term
                for slot in spec.required_slots
                for term in slot.match_terms
            )
        )
        role_mode = {
            "user": AssertionRoleMode.USER,
            "assistant": AssertionRoleMode.ASSISTANT,
        }.get(spec.required_evidence_role)
        hint = QuestionAssertionHint(
            dated_question_sha256=hashlib.sha256(
                prompt.encode("utf-8")
            ).hexdigest(),
            obligation_terms=obligation_terms,
            role_mode=role_mode,
        )
        if (
            row.get("typed_operator_spec") != spec.projection()
            or row.get("question_assertion_hint") != hint.projection()
        ):
            raise ValueError("question-only projection controls changed")
        compact_projection = _validate_compact_projection(
            row.get("assertion_projection")
        )
        expected_policy = ActivatedAssertionPolicy(
            include_proposed=spec.include_proposed
        ).projection()
        if (
            compact_projection.get("dated_question_sha256")
            != hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            or compact_projection.get("question_hint_receipt_sha256")
            != hint.receipt_sha256
            or compact_projection.get("policy") != expected_policy
            or len(compact_projection.get("active_source_bindings", []))
            != len(active_sources)
            or compact_projection.get("candidate_audit_count")
            != scan.get("scanned_chunk_count")
        ):
            raise ValueError("compact projection/control binding changed")
        projection_arm = _resolve_projection_arm(
            arms["assertion_projection"],
            compact_projection=compact_projection,
            active_source_ids=active_sources,
            dated_question=prompt,
        )
        fallback = _resolve_fallback_reference(
            arms["v7_fallback_ref"],
            v7_row=v7_row,
            v7_selection_sha=v7_selection_sha,
        )
        hot._validate_arm_payload(  # noqa: SLF001
            fallback,
            prompt_question=prompt,
            max_context_tokens=MAX_CONTEXT_TOKENS,
            max_prompt_tokens=MAX_PROMPT_TOKENS,
        )
        packing = _validate_compact_packing_audit(
            row.get("projection_packing_audit")
        )
        if (
            len(projection_arm["selected_chunk_ids"])
            > MAX_PROJECTION_CHUNKS
            or len(projection_arm["packed_chunk_ids"])
            > MAX_PROJECTION_CHUNKS
            or projection_arm["context_token_proxy"] > MAX_CONTEXT_TOKENS
            or projection_arm["prompt_workspace_token_proxy"]
            > MAX_PROMPT_TOKENS
            or packing.get("candidate_count")
            != len(projection_arm["selected_chunk_ids"])
            or packing.get("packed_count")
            != len(projection_arm["packed_chunk_ids"])
            or packing.get("dropped_count")
            != len(projection_arm["dropped_chunk_ids"])
        ):
            raise ValueError("projection packet cap changed")
        materialized.append(row)
    full100._assert_gold_free_rows(materialized)  # noqa: SLF001
    return materialized


def _load_selection(
    *,
    v7_root: Path,
    source_root: Path,
    output_root: Path,
) -> tuple[
    dict[str, Any],
    str,
    dict[str, Any],
    str,
    Path,
    dict[str, Any],
    str,
    dict[str, Any],
    str,
]:
    v7_selection, v7_sha = _load_v7_selection(v7_root)
    parent_root, probes, probes_sha, catalog, catalog_sha = _parent_material(
        v7_selection
    )
    body, digest = hot._read_json_artifact(  # noqa: SLF001
        output_root / SELECTION_NAME
    )
    rows = _validate_question_rows(
        body.get("questions"),
        v7_selection=v7_selection,
        v7_selection_sha=v7_sha,
        probes=probes,
    )
    expected = _selection_body(
        v7_root=v7_root,
        source_root=source_root,
        v7_selection=v7_selection,
        v7_selection_sha=v7_sha,
        probes_sha=probes_sha,
        catalog_sha=catalog_sha,
        rows=rows,
    )
    if body != expected:
        raise ValueError("assertion-projection selection receipt changed")
    return (
        body,
        digest,
        v7_selection,
        v7_sha,
        parent_root,
        probes,
        probes_sha,
        catalog,
        catalog_sha,
    )


def _validate_runtime(
    *,
    output_root: Path,
    selection_sha: str,
    v7_selection_sha: str,
) -> tuple[dict[str, Any], str]:
    body, digest = hot._read_json_artifact(  # noqa: SLF001
        output_root / RUNTIME_NAME
    )
    samples = body.get("samples")
    if (
        body.get("format") != RUNTIME_FORMAT
        or body.get("selection_sha256") != selection_sha
        or body.get("v7_selection_sha256") != v7_selection_sha
        or body.get("implementation_sha256")
        != _implementation_identity()["sha256"]
        or not isinstance(samples, list)
        or len(samples) != EXPECTED_QUESTION_COUNT
        or [row.get("ordinal") for row in samples]
        != list(range(EXPECTED_QUESTION_COUNT))
    ):
        raise ValueError("assertion-projection runtime changed")
    expected = _runtime_body(
        selection_sha=selection_sha,
        v7_selection_sha=v7_selection_sha,
        elapsed_ns=int(body["elapsed_ns"]),
        timings=samples,
    )
    if body != expected:
        raise ValueError("assertion-projection runtime fields changed")
    return body, digest


def _run_manifest_body(
    *,
    selection_sha: str,
    runtime_sha: str,
    v7_selection_sha: str,
) -> dict[str, Any]:
    return {
        "format": RUN_MANIFEST_FORMAT,
        "status": "complete_provider_free_compact_run_bundle",
        "selection": {"path": SELECTION_NAME, "sha256": selection_sha},
        "runtime": {"path": RUNTIME_NAME, "sha256": runtime_sha},
        "v7_selection_sha256": v7_selection_sha,
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


def _validate_run_manifest(
    *,
    output_root: Path,
    selection_sha: str,
    runtime_sha: str,
    v7_selection_sha: str,
) -> tuple[dict[str, Any], str]:
    body, digest = hot._read_json_artifact(  # noqa: SLF001
        output_root / RUN_MANIFEST_NAME
    )
    expected = _run_manifest_body(
        selection_sha=selection_sha,
        runtime_sha=runtime_sha,
        v7_selection_sha=v7_selection_sha,
    )
    if body != expected:
        raise ValueError("assertion-projection run manifest changed")
    return body, digest


def _replay_body(
    *,
    selection: Mapping[str, Any],
    selection_sha: str,
    v7_selection_sha: str,
) -> dict[str, Any]:
    selection_bytes = hot._canonical_json_bytes(selection)  # noqa: SLF001
    payload_sha = hashlib.sha256(selection_bytes).hexdigest()
    if payload_sha != selection_sha:
        raise ValueError("selection bytes differ from their sealed digest")
    return {
        "format": REPLAY_FORMAT,
        "status": "byte_identical_gold_blind_assertion_projection_replay",
        "selection_sha256": selection_sha,
        "v7_selection_sha256": v7_selection_sha,
        "population_identity_sha256": EXPECTED_POPULATION_SHA256,
        "implementation_sha256": _implementation_identity()["sha256"],
        "selection_payload_sha256": payload_sha,
        "replayed_payload_sha256": payload_sha,
        "selection_payload_utf8_bytes": len(selection_bytes),
        "replayed_payload_utf8_bytes": len(selection_bytes),
        "byte_identical": True,
        "question_count": EXPECTED_QUESTION_COUNT,
        "route_adoption": "undecided",
        "gold_fields_present": False,
        "qwen_calls": 0,
        "responder_calls": 0,
        "judge_calls": 0,
        "provider_calls": 0,
    }


def _validate_replay(
    *,
    output_root: Path,
    selection: Mapping[str, Any],
    selection_sha: str,
    v7_selection_sha: str,
) -> tuple[dict[str, Any], str]:
    body, digest = hot._read_json_artifact(  # noqa: SLF001
        output_root / REPLAY_NAME
    )
    expected = _replay_body(
        selection=selection,
        selection_sha=selection_sha,
        v7_selection_sha=v7_selection_sha,
    )
    if body != expected:
        raise ValueError("assertion-projection replay receipt changed")
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
        for path in (
            output_root / name,
            output_root / f"{name}.sha256",
        )
    )


def _assert_fresh_output_root(output_root: Path) -> None:
    present = [
        path.name
        for path in _known_artifact_paths(output_root)
        if path.exists()
    ]
    if present:
        raise FileExistsError(
            "refusing to publish a new selection into a non-empty artifact "
            f"lifecycle: {', '.join(present)}"
        )


def _remove_exact_artifact(path: Path, expected_sha: str) -> None:
    """Roll back only bytes written by the current failed publication."""

    sidecar = path.with_name(path.name + ".sha256")
    expected_sidecar = f"{expected_sha}  {path.name}\n".encode("ascii")
    if sidecar.is_file() and sidecar.read_bytes() == expected_sidecar:
        sidecar.unlink()
    if path.is_file() and file_sha256(path) == expected_sha:
        path.unlink()


def _revalidate_upstream_before_run_publish(
    *,
    v7_root: Path,
    expected_v7_selection: Mapping[str, Any],
    expected_v7_sha: str,
    expected_parent_root: Path,
    expected_probes_sha: str,
    expected_catalog_sha: str,
) -> None:
    current_v7, current_v7_sha = _load_v7_selection(v7_root)
    parent_root, _probes, probes_sha, _catalog, catalog_sha = _parent_material(
        current_v7
    )
    if (
        current_v7 != expected_v7_selection
        or current_v7_sha != expected_v7_sha
        or parent_root.resolve() != expected_parent_root.resolve()
        or probes_sha != expected_probes_sha
        or catalog_sha != expected_catalog_sha
    ):
        raise RuntimeError("run prerequisites changed during collection")


def _recover_or_validate_existing_run(
    *,
    v7_root: Path,
    source_root: Path,
    output_root: Path,
    v7_selection_sha: str,
) -> str | None:
    """Validate a complete run or recover only exact owned crash windows."""

    path = output_root / SELECTION_NAME
    _body, digest, *_rest = _load_selection(
        v7_root=v7_root,
        source_root=source_root,
        output_root=output_root,
    )
    runtime_path = output_root / RUNTIME_NAME
    runtime_sidecar = runtime_path.with_name(runtime_path.name + ".sha256")
    if not runtime_path.exists() and not runtime_sidecar.exists():
        downstream = (
            output_root / RUN_MANIFEST_NAME,
            output_root / f"{RUN_MANIFEST_NAME}.sha256",
            output_root / REPLAY_NAME,
            output_root / f"{REPLAY_NAME}.sha256",
            output_root / SCORE_NAME,
            output_root / f"{SCORE_NAME}.sha256",
        )
        if any(candidate.exists() for candidate in downstream):
            raise ValueError(
                "selection-only crash state has downstream artifacts"
            )
        _remove_exact_artifact(path, digest)
        print(
            "Recovered validated selection-only crash state; recomputing run",
            flush=True,
        )
        return None
    if runtime_path.exists() != runtime_sidecar.exists():
        raise ValueError("runtime artifact/sidecar crash state is not provable")

    _runtime, runtime_sha = _validate_runtime(
        output_root=output_root,
        selection_sha=digest,
        v7_selection_sha=v7_selection_sha,
    )
    manifest_path = output_root / RUN_MANIFEST_NAME
    manifest_sidecar = manifest_path.with_name(manifest_path.name + ".sha256")
    if manifest_sidecar.exists() and not manifest_path.exists():
        raise ValueError("run manifest sidecar is orphaned")
    if manifest_path.exists() and not manifest_sidecar.exists():
        expected = _run_manifest_body(
            selection_sha=digest,
            runtime_sha=runtime_sha,
            v7_selection_sha=v7_selection_sha,
        )
        # The immutable writer first proves the existing canonical bytes and
        # then creates only the missing digest sidecar.
        hot._atomic_write_json(manifest_path, expected)  # noqa: SLF001
    if manifest_path.exists():
        _validate_run_manifest(
            output_root=output_root,
            selection_sha=digest,
            runtime_sha=runtime_sha,
            v7_selection_sha=v7_selection_sha,
        )
    else:
        downstream = (
            output_root / REPLAY_NAME,
            output_root / f"{REPLAY_NAME}.sha256",
            output_root / SCORE_NAME,
            output_root / f"{SCORE_NAME}.sha256",
        )
        if any(candidate.exists() for candidate in downstream):
            raise ValueError(
                "cannot complete a run manifest above downstream artifacts"
            )
        manifest = _run_manifest_body(
            selection_sha=digest,
            runtime_sha=runtime_sha,
            v7_selection_sha=v7_selection_sha,
        )
        hot._atomic_write_json(manifest_path, manifest)  # noqa: SLF001
        _validate_run_manifest(
            output_root=output_root,
            selection_sha=digest,
            runtime_sha=runtime_sha,
            v7_selection_sha=v7_selection_sha,
        )
        print(
            "Recovered complete run manifest for validated immutable "
            f"selection/runtime pair: {manifest_path}",
            flush=True,
        )
    print(f"Assertion projection verified: {path} ({digest})", flush=True)
    return digest


def run(*, v7_root: Path, source_root: Path, output_root: Path) -> str:
    v7_selection, v7_sha = _load_v7_selection(v7_root)
    parent_root, probes, probes_sha, catalog, catalog_sha = _parent_material(
        v7_selection
    )
    path = output_root / SELECTION_NAME
    if path.exists():
        existing = _recover_or_validate_existing_run(
            v7_root=v7_root,
            source_root=source_root,
            output_root=output_root,
            v7_selection_sha=v7_sha,
        )
        if existing is not None:
            return existing
    _assert_fresh_output_root(output_root)
    started = time.perf_counter_ns()
    rows, timings = _collect(
        selection=v7_selection,
        v7_selection_sha=v7_sha,
        source_root=source_root,
        parent_root=parent_root,
        probes=probes,
        catalog=catalog,
    )
    elapsed_ns = time.perf_counter_ns() - started
    body = _selection_body(
        v7_root=v7_root,
        source_root=source_root,
        v7_selection=v7_selection,
        v7_selection_sha=v7_sha,
        probes_sha=probes_sha,
        catalog_sha=catalog_sha,
        rows=rows,
    )
    aggregate = body["aggregate"]
    if (
        aggregate["fallback_reference_count"] != EXPECTED_QUESTION_COUNT
        or aggregate["route_adoption_undecided_count"]
        != EXPECTED_QUESTION_COUNT
        or aggregate["projection_max_packed_chunks"] > MAX_PROJECTION_CHUNKS
        or not aggregate["projection_and_fallback_separate"]
    ):
        raise RuntimeError("assertion projection failed its structural gate")
    _revalidate_upstream_before_run_publish(
        v7_root=v7_root,
        expected_v7_selection=v7_selection,
        expected_v7_sha=v7_sha,
        expected_parent_root=parent_root,
        expected_probes_sha=probes_sha,
        expected_catalog_sha=catalog_sha,
    )
    # A second freshness check closes the collection-time window in which a
    # stale downstream receipt could otherwise appear under this run.
    _assert_fresh_output_root(output_root)
    selection_sha = hashlib.sha256(
        hot._canonical_json_bytes(body)  # noqa: SLF001
    ).hexdigest()
    runtime = _runtime_body(
        selection_sha=selection_sha,
        v7_selection_sha=v7_sha,
        elapsed_ns=elapsed_ns,
        timings=timings,
    )
    runtime_path = output_root / RUNTIME_NAME
    runtime_sha = hashlib.sha256(
        hot._canonical_json_bytes(runtime)  # noqa: SLF001
    ).hexdigest()
    manifest = _run_manifest_body(
        selection_sha=selection_sha,
        runtime_sha=runtime_sha,
        v7_selection_sha=v7_sha,
    )
    manifest_path = output_root / RUN_MANIFEST_NAME
    manifest_sha = hashlib.sha256(
        hot._canonical_json_bytes(manifest)  # noqa: SLF001
    ).hexdigest()
    try:
        published_selection_sha = hot._atomic_write_json(  # noqa: SLF001
            path, body
        )
        if published_selection_sha != selection_sha:
            raise RuntimeError("selection publication digest changed")
        published_runtime_sha = hot._atomic_write_json(  # noqa: SLF001
            runtime_path, runtime
        )
        if published_runtime_sha != runtime_sha:
            raise RuntimeError("runtime publication digest changed")
        published_manifest_sha = hot._atomic_write_json(  # noqa: SLF001
            manifest_path, manifest
        )
        if published_manifest_sha != manifest_sha:
            raise RuntimeError("run manifest publication digest changed")
        _load_selection(
            v7_root=v7_root,
            source_root=source_root,
            output_root=output_root,
        )
        _runtime, observed_runtime_sha = _validate_runtime(
            output_root=output_root,
            selection_sha=selection_sha,
            v7_selection_sha=v7_sha,
        )
        if observed_runtime_sha != runtime_sha:
            raise RuntimeError("runtime digest changed after publication")
        _validate_run_manifest(
            output_root=output_root,
            selection_sha=selection_sha,
            runtime_sha=runtime_sha,
            v7_selection_sha=v7_sha,
        )
    except BaseException:
        _remove_exact_artifact(manifest_path, manifest_sha)
        _remove_exact_artifact(runtime_path, runtime_sha)
        _remove_exact_artifact(path, selection_sha)
        raise
    print(
        f"Assertion projection published: {selection_sha}; "
        f"runtime={runtime_sha}; manifest={manifest_sha}; "
        "fallback=100/100 hash-resolvable",
        flush=True,
    )
    return selection_sha


def replay(*, v7_root: Path, source_root: Path, output_root: Path) -> str:
    (
        expected,
        selection_sha,
        v7_selection,
        v7_sha,
        parent_root,
        probes,
        probes_sha,
        catalog,
        catalog_sha,
    ) = _load_selection(
        v7_root=v7_root,
        source_root=source_root,
        output_root=output_root,
    )
    _runtime, runtime_sha = _validate_runtime(
        output_root=output_root,
        selection_sha=selection_sha,
        v7_selection_sha=v7_sha,
    )
    _validate_run_manifest(
        output_root=output_root,
        selection_sha=selection_sha,
        runtime_sha=runtime_sha,
        v7_selection_sha=v7_sha,
    )
    rows, _timings = _collect(
        selection=v7_selection,
        v7_selection_sha=v7_sha,
        source_root=source_root,
        parent_root=parent_root,
        probes=probes,
        catalog=catalog,
    )
    replayed = _selection_body(
        v7_root=v7_root,
        source_root=source_root,
        v7_selection=v7_selection,
        v7_selection_sha=v7_sha,
        probes_sha=probes_sha,
        catalog_sha=catalog_sha,
        rows=rows,
    )
    expected_bytes = hot._canonical_json_bytes(expected)  # noqa: SLF001
    replayed_bytes = hot._canonical_json_bytes(replayed)  # noqa: SLF001
    if expected_bytes != replayed_bytes:
        raise RuntimeError("assertion-projection replay differs from selection")
    # Re-open both prerequisites immediately before publication.  Without
    # this check, a replay process can retain them in memory through _collect
    # and publish an orphan after another process removes the run pair.
    current, current_sha, _v7, current_v7_sha, *_rest = _load_selection(
        v7_root=v7_root,
        source_root=source_root,
        output_root=output_root,
    )
    if (
        current != expected
        or current_sha != selection_sha
        or current_v7_sha != v7_sha
    ):
        raise RuntimeError("replay prerequisites changed during collection")
    _runtime, current_runtime_sha = _validate_runtime(
        output_root=output_root,
        selection_sha=selection_sha,
        v7_selection_sha=v7_sha,
    )
    if current_runtime_sha != runtime_sha:
        raise RuntimeError("replay runtime changed during collection")
    _validate_run_manifest(
        output_root=output_root,
        selection_sha=selection_sha,
        runtime_sha=runtime_sha,
        v7_selection_sha=v7_sha,
    )
    body = _replay_body(
        selection=expected,
        selection_sha=selection_sha,
        v7_selection_sha=v7_sha,
    )
    digest = hot._atomic_write_json(  # noqa: SLF001
        output_root / REPLAY_NAME, body
    )
    _validate_replay(
        output_root=output_root,
        selection=expected,
        selection_sha=selection_sha,
        v7_selection_sha=v7_sha,
    )
    print(f"Assertion projection replay: {digest}; byte_identical=true", flush=True)
    return digest


def _arm_score(arm: Mapping[str, Any], question: Any) -> dict[str, Any]:
    packed = arm["packed_evidence"]
    texts = [str(row["raw_text"]) for row in packed]
    retrieved_sources = {
        str(row["source_id"]) for row in packed if row.get("source_id")
    }
    expected_sources = {str(value) for value in question.evidence_sources}
    source_recall = (
        None
        if not expected_sources
        else len(expected_sources & retrieved_sources) / len(expected_sources)
    )
    components = answer_value_component_coverage(
        question.answer, len(expected_sources), texts
    )
    return {
        "packed_count": len(packed),
        "all_gold_source_ids_reached": expected_sources <= retrieved_sources,
        "gold_source_id_recall": source_recall,
        "literal_answer": contains_answer(texts, question.answer),
        "best_f1": best_f1(texts, question.answer),
        "answer_value_component_recall": (
            None if components is None else components.recall
        ),
        "all_answer_value_components": (
            None if components is None else components.all_components
        ),
        "answer_value_component_metric_kind": (
            None if components is None else components.metric_kind
        ),
    }


def score(
    *,
    dataset: Path,
    split_manifest: Path,
    v7_root: Path,
    source_root: Path,
    output_root: Path,
) -> str:
    selection, selection_sha, v7_selection, v7_sha, *_rest = _load_selection(
        v7_root=v7_root,
        source_root=source_root,
        output_root=output_root,
    )
    _runtime, runtime_sha = _validate_runtime(
        output_root=output_root,
        selection_sha=selection_sha,
        v7_selection_sha=v7_sha,
    )
    _validate_run_manifest(
        output_root=output_root,
        selection_sha=selection_sha,
        runtime_sha=runtime_sha,
        v7_selection_sha=v7_sha,
    )
    _replay_body_receipt, replay_sha = _validate_replay(
        output_root=output_root,
        selection=selection,
        selection_sha=selection_sha,
        v7_selection_sha=v7_sha,
    )
    samples, _identities, population = full100._load_population(  # noqa: SLF001
        dataset, split_manifest
    )
    questions = full100._flatten_questions(samples)  # noqa: SLF001
    rows: list[dict[str, Any]] = []
    for selected, v7_row, question in zip(
        selection["questions"],
        v7_selection["questions"],
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
        active_sources = _active_source_ids(v7_row)
        projection_arm = _resolve_projection_arm(
            selected["arms"]["assertion_projection"],
            compact_projection=selected["assertion_projection"],
            active_source_ids=active_sources,
            dated_question=question.dated_question,
        )
        fallback_arm = _resolve_fallback_reference(
            selected["arms"]["v7_fallback_ref"],
            v7_row=v7_row,
            v7_selection_sha=v7_sha,
        )
        rows.append(
            {
                "ordinal": int(selected["ordinal"]),
                "question_id": question.question_id,
                "assertion_projection": _arm_score(
                    projection_arm, question
                ),
                "v7_fallback": _arm_score(
                    fallback_arm, question
                ),
            }
        )
    aggregate: dict[str, Any] = {}
    for arm_id in ("assertion_projection", "v7_fallback"):
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
        "status": "post_replay_gold_join_assertion_projection_assay",
        "selection_sha256": selection_sha,
        "replay_sha256": replay_sha,
        "population_identity_sha256": population["population_identity_sha256"],
        "question_count": EXPECTED_QUESTION_COUNT,
        "route_adoption": "undecided",
        "questions": rows,
        "aggregate": aggregate,
        "gold_fields_present": True,
        "qwen_calls": 0,
        "responder_calls": 0,
        "judge_calls": 0,
        "provider_calls": 0,
    }
    # Prevent a score from becoming an orphan if its sealed prerequisites
    # disappear or change while the post-replay gold join is computed.
    current, current_sha, _v7, current_v7_sha, *_rest = _load_selection(
        v7_root=v7_root,
        source_root=source_root,
        output_root=output_root,
    )
    if (
        current != selection
        or current_sha != selection_sha
        or current_v7_sha != v7_sha
    ):
        raise RuntimeError("score prerequisites changed during gold join")
    _runtime, current_runtime_sha = _validate_runtime(
        output_root=output_root,
        selection_sha=selection_sha,
        v7_selection_sha=v7_sha,
    )
    if current_runtime_sha != runtime_sha:
        raise RuntimeError("score runtime changed during gold join")
    _validate_run_manifest(
        output_root=output_root,
        selection_sha=selection_sha,
        runtime_sha=runtime_sha,
        v7_selection_sha=v7_sha,
    )
    _current_replay, current_replay_sha = _validate_replay(
        output_root=output_root,
        selection=selection,
        selection_sha=selection_sha,
        v7_selection_sha=v7_sha,
    )
    if current_replay_sha != replay_sha:
        raise RuntimeError("score replay changed during gold join")
    digest = hot._atomic_write_json(  # noqa: SLF001
        output_root / SCORE_NAME, body
    )
    print(
        f"Assertion projection score: {digest}; "
        "projection gold source-ID reach="
        f"{aggregate['assertion_projection']['all_gold_source_id_reach_hits']}/100, "
        f"literal={aggregate['assertion_projection']['literal_answer_hits']}/100; "
        "fallback gold source-ID reach="
        f"{aggregate['v7_fallback']['all_gold_source_id_reach_hits']}/100, "
        f"literal={aggregate['v7_fallback']['literal_answer_hits']}/100",
        flush=True,
    )
    return digest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v7-root", type=Path, default=DEFAULT_V7_ROOT)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("run")
    commands.add_parser("replay")
    score = commands.add_parser("score")
    score.add_argument("--dataset", type=Path, required=True)
    score.add_argument("--split-manifest", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    v7_root = args.v7_root.resolve()
    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    if args.command == "run":
        run(
            v7_root=v7_root,
            source_root=source_root,
            output_root=output_root,
        )
    elif args.command == "replay":
        replay(
            v7_root=v7_root,
            source_root=source_root,
            output_root=output_root,
        )
    elif args.command == "score":
        score(
            dataset=args.dataset.resolve(),
            split_manifest=args.split_manifest.resolve(),
            v7_root=v7_root,
            source_root=source_root,
            output_root=output_root,
        )
    else:  # pragma: no cover
        raise AssertionError(f"unhandled command: {args.command}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
