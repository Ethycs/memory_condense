"""Authenticated provider-free bridge into the ``mem0-typed-v1`` epoch.

The official resumable terminal deliberately publishes the exact legacy V2
retrieval rows used by the fair Mem0 arm.  Those rows contain inferred memory
text but not the diagnostic request-window attribution retained by the
resumable add journal.  This module authenticates both views against the
locked ten-shard source population, joins them by opaque Mem0 memory ID, and
rebuilds V3 typed retrieval rows before that attribution boundary is lost.

Nothing in this module constructs a Mem0 client or calls a provider.  Request
windows remain diagnostic, are excluded from provider messages, and never
become exact fact provenance.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

from memory_condense.eval.mem0_adapter import MEM0_ATTRIBUTION_KIND
from tools.matched_eval.artifacts import (
    SealedArtifact,
    publish_sealed_json,
    read_sealed_json,
)
from tools.matched_eval.contracts import (
    MatchedEvalContractError,
    assert_gold_blind,
    canonical_json_bytes,
    identity_sha256,
    require_sha256,
)

from .policy import Mem0ComparisonPolicy, load_mem0_comparison_policy
from .preflight import SourceValidationPlan, load_source_validation_plan
from .prompt_pack import (
    MEM0_TYPED_RETRIEVAL_ROW_FORMAT,
    PromptRequestWindowRef,
    pack_mem0_typed_prompt,
)
from .protocol import (
    RawStressShard,
    build_raw_stress_shards,
    shard_receipt,
)
from .resumable import (
    AppendOnlyResumeJournal,
    ReplayState,
    ResumableShardError,
    _assert_no_link_ancestors,
    _path_identity_sha256,
    canonical_json_sha256,
    rehydration_material,
)
from .resumable_runner import _source_ref_dict, build_resume_plan, prepared_batch_sha256
from .run_shard import (
    RESUMABLE_RETRIEVAL_CERTIFICATION,
    _default_prompt_packer,
    _verify_retrieval_artifact,
    build_adapter_prepared_corpus,
)
from .source_compat import count_tokens


FORMAT = "memory-condense-mem0-typed-source-bridge-v1"
ROW_FORMAT = f"{FORMAT}-shard-v1"
SOURCE_FORMAT = f"{FORMAT}-locked-source-v1"
MANIFEST_NAME = "mem0-typed-source-bridge-v1.json"
EXPORT_NAME_TEMPLATE = "mem0-typed-retrieval-export-offset-{offset:03d}-v1.json"
FROZEN_OFFSETS = tuple(range(0, 100, 10))
FROZEN_QUESTION_COUNT = 100
FROZEN_QUESTIONS_PER_SHARD = 10
FROZEN_TARGET_TOKENS = 1_000_000
FROZEN_RAW_PAIRS = 24_928
FROZEN_SKIPPED_EMPTY_PAIRS = 5
FROZEN_ADD_OPERATIONS = 24_923
FROZEN_TRANSCRIPT_TOKENS = 10_441_617

# These are root-of-trust identities from the validation-v3 source freeze,
# not values learned from a caller-supplied manifest.  The policy and source
# readers still rebuild every constituent file; these pins prevent a
# self-consistent replacement dataset/policy/tool tree from minting a second
# population which merely has the same 10x10 shape and aggregate add count.
FROZEN_DATASET_SHA256 = (
    "d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442"
)
FROZEN_SPLIT_MANIFEST_SHA256 = (
    "8d5c1885903b199a4ab0859ccabc5ce41d9a105d0c755d3daf33cbfd959995f4"
)
FROZEN_SOURCE_POLICY_SHA256 = (
    "5263d5afd15298ec4088db9d6381ae243ddb685e9a3cf4d9892fc84e14fb9883"
)
FROZEN_SOURCE_IMPLEMENTATION_SHA256 = (
    "452be3bfa7524bb81676c7abcb032529a32a480311d24d1e17f8513c783ecd83"
)
FROZEN_SOURCE_ENVIRONMENT_LOCK_SHA256 = (
    "058083871240979257ada7ca4c71dd816fee64792b275ef11e4857c9f5ebba33"
)
FROZEN_SOURCE_EVALUATION_IDENTITY_SHA256 = (
    "bb5cbff63fcc6d27ddafce2fe63b54c2b4c66d6e43e33615224c6cb37f72fae4"
)
FROZEN_SHARD_POPULATION_SHA256 = (
    "f4b23065c597fbfe40666c47b232da06b26a9469b4d1fa84b41fc6099a31768e"
)
FROZEN_QUESTION_ORDER_SHA256 = (
    "7a67aa6f43ffb94d487fb9184f871735bd9edac1974a3154898846d1140c83a1"
)


class Mem0TypedSourceBridgeError(MatchedEvalContractError):
    """An official terminal, journal, source, or typed export changed."""


def _require(ok: object, message: str) -> None:
    if not ok:
        raise Mem0TypedSourceBridgeError(message)


def _strict_json(value: object, label: str) -> Any:
    try:
        return json.loads(
            json.dumps(
                value,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
    except (TypeError, ValueError) as exc:
        raise Mem0TypedSourceBridgeError(f"{label} must be strict JSON") from exc


def _plain_path(value: str | Path, label: str, *, directory: bool = False) -> Path:
    try:
        path = _assert_no_link_ancestors(
            value,
            label=label,
            require_final=True,
        )
    except ResumableShardError as exc:
        raise Mem0TypedSourceBridgeError(str(exc)) from exc
    if not path.is_dir() if directory else not path.is_file():
        kind = "directory" if directory else "regular file"
        raise Mem0TypedSourceBridgeError(f"{label} must be a plain {kind}: {path}")
    return path


def _file_receipt(value: str | Path, label: str) -> dict[str, Any]:
    path = _plain_path(value, label)
    raw = path.read_bytes()
    return {
        "bytes": len(raw),
        "path": str(path),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _same_file_receipt(observed: Mapping[str, Any], expected: Mapping[str, Any], label: str) -> None:
    _require(dict(observed) == dict(expected), f"{label} file receipt changed")


@dataclass(frozen=True, slots=True)
class ResumableTerminalInput:
    """Three official Stage-A authorities for one locked shard."""

    artifact_path: Path
    trace_path: Path
    journal_path: Path


@dataclass(frozen=True, slots=True)
class LockedSourceInputs:
    """All paths needed to independently reconstruct the frozen population."""

    benchmark_file: Path
    split_manifest: Path
    policy_manifest: Path
    repository_root: Path
    mem0_policy_manifest: Path
    mem0_environment_lock: Path
    mem0_tool_root: Path


@dataclass(frozen=True, slots=True)
class _LockedContext:
    source: LockedSourceInputs
    source_plan: SourceValidationPlan
    policy: Mem0ComparisonPolicy
    shards: tuple[RawStressShard, ...]
    population_identity_sha256: str
    population_projection: Mapping[str, Any]
    source_projection: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class SourceBridgeVerification:
    """Reopened bridge manifest and its byte-identical typed exports."""

    manifest: SealedArtifact
    exports: tuple[SealedArtifact, ...]
    population_identity_sha256: str
    question_order_sha256: str


def _verify_commits_against_locked_batches(
    state: ReplayState, shard: RawStressShard
) -> None:
    """Bind every durable add commit to the literal reconstructed add batch."""

    batches = build_adapter_prepared_corpus(shard).batches
    intents = {
        row["ordinal"]: row for row in state.entries if row.get("kind") == "intent"
    }
    _require(
        len(state.commits) == len(batches) == len(intents),
        "journal add population differs from the locked reconstructed corpus",
    )
    for ordinal, (commit, batch) in enumerate(zip(state.commits, batches, strict=True)):
        source_ref = _source_ref_dict(batch.ref)
        batch_sha256 = prepared_batch_sha256(batch)
        intent = intents.get(ordinal)
        raw_message_tokens = count_tokens(
            "\n".join(f"{role}: {content}" for role, content in batch.messages)
        )
        _require(
            commit.get("ordinal") == ordinal
            and intent is not None
            and commit.get("intent_entry_sha256") == intent.get("entry_sha256")
            and commit.get("batch_sha256") == batch_sha256
            and intent.get("batch_sha256") == batch_sha256
            and commit.get("source_ref") == source_ref
            and intent.get("session_sha256") == canonical_json_sha256(source_ref)
            and commit.get("raw_message_tokens") == raw_message_tokens,
            f"journal commit {ordinal} differs from its locked add batch",
        )


def _normalize_source_inputs(source: LockedSourceInputs) -> LockedSourceInputs:
    if type(source) is not LockedSourceInputs:
        raise TypeError("source must be an exact LockedSourceInputs")
    return LockedSourceInputs(
        benchmark_file=_plain_path(source.benchmark_file, "benchmark file"),
        split_manifest=_plain_path(source.split_manifest, "split manifest"),
        policy_manifest=_plain_path(source.policy_manifest, "source policy"),
        repository_root=_plain_path(
            source.repository_root, "frozen source repository", directory=True
        ),
        mem0_policy_manifest=_plain_path(
            source.mem0_policy_manifest, "Mem0 policy"
        ),
        mem0_environment_lock=_plain_path(
            source.mem0_environment_lock, "Mem0 environment lock"
        ),
        mem0_tool_root=_plain_path(
            source.mem0_tool_root, "frozen Mem0 tool root", directory=True
        ),
    )


def _source_projection(source: LockedSourceInputs, plan: SourceValidationPlan, policy: Mem0ComparisonPolicy) -> dict[str, Any]:
    return {
        "benchmark_file": _file_receipt(source.benchmark_file, "benchmark file"),
        "format": SOURCE_FORMAT,
        "mem0_environment_lock": _file_receipt(
            source.mem0_environment_lock, "Mem0 environment lock"
        ),
        "mem0_policy": _file_receipt(source.mem0_policy_manifest, "Mem0 policy"),
        "mem0_tool_implementation_sha256": policy.tool_implementation_sha256,
        "mem0_tool_root": str(source.mem0_tool_root),
        "policy_manifest": _file_receipt(source.policy_manifest, "source policy"),
        "repository_root": str(source.repository_root),
        "source_environment_lock_sha256": plan.environment_lock_sha256,
        "source_implementation_sha256": plan.implementation_sha256,
        "split_manifest": _file_receipt(source.split_manifest, "split manifest"),
    }


def _locked_context(source_value: LockedSourceInputs) -> _LockedContext:
    source = _normalize_source_inputs(source_value)
    plan = load_source_validation_plan(
        benchmark_file=source.benchmark_file,
        split_manifest=source.split_manifest,
        policy_manifest=source.policy_manifest,
        repository_root=source.repository_root,
    )
    frozen_source = {
        "dataset_sha256": FROZEN_DATASET_SHA256,
        "split_manifest_sha256": FROZEN_SPLIT_MANIFEST_SHA256,
        "policy_manifest_sha256": FROZEN_SOURCE_POLICY_SHA256,
        "implementation_sha256": FROZEN_SOURCE_IMPLEMENTATION_SHA256,
        "environment_lock_sha256": FROZEN_SOURCE_ENVIRONMENT_LOCK_SHA256,
    }
    _require(
        all(getattr(plan, field) == expected for field, expected in frozen_source.items())
        and canonical_json_sha256(plan.evaluation_identity)
        == FROZEN_SOURCE_EVALUATION_IDENTITY_SHA256,
        "source inputs are not the exact validation-v3 freeze",
    )
    _require(plan.sample_offsets == FROZEN_OFFSETS, "source offsets are not 0..90")
    _require(
        plan.target_tokens == FROZEN_TARGET_TOKENS,
        "source workload is not the locked 1M-token workload",
    )
    _require(
        plan.questions_per_shard == FROZEN_QUESTIONS_PER_SHARD,
        "source workload is not ten questions per shard",
    )
    shards = build_raw_stress_shards(
        benchmark_file=source.benchmark_file,
        split_manifest=source.split_manifest,
        sample_offsets=plan.sample_offsets,
        target_tokens=plan.target_tokens,
        max_questions=plan.questions_per_shard,
    )
    _require(
        len(shards) == len(FROZEN_OFFSETS)
        and tuple(row.sample_offset for row in shards) == FROZEN_OFFSETS,
        "locked shard population changed",
    )
    _require(
        all(len(row.question_ids) == FROZEN_QUESTIONS_PER_SHARD for row in shards),
        "a locked shard does not contain ten questions",
    )
    all_question_ids = [question_id for shard in shards for question_id in shard.question_ids]
    _require(
        len(all_question_ids) == FROZEN_QUESTION_COUNT
        and len(set(all_question_ids)) == FROZEN_QUESTION_COUNT,
        "locked question population is not exactly 100 distinct questions",
    )
    totals = (
        sum(row.add_counts.raw_pairs for row in shards),
        sum(row.add_counts.skipped_empty_pairs for row in shards),
        sum(row.add_counts.add_requests for row in shards),
    )
    _require(
        totals
        == (
            FROZEN_RAW_PAIRS,
            FROZEN_SKIPPED_EMPTY_PAIRS,
            FROZEN_ADD_OPERATIONS,
        ),
        "locked raw/add totals changed",
    )
    policy = load_mem0_comparison_policy(
        source.mem0_policy_manifest,
        source_plan=plan,
        mem0_environment_lock=source.mem0_environment_lock,
        expected_shards=shards,
        tool_root=source.mem0_tool_root,
    )
    shard_rows = []
    for shard in shards:
        receipt = shard_receipt(shard)
        shard_rows.append(
            {
                "add_batches_sha256": receipt["add_batches_sha256"],
                "add_operations": shard.add_counts.add_requests,
                "history_sample_ids_sha256": canonical_json_sha256(
                    list(shard.history_sample_ids)
                ),
                "question_ids": list(shard.question_ids),
                "question_ids_sha256": canonical_json_sha256(
                    list(shard.question_ids)
                ),
                "raw_history_bundle_sha256": shard.raw_history_bundle_sha256,
                "raw_pairs": shard.add_counts.raw_pairs,
                "sample_id": shard.parsed_sample.sample_id,
                "sample_offset": shard.sample_offset,
                "sample_sha256": shard.sample_sha256,
                "skipped_empty_pairs": shard.add_counts.skipped_empty_pairs,
                "transcript_tokens": receipt["transcript_tokens"],
            }
        )
    _require(
        identity_sha256(shard_rows) == FROZEN_SHARD_POPULATION_SHA256
        and identity_sha256(all_question_ids) == FROZEN_QUESTION_ORDER_SHA256
        and sum(row["transcript_tokens"] for row in shard_rows)
        == FROZEN_TRANSCRIPT_TOKENS,
        "sample, question, transcript, or per-shard add identity changed",
    )
    population = {
        "format": f"{FORMAT}-population-v1",
        "mem0_environment_lock_sha256": policy.environment_lock_sha256,
        "mem0_policy_sha256": policy.sha256,
        "mem0_tool_implementation_sha256": policy.tool_implementation_sha256,
        "question_count": FROZEN_QUESTION_COUNT,
        "question_order_sha256": identity_sha256(all_question_ids),
        "questions_per_shard": FROZEN_QUESTIONS_PER_SHARD,
        "sample_offsets": list(FROZEN_OFFSETS),
        "shards": shard_rows,
        "source_environment_lock_sha256": plan.environment_lock_sha256,
        "source_implementation_sha256": plan.implementation_sha256,
        "source_policy_sha256": plan.policy_manifest_sha256,
        "split_manifest_sha256": plan.split_manifest_sha256,
        "target_tokens": FROZEN_TARGET_TOKENS,
        "totals": {
            "add_operations": totals[2],
            "raw_pairs": totals[0],
            "skipped_empty_pairs": totals[1],
        },
    }
    return _LockedContext(
        source=source,
        source_plan=plan,
        policy=policy,
        shards=tuple(shards),
        population_identity_sha256=identity_sha256(population),
        population_projection=population,
        source_projection=_source_projection(source, plan, policy),
    )


def _terminal_input(value: ResumableTerminalInput) -> ResumableTerminalInput:
    if type(value) is not ResumableTerminalInput:
        raise TypeError("terminal inputs must be exact ResumableTerminalInput rows")
    return ResumableTerminalInput(
        artifact_path=_plain_path(value.artifact_path, "official terminal artifact"),
        trace_path=_plain_path(value.trace_path, "official terminal trace"),
        journal_path=_plain_path(value.journal_path, "official resume journal"),
    )


def _artifact_offset(path: Path) -> int:
    try:
        value = json.loads(path.read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Mem0TypedSourceBridgeError("official terminal artifact is not JSON") from exc
    if type(value) is not dict or type(value.get("sample_offset")) is not int:
        raise Mem0TypedSourceBridgeError("official terminal artifact has no exact offset")
    return value["sample_offset"]


def _journal_binding(
    *,
    terminal: ResumableTerminalInput,
    artifact: Mapping[str, Any],
    shard: RawStressShard,
    policy: Mem0ComparisonPolicy,
    artifact_sha256: str,
    trace_sha256: str,
) -> tuple[ReplayState, dict[str, tuple[PromptRequestWindowRef, ...]], dict[str, Any]]:
    retrieval_authorization = policy.retrieval_authorization(shard)
    authorization_sha256 = canonical_json_sha256(asdict(retrieval_authorization))
    plan = build_resume_plan(
        shard=shard,
        authorization=retrieval_authorization,
        authorization_sha256=authorization_sha256,
    )
    journal = AppendOnlyResumeJournal(terminal.journal_path, plan)
    state = journal.replay()
    _require(
        state.terminal_search is not None
        and state.active_state_removed is not None
        and state.terminal_published is not None
        and state.checkpoint_gc is not None
        and state.cleanup_closed is not None
        and state.entries[-1]["kind"] == "cleanup_closed",
        "resume journal is not terminally published and cleaned",
    )
    header = state.entries[0]
    _require(
        header.get("journal_path_sha256")
        == _path_identity_sha256(terminal.journal_path),
        "resume journal escaped its original path identity",
    )
    published = state.terminal_published
    assert published is not None
    artifact_target = Path(
        os.path.abspath(
            terminal.journal_path.parent / Path(published["official_artifact_path"])
        )
    )
    trace_target = Path(
        os.path.abspath(
            terminal.journal_path.parent / Path(published["official_trace_path"])
        )
    )
    _require(
        artifact_target == Path(os.path.abspath(terminal.artifact_path))
        and trace_target == Path(os.path.abspath(terminal.trace_path)),
        "journal publication paths differ from the supplied official outputs",
    )
    _require(
        published["official_artifact_sha256"] == artifact_sha256
        and published["official_trace_sha256"] == trace_sha256,
        "journal publication digests differ from the official outputs",
    )
    closure = artifact.get("resumable_closure")
    terminal_binding = artifact.get("resumable_terminal")
    _require(
        type(closure) is dict and type(terminal_binding) is dict,
        "official artifact omitted resumable bindings",
    )
    seal = state.latest_prefix_seal
    terminal_search = state.terminal_search
    _require(
        seal is not None
        and terminal_search is not None
        and dict(closure["full_prefix_seal"]) == dict(seal),
        "artifact full-prefix seal differs from the journal",
    )
    seal_sequence = int(seal["sequence"])
    prefix_entries = state.entries[: seal_sequence + 1]
    _require(
        closure["journal_tail_entry_sha256"] == seal["entry_sha256"]
        and closure["journal_chain_sha256"]
        == canonical_json_sha256(
            [row["entry_sha256"] for row in prefix_entries]
        )
        and closure["commit_population_sha256"]
        == canonical_json_sha256([dict(row) for row in state.commits]),
        "artifact journal-chain projection differs from the official journal",
    )
    _require(
        terminal_binding["terminal_stage_file_sha256"]
        == terminal_search["terminal_stage_sha256"]
        and terminal_binding["stage_result_sha256"]
        == terminal_search["terminal_result_sha256"]
        and terminal_binding["stage_trace_sha256"]
        == terminal_search["terminal_trace_sha256"]
        and terminal_binding["checkpoint_authority_sha256"]
        == state.checkpoint_authority_sha256,
        "artifact terminal binding differs from the journal",
    )
    material = rehydration_material(state)
    ledger: dict[str, tuple[PromptRequestWindowRef, ...]] = {}
    for row in material["ledger_projection"]:
        memory_id = row["memory_id"]
        windows = tuple(
            PromptRequestWindowRef(
                sample_id=ref["sample_id"],
                source=ref["source"],
                session=ref["session"],
                session_index=ref["session_index"],
                original_session_index=ref["original_session_index"],
                batch_index=ref["batch_index"],
                date=ref["date"],
                turn_start=ref["turn_start"],
                turn_count=ref["turn_count"],
                roles=tuple(ref["roles"]),
            )
            for ref in row["source_refs"]
        )
        _require(memory_id not in ledger and bool(windows), "journal ledger changed")
        ledger[memory_id] = windows
    journal_receipt = _file_receipt(terminal.journal_path, "resume journal")
    journal_receipt.update(
        {
            "checkpoint_authority_sha256": state.checkpoint_authority_sha256,
            "cleanup_entry_sha256": state.entries[-1]["entry_sha256"],
            "commit_population_sha256": canonical_json_sha256(
                [dict(row) for row in state.commits]
            ),
            "journal_chain_sha256": canonical_json_sha256(
                [row["entry_sha256"] for row in state.entries]
            ),
            "plan_sha256": state.plan.sha256,
        }
    )
    return state, ledger, journal_receipt


def _typed_rows(
    *,
    artifact: Mapping[str, Any],
    shard: RawStressShard,
    ledger: Mapping[str, tuple[PromptRequestWindowRef, ...]],
) -> tuple[list[dict[str, Any]], int]:
    source_rows = artifact.get("retrieval_rows")
    _require(
        type(source_rows) is list and len(source_rows) == FROZEN_QUESTIONS_PER_SHARD,
        "official terminal retrieval-row population changed",
    )
    runtime_identity = artifact["identity"]["runtime_identity"]
    evaluation_identity = artifact["identity"]["source_evaluation_identity"]
    rows: list[dict[str, Any]] = []
    window_count = 0
    for source_row, question in zip(
        source_rows, shard.parsed_sample.questions, strict=True
    ):
        raw_pool = source_row.get("raw_pool")
        _require(type(raw_pool) is list, "official raw pool changed type")
        candidates: list[SimpleNamespace] = []
        for rank, raw_candidate in enumerate(raw_pool, start=1):
            _require(type(raw_candidate) is dict, "official candidate changed type")
            memory_id = raw_candidate.get("memory_id")
            windows = ledger.get(memory_id)
            _require(
                windows is not None and bool(windows),
                "retrieved memory is absent from the authenticated journal ledger",
            )
            window_count += len(windows)
            candidates.append(
                SimpleNamespace(
                    **dict(raw_candidate),
                    request_window_attribution=windows,
                )
            )
        result = SimpleNamespace(
            query=question.dated_question,
            raw_pool=tuple(candidates),
            official_longmemeval_protocol=True,
            official_search_protocol=True,
            rendering_mode=artifact["protocol"]["rendering_mode"],
            certified_rendering=True,
            comparison_certified=True,
            runtime_identity=runtime_identity,
            attribution_kind=MEM0_ATTRIBUTION_KIND,
            supports_exact_source_provenance=False,
        )
        pack = pack_mem0_typed_prompt(
            question.dated_question,
            result,
            evaluation_identity=evaluation_identity,
            max_prompt_tokens=8_000,
        )
        row = pack.to_retrieval_row(
            question_id=question.question_id,
            search_latency_s=float(source_row["search_latency_s"]),
        )
        _require(
            row.get("format") == MEM0_TYPED_RETRIEVAL_ROW_FORMAT,
            "typed bridge did not produce a V3 row",
        )
        rows.append(row)
    return rows, window_count


def _write_observation(artifact: Mapping[str, Any]) -> dict[str, Any]:
    """Project the terminal's authenticated complete write attestation."""

    from .typed_epoch_campaign import _validate_write_observation

    attestation = artifact.get("write_usage_attestation")
    closure = artifact.get("resumable_closure")
    execution = artifact.get("execution_binding")
    if (
        type(attestation) is not dict
        or type(closure) is not dict
        or type(execution) is not dict
    ):
        raise Mem0TypedSourceBridgeError(
            "official terminal omitted complete write usage"
        )
    body = dict(attestation)
    receipt_sha = body.pop("receipt_sha256", None)
    if (
        receipt_sha != canonical_json_sha256(body)
        or closure.get("write_usage_attestation") != attestation
        or closure.get("write_usage_attestation_sha256") != receipt_sha
        or execution.get("write_usage_attestation_sha256") != receipt_sha
    ):
        raise Mem0TypedSourceBridgeError(
            "official terminal write-usage chain changed"
        )
    observed = attestation.get("observed")
    if type(observed) is not dict or attestation.get(
        "observed_sha256"
    ) != canonical_json_sha256(observed):
        raise Mem0TypedSourceBridgeError(
            "official terminal observed write usage changed"
        )
    return _validate_write_observation(observed)


def _retrieval_cleanup(trace: Mapping[str, Any]) -> dict[str, Any]:
    cleanup = trace.get("cleanup")
    _require(type(cleanup) is dict, "official trace omitted cleanup")
    return {
        "active_scope_cleared": cleanup["active_scope_cleared"],
        "adapter_closed": cleanup["adapter_closed"],
        "external_provider_persistence_certified": cleanup[
            "external_provider_persistence_certified"
        ],
        "extraction_meter_restored_before_cleanup": cleanup[
            "extraction_meter_restored_before_cleanup"
        ],
        "ledger_empty": cleanup["ledger_empty"],
        "owned_state_path_absent": cleanup["owned_state_path_absent"],
        "persisted_request_token_state": cleanup[
            "persisted_request_token_state"
        ],
        "retained_request_token_state_bytes": cleanup[
            "retained_request_token_state_bytes"
        ],
        "state_absent_after": cleanup["state_absent_after"],
    }


def _load_json_file(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Mem0TypedSourceBridgeError(f"{label} is not JSON") from exc
    if type(value) is not dict:
        raise Mem0TypedSourceBridgeError(f"{label} must be an object")
    return value


def _verify_terminal_shard(
    *,
    terminal: ResumableTerminalInput,
    shard: RawStressShard,
    context: _LockedContext,
) -> tuple[dict[str, Any], dict[str, Any]]:
    artifact_receipt = _file_receipt(terminal.artifact_path, "official artifact")
    trace_receipt = _file_receipt(terminal.trace_path, "official trace")
    authorization = context.policy.scoring_authorization(
        shard,
        retrieval_artifact_sha256=artifact_receipt["sha256"],
    )
    verified_artifact, artifact_bytes, _verified_rows = _verify_retrieval_artifact(
        artifact_path=terminal.artifact_path,
        trace_path=terminal.trace_path,
        shard=shard,
        authorization=authorization,
        prompt_packer=_default_prompt_packer,
    )
    _require(
        hashlib.sha256(artifact_bytes).hexdigest() == artifact_receipt["sha256"]
        and verified_artifact.get("certification_status")
        == RESUMABLE_RETRIEVAL_CERTIFICATION
        and verified_artifact.get("comparison_certified") is True,
        "official terminal is not exact resumable production",
    )
    state, ledger, journal_receipt = _journal_binding(
        terminal=terminal,
        artifact=verified_artifact,
        shard=shard,
        policy=context.policy,
        artifact_sha256=artifact_receipt["sha256"],
        trace_sha256=trace_receipt["sha256"],
    )
    typed_rows, window_count = _typed_rows(
        artifact=verified_artifact,
        shard=shard,
        ledger=ledger,
    )
    trace = _load_json_file(terminal.trace_path, "official trace")
    # Import lazily to keep the bridge/campaign modules acyclic.
    from .typed_epoch_campaign import build_retrieval_export_payload

    export = build_retrieval_export_payload(
        population_identity_sha256=context.population_identity_sha256,
        source_shard_sha256=artifact_receipt["sha256"],
        retrieval_trace_sha256=trace_receipt["sha256"],
        question_offset=shard.sample_offset,
        retrieval_rows=typed_rows,
        write_observation=_write_observation(verified_artifact),
        retrieval_cleanup=_retrieval_cleanup(trace),
    )
    row_body = {
        "artifact": artifact_receipt,
        "diagnostic_request_window_count": window_count,
        "export_payload_sha256": hashlib.sha256(canonical_json_bytes(export)).hexdigest(),
        "format": ROW_FORMAT,
        "journal": journal_receipt,
        "population_identity_sha256": context.population_identity_sha256,
        "provenance": {
            "attribution_kind": MEM0_ATTRIBUTION_KIND,
            "request_windows_are_fact_evidence": False,
            "supports_exact_source_provenance": False,
        },
        "question_count": len(typed_rows),
        "question_ids": list(shard.question_ids),
        "question_offset": shard.sample_offset,
        "raw_history_bundle_sha256": shard.raw_history_bundle_sha256,
        "sample_id": shard.parsed_sample.sample_id,
        "sample_offset": shard.sample_offset,
        "sample_sha256": shard.sample_sha256,
        "source_add_operations": shard.add_counts.add_requests,
        "source_raw_pairs": shard.add_counts.raw_pairs,
        "source_skipped_empty_pairs": shard.add_counts.skipped_empty_pairs,
        "trace": trace_receipt,
        "typed_retrieval_rows_sha256": identity_sha256(typed_rows),
        "zero_persisted_transformer_token_state": True,
    }
    return export, {**row_body, "row_receipt_sha256": identity_sha256(row_body)}


def _manifest_value(value: object) -> dict[str, Any]:
    _require(type(value) is dict, "source bridge manifest must be an exact object")
    manifest = _strict_json(value, "source bridge manifest")
    expected_keys = {
        "export_count",
        "format",
        "gold_loaded",
        "physical_provider_calls",
        "population_identity_sha256",
        "population_projection",
        "provenance",
        "question_count",
        "question_order_sha256",
        "retained_transformer_token_state_bytes",
        "sample_offsets",
        "shards",
        "source",
    }
    _require(set(manifest) == expected_keys, "source bridge manifest fields changed")
    _require(
        manifest["format"] == FORMAT
        and manifest["gold_loaded"] is False
        and manifest["physical_provider_calls"] == 0
        and manifest["retained_transformer_token_state_bytes"] == 0
        and manifest["export_count"] == len(FROZEN_OFFSETS)
        and manifest["question_count"] == FROZEN_QUESTION_COUNT
        and manifest["sample_offsets"] == list(FROZEN_OFFSETS),
        "source bridge fixed population contract changed",
    )
    require_sha256(manifest["population_identity_sha256"], "bridge population")
    require_sha256(manifest["question_order_sha256"], "bridge question order")
    _require(
        type(manifest["population_projection"]) is dict
        and type(manifest["source"]) is dict,
        "source bridge projections changed type",
    )
    _require(
        type(manifest["shards"]) is list
        and len(manifest["shards"]) == len(FROZEN_OFFSETS),
        "source bridge must contain exactly ten shard rows",
    )
    _require(
        all(type(row) is dict for row in manifest["shards"]),
        "source bridge shard rows changed type",
    )
    _require(
        [row.get("sample_offset") for row in manifest["shards"]]
        == list(FROZEN_OFFSETS)
        and [row.get("question_offset") for row in manifest["shards"]]
        == list(FROZEN_OFFSETS)
        and all(row.get("question_count") == 10 for row in manifest["shards"]),
        "source bridge shard offsets/counts changed",
    )
    question_ids: list[str] = []
    for index, row in enumerate(manifest["shards"]):
        expected_row_keys = {
            "artifact",
            "diagnostic_request_window_count",
            "export",
            "export_payload_sha256",
            "format",
            "journal",
            "population_identity_sha256",
            "provenance",
            "question_count",
            "question_ids",
            "question_offset",
            "raw_history_bundle_sha256",
            "row_receipt_sha256",
            "sample_id",
            "sample_offset",
            "sample_sha256",
            "source_add_operations",
            "source_raw_pairs",
            "source_skipped_empty_pairs",
            "trace",
            "typed_retrieval_rows_sha256",
            "zero_persisted_transformer_token_state",
        }
        _require(set(row) == expected_row_keys, f"bridge shard {index} fields changed")
        for field in ("artifact", "trace"):
            receipt = row[field]
            _require(
                type(receipt) is dict
                and set(receipt) == {"bytes", "path", "sha256"}
                and type(receipt["bytes"]) is int
                and receipt["bytes"] > 0
                and type(receipt["path"]) is str
                and bool(receipt["path"]),
                f"bridge shard {index} {field} receipt changed",
            )
            require_sha256(receipt["sha256"], f"bridge shard {index} {field}")
        journal = row["journal"]
        _require(
            type(journal) is dict
            and set(journal)
            == {
                "bytes",
                "checkpoint_authority_sha256",
                "cleanup_entry_sha256",
                "commit_population_sha256",
                "journal_chain_sha256",
                "path",
                "plan_sha256",
                "sha256",
            }
            and type(journal["bytes"]) is int
            and journal["bytes"] > 0
            and type(journal["path"]) is str
            and bool(journal["path"]),
            f"bridge shard {index} journal receipt changed",
        )
        for field in (
            "checkpoint_authority_sha256",
            "cleanup_entry_sha256",
            "commit_population_sha256",
            "journal_chain_sha256",
            "plan_sha256",
            "sha256",
        ):
            require_sha256(journal[field], f"bridge shard {index} journal {field}")
        export = row["export"]
        _require(
            type(export) is dict
            and set(export) == {"filename", "sha256"}
            and export["filename"]
            == EXPORT_NAME_TEMPLATE.format(offset=FROZEN_OFFSETS[index]),
            f"bridge shard {index} export locator changed",
        )
        require_sha256(export["sha256"], f"bridge shard {index} export")
        for field in (
            "export_payload_sha256",
            "population_identity_sha256",
            "raw_history_bundle_sha256",
            "row_receipt_sha256",
            "sample_sha256",
            "typed_retrieval_rows_sha256",
        ):
            require_sha256(row[field], f"bridge shard {index} {field}")
        ids = row["question_ids"]
        _require(
            type(ids) is list
            and len(ids) == FROZEN_QUESTIONS_PER_SHARD
            and len(set(ids)) == len(ids)
            and all(type(question_id) is str and bool(question_id) for question_id in ids)
            and row["format"] == ROW_FORMAT
            and row["population_identity_sha256"]
            == manifest["population_identity_sha256"]
            and row["export_payload_sha256"] == export["sha256"]
            and row["zero_persisted_transformer_token_state"] is True
            and row["provenance"] == manifest["provenance"],
            f"bridge shard {index} identity/provenance changed",
        )
        for field in (
            "diagnostic_request_window_count",
            "source_add_operations",
            "source_raw_pairs",
            "source_skipped_empty_pairs",
        ):
            _require(
                type(row[field]) is int and row[field] >= 0,
                f"bridge shard {index} {field} changed",
            )
        body = dict(row)
        receipt_sha = body.pop("row_receipt_sha256")
        _require(
            identity_sha256(body) == receipt_sha,
            f"bridge shard {index} row receipt changed",
        )
        question_ids.extend(ids)
    _require(
        len(question_ids) == FROZEN_QUESTION_COUNT
        and len(set(question_ids)) == FROZEN_QUESTION_COUNT
        and identity_sha256(question_ids) == manifest["question_order_sha256"],
        "source bridge question population changed",
    )
    _require(
        len(
            {
                row[k]["sha256"]
                for row in manifest["shards"]
                for k in ("artifact", "trace", "journal")
            }
        )
        == 30,
        "source bridge reused an official artifact, trace, or journal",
    )
    _require(
        manifest["provenance"]
        == {
            "attribution_kind": MEM0_ATTRIBUTION_KIND,
            "request_windows_are_fact_evidence": False,
            "supports_exact_source_provenance": False,
        },
        "source bridge provenance was overstated",
    )
    assert_gold_blind(manifest, path="mem0_typed_source_bridge")
    return manifest


def build_source_bridge(
    *,
    source: LockedSourceInputs,
    terminals: Sequence[ResumableTerminalInput],
    output_root: str | Path,
    dry_run: bool = False,
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    """Authenticate ten official terminals and publish their typed exports.

    All digests are derived from files opened here.  The caller supplies paths,
    never authority SHA strings.
    """

    if not isinstance(terminals, Sequence) or isinstance(terminals, (str, bytes)):
        raise TypeError("terminals must be a sequence")
    _require(len(terminals) == len(FROZEN_OFFSETS), "exactly ten terminals are required")
    normalized = tuple(_terminal_input(row) for row in terminals)
    all_paths = [
        path
        for row in normalized
        for path in (row.artifact_path, row.trace_path, row.journal_path)
    ]
    _require(len(set(all_paths)) == 30, "terminal paths must be pairwise distinct")
    by_offset: dict[int, ResumableTerminalInput] = {}
    for row in normalized:
        offset = _artifact_offset(row.artifact_path)
        _require(offset not in by_offset, "terminal population repeats an offset")
        by_offset[offset] = row
    _require(set(by_offset) == set(FROZEN_OFFSETS), "terminal offsets are not exactly 0..90")
    context = _locked_context(source)
    exports: list[dict[str, Any]] = []
    bridge_rows: list[dict[str, Any]] = []
    root = Path(output_root).resolve()
    for shard in context.shards:
        export, bridge_row = _verify_terminal_shard(
            terminal=by_offset[shard.sample_offset],
            shard=shard,
            context=context,
        )
        filename = EXPORT_NAME_TEMPLATE.format(offset=shard.sample_offset)
        export_sha = hashlib.sha256(canonical_json_bytes(export)).hexdigest()
        bridge_body = dict(bridge_row)
        bridge_body["export"] = {
            "filename": filename,
            "sha256": export_sha,
        }
        bridge_body.pop("row_receipt_sha256")
        bridge_row = {
            **bridge_body,
            "row_receipt_sha256": identity_sha256(bridge_body),
        }
        exports.append(export)
        bridge_rows.append(bridge_row)
        if not dry_run:
            artifact, _ = publish_sealed_json(root / filename, export)
            _require(artifact.sha256 == export_sha, "typed export publication changed")
    question_ids = [question_id for row in bridge_rows for question_id in row["question_ids"]]
    manifest = {
        "export_count": len(exports),
        "format": FORMAT,
        "gold_loaded": False,
        "physical_provider_calls": 0,
        "population_identity_sha256": context.population_identity_sha256,
        "population_projection": dict(context.population_projection),
        "provenance": {
            "attribution_kind": MEM0_ATTRIBUTION_KIND,
            "request_windows_are_fact_evidence": False,
            "supports_exact_source_provenance": False,
        },
        "question_count": len(question_ids),
        "question_order_sha256": identity_sha256(question_ids),
        "retained_transformer_token_state_bytes": 0,
        "sample_offsets": list(FROZEN_OFFSETS),
        "shards": bridge_rows,
        "source": dict(context.source_projection),
    }
    _manifest_value(manifest)
    if not dry_run:
        publish_sealed_json(root / MANIFEST_NAME, manifest)
        # Re-open every external source after publication to close replacement
        # races at the bridge boundary.
        reopened = _locked_context(source)
        _require(
            reopened.population_identity_sha256 == context.population_identity_sha256
            and reopened.source_projection == context.source_projection,
            "locked source changed during source-bridge publication",
        )
    return manifest, tuple(exports)


def _source_from_projection(value: Mapping[str, Any]) -> LockedSourceInputs:
    def path(field: str, *, nested: str | None = None) -> Path:
        raw = value.get(field)
        if nested is not None:
            _require(type(raw) is dict, f"bridge source {field} changed")
            raw = raw.get(nested)
        _require(type(raw) is str and bool(raw), f"bridge source {field} path changed")
        return Path(raw)

    return LockedSourceInputs(
        benchmark_file=path("benchmark_file", nested="path"),
        split_manifest=path("split_manifest", nested="path"),
        policy_manifest=path("policy_manifest", nested="path"),
        repository_root=path("repository_root"),
        mem0_policy_manifest=path("mem0_policy", nested="path"),
        mem0_environment_lock=path("mem0_environment_lock", nested="path"),
        mem0_tool_root=path("mem0_tool_root"),
    )


def reopen_source_bridge(
    manifest_path: str | Path,
    *,
    expected_manifest_sha256: str | None = None,
) -> SourceBridgeVerification:
    """Reopen sources, terminals, journals, and exports from one manifest."""

    manifest_artifact = read_sealed_json(manifest_path)
    if expected_manifest_sha256 is not None:
        _require(
            manifest_artifact.sha256
            == require_sha256(expected_manifest_sha256, "expected source bridge"),
            "source bridge manifest digest changed",
        )
    manifest = _manifest_value(manifest_artifact.payload)
    context = _locked_context(_source_from_projection(manifest["source"]))
    _require(
        context.population_identity_sha256 == manifest["population_identity_sha256"]
        and context.population_projection == manifest["population_projection"]
        and context.source_projection == manifest["source"],
        "source bridge no longer reconstructs its locked source",
    )
    exports: list[SealedArtifact] = []
    question_ids: list[str] = []
    root = manifest_artifact.path.parent
    for bridge_row, shard in zip(manifest["shards"], context.shards, strict=True):
        row_body = dict(bridge_row)
        row_receipt = row_body.pop("row_receipt_sha256", None)
        _require(
            identity_sha256(row_body) == row_receipt,
            "source bridge shard receipt changed",
        )
        terminal = ResumableTerminalInput(
            artifact_path=Path(bridge_row["artifact"]["path"]),
            trace_path=Path(bridge_row["trace"]["path"]),
            journal_path=Path(bridge_row["journal"]["path"]),
        )
        normalized_terminal = _terminal_input(terminal)
        for field, path_value in (
            ("artifact", normalized_terminal.artifact_path),
            ("trace", normalized_terminal.trace_path),
            ("journal", normalized_terminal.journal_path),
        ):
            observed = _file_receipt(path_value, f"reopened {field}")
            expected = {
                key: bridge_row[field][key]
                for key in ("bytes", "path", "sha256")
            }
            _same_file_receipt(observed, expected, f"reopened {field}")
        rebuilt_export, rebuilt_row = _verify_terminal_shard(
            terminal=normalized_terminal,
            shard=shard,
            context=context,
        )
        filename = bridge_row["export"]["filename"]
        _require(
            filename == EXPORT_NAME_TEMPLATE.format(offset=shard.sample_offset)
            and Path(filename).name == filename,
            "source bridge export filename changed",
        )
        export_artifact = read_sealed_json(root / filename)
        _require(
            export_artifact.sha256 == bridge_row["export"]["sha256"]
            and export_artifact.payload == rebuilt_export,
            "typed export is not the exact reopened bridge projection",
        )
        # Compare every semantic bridge field.  The export locator is added
        # only after terminal verification during initial publication.
        expected_row = dict(rebuilt_row)
        expected_row.pop("row_receipt_sha256")
        expected_row["export"] = dict(bridge_row["export"])
        expected_row["row_receipt_sha256"] = identity_sha256(expected_row)
        _require(expected_row == bridge_row, "source bridge shard projection changed")
        exports.append(export_artifact)
        question_ids.extend(bridge_row["question_ids"])
    _require(
        identity_sha256(question_ids) == manifest["question_order_sha256"],
        "source bridge question order changed",
    )
    return SourceBridgeVerification(
        manifest=manifest_artifact,
        exports=tuple(exports),
        population_identity_sha256=manifest["population_identity_sha256"],
        question_order_sha256=manifest["question_order_sha256"],
    )


__all__ = [
    "EXPORT_NAME_TEMPLATE",
    "FORMAT",
    "FROZEN_OFFSETS",
    "FROZEN_QUESTION_COUNT",
    "LockedSourceInputs",
    "MANIFEST_NAME",
    "Mem0TypedSourceBridgeError",
    "ResumableTerminalInput",
    "SourceBridgeVerification",
    "build_source_bridge",
    "reopen_source_bridge",
]
