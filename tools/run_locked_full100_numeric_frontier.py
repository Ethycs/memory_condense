#!/usr/bin/env python3
"""Materialize replayable numeric-policy frontiers for locked full100.

The lifecycle is provider-free and gold-blind.  It authenticates the sealed
full100 answer lineage, derives applicability from each terminal provider
input, and streams every required namespace exactly once.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

if __package__ in {None, ""}:
    _ROOT = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(_ROOT / "src"), str(_ROOT)]

from memory_condense.persistence.db import Database  # noqa: E402
from tools import run_locked_semantic_global_terminal_full100_answer as answer_cli  # noqa: E402
from tools import run_locked_semantic_global_terminal_full100_construction as full100_cli  # noqa: E402
from tools import run_reduced_second_read_retrieval_assay as resident_cli  # noqa: E402
from tools.matched_eval.artifacts import (  # noqa: E402
    SealedArtifact,
    publish_sealed_json,
    read_sealed_json,
)
from tools.matched_eval.contracts import (  # noqa: E402
    MatchedEvalContractError,
    assert_gold_blind,
    identity_sha256,
    require_sha256,
)
from tools.matched_eval.full_store_slot_closure import (  # noqa: E402
    FullStoreWindowIndex,
    build_full_store_window_index,
)
from tools.matched_eval.numeric_operand_specialist import (  # noqa: E402
    NumericOperandClosureResult,
    scan_numeric_operand_closure,
)
from tools.matched_eval.numeric_policy_frontier_bridge import (  # noqa: E402
    BRIDGE_FORMAT,
    EXTENDED_SUPPORTED_DOMAINS,
    POLICY_GRAMMAR_ID,
    SUPPORTED_DOMAINS,
    NumericPolicyFrontierBridgeResult,
    build_operator_first_numeric_frontier,
    operator_first_numeric_frontier_applicable,
)
from tools.matched_eval.operator_first_numeric_policy import (  # noqa: E402
    FRONTIER_FORMAT,
    RelevantNumericFrontier,
)
from tools.matched_eval.query_guided_scan import cache_namespace_partitions  # noqa: E402


FORMAT = "memory-condense-locked-full100-numeric-frontier-v2"
ROW_FORMAT = f"{FORMAT}-row-v2"
LIFECYCLE_FORMAT = f"{FORMAT}-namespace-lifecycle-v2"
MATERIALIZATION_NAME = "locked-full100-numeric-frontier-v2.json"
REPLAY_NAME = "locked-full100-numeric-frontier-replay-v2.json"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = REPOSITORY_ROOT / (
    "eval_results/matched_eval_100/locked-full100-numeric-frontier-v2"
)
OPERATOR_MATERIAL_PROFILE = "operator-material-v3"
STRICT_PROFILE = "strict-v2"
V3_FORMAT = "memory-condense-locked-full100-numeric-frontier-v3"
V3_ROW_FORMAT = f"{V3_FORMAT}-row-v3"
V3_LIFECYCLE_FORMAT = f"{V3_FORMAT}-namespace-lifecycle-v3"
V3_MATERIALIZATION_NAME = "locked-full100-numeric-frontier-v3.json"
V3_REPLAY_NAME = "locked-full100-numeric-frontier-replay-v3.json"
V3_DEFAULT_OUTPUT_ROOT = REPOSITORY_ROOT / (
    "eval_results/matched_eval_100/locked-full100-numeric-frontier-v3"
)


@dataclass(frozen=True, slots=True)
class NumericFrontierLifecycleProfile:
    profile_id: str
    format: str
    row_format: str
    lifecycle_format: str
    materialization_name: str
    replay_name: str
    default_output_root: Path
    applicability: str
    extended_domains: bool
    operator_material_status: bool


_PROFILES = {
    STRICT_PROFILE: NumericFrontierLifecycleProfile(
        profile_id=STRICT_PROFILE,
        format=FORMAT,
        row_format=ROW_FORMAT,
        lifecycle_format=LIFECYCLE_FORMAT,
        materialization_name=MATERIALIZATION_NAME,
        replay_name=REPLAY_NAME,
        default_output_root=DEFAULT_OUTPUT_ROOT,
        applicability="operator_first_supported_mode_domain_and_question_action_v2",
        extended_domains=False,
        operator_material_status=False,
    ),
    OPERATOR_MATERIAL_PROFILE: NumericFrontierLifecycleProfile(
        profile_id=OPERATOR_MATERIAL_PROFILE,
        format=V3_FORMAT,
        row_format=V3_ROW_FORMAT,
        lifecycle_format=V3_LIFECYCLE_FORMAT,
        materialization_name=V3_MATERIALIZATION_NAME,
        replay_name=V3_REPLAY_NAME,
        default_output_root=V3_DEFAULT_OUTPUT_ROOT,
        applicability=(
            "operator_first_extended_domain_and_operator_material_status_v3"
        ),
        extended_domains=True,
        operator_material_status=True,
    ),
}


def lifecycle_profile(value: str = STRICT_PROFILE) -> NumericFrontierLifecycleProfile:
    try:
        return _PROFILES[value]
    except KeyError as exc:
        raise LockedFull100NumericFrontierError(
            f"unknown numeric frontier policy profile: {value}"
        ) from exc


class LockedFull100NumericFrontierError(MatchedEvalContractError):
    """Raised when numeric-frontier lineage or replay changes."""


def _require(ok: object, message: str) -> None:
    if not ok:
        raise LockedFull100NumericFrontierError(message)


def _exact_dict(value: object, label: str) -> dict[str, Any]:
    _require(type(value) is dict, f"{label} must be an exact object")
    return value  # type: ignore[return-value]


def _unique_sha_list(value: object, label: str) -> tuple[str, ...]:
    _require(type(value) is list, f"{label} must be an exact array")
    rows = tuple(require_sha256(row, label) for row in value)
    _require(len(rows) == len(set(rows)), f"{label} must be ordered and unique")
    return rows


@dataclass(frozen=True, slots=True)
class VerifiedInputs:
    preflight: SealedArtifact
    answer_run: SealedArtifact
    answer_replay: SealedArtifact
    full100: Any
    provider_plans: tuple[dict[str, Any], ...]
    namespace_by_ordinal: Mapping[int, str]


def _applicable(
    provider_input: Mapping[str, Any],
    profile: NumericFrontierLifecycleProfile,
) -> bool:
    return operator_first_numeric_frontier_applicable(
        provider_input,
        supported_domains=(
            EXTENDED_SUPPORTED_DOMAINS
            if profile.extended_domains
            else SUPPORTED_DOMAINS
        ),
    )


def _load_verified_inputs(args: argparse.Namespace) -> VerifiedInputs:
    preflight, _prompts, prompt_rows, _passthroughs = answer_cli._read_preflight(  # noqa: SLF001
        args.answer_root, args.expected_answer_preflight_sha256
    )
    answer_run, answer_replay, _answer_rows = answer_cli.load_verified_answer_run(
        args.answer_root,
        expected_preflight_sha256=args.expected_answer_preflight_sha256,
        expected_run_sha256=args.expected_answer_run_sha256,
        expected_replay_sha256=args.expected_answer_replay_sha256,
        postseal_audit=args.postseal_audit,
        expected_postseal_audit_sha256=args.expected_postseal_audit_sha256,
    )
    construction_sha = require_sha256(
        preflight.payload.get("full100_construction_artifact_sha256"),
        "answer-bound full100 construction",
    )
    replay_sha = require_sha256(
        preflight.payload.get("full100_replay_artifact_sha256"),
        "answer-bound full100 replay",
    )
    root = Path(args.full100_root)
    construction = read_sealed_json(root / full100_cli.CONSTRUCTION_NAME)
    replay = read_sealed_json(root / full100_cli.REPLAY_NAME)
    _require(
        construction.sha256 == construction_sha
        and replay.sha256 == replay_sha
        and construction.sha256 == replay.sha256
        and construction.payload == replay.payload,
        "answer-bound compact full100 construction/replay changed",
    )
    questions = construction.payload.get("questions")
    _require(type(questions) is list and len(questions) == 100, "full100 rows changed")
    prompt_by_ordinal = {int(row["ordinal"]): row for row in prompt_rows}
    plans_list: list[dict[str, Any]] = []
    for ordinal, prompt in sorted(prompt_by_ordinal.items()):
        source_row = full100_cli._validate_receipt(  # noqa: SLF001
            questions[ordinal],
            key="question_construction_receipt_sha256",
            label=f"numeric frontier full100 row {ordinal}",
        )
        compact = full100_cli._validate_receipt(  # noqa: SLF001
            source_row.get("terminal_answer_plan"),
            key="compact_plan_receipt_sha256",
            label=f"numeric frontier compact plan {ordinal}",
        )
        provider = _exact_dict(compact.get("provider_plan"), "compact provider plan")
        _require(
            source_row.get("ordinal") == ordinal
            and source_row.get("mode") == full100_cli.TERMINAL_MODE
            and compact.get("provider_plan_sha256") == identity_sha256(provider)
            and prompt.get("provider_input_sha256")
            == provider.get("provider_input_sha256"),
            f"answer/full100 compact plan binding changed at ordinal {ordinal}",
        )
        plans_list.append(
            answer_cli._validate_provider_plan(provider, source_row)  # noqa: SLF001
        )
    plans = tuple(plans_list)
    plan_by_ordinal = {int(row["ordinal"]): row for row in plans}
    _require(
        len(plan_by_ordinal) == len(plans)
        and {
            (int(row["ordinal"]), str(row["provider_input_sha256"]))
            for row in prompt_rows
        }
        == {
            (ordinal, str(plan["provider_input_sha256"]))
            for ordinal, plan in plan_by_ordinal.items()
        },
        "answer preflight and full100 provider-plan populations differ",
    )
    namespace_by_ordinal = {
        int(row["ordinal"]): str(row["namespace_id"])
        for raw in questions
        for row in (_exact_dict(raw, "full100 question"),)
        if int(row["ordinal"]) in plan_by_ordinal
    }
    _require(
        set(namespace_by_ordinal) == set(plan_by_ordinal),
        "terminal namespace population changed",
    )
    full100 = SimpleNamespace(construction=construction, replay=replay)
    return VerifiedInputs(
        preflight,
        answer_run,
        answer_replay,
        full100,
        plans,
        namespace_by_ordinal,
    )


IndexLoader = Callable[[str], tuple[FullStoreWindowIndex, Mapping[str, Any]]]


@dataclass(frozen=True, slots=True)
class _ResidentNamespaceLocation:
    namespace: Any
    shard_offset: int
    store_dir: Path
    database_sha256: str
    index_sha256: str


@dataclass(frozen=True, slots=True)
class _ResidentPopulationManifest:
    locations_by_namespace: Mapping[
        str, tuple[_ResidentNamespaceLocation, ...]
    ]


def _prepare_resident_population(
    args: argparse.Namespace,
) -> _ResidentPopulationManifest:
    """Authenticate the shared population once before streaming its stores.

    ``resident_cli._scoped_guided_context`` intentionally reconstructs and
    authenticates the complete query population on every invocation.  That is
    appropriate for isolated workers but needlessly reparses the same sealed
    million-token retrieval when this lifecycle streams several namespaces in
    one process.  This is the same validation factored into a loader-local
    manifest; store bytes remain authenticated lazily for each requested
    namespace.
    """

    population, preflight = (
        resident_cli.load_preflighted_query_expansion_population(
            Path(args.retrieval),
            output_root=Path(args.query_parent_output_root),
            expected_retrieval_sha256=args.expected_retrieval_sha256,
            expected_question_count=100,
        )
    )
    _require(
        preflight.sha256
        == require_sha256(
            args.expected_query_parent_preflight_sha256,
            "expected query parent preflight",
        ),
        "parent query preflight changed",
    )
    retrieval = read_sealed_json(Path(args.retrieval))
    _require(
        retrieval.sha256 == population.source_population.retrieval_sha256,
        "locked retrieval changed",
    )
    raw_shards = resident_cli._exact_list(  # noqa: SLF001
        retrieval.payload.get("shards"), "locked retrieval shards"
    )
    raw_questions = resident_cli._exact_list(  # noqa: SLF001
        retrieval.payload.get("questions"), "locked retrieval questions"
    )
    _require(
        len(raw_questions) == len(population.rows)
        and all(type(row) is dict for row in (*raw_shards, *raw_questions)),
        "locked retrieval shard/question population changed",
    )
    namespace_by_receipt = {
        row.combined_store_receipt_sha256: row
        for row in population.namespaces
    }
    _require(
        len(namespace_by_receipt) == len(population.namespaces),
        "namespace store receipts must be unique",
    )
    namespace_by_offset: dict[int, Any] = {}
    mutable_locations: dict[str, list[_ResidentNamespaceLocation]] = (
        defaultdict(list)
    )
    for value in raw_shards:
        raw = resident_cli._exact_dict(  # noqa: SLF001
            value, "locked retrieval shard"
        )
        offset = raw.get("shard_offset")
        receipt_sha = raw.get("combined_store_receipt_sha256")
        receipt = resident_cli._exact_dict(  # noqa: SLF001
            raw.get("combined_store_receipt"), "combined store receipt"
        )
        _require(
            type(offset) is int
            and offset >= 0
            and offset % 10 == 0
            and receipt.get("receipt_sha256") == receipt_sha
            and receipt_sha in namespace_by_receipt,
            "frozen shard/store receipt changed",
        )
        namespace = namespace_by_receipt[str(receipt_sha)]
        _require(offset not in namespace_by_offset, "shard offset repeated")
        namespace_by_offset[offset] = namespace
        mutable_locations[namespace.namespace_id].append(
            _ResidentNamespaceLocation(
                namespace=namespace,
                shard_offset=offset,
                store_dir=(
                    Path(args.store_root)
                    / "shards"
                    / f"offset-{offset:03d}"
                    / "combined-store"
                ),
                database_sha256=require_sha256(
                    receipt.get("target_database_sha256"),
                    "frozen database SHA-256",
                ),
                index_sha256=require_sha256(
                    receipt.get("target_index_sha256"),
                    "frozen index SHA-256",
                ),
            )
        )
    prompt_rows_by_question: dict[str, Any] = {}
    for prompt, value in zip(population.rows, raw_questions, strict=True):
        raw = resident_cli._exact_dict(  # noqa: SLF001
            value, "locked retrieval question"
        )
        offset = raw.get("shard_offset")
        question_id = resident_cli.require_text(
            prompt.source.packet.question_id,
            "query population question ID",
        )
        _require(
            type(offset) is int
            and offset in namespace_by_offset
            and raw.get("question_id") == question_id
            and namespace_by_offset[offset].namespace_id
            == prompt.namespace.namespace_id
            and question_id not in prompt_rows_by_question,
            "question changed its frozen store binding",
        )
        prompt_rows_by_question[question_id] = prompt
    return _ResidentPopulationManifest(
        {
            namespace_id: tuple(locations)
            for namespace_id, locations in mutable_locations.items()
        }
    )


def build_materialization_payload(
    inputs: VerifiedInputs,
    /,
    *,
    index_loader: IndexLoader,
    specialist_scanner: Callable[..., NumericOperandClosureResult] = (
        scan_numeric_operand_closure
    ),
    frontier_builder: Callable[..., NumericPolicyFrontierBridgeResult] = (
        build_operator_first_numeric_frontier
    ),
    policy_profile: str = STRICT_PROFILE,
) -> dict[str, Any]:
    """Build the compact artifact from already authenticated inputs."""

    profile = lifecycle_profile(policy_profile)
    applicable: list[tuple[int, dict[str, Any], str]] = []
    for plan in inputs.provider_plans:
        provider_input = _exact_dict(plan.get("provider_input"), "provider input")
        if _applicable(provider_input, profile):
            ordinal = int(plan["ordinal"])
            applicable.append(
                (ordinal, plan, inputs.namespace_by_ordinal[ordinal])
            )
    by_namespace: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for ordinal, plan, namespace_id in applicable:
        by_namespace[namespace_id].append((ordinal, plan))

    rows: list[dict[str, Any]] = []
    lifecycle: list[dict[str, Any]] = []
    for namespace_id in sorted(by_namespace):
        index, raw_lifecycle = index_loader(namespace_id)
        _require(
            type(index) is FullStoreWindowIndex
            and index.cache.namespace_id == namespace_id,
            "resident index escaped its requested namespace",
        )
        namespace_receipts: list[str] = []
        for ordinal, plan in sorted(by_namespace[namespace_id]):
            provider_input = _exact_dict(plan["provider_input"], "provider input")
            dated_question = str(provider_input.get("dated_question"))
            specialist = specialist_scanner(index, dated_question)
            if profile.profile_id == STRICT_PROFILE:
                bridge = frontier_builder(
                    provider_input, index=index, specialist_result=specialist
                )
            else:
                bridge = frontier_builder(
                    provider_input,
                    index=index,
                    specialist_result=specialist,
                    supported_domains=EXTENDED_SUPPORTED_DOMAINS,
                    operator_material_status=profile.operator_material_status,
                )
            _require(
                type(bridge) is NumericPolicyFrontierBridgeResult,
                "frontier builder returned another result type",
            )
            body = {
                "bridge": bridge.projection(),
                "bridge_receipt_sha256": bridge.receipt_sha256,
                "closed": bridge.closed,
                "format": profile.row_format,
                "namespace_id": namespace_id,
                "ordinal": ordinal,
                "provider_input_sha256": plan["provider_input_sha256"],
                "question_id": plan["question_id"],
                "question_sha256": plan["question_sha256"],
                "specialist_receipt_sha256": specialist.receipt.receipt_sha256,
                "window_index_receipt_sha256": index.receipt_sha256,
            }
            row = {**body, "row_receipt_sha256": identity_sha256(body)}
            rows.append(row)
            namespace_receipts.append(row["row_receipt_sha256"])
        life_body = {
            **dict(raw_lifecycle),
            "format": profile.lifecycle_format,
            "namespace_id": namespace_id,
            "numeric_row_receipt_sha256s": namespace_receipts,
            "window_index_receipt_sha256": index.receipt_sha256,
        }
        lifecycle.append(
            {**life_body, "receipt_sha256": identity_sha256(life_body)}
        )
        del index
        gc.collect()

    rows.sort(key=lambda row: int(row["ordinal"]))
    body = {
        "answer_preflight_artifact_sha256": inputs.preflight.sha256,
        "answer_replay_artifact_sha256": inputs.answer_replay.sha256,
        "answer_run_artifact_sha256": inputs.answer_run.sha256,
        "applicability": profile.applicability,
        "closed_count": sum(bool(row["closed"]) for row in rows),
        "format": profile.format,
        "frontier_count": len(rows),
        "frontier_rows": rows,
        "full100_construction_artifact_sha256": inputs.full100.construction.sha256,
        "full100_replay_artifact_sha256": inputs.full100.replay.sha256,
        "gold_loaded": False,
        "namespace_lifecycle": lifecycle,
        "new_provider_calls": 0,
        "ordinal_cli_routing_available": False,
        "ordinals": [row["ordinal"] for row in rows],
        "retained_transformer_token_state_bytes": 0,
    }
    assert_gold_blind(body, path="locked_full100_numeric_frontier")
    return {**body, "identity_sha256": identity_sha256(body)}


def _resident_index_loader(args: argparse.Namespace) -> IndexLoader:
    manifest: _ResidentPopulationManifest | None = None

    def load(namespace_id: str) -> tuple[FullStoreWindowIndex, Mapping[str, Any]]:
        nonlocal manifest
        namespace_id = require_sha256(namespace_id, "streamed worker namespace")
        if manifest is None:
            manifest = _prepare_resident_population(args)
        locations = manifest.locations_by_namespace.get(namespace_id, ())
        _require(locations, "streamed namespace is absent from retrieval")
        _require(len(locations) == 1, "streamed namespace store repeated")
        scoped = locations[0]
        database_path = scoped.store_dir / "memory.db"
        index_path = scoped.store_dir / "hnsw_index.bin"
        _require(
            database_path.is_file()
            and not database_path.is_symlink()
            and resident_cli.file_sha256(database_path)
            == scoped.database_sha256,
            f"frozen selected database changed: {namespace_id}",
        )
        _require(
            index_path.is_file()
            and not index_path.is_symlink()
            and resident_cli.file_sha256(index_path) == scoped.index_sha256,
            f"frozen selected HNSW index changed: {namespace_id}",
        )
        _require(
            scoped.namespace.combined_store_receipt_sha256
            in {
                location.namespace.combined_store_receipt_sha256
                for values in manifest.locations_by_namespace.values()
                for location in values
            },
            "selected combined-store receipt changed",
        )
        with Database(database_path, read_only=True) as database:
            cache = cache_namespace_partitions(
                database,
                scoped.namespace,
                source_database_sha256=scoped.database_sha256,
                source_store_receipt_sha256=(
                    scoped.namespace.combined_store_receipt_sha256
                ),
            )
            index = build_full_store_window_index(cache)
        return index, {
            "cache_receipt_sha256": cache.cache_receipt_sha256,
            "physical_content_row_count": len(index.rows),
            "physical_sentence_window_count": len(index.windows),
        }

    return load


def _build(args: argparse.Namespace) -> dict[str, Any]:
    return build_materialization_payload(
        _load_verified_inputs(args),
        index_loader=_resident_index_loader(args),
        policy_profile=str(getattr(args, "policy_profile", STRICT_PROFILE)),
    )


def _profile_output_root(args: argparse.Namespace) -> Path:
    profile = lifecycle_profile(str(getattr(args, "policy_profile", STRICT_PROFILE)))
    raw = getattr(args, "output_root", None)
    return profile.default_output_root if raw is None else Path(raw)


def run_materialize(args: argparse.Namespace) -> dict[str, Any]:
    profile = lifecycle_profile(str(getattr(args, "policy_profile", STRICT_PROFILE)))
    payload = _build(args)
    artifact, created = publish_sealed_json(
        _profile_output_root(args) / profile.materialization_name, payload
    )
    return {"created": created, "frontier_count": payload["frontier_count"], "sha256": artifact.sha256}


def run_replay(args: argparse.Namespace) -> dict[str, Any]:
    profile = lifecycle_profile(str(getattr(args, "policy_profile", STRICT_PROFILE)))
    root = _profile_output_root(args)
    source = read_sealed_json(root / profile.materialization_name)
    _require(
        source.sha256
        == require_sha256(args.expected_materialization_sha256, "materialization"),
        "numeric-frontier materialization changed",
    )
    payload = _build(args)
    _require(payload == source.payload, "numeric-frontier replay is not byte-identical")
    replay, created = publish_sealed_json(root / profile.replay_name, payload)
    _require(replay.sha256 == source.sha256, "numeric-frontier replay SHA changed")
    return {"byte_identical": True, "created": created, "sha256": replay.sha256}


def _frontier_from_projection(value: object) -> RelevantNumericFrontier:
    row = _exact_dict(value, "relevant numeric frontier")
    _require(row.get("format") == FRONTIER_FORMAT, "numeric frontier format changed")
    frontier = RelevantNumericFrontier(
        policy_input_sha256=str(row.get("policy_input_sha256")),
        candidate_population_receipt_sha256=str(
            row.get("candidate_population_receipt_sha256")
        ),
        represented_handle_ids=tuple(row.get("represented_handle_ids", ())),
        unresolved_candidate_keys=tuple(row.get("unresolved_candidate_keys", ())),
        selection_truncated=row.get("selection_truncated"),
        closed=row.get("closed"),
        provider_prompt_count=row.get("provider_prompt_count"),
        retained_transformer_token_state_bytes=row.get(
            "retained_transformer_token_state_bytes"
        ),
        receipt_sha256=str(row.get("receipt_sha256")),
    )
    _require(frontier.projection() == row, "numeric frontier projection changed")
    return frontier


def load_verified_numeric_frontiers(
    output_root: str | Path,
    expected_materialization_sha256: str,
    expected_replay_sha256: str,
    *,
    policy_profile: str = STRICT_PROFILE,
) -> tuple[
    SealedArtifact,
    SealedArtifact,
    Mapping[int, RelevantNumericFrontier],
]:
    """Load byte-identical lifecycle artifacts as typed overlay inputs."""

    profile = lifecycle_profile(policy_profile)
    root = Path(output_root)
    materialization = read_sealed_json(root / profile.materialization_name)
    replay = read_sealed_json(root / profile.replay_name)
    _require(
        materialization.sha256
        == require_sha256(expected_materialization_sha256, "numeric materialization")
        and replay.sha256
        == require_sha256(expected_replay_sha256, "numeric replay")
        and materialization.sha256 == replay.sha256
        and materialization.payload == replay.payload,
        "numeric frontier materialization/replay are not byte-identical",
    )
    payload = materialization.payload
    unsigned = dict(payload)
    declared = require_sha256(unsigned.pop("identity_sha256", None), "numeric identity")
    raw_rows = payload.get("frontier_rows")
    _require(
        payload.get("format") == profile.format
        and payload.get("applicability") == profile.applicability
        and payload.get("gold_loaded") is False
        and payload.get("new_provider_calls") == 0
        and payload.get("retained_transformer_token_state_bytes") == 0
        and payload.get("ordinal_cli_routing_available") is False
        and identity_sha256(unsigned) == declared
        and type(raw_rows) is list
        and payload.get("frontier_count") == len(raw_rows),
        "numeric frontier lifecycle envelope changed",
    )
    output: dict[int, RelevantNumericFrontier] = {}
    row_receipts_by_namespace: dict[str, list[str]] = defaultdict(list)
    window_receipts_by_namespace: dict[str, set[str]] = defaultdict(set)
    for raw in raw_rows:
        row = _exact_dict(raw, "numeric frontier row")
        ordinal = row.get("ordinal")
        body = dict(row)
        receipt = require_sha256(body.pop("row_receipt_sha256", None), "numeric row")
        namespace_id = require_sha256(
            row.get("namespace_id"), "numeric row namespace"
        )
        window_index_receipt = require_sha256(
            row.get("window_index_receipt_sha256"),
            "numeric row window index",
        )
        bridge = _exact_dict(row.get("bridge"), "numeric bridge")
        frontier = _frontier_from_projection(bridge.get("frontier"))
        bridge_body = dict(bridge)
        bridge_receipt = require_sha256(
            bridge_body.pop("receipt_sha256", None), "numeric bridge"
        )
        census_semantics = _unique_sha_list(
            bridge.get("census_semantic_key_sha256s"),
            "numeric census semantic keys",
        )
        provider_semantics = _unique_sha_list(
            bridge.get("provider_semantic_key_sha256s"),
            "numeric provider semantic keys",
        )
        represented_semantics = _unique_sha_list(
            bridge.get("represented_semantic_key_sha256s"),
            "numeric represented semantic keys",
        )
        census_facts = _unique_sha_list(
            bridge.get("census_material_fact_sha256s"),
            "numeric census material facts",
        )
        provider_facts = _unique_sha_list(
            bridge.get("provider_material_fact_sha256s"),
            "numeric provider material facts",
        )
        represented_facts = _unique_sha_list(
            bridge.get("represented_material_fact_sha256s"),
            "numeric represented material facts",
        )
        raw_census_atoms = bridge.get("census_atoms")
        _require(
            type(raw_census_atoms) is list
            and all(type(atom) is dict for atom in raw_census_atoms),
            "numeric census atom inventory changed",
        )
        if profile.operator_material_status:
            _require(
                all(
                    atom.get("status") == "operator_eligible"
                    for atom in raw_census_atoms
                ),
                "operator-material census status changed",
            )
        unresolved = tuple(bridge.get("unresolved_candidate_keys", ()))
        represented_handles = tuple(bridge.get("represented_handle_ids", ()))
        bridge_closed = bool(
            bridge.get("applicable") is True
            and census_semantics
            and not unresolved
            and set(census_semantics)
            == set(provider_semantics)
            == set(represented_semantics)
            and set(census_facts)
            == set(provider_facts)
            == set(represented_facts)
        )
        _require(
            type(ordinal) is int
            and ordinal not in output
            and row.get("format") == profile.row_format
            and identity_sha256(body) == receipt
            and identity_sha256(bridge_body) == bridge_receipt
            and bridge_receipt == row.get("bridge_receipt_sha256")
            and bridge.get("window_index_receipt_sha256")
            == row.get("window_index_receipt_sha256")
            and bridge.get("specialist_receipt_sha256")
            == row.get("specialist_receipt_sha256")
            and frontier.closed is row.get("closed")
            and bridge.get("format") == BRIDGE_FORMAT
            and bridge.get("policy_grammar_id") == POLICY_GRAMMAR_ID
            and bridge.get("policy_semantic_completeness_scope")
            == "versioned_supported_grammar"
            and bridge.get("policy_semantic_census_unit")
            == "full_immutable_content_row"
            and bridge.get("specialist_semantic_completeness_status")
            == "not_claimed"
            and bridge.get("physical_scan_exhaustive") is True
            and bridge.get("provider_prompt_count") == 0
            and bridge.get("retained_transformer_token_state_bytes") == 0
            and bridge.get("gold_loaded") is False
            and bridge.get("candidate_population_receipt_sha256")
            == frontier.candidate_population_receipt_sha256
            and unresolved == frontier.unresolved_candidate_keys
            and represented_handles == frontier.represented_handle_ids
            and frontier.closed is bridge_closed,
            "numeric frontier row binding changed",
        )
        output[ordinal] = frontier
        row_receipts_by_namespace[namespace_id].append(receipt)
        window_receipts_by_namespace[namespace_id].add(window_index_receipt)
    raw_lifecycle = payload.get("namespace_lifecycle")
    _require(type(raw_lifecycle) is list, "numeric namespace lifecycle changed")
    lifecycle_by_namespace: dict[
        str, tuple[tuple[str, ...], str]
    ] = {}
    for raw in raw_lifecycle:
        row = _exact_dict(raw, "numeric namespace lifecycle")
        body = dict(row)
        receipt = require_sha256(
            body.pop("receipt_sha256", None), "numeric namespace lifecycle"
        )
        namespace_id = require_sha256(
            row.get("namespace_id"), "numeric lifecycle namespace"
        )
        numeric_row_receipts = _unique_sha_list(
            row.get("numeric_row_receipt_sha256s"),
            "numeric lifecycle row receipts",
        )
        window_index_receipt = require_sha256(
            row.get("window_index_receipt_sha256"),
            "numeric lifecycle window index",
        )
        _require(
            row.get("format") == profile.lifecycle_format
            and identity_sha256(body) == receipt
            and namespace_id not in lifecycle_by_namespace,
            "numeric namespace lifecycle receipt changed",
        )
        lifecycle_by_namespace[namespace_id] = (
            numeric_row_receipts,
            window_index_receipt,
        )
    _require(
        set(lifecycle_by_namespace) == set(row_receipts_by_namespace)
        and all(
            tuple(row_receipts_by_namespace[namespace_id])
            == lifecycle_by_namespace[namespace_id][0]
            and window_receipts_by_namespace[namespace_id]
            == {lifecycle_by_namespace[namespace_id][1]}
            for namespace_id in lifecycle_by_namespace
        ),
        "numeric namespace lifecycle binding changed",
    )
    _require(
        tuple(output) == tuple(payload.get("ordinals", ()))
        and payload.get("closed_count") == sum(row.closed for row in output.values()),
        "numeric frontier ordered population changed",
    )
    return materialization, replay, output


def _add_inputs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--full100-root", type=Path, required=True)
    parser.add_argument("--answer-root", type=Path, required=True)
    parser.add_argument("--expected-answer-preflight-sha256", required=True)
    parser.add_argument("--expected-answer-run-sha256", required=True)
    parser.add_argument("--expected-answer-replay-sha256", required=True)
    parser.add_argument("--postseal-audit", type=Path, required=True)
    parser.add_argument("--expected-postseal-audit-sha256", required=True)
    resident_cli._add_store_args(parser)  # noqa: SLF001
    parser.add_argument(
        "--policy-profile",
        choices=tuple(_PROFILES),
        default=STRICT_PROFILE,
    )
    parser.add_argument("--output-root", type=Path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    materialize = commands.add_parser("materialize")
    _add_inputs(materialize)
    materialize.set_defaults(handler=run_materialize)
    replay = commands.add_parser("replay")
    _add_inputs(replay)
    replay.add_argument("--expected-materialization-sha256", required=True)
    replay.set_defaults(handler=run_replay)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(json.dumps(args.handler(args), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
