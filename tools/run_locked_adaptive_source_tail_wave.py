#!/usr/bin/env python3
"""Run the protocol-honest wave-2 recovery over the sealed D1+G1 base.

Planning is provider-free and revalidates the locked source stores.  Provider
execution reads only the sealed prompt population.  Materialization reads the
tail work/cache manifests plus immutable completion journals, and never opens a
store or invokes a provider.  Full replay rebuilds planning and revalidates the
stores before requiring byte identity.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:
    repository = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(repository / "src"), str(repository)]

from dotenv import load_dotenv  # noqa: E402

from memory_condense.eval.fast_completion_runtime import (  # noqa: E402
    FastCompletionBatch,
    FastCompletionRuntime,
    FastPromptPopulation,
    preflight_fast_completion_prompts,
)
from tools import run_locked_adaptive_source_map as source_cli  # noqa: E402
from tools.matched_eval import live  # noqa: E402
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
    require_text,
)
from tools.matched_eval.locked_source_gate_adapter import (  # noqa: E402
    DIRECT_STREAM_PROFILE_REPACK_V2,
    DIRECT_STREAM_PROFILE_V1,
    LockedSourceGateAdapterPopulation,
    LockedSourceGatePins,
    LockedSourceGateQuestion,
    load_locked_source_gate_adapter,
)
from tools.matched_eval.query_map_source_gate_adapter import (  # noqa: E402
    CONSOLIDATED_OBLIGATION_MODE,
    STATE_CHAIN_DIRECT_AUTHORITY_PROFILE,
)
from tools.matched_eval.source_gate_controller import (  # noqa: E402
    GateRoundKind,
    ObligationCoverageReceipt,
    QuestionBoundMappingPlan,
    SourceGateCandidate,
    SourceGatePlan,
    SourceGateRound,
    assess_obligation_coverage,
    build_question_bound_mapping_plan,
    coverage_facts_from_fact_union,
    start_source_gate,
)
from tools.matched_eval.source_history_fact_union import (  # noqa: E402
    FactLane,
    HydratedSourceHistory,
    PostMapFactUnion,
    SourceHistoryHydrationPlan,
    build_post_map_fact_union,
    plan_source_history_hydration,
)
from tools.matched_eval.source_history_mapper_live import (  # noqa: E402
    HARD_CONTEXT_TOKEN_CAP,
    MAPPER_CONTRACT_SHA256,
    MAX_PROMPT_TOKENS,
    OUTPUT_TOKEN_RESERVE,
    SourceMapperCachedCompletion,
    SourceMapperMaterialization,
    SourceMapperPreflight,
    SourceMapperProviderJournal,
    SourceHistoryMapperError,
    WorkDisposition,
    build_source_history_mapper_preflight,
    materialize_source_history_mapper,
)
from tools._routed_repair_routing import (  # noqa: E402
    RoutedRepairStyle,
)


CAMPAIGN_ID = "tail-wave-2-recovery-v1"
FORMAT = "memory-condense-locked-adaptive-source-tail-wave-2-recovery-v1"
PREFLIGHT_FORMAT = f"{FORMAT}-preflight"
CACHE_FORMAT = f"{FORMAT}-base-cache"
MATERIALIZATION_FORMAT = f"{FORMAT}-materialization"
REPLAY_FORMAT = f"{FORMAT}-replay"
INCIDENT_FORMAT = f"{FORMAT}-invalid-wave-1-incident"
PREFLIGHT_NAME = "adaptive-source-tail-wave-2-recovery-v1-preflight.json"
WORK_MANIFEST_NAME = "adaptive-source-tail-wave-2-recovery-v1-work-manifest.json"
CACHE_NAME = "adaptive-source-tail-wave-2-recovery-v1-base-cache.json"
INCIDENT_NAME = "adaptive-source-tail-wave-2-recovery-v1-incident.json"
MATERIALIZATION_NAME = "adaptive-source-tail-wave-2-recovery-v1-materialization.json"
REPLAY_NAME = "adaptive-source-tail-wave-2-recovery-v1-replay.json"
CHECKPOINT_DIR_NAME = "terra-source-history-tail-wave-2-recovery-v1-calls"
INVALID_WAVE1_DIR_NAME = "tail-wave-1"
INVALID_WAVE1_PREFLIGHT_NAME = "adaptive-source-tail-wave-1-preflight-v1.json"
INVALID_WAVE1_WORK_MANIFEST_NAME = (
    "adaptive-source-tail-wave-1-work-manifest-v1.json"
)
INVALID_WAVE1_CACHE_NAME = "adaptive-source-tail-wave-1-base-cache-v1.json"
INVALID_WAVE1_CHECKPOINT_DIR_NAME = "terra-source-history-tail-wave-1-calls"
PROTOCOL_INVALID_SENTINEL_NAME = "PROTOCOL_INVALID.md"

DEFAULT_BASE_ROOT = (
    Path("eval_results")
    / "matched_eval_100"
    / "adaptive-source-pareto-consolidated-authority-v1"
    / "d1-p0-g1"
)
DEFAULT_OUTPUT = DEFAULT_BASE_ROOT / CAMPAIGN_ID
EXPECTED_BASE_PREFLIGHT_SHA256 = (
    "216be985c901e47b2bc8ae21917f7417e1443704051f150aba7a4b40dec1a3e6"
)
EXPECTED_BASE_MATERIALIZATION_SHA256 = (
    "21f4c79c1c0d4d663bca8fffbfb3f38933ae5ab72492434b2af860babfdd03e6"
)
EXPECTED_PENDING_SOLVER_QUESTION_IDS = (
    "06878be2",
    "d23cf73b",
    "6b7dfb22",
    "32260d93",
    "9d25d4e0",
    "2e6d26dc",
    "2788b940",
)
MAX_NEW_PROVIDER_CALLS = 128
DIRECT_REPACK_MIN_RANK = 4
EXPECTED_INVALID_WAVE1_PREFLIGHT_SHA256 = (
    "11309cff569158e7ba66b454f71e47ef64d0a27ccad9c34cca7b4db80220f1ae"
)
EXPECTED_INVALID_WAVE1_WORK_MANIFEST_SHA256 = (
    "1db344cc785b51866af374b7250481d03e16cec4377bb7beca0236080b13df52"
)
EXPECTED_INVALID_WAVE1_CACHE_SHA256 = (
    "1708871472b710755cdda36837c508e7a8a6f6134a1dad43d7d387a93a2cfdc4"
)
EXPECTED_INVALID_WAVE1_RUNTIME_IDENTITY_SHA256 = (
    "94560d40ae6f97276a657d483a2a57372ed9380518900141a62b605402b18fed"
)
EXPECTED_UNAFFECTED_SELECTION_PROJECTION_SHA256 = (
    "10599cdeef72b1be6e6965cc6d19667293dbd78ebabff4b87deeead32d0053e3"
)
UNAFFECTED_SELECTION_FORMAT = "tail-wave-1-unaffected-logical-selection-projection-v1"

# Immutable reference receipts for the completed wave-2 recovery.  Downstream
# callers still have to pass the receipts they intend to consume; these names
# make the one audited population discoverable without weakening that explicit
# expected-hash boundary.
REFERENCE_RECOVERY_PREFLIGHT_SHA256 = (
    "c6618f8c1050ec64d0f744c1666484b43e4ac814331dedce7138a3e47d3ea335"
)
REFERENCE_RECOVERY_WORK_MANIFEST_SHA256 = (
    "5b9746dcf685075726b4418faa05abdf53bb8b9d83548bcea189a9c1756dc4ab"
)
REFERENCE_RECOVERY_MATERIALIZATION_SHA256 = (
    "e482c600ae89b85381d0d9b842ed5bb053770c1d544633241e6da7769c5d52ee"
)
REFERENCE_RECOVERY_REPLAY_SHA256 = (
    "ae2f7d4ffe1a89f790c12e9256e472a38209717523569ac44cecfc849b338e64"
)


class LockedAdaptiveSourceTailError(MatchedEvalContractError):
    """A base binding, tail decision, cache, prompt, or journal changed."""


@dataclass(frozen=True, slots=True)
class TerminalCallIdentity:
    provider_ordinal: int
    question_ordinal: int
    question_id: str
    lane: FactLane
    source_id: str
    selection_id: str
    window_id: str
    physical_work_id: str
    prompt_id: str
    messages_sha256: str
    call_key_sha256: str

    def projection(self) -> dict[str, Any]:
        return {
            "call_key_sha256": self.call_key_sha256,
            "lane": self.lane.value,
            "messages_sha256": self.messages_sha256,
            "physical_work_id": self.physical_work_id,
            "prompt_id": self.prompt_id,
            "provider_ordinal": self.provider_ordinal,
            "question_id": self.question_id,
            "question_ordinal": self.question_ordinal,
            "selection_id": self.selection_id,
            "source_id": self.source_id,
            "window_id": self.window_id,
        }


TERMINAL_CALL_IDENTITIES = (
    TerminalCallIdentity(
        0,
        1,
        "37d43f65",
        FactLane.DIRECT,
        "37d43f65::ultrachat_548962",
        "f13b64a786551124a3e8f04aa09902582395500e30f25174466adc4f00ff62a3",
        "1d36ac4c46efe164baebfffc31663ba462acff589af463ab494a31cb6cd72e5a",
        "87f1bfab1b803a04143666a21b85c76b2d6c31946ac15f500fc0ee9b6ebef9ed",
        "fd4f5b67f7f22f330d8e0a41789835d6fdebca47b05859b77c3af626d8eafe80",
        "d82ec59123ec2a95c11b78b74cd44f86e69c6b30f2699615f382781ab0836152",
        "151768481d56af2e04cd541ea648715c82ed90aba104f4681d1d1a0e32fd913d",
    ),
    TerminalCallIdentity(
        1,
        2,
        "e56a43b9",
        FactLane.DIRECT,
        "8a137a7f::683fdb17",
        "b96a479e9c47354908ef6fafb101306f5bb7984685e9edb29af972041506851d",
        "343babb34108673e23ae19a9d99a379cb6d5a94faaec5391c7759329f730a8f0",
        "cc2d3826293673a225f641b37c91c205775e07d79d13771a4a8c36a3989f39ad",
        "3d0ae77029b0a0ecd3fe0ff057eb54c160053965d54ff0a7e1bcff0f8373aaa2",
        "d6e4d0854da968878d5eff1d94b35a51d8b91fca21b02ccd931e495d34f0ba26",
        "923df27d297eb0a7220fe574f98645cf4e7f2a9226fc6b17b3026459278d18f1",
    ),
    TerminalCallIdentity(
        2,
        3,
        "gpt4_2f91af09",
        FactLane.PARTITION,
        "45dc21b6::60147f90_1",
        "81ca56e94f2e5e7da1b6993248d64c4e2839068ee4a8148144b0dfefe3db151a",
        "996bc0260d26340ebb8fe34ae8bfb7b0edf0cd6a7abd3f31cd8255c6c91b5a86",
        "59f94e717b3266e85056e92ee750105eee3f23edbdc80244cdb05dbbcd100238",
        "d62bd2324808b3969483d32a5ee16fb184676ff5facb0ddd0c121dff545890b7",
        "3a5f34d9e7a32082f32ebe4b4b43573122c31517b06da7368b7f13bebda9caec",
        "b8c43f66601a835d6d07d8fcfe306886bb6219351fad945c0b93c43b9ffe0a47",
    ),
    TerminalCallIdentity(
        3,
        4,
        "45dc21b6",
        FactLane.PARTITION,
        "45dc21b6::answer_07664d43_1",
        "8bb90d7314332427038f9e5c8ac9beae7a4f3950f631ddf27de312c9bf5e5071",
        "d5173e4b3b5d32734c627405b572b5201a3aa20754d3ec9af6eacf8b6be0b7b1",
        "fde61da72ade273e3a2a935164e084a2557c49af208fcc83338bbfb4dd5e8f35",
        "d007022139428433b25131c200e576d485564bcb43fb1483194af5c0c70d526d",
        "6867d210b87bb888d3f98f0323d58adeac8c5024bdff0434c38a9a6b764ce6e4",
        "ad5be2112c52ff5896d2e042a5c03f0018a1ec724aa5b3a2a49a893505c95f62",
    ),
)
_TERMINAL_BY_QUESTION_ID = {
    row.question_id: row for row in TERMINAL_CALL_IDENTITIES
}
_BANNED_SOURCE_IDS = frozenset(row.source_id for row in TERMINAL_CALL_IDENTITIES)
_BANNED_SELECTION_IDS = frozenset(
    row.selection_id for row in TERMINAL_CALL_IDENTITIES
)
_BANNED_WINDOW_IDS = frozenset(row.window_id for row in TERMINAL_CALL_IDENTITIES)
_BANNED_WORK_IDS = frozenset(
    row.physical_work_id for row in TERMINAL_CALL_IDENTITIES
)
_BANNED_PROMPT_IDS = frozenset(row.prompt_id for row in TERMINAL_CALL_IDENTITIES)
_BANNED_MESSAGE_SHA256S = frozenset(
    row.messages_sha256 for row in TERMINAL_CALL_IDENTITIES
)
_BANNED_CALL_KEYS = frozenset(
    row.call_key_sha256 for row in TERMINAL_CALL_IDENTITIES
)
EXPECTED_RECOVERY_REPLACEMENTS: Mapping[str, tuple[FactLane, int, str]] = {
    "37d43f65": (FactLane.DIRECT, 5, "8a137a7f::e229be9e_1"),
    "e56a43b9": (FactLane.DIRECT, 5, "37d43f65::1cafd864_3"),
    "gpt4_2f91af09": (FactLane.PARTITION, 1, "45dc21b6::ef69c258"),
    "45dc21b6": (FactLane.PARTITION, 1, "45dc21b6::answer_07664d43_2"),
}


def recovery_incident_projection() -> dict[str, Any]:
    """Return the immutable, gold-blind terminal-uncertainty incident receipt."""

    body: dict[str, Any] = {
        "campaign_id": CAMPAIGN_ID,
        "checkpoint_state": "compromised_request_journals_removed",
        "deletion_incident": {
            "cleanup_action_completed_normally": False,
            "cleanup_action_interrupted": True,
            "request_journals_removed": True,
            "response_journals_removed": False,
        },
        "format": INCIDENT_FORMAT,
        "invalid_wave1_cache_sha256": EXPECTED_INVALID_WAVE1_CACHE_SHA256,
        "invalid_wave1_checkpoint_dir_name": INVALID_WAVE1_CHECKPOINT_DIR_NAME,
        "invalid_wave1_preflight_sha256": EXPECTED_INVALID_WAVE1_PREFLIGHT_SHA256,
        "invalid_wave1_runtime_identity_sha256": (
            EXPECTED_INVALID_WAVE1_RUNTIME_IDENTITY_SHA256
        ),
        "invalid_wave1_status": "abandoned_terminal_uncertainty",
        "invalid_wave1_work_manifest_sha256": (
            EXPECTED_INVALID_WAVE1_WORK_MANIFEST_SHA256
        ),
        "provider_attempts": [
            {
                "client_entered": True,
                "outcome": "os_tcp_connect_denied_winerror_10013",
                "transport_entered": True,
            },
            {
                "client_entered": False,
                "outcome": "fail_closed_on_response_less_request",
                "transport_entered": False,
            },
        ],
        "request_journal_count_observed_before_deletion": 4,
        "response_journal_count_observed": 0,
        "terminal_calls": [row.projection() for row in TERMINAL_CALL_IDENTITIES],
        "wave1_checkpoint_reuse_permitted": False,
        "wave1_materialization_permitted": False,
        "wave1_replay_permitted": False,
    }
    body["receipt_sha256"] = identity_sha256(
        {"format": f"{INCIDENT_FORMAT}-receipt", **body}
    )
    assert_gold_blind(body, path="adaptive_source_tail_recovery_incident")
    return body


def _validate_incident_artifact(artifact: SealedArtifact) -> None:
    _require(
        artifact.payload == recovery_incident_projection(),
        "recovery incident receipt changed",
    )


def _invalid_wave1_root(base_source_root: Path) -> Path:
    return Path(base_source_root) / INVALID_WAVE1_DIR_NAME


def _assert_recovery_root_isolated(output_root: Path, base_source_root: Path) -> None:
    output = Path(output_root).resolve()
    base = Path(base_source_root).resolve()
    invalid = _invalid_wave1_root(base).resolve()
    checkpoint = (output / CHECKPOINT_DIR_NAME).resolve()
    _require(output != base, "recovery output must not overwrite its immutable base")
    _require(
        output != invalid and not output.is_relative_to(invalid),
        "wave-2 output cannot reside in the invalid wave-1 root",
    )
    _require(
        checkpoint != (invalid / INVALID_WAVE1_CHECKPOINT_DIR_NAME).resolve()
        and not checkpoint.is_relative_to(invalid),
        "wave-2 checkpoint namespace escaped into invalid wave 1",
    )
    _require(
        CHECKPOINT_DIR_NAME != INVALID_WAVE1_CHECKPOINT_DIR_NAME,
        "wave-2 checkpoint namespace aliases invalid wave 1",
    )


def _revalidate_invalid_wave1_lineage(base_source_root: Path) -> None:
    """Read only wave-1 sealed construction artifacts, never its checkpoints."""

    invalid = _invalid_wave1_root(base_source_root)
    preflight = read_sealed_json(invalid / INVALID_WAVE1_PREFLIGHT_NAME)
    work = read_sealed_json(invalid / INVALID_WAVE1_WORK_MANIFEST_NAME)
    cache = read_sealed_json(invalid / INVALID_WAVE1_CACHE_NAME)
    _require(
        preflight.sha256 == EXPECTED_INVALID_WAVE1_PREFLIGHT_SHA256
        and work.sha256 == EXPECTED_INVALID_WAVE1_WORK_MANIFEST_SHA256
        and cache.sha256 == EXPECTED_INVALID_WAVE1_CACHE_SHA256
        and (invalid / PROTOCOL_INVALID_SENTINEL_NAME).is_file(),
        "invalid wave-1 lineage changed or lacks its terminal sentinel",
    )


def _reject_protocol_invalid_root(output_root: Path) -> None:
    sentinel = Path(output_root) / PROTOCOL_INVALID_SENTINEL_NAME
    _require(
        not sentinel.exists(),
        "tail output root is permanently protocol-invalid: " + sentinel.as_posix(),
    )


def _require(ok: object, message: str) -> None:
    if not ok:
        raise LockedAdaptiveSourceTailError(message)


def _plain_messages(value: Sequence[Any]) -> tuple[dict[str, str], ...]:
    return tuple({"role": row.role, "content": row.content} for row in value)


class TailDisposition(str, Enum):
    SELECTED = "selected_zero_new_fact_unresolved"
    RECOVERY_SKIPPED = "recovery_same_lane_frontier_exhausted_no_advance"
    PENDING_SOLVER = "pending_solver_result_no_advance"
    SATISFIED = "obligations_satisfied_no_advance"
    EXHAUSTED = "route_frontier_exhausted_no_advance"
    REPACK_REQUIRED = "direct_rank_ge_4_requires_repack_v2_no_advance"


_ROUTE_LANES: Mapping[RoutedRepairStyle, tuple[FactLane, ...]] = {
    RoutedRepairStyle.NUMERIC_REDUCE: (
        FactLane.PARTITION,
        FactLane.GUIDED,
        FactLane.DIRECT,
    ),
    RoutedRepairStyle.SET_JOIN: (
        FactLane.PARTITION,
        FactLane.GUIDED,
        FactLane.DIRECT,
    ),
    RoutedRepairStyle.TIMELINE: (
        FactLane.GUIDED,
        FactLane.DIRECT,
        FactLane.PARTITION,
    ),
    RoutedRepairStyle.STATE_CHAIN: (
        FactLane.GUIDED,
        FactLane.DIRECT,
        FactLane.PARTITION,
    ),
    RoutedRepairStyle.EXTRACT: (
        FactLane.DIRECT,
        FactLane.GUIDED,
        FactLane.PARTITION,
    ),
    RoutedRepairStyle.SYNTHESIZE: (
        FactLane.DIRECT,
        FactLane.GUIDED,
        FactLane.PARTITION,
    ),
}


def route_lane_order(style: RoutedRepairStyle) -> tuple[FactLane, ...]:
    """Return the explicit specialized lane order for one bounded wave."""

    _require(type(style) is RoutedRepairStyle, "tail route style changed")
    return _ROUTE_LANES[style]


def direct_stream_profile_for_rank(lane: FactLane, rank: int) -> str:
    """Keep V1 shallow; deeper direct ranks explicitly require repack V2."""

    _require(type(lane) is FactLane, "tail lane changed")
    _require(type(rank) is int and rank >= 0, "tail candidate rank changed")
    if lane is FactLane.DIRECT and rank >= DIRECT_REPACK_MIN_RANK:
        return DIRECT_STREAM_PROFILE_REPACK_V2
    return DIRECT_STREAM_PROFILE_V1


def select_one_tail_candidate(
    plan: SourceGatePlan,
    base_round: SourceGateRound,
    *,
    direct_plan: SourceGatePlan | None = None,
    excluded_source_ids: frozenset[str] = frozenset(),
    required_lane: FactLane | None = None,
) -> tuple[SourceGatePlan, FactLane, SourceGateCandidate, str] | None:
    """Choose one method-local candidate before any physical-source dedup."""

    _require(type(plan) is SourceGatePlan, "tail gate plan changed")
    _require(
        type(base_round) is SourceGateRound
        and base_round == start_source_gate(plan),
        "tail base round differs from the locked base selection",
    )
    if direct_plan is not None:
        _require(
            type(direct_plan) is SourceGatePlan
            and direct_plan.question_id == plan.question_id
            and direct_plan.parent == plan.parent,
            "repack direct supplement escaped the base question/parent",
        )
    _require(
        type(excluded_source_ids) is frozenset
        and all(type(value) is str and value for value in excluded_source_ids),
        "tail excluded source identities changed",
    )
    _require(
        required_lane is None or type(required_lane) is FactLane,
        "tail required lane changed",
    )
    base_source_keys = {
        plan.candidate_by_id(candidate_id).source_key
        for candidate_id in base_round.cumulative_selected_candidate_ids
    }
    lanes = (
        (required_lane,)
        if required_lane is not None
        else route_lane_order(plan.route.style)
    )
    for lane in lanes:
        assert lane is not None
        lane_plan = direct_plan if lane is FactLane.DIRECT and direct_plan is not None else plan
        assert lane_plan is not None
        hard_cap = lane_plan.policy.budget_for(lane).hard_source_cap
        available = tuple(
            row
            for row in lane_plan.candidates_for(lane)[:hard_cap]
            if row.source_key not in base_source_keys
            and row.source_id not in excluded_source_ids
            and (
                lane is not FactLane.DIRECT
                or direct_plan is None
                or row.rank >= DIRECT_REPACK_MIN_RANK
            )
        )
        if available:
            profile = (
                DIRECT_STREAM_PROFILE_REPACK_V2
                if lane is FactLane.DIRECT and direct_plan is not None
                else DIRECT_STREAM_PROFILE_V1
            )
            return lane_plan, lane, available[0], profile
    return None


def _tail_round(
    plan: SourceGatePlan,
    base_round: SourceGateRound,
    lane: FactLane,
    candidate: SourceGateCandidate,
) -> SourceGateRound:
    _require(candidate.lane is lane, "tail candidate changed selected lane")
    same_gate = plan.receipt_sha256 == base_round.gate_plan_receipt_sha256
    cumulative = (
        base_round.cumulative_selected_candidate_ids + (candidate.candidate_id,)
        if same_gate
        else (candidate.candidate_id,)
    )
    source_keys = {
        (plan.candidate_by_id(candidate_id).namespace_id, plan.candidate_by_id(candidate_id).source_id)
        for candidate_id in cumulative
    }
    _require(
        len(source_keys) <= plan.policy.global_unique_source_cap,
        "tail selection exceeds the global unique-source cap",
    )
    return SourceGateRound(
        plan.receipt_sha256,
        base_round.round_index + 1,
        GateRoundKind.TAIL,
        lane,
        base_round.receipt_sha256,
        (candidate,),
        cumulative,
        len(source_keys),
    )


@dataclass(frozen=True, slots=True)
class TailQuestionDecision:
    ordinal: int
    question_id: str
    gate_plan_receipt_sha256: str
    base_round_receipt_sha256: str
    base_coverage_receipt_sha256: str
    base_fact_union_receipt_sha256: str
    retained_base_fact_count: int
    unresolved_obligation_ids: tuple[str, ...]
    route: str
    disposition: TailDisposition
    selected_lane: FactLane | None = None
    selected_candidate_id: str | None = None
    selected_source_id: str | None = None
    selected_rank: int | None = None
    tail_round_receipt_sha256: str | None = None
    tail_gate_plan_receipt_sha256: str | None = None
    selected_direct_stream_profile: str | None = None
    replaced_terminal_call_key_sha256: str | None = None

    @property
    def receipt_sha256(self) -> str:
        return identity_sha256({"format": f"{FORMAT}-decision", **self._body()})

    def _body(self) -> dict[str, Any]:
        return {
            "base_coverage_receipt_sha256": self.base_coverage_receipt_sha256,
            "base_fact_union_receipt_sha256": self.base_fact_union_receipt_sha256,
            "base_round_receipt_sha256": self.base_round_receipt_sha256,
            "disposition": self.disposition.value,
            "gate_plan_receipt_sha256": self.gate_plan_receipt_sha256,
            "ordinal": self.ordinal,
            "question_id": self.question_id,
            "retained_base_fact_count": self.retained_base_fact_count,
            "route": self.route,
            "selected_candidate_id": self.selected_candidate_id,
            "selected_lane": (
                None if self.selected_lane is None else self.selected_lane.value
            ),
            "selected_rank": self.selected_rank,
            "selected_source_id": self.selected_source_id,
            "tail_round_receipt_sha256": self.tail_round_receipt_sha256,
            "tail_gate_plan_receipt_sha256": self.tail_gate_plan_receipt_sha256,
            "selected_direct_stream_profile": self.selected_direct_stream_profile,
            "replaced_terminal_call_key_sha256": (
                self.replaced_terminal_call_key_sha256
            ),
            "unresolved_obligation_ids": list(self.unresolved_obligation_ids),
        }

    def projection(self) -> dict[str, Any]:
        value = {**self._body(), "receipt_sha256": self.receipt_sha256}
        assert_gold_blind(value, path="adaptive_source_tail_decision")
        return value


@dataclass(frozen=True, slots=True)
class TailQuestionWork:
    ordinal: int
    question_id: str
    decision: TailQuestionDecision
    gate_round: SourceGateRound
    hydration_plan: SourceHistoryHydrationPlan
    mapping_plan: QuestionBoundMappingPlan
    mapper_preflight: SourceMapperPreflight
    cached_completions: tuple[SourceMapperCachedCompletion, ...]
    base_window_token_cap: int


@dataclass(frozen=True, slots=True)
class LockedTailWavePlan:
    base_preflight: SealedArtifact
    base_work_manifest: SealedArtifact
    base_materialization: SealedArtifact
    source_population: LockedSourceGateAdapterPopulation
    repack_source_population: LockedSourceGateAdapterPopulation
    hydration_batches: tuple[source_cli.NamespaceHydrationBatch, ...]
    decisions: tuple[TailQuestionDecision, ...]
    questions: tuple[TailQuestionWork, ...]
    provider_population: FastPromptPopulation

    @property
    def required_provider_calls(self) -> int:
        return self.provider_population.unique_prompt_count

    @property
    def all_prompt_rows(self) -> tuple[Any, ...]:
        return tuple(
            prompt
            for question in self.questions
            for prompt in question.mapper_preflight.prompt_rows
        )


def _base_fact_unions(
    questions: Sequence[source_cli.FastMaterializationQuestionPlan],
    materializations: Sequence[SourceMapperMaterialization],
) -> dict[str, PostMapFactUnion]:
    _require(
        len(questions) == len(materializations),
        "base question/materialization populations differ",
    )
    result: dict[str, PostMapFactUnion] = {}
    for question, materialization in zip(
        questions, materializations, strict=True
    ):
        _require(
            materialization.hydration_plan_receipt_sha256
            == question.hydration_plan.receipt_sha256
            and materialization.mapping_plan_receipt_sha256
            == question.mapping_plan.receipt_sha256,
            "base materialization escaped its sealed question plan",
        )
        result[question.question_id] = build_post_map_fact_union(
            question.hydration_plan,
            batches=materialization.batches,
            direct_evidence=question.direct_evidence,
        )
    return result


def _base_cache(
    questions: Sequence[source_cli.FastMaterializationQuestionPlan],
    materializations: Sequence[SourceMapperMaterialization],
    batch: FastCompletionBatch,
) -> dict[str, SourceMapperCachedCompletion]:
    result: dict[str, SourceMapperCachedCompletion] = {}
    for question, materialization in zip(
        questions, materializations, strict=True
    ):
        journals = source_cli.provider_journals_for_question(
            question.mapper_preflight, batch
        )
        work_results = {
            row.physical_work_id: row for row in materialization.work_results
        }
        for journal in journals:
            original = work_results.get(journal.physical_work_id)
            _require(original is not None, "base cache lost materialized work")
            assert original is not None
            cached = SourceMapperCachedCompletion(
                journal.physical_work_id,
                journal.prompt_id,
                journal.messages_sha256,
                journal.completion,
                journal.completion_sha256,
                original.receipt_sha256,
                0,
            )
            _require(
                cached.physical_work_id not in result,
                "base cache repeated physical work",
            )
            result[cached.physical_work_id] = cached
    return result


def cap_mapping_plan_new_calls(
    plan: QuestionBoundMappingPlan,
    remaining: int,
) -> tuple[QuestionBoundMappingPlan, int]:
    """Move the global-budget suffix to deferred work without changing work."""

    _require(type(plan) is QuestionBoundMappingPlan, "tail mapping plan changed")
    _require(type(remaining) is int and remaining >= 0, "tail budget changed")
    allowed = plan.new_call_work_ids[:remaining]
    overflow = plan.new_call_work_ids[len(allowed) :]
    capped = QuestionBoundMappingPlan(
        plan.gate_plan_receipt_sha256,
        plan.gate_round_receipt_sha256,
        plan.hydration_plan_receipt_sha256,
        plan.work_items,
        plan.aliases,
        plan.reused_work_ids,
        allowed,
        plan.deferred_work_ids + overflow,
        plan.prior_call_work_ids,
    )
    return capped, remaining - len(allowed)


def enforce_structural_recovery_denylist(
    decisions: Sequence[TailQuestionDecision],
    questions: Sequence[TailQuestionWork],
) -> None:
    """Stage 1: exclude every terminal source/selection/window/work identity."""

    selected_sources = {
        row.selected_source_id
        for row in decisions
        if row.selected_source_id is not None
    }
    selection_ids = {
        selection.selection_id
        for question in questions
        for selection in question.gate_round.selections
    }
    window_ids = {
        window.window_id
        for question in questions
        for window in question.hydration_plan.windows
    }
    work_ids = {
        work.work_id
        for question in questions
        for work in question.mapping_plan.work_items
    }
    _require(
        selected_sources.isdisjoint(_BANNED_SOURCE_IDS)
        and selection_ids.isdisjoint(_BANNED_SELECTION_IDS)
        and window_ids.isdisjoint(_BANNED_WINDOW_IDS)
        and work_ids.isdisjoint(_BANNED_WORK_IDS),
        "wave-2 structural population reused a terminal wave-1 identity",
    )


def enforce_prompt_recovery_denylist(
    prompts: Sequence[Any],
) -> None:
    """Stage 2: exact old prompt/message identities can never be rendered."""

    prompt_ids = {row.prompt_id for row in prompts}
    messages = {row.messages_sha256 for row in prompts}
    _require(
        prompt_ids.isdisjoint(_BANNED_PROMPT_IDS)
        and messages.isdisjoint(_BANNED_MESSAGE_SHA256S),
        "wave-2 rendered a terminal wave-1 prompt or message",
    )


def enforce_runtime_recovery_denylist(runtime: FastCompletionRuntime) -> None:
    """Stage 3: fail before run() if a fresh runtime collides with wave 1."""

    raw = getattr(runtime, "_call_keys", None)
    _require(type(raw) is dict, "fast runtime call-key projection changed")
    call_keys = set(raw.values())
    _require(
        runtime.runtime_identity_sha256
        != EXPECTED_INVALID_WAVE1_RUNTIME_IDENTITY_SHA256
        and call_keys.isdisjoint(_BANNED_CALL_KEYS),
        "wave-2 runtime reused a terminal wave-1 identity",
    )


def _decision(
    question: LockedSourceGateQuestion,
    base_round: SourceGateRound,
    coverage: ObligationCoverageReceipt,
    union: PostMapFactUnion,
    disposition: TailDisposition,
    *,
    tail_round: SourceGateRound | None = None,
    tail_plan: SourceGatePlan | None = None,
    direct_stream_profile: str | None = None,
    terminal_identity: TerminalCallIdentity | None = None,
) -> TailQuestionDecision:
    selected = None if tail_round is None else tail_round.selected_candidates[0]
    return TailQuestionDecision(
        question.ordinal,
        question.plan.question_id,
        question.plan.receipt_sha256,
        base_round.receipt_sha256,
        coverage.receipt_sha256,
        union.receipt_sha256,
        len(union.retained_facts),
        coverage.unresolved_obligation_ids,
        question.plan.route.style.value,
        disposition,
        None if selected is None else selected.lane,
        None if selected is None else selected.candidate_id,
        None if selected is None else selected.source_id,
        None if selected is None else selected.rank,
        None if tail_round is None else tail_round.receipt_sha256,
        None if tail_plan is None else tail_plan.receipt_sha256,
        direct_stream_profile,
        None if terminal_identity is None else terminal_identity.call_key_sha256,
    )


def unaffected_selection_projection_sha256(
    decisions: Sequence[TailQuestionDecision],
) -> str:
    rows = [
        {
            "ordinal": row.ordinal,
            "question_id": row.question_id,
            "selected_candidate_id": row.selected_candidate_id,
            "selected_lane": (
                None if row.selected_lane is None else row.selected_lane.value
            ),
            "selected_rank": row.selected_rank,
            "selected_source_id": row.selected_source_id,
            "selected_direct_stream_profile": row.selected_direct_stream_profile,
        }
        for row in decisions
        if row.disposition is TailDisposition.SELECTED
        and row.question_id not in _TERMINAL_BY_QUESTION_ID
    ]
    return identity_sha256(
        {"format": UNAFFECTED_SELECTION_FORMAT, "rows": rows}
    )


def _load_base_and_population(args: argparse.Namespace) -> tuple[Any, ...]:
    loaded = source_cli.load_typed_materialization_root_with_batch(
        args.base_source_root,
        expected_preflight_sha256=args.expected_base_preflight_sha256,
        expected_materialization_sha256=args.expected_base_materialization_sha256,
        model=args.model,
        gateway_url=args.gateway_url,
        max_concurrency=args.max_concurrency,
        direct_base_cap=1,
        partition_base_cap=0,
        guided_base_cap=1,
    )
    base_preflight = loaded[0]
    _require(
        base_preflight.payload.get("obligation_compilation_mode")
        == CONSOLIDATED_OBLIGATION_MODE
        and base_preflight.payload.get("state_chain_profile")
        == STATE_CHAIN_DIRECT_AUTHORITY_PROFILE,
        "tail requires the consolidated state-chain-authority base",
    )
    _query, _map_plan, _map_plane, adapter = source_cli.load_locked_query_map(
        max_concurrency=args.max_concurrency,
        gateway_url=args.gateway_url,
        obligation_mode=CONSOLIDATED_OBLIGATION_MODE,
        state_chain_profile=STATE_CHAIN_DIRECT_AUTHORITY_PROFILE,
    )
    activations = source_cli.activation_inputs_from_query_map(adapter)
    population = load_locked_source_gate_adapter(
        activations,
        pins=LockedSourceGatePins(),
        policy=source_cli.source_gate_policy(1, 0, 1),
        direct_stream_profile=DIRECT_STREAM_PROFILE_V1,
    )
    repack_population = load_locked_source_gate_adapter(
        activations,
        pins=LockedSourceGatePins(),
        policy=source_cli.source_gate_policy(1, 0, 1),
        direct_stream_profile=DIRECT_STREAM_PROFILE_REPACK_V2,
    )
    _require(
        population.receipt_sha256
        == base_preflight.payload.get("source_gate_population_receipt_sha256"),
        "tail source population differs from the sealed D1+G1 base",
    )
    _require(
        tuple(row.plan.question_id for row in repack_population.questions)
        == tuple(row.plan.question_id for row in population.questions),
        "repack V2 supplement changed the activated question population",
    )
    return (*loaded, population, repack_population)


def build_tail_wave(args: argparse.Namespace) -> LockedTailWavePlan:
    """Revalidate the base, select one source per eligible question, and seal work."""

    (
        base_preflight,
        base_work_manifest,
        base_materialization,
        base_questions,
        base_materializations,
        base_batch,
        source_population,
        repack_source_population,
    ) = _load_base_and_population(args)
    base_by_id = {row.question_id: row for row in base_questions}
    materialization_by_id = {
        question.question_id: result
        for question, result in zip(
            base_questions, base_materializations, strict=True
        )
    }
    unions = _base_fact_unions(base_questions, base_materializations)
    cache_by_work = _base_cache(
        base_questions, base_materializations, base_batch
    )
    repack_by_id = {
        row.plan.question_id: row for row in repack_source_population.questions
    }
    decisions: list[TailQuestionDecision] = []
    selected: list[
        tuple[
            LockedSourceGateQuestion,
            SourceGateRound,
            TailQuestionDecision,
        ]
    ] = []
    for question in source_population.questions:
        gate = question.plan
        base_question = base_by_id.get(gate.question_id)
        base_result = materialization_by_id.get(gate.question_id)
        union = unions.get(gate.question_id)
        _require(
            base_question is not None
            and base_result is not None
            and union is not None,
            "tail source question escaped the sealed base materialization",
        )
        assert base_question is not None and base_result is not None and union is not None
        base_round = start_source_gate(gate)
        _require(
            base_round.receipt_sha256
            == base_question.mapping_plan.gate_round_receipt_sha256
            and base_round.selections == base_question.hydration_plan.selections,
            "tail base round changed its sealed selection/window binding",
        )
        coverage = assess_obligation_coverage(
            gate,
            base_round,
            coverage_facts_from_fact_union(union),
            mapping_plan_receipt_sha256s=(
                base_question.mapping_plan.receipt_sha256,
            ),
            cumulative_physical_work_call_ids=tuple(
                row.physical_work_id for row in base_result.work_results
            ),
            pending_physical_work_ids=base_result.deferred_work_ids,
        )
        if union.retained_facts:
            decisions.append(
                _decision(
                    question,
                    base_round,
                    coverage,
                    union,
                    TailDisposition.PENDING_SOLVER,
                )
            )
            continue
        if coverage.all_satisfied:
            decisions.append(
                _decision(
                    question,
                    base_round,
                    coverage,
                    union,
                    TailDisposition.SATISFIED,
                )
            )
            continue
        repack_question = repack_by_id.get(gate.question_id)
        _require(repack_question is not None, "tail lost its repack V2 question")
        assert repack_question is not None
        terminal_identity = _TERMINAL_BY_QUESTION_ID.get(gate.question_id)
        choice = select_one_tail_candidate(
            gate,
            base_round,
            direct_plan=repack_question.plan,
            excluded_source_ids=_BANNED_SOURCE_IDS,
            required_lane=(
                None if terminal_identity is None else terminal_identity.lane
            ),
        )
        if choice is None:
            decisions.append(
                _decision(
                    question,
                    base_round,
                    coverage,
                    union,
                    (
                        TailDisposition.EXHAUSTED
                        if terminal_identity is None
                        else TailDisposition.RECOVERY_SKIPPED
                    ),
                    terminal_identity=terminal_identity,
                )
            )
            continue
        tail_plan, lane, candidate, direct_profile = choice
        if terminal_identity is not None:
            expected_lane, expected_rank, expected_source_id = (
                EXPECTED_RECOVERY_REPLACEMENTS[gate.question_id]
            )
            _require(
                (lane, candidate.rank, candidate.source_id)
                == (expected_lane, expected_rank, expected_source_id),
                "recovery next same-lane source changed",
            )
        tail_round = _tail_round(tail_plan, base_round, lane, candidate)
        hydration_question = (
            repack_question if lane is FactLane.DIRECT else question
        )
        decision = _decision(
            question,
            base_round,
            coverage,
            union,
            TailDisposition.SELECTED,
            tail_round=tail_round,
            tail_plan=tail_plan,
            direct_stream_profile=direct_profile,
            terminal_identity=terminal_identity,
        )
        decisions.append(decision)
        selected.append((hydration_question, tail_round, decision))

    pending = tuple(
        row.question_id
        for row in decisions
        if row.disposition is TailDisposition.PENDING_SOLVER
    )
    _require(
        pending == EXPECTED_PENDING_SOLVER_QUESTION_IDS,
        "tail pending-solver population changed from the sealed seven rows",
    )
    selected_decisions = tuple(
        row for row in decisions if row.disposition is TailDisposition.SELECTED
    )
    unaffected_selected = tuple(
        row
        for row in selected_decisions
        if row.question_id not in _TERMINAL_BY_QUESTION_ID
    )
    recovery_outcomes = tuple(
        row
        for row in decisions
        if row.question_id in _TERMINAL_BY_QUESTION_ID
    )
    _require(
        len(unaffected_selected) == 75
        and len(recovery_outcomes) == 4
        and all(
            row.disposition
            in {TailDisposition.SELECTED, TailDisposition.RECOVERY_SKIPPED}
            for row in recovery_outcomes
        )
        and all(
            row.selected_rank == 4
            for row in unaffected_selected
            if row.selected_lane is FactLane.DIRECT
        ),
        "tail eligibility/lane/rank population changed from the locked wave",
    )
    _require(
        unaffected_selection_projection_sha256(selected_decisions)
        == EXPECTED_UNAFFECTED_SELECTION_PROJECTION_SHA256,
        "the 75 unaffected logical selections changed",
    )
    hydration_batches, histories_by_key = source_cli.hydrate_namespace_batches(
        tuple((question, round_plan) for question, round_plan, _row in selected)
    )
    remaining = args.max_new_calls
    work_rows: list[TailQuestionWork] = []
    for question, tail_round, decision in selected:
        base_question = base_by_id[question.plan.question_id]
        hydration_input = question.hydration_input(tail_round)
        histories: tuple[HydratedSourceHistory, ...] = tuple(
            histories_by_key[(hydration_input.namespace_id, row.source_id)]
            for row in hydration_input.memberships
        )
        base_work_ids = tuple(
            row.work_id for row in base_question.mapping_plan.work_items
        )
        base_cap = base_question.hydration_plan.max_window_tokens
        largest_chunk = max(
            chunk.token_count
            for history in histories
            for chunk in history.chunks
            if not chunk.metadata_chunk
        )
        candidate_caps = tuple(
            dict.fromkeys(
                value
                for value in (
                    base_cap,
                    min(base_cap, 4_800),
                    min(base_cap, 4_000),
                    min(base_cap, 3_200),
                    min(base_cap, 2_400),
                    min(base_cap, 1_600),
                    min(base_cap, 800),
                    largest_chunk,
                )
                if value >= largest_chunk
            )
        )
        last_overflow: SourceHistoryMapperError | None = None
        for cap in candidate_caps:
            hydration = plan_source_history_hydration(
                question.plan.parent,
                selections=tail_round.selections,
                histories=histories,
                max_window_tokens=cap,
            )
            raw_mapping = build_question_bound_mapping_plan(
                question.plan,
                tail_round,
                hydration,
                mapper_contract_sha256=MAPPER_CONTRACT_SHA256,
                cached_work_ids=base_work_ids,
                prior_call_work_ids=tuple(
                    row.physical_work_id
                    for row in materialization_by_id[
                        question.plan.question_id
                    ].work_results
                ),
            )
            mapping, next_remaining = cap_mapping_plan_new_calls(
                raw_mapping, remaining
            )
            try:
                mapper_preflight = build_source_history_mapper_preflight(
                    hydration, mapping
                )
            except SourceHistoryMapperError as exc:
                if "envelope overflow" not in str(exc):
                    raise
                last_overflow = exc
                continue
            remaining = next_remaining
            break
        else:
            assert last_overflow is not None
            raise last_overflow
        _require(
            mapper_preflight.maximum_combined_token_proxy
            <= HARD_CONTEXT_TOKEN_CAP,
            "tail mapper prompt escaped the hard context envelope",
        )
        caches = tuple(
            cache_by_work[work_id] for work_id in mapping.reused_work_ids
        )
        work_rows.append(
            TailQuestionWork(
                question.ordinal,
                question.plan.question_id,
                decision,
                tail_round,
                hydration,
                mapping,
                mapper_preflight,
                caches,
                base_cap,
            )
        )
    enforce_structural_recovery_denylist(decisions, work_rows)
    enforce_prompt_recovery_denylist(
        tuple(
            prompt
            for row in work_rows
            for prompt in row.mapper_preflight.prompt_rows
        )
    )
    submitted = tuple(
        _plain_messages(prompt.messages)
        for row in work_rows
        for prompt in row.mapper_preflight.prompt_rows
        if prompt.disposition is WorkDisposition.NEW_CALL
    )
    _require(bool(submitted), "bounded tail wave has no new provider work")
    provider_population = preflight_fast_completion_prompts(
        submitted, max_prompt_tokens=MAX_PROMPT_TOKENS
    )
    _require(
        provider_population.logical_prompt_count
        == provider_population.unique_prompt_count
        == sum(row.mapping_plan.planned_provider_calls for row in work_rows)
        <= args.max_new_calls
        <= MAX_NEW_PROVIDER_CALLS,
        "tail provider population escaped its exact 128-call ceiling",
    )
    return LockedTailWavePlan(
        base_preflight,
        base_work_manifest,
        base_materialization,
        source_population,
        repack_source_population,
        hydration_batches,
        tuple(decisions),
        tuple(work_rows),
        provider_population,
    )


def _as_base_plan(plan: LockedTailWavePlan) -> source_cli.LockedAdaptiveBasePlan:
    route_counts = Counter(
        row.decision.route for row in plan.questions
    )
    questions = tuple(
        source_cli.BaseQuestionSourceMap(
            row.ordinal,
            row.question_id,
            row.gate_round,
            row.hydration_plan,
            row.mapping_plan,
            row.mapper_preflight,
        )
        for row in plan.questions
    )
    return source_cli.LockedAdaptiveBasePlan(
        None,
        plan.source_population,
        plan.hydration_batches,
        questions,
        tuple(sorted(route_counts.items())),
        plan.provider_population,
    )


def cache_projection(plan: LockedTailWavePlan) -> dict[str, Any]:
    entries: list[SourceMapperCachedCompletion] = []
    for question in plan.questions:
        entries.extend(question.cached_completions)
    _require(
        len({row.physical_work_id for row in entries}) == len(entries),
        "tail cache repeated physical work",
    )
    payload = {
        "base_materialization_sha256": plan.base_materialization.sha256,
        "base_preflight_sha256": plan.base_preflight.sha256,
        "campaign_id": CAMPAIGN_ID,
        "cached_completion_count": len(entries),
        "cached_completions": [row.projection() for row in entries],
        "format": CACHE_FORMAT,
        "gold_loaded": False,
        "invalid_wave1_checkpoint_reads": 0,
        "invalid_wave1_checkpoint_reuse": False,
        "provider_calls": 0,
        "retained_transformer_token_state_bytes": 0,
    }
    assert_gold_blind(payload, path="adaptive_source_tail_cache")
    return payload


def _parse_cache(artifact: SealedArtifact) -> dict[str, SourceMapperCachedCompletion]:
    payload = artifact.payload
    _require(
        payload.get("format") == CACHE_FORMAT
        and payload.get("campaign_id") == CAMPAIGN_ID
        and payload.get("base_preflight_sha256")
        == EXPECTED_BASE_PREFLIGHT_SHA256
        and payload.get("base_materialization_sha256")
        == EXPECTED_BASE_MATERIALIZATION_SHA256
        and payload.get("gold_loaded") is False
        and payload.get("invalid_wave1_checkpoint_reads") == 0
        and payload.get("invalid_wave1_checkpoint_reuse") is False
        and payload.get("provider_calls") == 0
        and payload.get("retained_transformer_token_state_bytes") == 0,
        "tail cache changed parent or firewall",
    )
    raw = payload.get("cached_completions")
    _require(type(raw) is list, "tail cache entries changed type")
    result: dict[str, SourceMapperCachedCompletion] = {}
    for value in raw:
        _require(type(value) is dict, "tail cache entry changed type")
        assert type(value) is dict
        cached = SourceMapperCachedCompletion(
            require_sha256(value.get("physical_work_id"), "cached work"),
            require_sha256(value.get("prompt_id"), "cached prompt"),
            require_sha256(value.get("messages_sha256"), "cached messages"),
            require_text(value.get("completion"), "cached completion"),
            require_sha256(value.get("completion_sha256"), "cached completion"),
            require_sha256(
                value.get("original_work_result_receipt_sha256"),
                "cached original result",
            ),
            0,
        )
        _require(
            value == cached.projection()
            and cached.physical_work_id not in result,
            "tail cache entry changed seal or repeated work",
        )
        result[cached.physical_work_id] = cached
    _require(
        len(result) == payload.get("cached_completion_count"),
        "tail cache count changed",
    )
    return result


def preflight_projection(
    plan: LockedTailWavePlan,
    *,
    work_manifest: SealedArtifact,
    cache: SealedArtifact,
    incident: SealedArtifact,
    gateway_url: str,
    model: str,
    max_concurrency: int,
    max_new_calls: int,
) -> dict[str, Any]:
    prompts = [
        row.projection(include_messages=True) for row in plan.all_prompt_rows
    ]
    selected = tuple(
        row for row in plan.decisions if row.disposition is TailDisposition.SELECTED
    )
    pending = tuple(
        row
        for row in plan.decisions
        if row.disposition is TailDisposition.PENDING_SOLVER
    )
    repack = tuple(
        row
        for row in selected
        if row.selected_direct_stream_profile
        == DIRECT_STREAM_PROFILE_REPACK_V2
    )
    route_counts = Counter(row.route for row in selected)
    lane_counts = Counter(
        row.selected_lane.value for row in selected if row.selected_lane is not None
    )
    window_cap_rows = [
        {
            "actual_window_token_cap": row.hydration_plan.max_window_tokens,
            "base_window_token_cap": row.base_window_token_cap,
            "fallback_applied": (
                row.hydration_plan.max_window_tokens != row.base_window_token_cap
            ),
            "question_id": row.question_id,
        }
        for row in plan.questions
    ]
    frontier_body = {
        "base_round_receipt_sha256s": [
            row.base_round_receipt_sha256 for row in plan.decisions
        ],
        "direct_repack_min_rank": DIRECT_REPACK_MIN_RANK,
        "direct_repack_v2_profile": DIRECT_STREAM_PROFILE_REPACK_V2,
        "repack_source_gate_population_receipt_sha256": (
            plan.repack_source_population.receipt_sha256
        ),
        "route_lane_order": {
            style.value: [lane.value for lane in lanes]
            for style, lanes in _ROUTE_LANES.items()
        },
        "selected_source_exclusion": (
            "all_namespaced_sources_selected_in_base_round_plus_terminal_wave1_sources"
        ),
    }
    frontier_receipt = identity_sha256(
        {"format": f"{FORMAT}-tail-frontier", **frontier_body}
    )
    payload: dict[str, Any] = {
        "campaign_id": CAMPAIGN_ID,
        "base_materialization_sha256": plan.base_materialization.sha256,
        "base_preflight_sha256": plan.base_preflight.sha256,
        "base_work_manifest_sha256": plan.base_work_manifest.sha256,
        "cache_name": CACHE_NAME,
        "cache_sha256": cache.sha256,
        "deferred_physical_work_count": sum(
            len(row.mapping_plan.deferred_work_ids) for row in plan.questions
        ),
        "direct_repack_min_rank": DIRECT_REPACK_MIN_RANK,
        "direct_repack_v2_profile": DIRECT_STREAM_PROFILE_REPACK_V2,
        "direct_repack_v2_question_count": len(repack),
        "direct_repack_v2_question_ids": [row.question_id for row in repack],
        "direct_stream_profile": DIRECT_STREAM_PROFILE_V1,
        "format": PREFLIGHT_FORMAT,
        "gateway_url": gateway_url,
        "gold_loaded": False,
        "incident_name": INCIDENT_NAME,
        "incident_receipt_sha256": require_sha256(
            incident.payload.get("receipt_sha256"), "recovery incident receipt"
        ),
        "incident_sha256": incident.sha256,
        "invalid_wave1_checkpoint_reads": 0,
        "invalid_wave1_checkpoint_reuse": False,
        "logical_source_count": len(selected),
        "logical_source_selected_before_physical_dedup": True,
        "logical_window_count": sum(
            len(row.hydration_plan.windows) for row in plan.questions
        ),
        "mapper_contract_sha256": MAPPER_CONTRACT_SHA256,
        "max_concurrency": max_concurrency,
        "max_new_provider_calls": max_new_calls,
        "maximum_prompt_and_output_token_envelope": max(
            row.mapper_preflight.maximum_combined_token_proxy
            for row in plan.questions
        ),
        "model": model,
        "namespace_batch_count": len(plan.hydration_batches),
        "obligation_compilation_mode": CONSOLIDATED_OBLIGATION_MODE,
        "output_token_reserve": OUTPUT_TOKEN_RESERVE,
        "pending_solver_question_count": len(pending),
        "pending_solver_question_ids": [row.question_id for row in pending],
        "physical_prompt_count": len(prompts),
        "physical_prompt_rows": prompts,
        "provider_calls": 0,
        "provider_population": plan.provider_population.model_dump(),
        "all_prompt_and_output_envelopes_within_cap": all(
            row.get("combined_token_proxy", HARD_CONTEXT_TOKEN_CAP + 1)
            <= HARD_CONTEXT_TOKEN_CAP
            for row in prompts
        ),
        "question_decisions": [row.projection() for row in plan.decisions],
        "recovery_replacement_question_count": sum(
            row.question_id in _TERMINAL_BY_QUESTION_ID
            and row.disposition is TailDisposition.SELECTED
            for row in plan.decisions
        ),
        "recovery_skipped_question_count": sum(
            row.disposition is TailDisposition.RECOVERY_SKIPPED
            for row in plan.decisions
        ),
        "recovery_terminal_question_ids": list(_TERMINAL_BY_QUESTION_ID),
        "required_authorized_provider_calls": plan.required_provider_calls,
        "retained_transformer_token_state_bytes": 0,
        "route_counts": dict(sorted(route_counts.items())),
        "selected_lane_counts": dict(sorted(lane_counts.items())),
        "selection_rule": "unresolved_and_zero_retained_base_facts_only",
        "solver_dependency": "retained_base_facts_pending_until_sealed_solver_result",
        "repack_source_gate_population_receipt_sha256": (
            plan.repack_source_population.receipt_sha256
        ),
        "repack_source_input_artifacts": [
            row.projection()
            for row in plan.repack_source_population.source_artifacts
            if row.role in {"query_repack_v2_run", "query_repack_v2_runtime"}
        ],
        "source_gate_population_receipt_sha256": (
            plan.source_population.receipt_sha256
        ),
        "stable_base_window_token_caps": all(
            not row["fallback_applied"] for row in window_cap_rows
        ),
        "stable_windowing_policy": (
            "base_cap_then_4800_4000_3200_2400_1600_800_largest_chunk"
        ),
        "state_chain_profile": STATE_CHAIN_DIRECT_AUTHORITY_PROFILE,
        "tail_frontier_receipt_sha256": frontier_receipt,
        "tail_frontier": frontier_body,
        "unique_namespaced_source_count": len(
            {
                (selection.namespace_id, selection.source_id)
                for row in plan.questions
                for selection in row.gate_round.selections
            }
        ),
        "unique_physical_window_count": len(prompts),
        "window_cap_fallback_question_count": sum(
            row["fallback_applied"] for row in window_cap_rows
        ),
        "window_cap_rows": window_cap_rows,
        "work_manifest_name": WORK_MANIFEST_NAME,
        "work_manifest_sha256": work_manifest.sha256,
        "unaffected_logical_selection_count": 75,
        "unaffected_selection_projection_sha256": (
            unaffected_selection_projection_sha256(plan.decisions)
        ),
        "zero_new_fact_selected_question_count": len(selected),
    }
    _require(
        payload["maximum_prompt_and_output_token_envelope"]
        <= HARD_CONTEXT_TOKEN_CAP,
        "tail global mapper envelope exceeds 8K",
    )
    assert_gold_blind(payload, path="adaptive_source_tail_preflight")
    return payload


def _publish_preflight(
    plan: LockedTailWavePlan,
    args: argparse.Namespace,
) -> tuple[SealedArtifact, SealedArtifact, SealedArtifact, SealedArtifact]:
    output = Path(args.output_root)
    incident, _created = publish_sealed_json(
        output / INCIDENT_NAME, recovery_incident_projection()
    )
    _validate_incident_artifact(incident)
    work, _created = publish_sealed_json(
        output / WORK_MANIFEST_NAME,
        source_cli.work_manifest_projection(_as_base_plan(plan)),
    )
    cache, _created = publish_sealed_json(
        output / CACHE_NAME, cache_projection(plan)
    )
    preflight, _created = publish_sealed_json(
        output / PREFLIGHT_NAME,
        preflight_projection(
            plan,
            work_manifest=work,
            cache=cache,
            incident=incident,
            gateway_url=args.gateway_url,
            model=args.model,
            max_concurrency=args.max_concurrency,
            max_new_calls=args.max_new_calls,
        ),
    )
    return preflight, work, cache, incident


def _validate_preflight(
    artifact: SealedArtifact,
) -> tuple[FastPromptPopulation, tuple[tuple[dict[str, str], ...], ...]]:
    payload = artifact.payload
    assert_gold_blind(payload, path="adaptive_source_tail_provider")
    _require(
        payload.get("format") == PREFLIGHT_FORMAT
        and payload.get("campaign_id") == CAMPAIGN_ID
        and payload.get("base_preflight_sha256")
        == EXPECTED_BASE_PREFLIGHT_SHA256
        and payload.get("base_materialization_sha256")
        == EXPECTED_BASE_MATERIALIZATION_SHA256
        and payload.get("gold_loaded") is False
        and payload.get("provider_calls") == 0
        and payload.get("retained_transformer_token_state_bytes") == 0
        and payload.get("mapper_contract_sha256") == MAPPER_CONTRACT_SHA256
        and payload.get("pending_solver_question_ids")
        == list(EXPECTED_PENDING_SOLVER_QUESTION_IDS)
        and payload.get("direct_stream_profile") == DIRECT_STREAM_PROFILE_V1
        and payload.get("direct_repack_min_rank") == DIRECT_REPACK_MIN_RANK
        and payload.get("invalid_wave1_checkpoint_reads") == 0
        and payload.get("invalid_wave1_checkpoint_reuse") is False
        and payload.get("unaffected_logical_selection_count") == 75
        and payload.get("unaffected_selection_projection_sha256")
        == EXPECTED_UNAFFECTED_SELECTION_PROJECTION_SHA256
        and payload.get("recovery_terminal_question_ids")
        == list(_TERMINAL_BY_QUESTION_ID)
        and payload.get("recovery_replacement_question_count", 0)
        + payload.get("recovery_skipped_question_count", 0)
        == 4
        and payload.get("all_prompt_and_output_envelopes_within_cap") is True
        and type(payload.get("maximum_prompt_and_output_token_envelope")) is int
        and payload.get("maximum_prompt_and_output_token_envelope")
        <= HARD_CONTEXT_TOKEN_CAP,
        "tail preflight changed locked parent, selection policy, or firewall",
    )
    raw_rows = payload.get("physical_prompt_rows")
    _require(type(raw_rows) is list and bool(raw_rows), "tail prompts are missing")
    prompts: list[tuple[dict[str, str], ...]] = []
    seen: set[str] = set()
    seen_prompt_ids: set[str] = set()
    seen_messages: set[str] = set()
    for raw in raw_rows:
        _require(type(raw) is dict, "tail prompt row changed type")
        assert type(raw) is dict
        work_id = require_sha256(raw.get("physical_work_id"), "tail work")
        prompt_id = require_sha256(raw.get("prompt_id"), "tail prompt")
        messages_sha = require_sha256(
            raw.get("messages_sha256"), "tail messages"
        )
        _require(work_id not in seen, "tail physical work repeats")
        seen.add(work_id)
        seen_prompt_ids.add(prompt_id)
        seen_messages.add(messages_sha)
        messages = raw.get("messages")
        _require(
            type(messages) is list
            and len(messages) == 2
            and all(
                type(value) is dict
                and set(value) == {"role", "content"}
                and type(value.get("role")) is str
                and type(value.get("content")) is str
                for value in messages
            )
            and identity_sha256(messages) == raw.get("messages_sha256"),
            "tail prompt messages changed",
        )
        if raw.get("disposition") == WorkDisposition.NEW_CALL.value:
            prompts.append(tuple(dict(value) for value in messages))
        else:
            _require(
                raw.get("disposition")
                in {
                    WorkDisposition.REUSED.value,
                    WorkDisposition.DEFERRED.value,
                },
                "tail prompt disposition changed",
            )
    _require(
        seen.isdisjoint(_BANNED_WORK_IDS)
        and seen_prompt_ids.isdisjoint(_BANNED_PROMPT_IDS)
        and seen_messages.isdisjoint(_BANNED_MESSAGE_SHA256S),
        "tail preflight reused a terminal wave-1 work/prompt/message identity",
    )
    _require(bool(prompts), "tail authorized provider population is empty")
    population = preflight_fast_completion_prompts(
        prompts, max_prompt_tokens=MAX_PROMPT_TOKENS
    )
    required = payload.get("required_authorized_provider_calls")
    _require(
        type(required) is int
        and 0 < required <= payload.get("max_new_provider_calls") <= 128
        and population.logical_prompt_count
        == population.unique_prompt_count
        == required
        and population.model_dump() == payload.get("provider_population"),
        "tail provider population differs from exact authorization",
    )
    return population, tuple(prompts)


def _runtime(
    *,
    artifact: SealedArtifact,
    prompts: tuple[tuple[dict[str, str], ...], ...],
    checkpoint_dir: Path,
    model: str,
    gateway_url: str,
    max_concurrency: int,
    client: Any | None,
) -> FastCompletionRuntime:
    _require(
        artifact.payload.get("model") == model
        and artifact.payload.get("gateway_url") == gateway_url
        and artifact.payload.get("max_concurrency") == max_concurrency,
        "tail runtime configuration differs from sealed preflight",
    )
    runtime = FastCompletionRuntime(
        checkpoint_dir=checkpoint_dir,
        prompt_population=prompts,
        model=model,
        client=client,
        max_prompt_tokens=MAX_PROMPT_TOKENS,
        max_new_tokens=OUTPUT_TOKEN_RESERVE,
        max_concurrency=max_concurrency,
        retries=0,
        benchmark_provenance={
            "arm": "locked_adaptive_source_tail_wave_2_recovery_v1",
            "authorized_unique_calls": len(prompts),
            "base_materialization_sha256": EXPECTED_BASE_MATERIALIZATION_SHA256,
            "base_preflight_sha256": EXPECTED_BASE_PREFLIGHT_SHA256,
            "gateway_url": gateway_url,
            "gold_loaded": False,
            "mapper_contract_sha256": MAPPER_CONTRACT_SHA256,
            "preflight_artifact_sha256": artifact.sha256,
            "recovery_campaign_id": CAMPAIGN_ID,
            "recovery_incident_sha256": artifact.payload.get("incident_sha256"),
            "invalid_wave1_checkpoint_reuse": False,
        },
    )
    enforce_runtime_recovery_denylist(runtime)
    return runtime


def _read_preflight(
    args: argparse.Namespace,
) -> tuple[SealedArtifact, tuple[tuple[dict[str, str], ...], ...]]:
    expected = require_sha256(
        args.expected_preflight_sha256, "expected tail preflight"
    )
    artifact = read_sealed_json(Path(args.output_root) / PREFLIGHT_NAME)
    _require(artifact.sha256 == expected, "tail preflight changed")
    incident = read_sealed_json(Path(args.output_root) / INCIDENT_NAME)
    _validate_incident_artifact(incident)
    _require(
        artifact.payload.get("incident_sha256") == incident.sha256
        and artifact.payload.get("incident_receipt_sha256")
        == incident.payload.get("receipt_sha256"),
        "tail preflight lost its invalid-wave-1 incident binding",
    )
    _population, prompts = _validate_preflight(artifact)
    return artifact, prompts


def _journal_batch(
    args: argparse.Namespace,
    artifact: SealedArtifact,
    prompts: tuple[tuple[dict[str, str], ...], ...],
) -> FastCompletionBatch:
    runtime = _runtime(
        artifact=artifact,
        prompts=prompts,
        checkpoint_dir=Path(args.output_root) / CHECKPOINT_DIR_NAME,
        model=args.model,
        gateway_url=args.gateway_url,
        max_concurrency=args.max_concurrency,
        client=None,
    )
    try:
        return runtime.run()
    finally:
        runtime.close()


def _read_store_free_inputs(
    args: argparse.Namespace,
    preflight: SealedArtifact,
) -> tuple[
    SealedArtifact,
    SealedArtifact,
    tuple[source_cli.FastMaterializationQuestionPlan, ...],
    dict[str, SourceMapperCachedCompletion],
]:
    payload = preflight.payload
    work = read_sealed_json(Path(args.output_root) / WORK_MANIFEST_NAME)
    cache = read_sealed_json(Path(args.output_root) / CACHE_NAME)
    _require(
        work.sha256 == payload.get("work_manifest_sha256")
        and cache.sha256 == payload.get("cache_sha256"),
        "tail store-free artifacts changed",
    )
    questions = source_cli.load_fast_materialization_manifest(
        work,
        expected_source_population_receipt_sha256=require_sha256(
            payload.get("source_gate_population_receipt_sha256"),
            "tail source population",
        ),
    )
    _require(
        [
            prompt.projection(include_messages=True)
            for question in questions
            for prompt in question.mapper_preflight.prompt_rows
        ]
        == payload.get("physical_prompt_rows"),
        "tail work manifest differs from sealed prompts",
    )
    return work, cache, questions, _parse_cache(cache)


def _materialize_results(
    questions: tuple[source_cli.FastMaterializationQuestionPlan, ...],
    batch: FastCompletionBatch,
    cache: Mapping[str, SourceMapperCachedCompletion],
) -> tuple[SourceMapperMaterialization, ...]:
    results: list[SourceMapperMaterialization] = []
    for question in questions:
        cached = tuple(
            cache[work_id] for work_id in question.mapping_plan.reused_work_ids
        )
        journals: tuple[SourceMapperProviderJournal, ...] = (
            source_cli.provider_journals_for_question(
                question.mapper_preflight, batch
            )
        )
        results.append(
            materialize_source_history_mapper(
                question.mapper_preflight,
                question.hydration_plan,
                question.mapping_plan,
                provider_journals=journals,
                cached_completions=cached,
            )
        )
    return tuple(results)


def materialization_projection(
    preflight: SealedArtifact,
    work: SealedArtifact,
    cache: SealedArtifact,
    results: tuple[SourceMapperMaterialization, ...],
    batch: FastCompletionBatch,
) -> dict[str, Any]:
    payload = {
        "accepted_before_post_map_dedup_count": sum(
            row.accepted_before_post_map_dedup_count
            for result in results
            for row in result.work_results
        ),
        "base_materialization_sha256": EXPECTED_BASE_MATERIALIZATION_SHA256,
        "base_preflight_sha256": EXPECTED_BASE_PREFLIGHT_SHA256,
        "campaign_id": CAMPAIGN_ID,
        "cache_sha256": cache.sha256,
        "format": MATERIALIZATION_FORMAT,
        "gold_loaded": False,
        "historical_checkpoint_hits": batch.usage.checkpoint_hits,
        "incident_sha256": preflight.payload.get("incident_sha256"),
        "invalid_wave1_checkpoint_reads": 0,
        "invalid_wave1_checkpoint_reuse": False,
        "materializations": [row.projection() for row in results],
        "post_map_dedup_performed": False,
        "preflight_artifact_sha256": preflight.sha256,
        "provider_calls_during_materialization": 0,
        "question_count": len(results),
        "rejected_item_count": sum(
            row.rejected_item_count
            for result in results
            for row in result.work_results
        ),
        "retained_transformer_token_state_bytes": 0,
        "source_mapper_materialization_receipt_sha256s": [
            row.receipt_sha256 for row in results
        ],
        "store_reads_during_materialization": 0,
        "work_manifest_sha256": work.sha256,
    }
    assert_gold_blind(payload, path="adaptive_source_tail_materialization")
    return payload


def load_typed_tail_materialization_root(
    output_root: str | Path,
    *,
    expected_preflight_sha256: str,
    expected_work_manifest_sha256: str,
    expected_materialization_sha256: str,
    expected_replay_sha256: str,
    model: str,
    gateway_url: str,
    max_concurrency: int,
) -> tuple[
    SealedArtifact,
    SealedArtifact,
    SealedArtifact,
    SealedArtifact,
    tuple[source_cli.FastMaterializationQuestionPlan, ...],
    tuple[SourceMapperMaterialization, ...],
]:
    """Load the sealed recovery as typed objects using checkpoints only.

    The four caller-supplied receipts form the downstream authority boundary.
    Work/cache reconstruction and all mapper results are replayed from the
    wave-2 checkpoint namespace with ``client=None``.  No memory store, gold
    answer, or provider client is opened here.  The invalid wave-1 namespace
    and every terminal wave-1 call identity remain fail-closed.
    """

    root = Path(output_root)
    expected_preflight = require_sha256(
        expected_preflight_sha256, "expected tail preflight"
    )
    expected_work = require_sha256(
        expected_work_manifest_sha256, "expected tail work manifest"
    )
    expected_materialization = require_sha256(
        expected_materialization_sha256, "expected tail materialization"
    )
    expected_replay = require_sha256(
        expected_replay_sha256, "expected tail replay"
    )
    _reject_protocol_invalid_root(root)
    _require(
        INVALID_WAVE1_DIR_NAME.casefold()
        not in {part.casefold() for part in root.resolve().parts},
        "typed tail loader cannot read from the invalid wave-1 root",
    )

    preflight = read_sealed_json(root / PREFLIGHT_NAME)
    work = read_sealed_json(root / WORK_MANIFEST_NAME)
    terminal = read_sealed_json(root / MATERIALIZATION_NAME)
    replay = read_sealed_json(root / REPLAY_NAME)
    _require(preflight.sha256 == expected_preflight, "typed tail preflight changed")
    _require(work.sha256 == expected_work, "typed tail work manifest changed")
    _require(
        terminal.sha256 == expected_materialization,
        "typed tail materialization changed",
    )
    _require(replay.sha256 == expected_replay, "typed tail replay changed")
    _population, prompts = _validate_preflight(preflight)

    incident = read_sealed_json(root / INCIDENT_NAME)
    _validate_incident_artifact(incident)
    _require(
        preflight.payload.get("incident_sha256") == incident.sha256
        and preflight.payload.get("incident_receipt_sha256")
        == incident.payload.get("receipt_sha256"),
        "typed tail loader lost its invalid-wave-1 incident binding",
    )

    cache_artifact = read_sealed_json(root / CACHE_NAME)
    _require(
        work.sha256 == preflight.payload.get("work_manifest_sha256")
        and cache_artifact.sha256 == preflight.payload.get("cache_sha256"),
        "typed tail work/cache lineage changed",
    )
    questions = source_cli.load_fast_materialization_manifest(
        work,
        expected_source_population_receipt_sha256=require_sha256(
            preflight.payload.get("source_gate_population_receipt_sha256"),
            "typed tail source population",
        ),
    )
    _require(
        [
            prompt.projection(include_messages=True)
            for question in questions
            for prompt in question.mapper_preflight.prompt_rows
        ]
        == preflight.payload.get("physical_prompt_rows"),
        "typed tail work manifest differs from its sealed prompts",
    )
    cache = _parse_cache(cache_artifact)

    expected_replay_payload = {
        "byte_identical": True,
        "campaign_id": CAMPAIGN_ID,
        "expected_materialization_sha256": expected_materialization,
        "format": REPLAY_FORMAT,
        "gold_loaded": False,
        "incident_sha256": incident.sha256,
        "invalid_wave1_checkpoint_reads": 0,
        "invalid_wave1_checkpoint_reuse": False,
        "preflight_artifact_sha256": preflight.sha256,
        "provider_calls_during_replay": 0,
        "replayed_materialization_sha256": terminal.sha256,
        "retained_transformer_token_state_bytes": 0,
        "stores_revalidated_during_replay": True,
    }
    _require(
        replay.payload == expected_replay_payload,
        "typed tail replay lineage or firewall changed",
    )

    runtime = _runtime(
        artifact=preflight,
        prompts=prompts,
        checkpoint_dir=root / CHECKPOINT_DIR_NAME,
        model=model,
        gateway_url=gateway_url,
        max_concurrency=max_concurrency,
        client=None,
    )
    try:
        batch = runtime.run()
    finally:
        runtime.close()
    _require(
        batch.usage.checkpoint_hits == len(prompts)
        and batch.usage.physical_calls == 0
        and batch.provenance.retained_transformer_token_state_bytes == 0
        and batch.provenance.persisted_transformer_token_state is False,
        "typed tail loader requires complete checkpoint-only zero-state results",
    )
    materializations = _materialize_results(questions, batch, cache)
    expected_terminal_payload = materialization_projection(
        preflight, work, cache_artifact, materializations, batch
    )
    _require(
        terminal.payload == expected_terminal_payload,
        "typed tail checkpoint replay differs from terminal materialization",
    )
    return (
        preflight,
        work,
        terminal,
        replay,
        questions,
        materializations,
    )


def _preflight(args: argparse.Namespace) -> dict[str, Any]:
    _assert_recovery_root_isolated(args.output_root, args.base_source_root)
    _reject_protocol_invalid_root(Path(args.output_root))
    _require(
        0 < args.max_new_calls <= MAX_NEW_PROVIDER_CALLS,
        "tail max-new-calls must be within 1..128",
    )
    _revalidate_invalid_wave1_lineage(args.base_source_root)
    plan = build_tail_wave(args)
    artifact, work, cache, incident = _publish_preflight(plan, args)
    return {
        "artifact": artifact.path.as_posix(),
        "base_materialization_sha256": plan.base_materialization.sha256,
        "base_preflight_sha256": plan.base_preflight.sha256,
        "cache_sha256": cache.sha256,
        "gold_loaded": False,
        "incident_sha256": incident.sha256,
        "logical_source_count": artifact.payload["logical_source_count"],
        "logical_window_count": artifact.payload["logical_window_count"],
        "maximum_prompt_and_output_token_envelope": artifact.payload[
            "maximum_prompt_and_output_token_envelope"
        ],
        "pending_solver_question_count": artifact.payload[
            "pending_solver_question_count"
        ],
        "physical_prompt_count": artifact.payload["physical_prompt_count"],
        "preflight_sha256": artifact.sha256,
        "provider_calls": 0,
        "required_authorized_provider_calls": plan.required_provider_calls,
        "route_counts": artifact.payload["route_counts"],
        "selected_lane_counts": artifact.payload["selected_lane_counts"],
        "work_manifest_sha256": work.sha256,
    }


def _provider(args: argparse.Namespace) -> dict[str, Any]:
    _assert_recovery_root_isolated(args.output_root, args.base_source_root)
    _reject_protocol_invalid_root(Path(args.output_root))
    artifact, prompts = _read_preflight(args)
    required = len(prompts)
    _require(
        args.enable_provider is True
        and args.authorized_provider_calls == required,
        f"provider-run requires exact authorization for {required} calls",
    )
    load_dotenv()
    api_key = os.environ.get(args.api_key_env, "").strip()
    _require(bool(api_key), f"provider API key is empty: {args.api_key_env}")
    client = live._make_provider_client(api_key, args.gateway_url)  # noqa: SLF001
    runtime = _runtime(
        artifact=artifact,
        prompts=prompts,
        checkpoint_dir=Path(args.output_root) / CHECKPOINT_DIR_NAME,
        model=args.model,
        gateway_url=args.gateway_url,
        max_concurrency=args.max_concurrency,
        client=client,
    )
    try:
        batch = runtime.run()
    finally:
        runtime.close()
    return {
        "checkpoint_hits": batch.usage.checkpoint_hits,
        "gold_loaded": False,
        "physical_provider_calls": batch.usage.physical_calls,
        "preflight_sha256": artifact.sha256,
        "required_authorized_provider_calls": required,
        "retained_transformer_token_state_bytes": 0,
    }


def _materialize(args: argparse.Namespace) -> dict[str, Any]:
    _assert_recovery_root_isolated(args.output_root, args.base_source_root)
    _reject_protocol_invalid_root(Path(args.output_root))
    artifact, prompts = _read_preflight(args)
    work, cache_artifact, questions, cache = _read_store_free_inputs(
        args, artifact
    )
    batch = _journal_batch(args, artifact, prompts)
    _require(
        batch.usage.checkpoint_hits == len(prompts)
        and batch.usage.physical_calls == 0,
        "tail materialization requires checkpoint-only completions",
    )
    results = _materialize_results(questions, batch, cache)
    payload = materialization_projection(
        artifact, work, cache_artifact, results, batch
    )
    terminal, created = publish_sealed_json(
        Path(args.output_root) / MATERIALIZATION_NAME, payload
    )
    return {
        "accepted_before_post_map_dedup_count": payload[
            "accepted_before_post_map_dedup_count"
        ],
        "checkpoint_hits": batch.usage.checkpoint_hits,
        "gold_loaded": False,
        "materialization_sha256": terminal.sha256,
        "physical_provider_calls": 0,
        "store_reads_during_materialization": 0,
        "terminal_materialization_replayed": not created,
    }


def _replay(args: argparse.Namespace) -> dict[str, Any]:
    _assert_recovery_root_isolated(args.output_root, args.base_source_root)
    _reject_protocol_invalid_root(Path(args.output_root))
    expected_materialization = require_sha256(
        args.expected_materialization_sha256,
        "expected tail materialization",
    )
    artifact, prompts = _read_preflight(args)
    plan = build_tail_wave(args)
    rebuilt, _work, _cache, _incident = _publish_preflight(plan, args)
    _require(rebuilt.sha256 == artifact.sha256, "tail preflight replay changed")
    work, cache_artifact, questions, cache = _read_store_free_inputs(
        args, artifact
    )
    batch = _journal_batch(args, artifact, prompts)
    _require(
        batch.usage.checkpoint_hits == len(prompts)
        and batch.usage.physical_calls == 0,
        "tail replay requires checkpoint-only completions",
    )
    results = _materialize_results(questions, batch, cache)
    payload = materialization_projection(
        artifact, work, cache_artifact, results, batch
    )
    terminal = read_sealed_json(Path(args.output_root) / MATERIALIZATION_NAME)
    _require(
        terminal.sha256 == expected_materialization
        and terminal.payload == payload,
        "tail materialization replay changed bytes",
    )
    replay_payload = {
        "byte_identical": True,
        "campaign_id": CAMPAIGN_ID,
        "expected_materialization_sha256": expected_materialization,
        "format": REPLAY_FORMAT,
        "gold_loaded": False,
        "incident_sha256": artifact.payload.get("incident_sha256"),
        "invalid_wave1_checkpoint_reads": 0,
        "invalid_wave1_checkpoint_reuse": False,
        "preflight_artifact_sha256": artifact.sha256,
        "provider_calls_during_replay": 0,
        "replayed_materialization_sha256": terminal.sha256,
        "retained_transformer_token_state_bytes": 0,
        "stores_revalidated_during_replay": True,
    }
    replay, _created = publish_sealed_json(
        Path(args.output_root) / REPLAY_NAME, replay_payload
    )
    return {
        "byte_identical": True,
        "gold_loaded": False,
        "materialization_sha256": terminal.sha256,
        "physical_provider_calls": 0,
        "replay_sha256": replay.sha256,
    }


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--base-source-root", type=Path, default=DEFAULT_BASE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--expected-base-preflight-sha256",
        default=EXPECTED_BASE_PREFLIGHT_SHA256,
    )
    parser.add_argument(
        "--expected-base-materialization-sha256",
        default=EXPECTED_BASE_MATERIALIZATION_SHA256,
    )
    parser.add_argument("--gateway-url", default=live.DEFAULT_GATEWAY_URL)
    parser.add_argument("--model", default=live.DEFAULT_TERRA_GATEWAY_MODEL)
    parser.add_argument("--max-concurrency", type=int, default=4)
    parser.add_argument("--max-new-calls", type=int, default=MAX_NEW_PROVIDER_CALLS)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    preflight = commands.add_parser("preflight")
    _common(preflight)
    provider = commands.add_parser("provider-run")
    _common(provider)
    provider.add_argument("--expected-preflight-sha256", required=True)
    provider.add_argument("--enable-provider", action="store_true")
    provider.add_argument("--authorized-provider-calls", type=int, default=0)
    provider.add_argument("--api-key-env", default=live.DEFAULT_API_KEY_ENV)
    materialize = commands.add_parser("materialize")
    _common(materialize)
    materialize.add_argument("--expected-preflight-sha256", required=True)
    replay = commands.add_parser("replay")
    _common(replay)
    replay.add_argument("--expected-preflight-sha256", required=True)
    replay.add_argument("--expected-materialization-sha256", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "preflight":
        result = _preflight(args)
    elif args.command == "provider-run":
        result = _provider(args)
    elif args.command == "materialize":
        result = _materialize(args)
    elif args.command == "replay":
        result = _replay(args)
    else:  # pragma: no cover
        raise AssertionError("unreachable command")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CAMPAIGN_ID",
    "CACHE_NAME",
    "CHECKPOINT_DIR_NAME",
    "DEFAULT_BASE_ROOT",
    "DEFAULT_OUTPUT",
    "DIRECT_REPACK_MIN_RANK",
    "EXPECTED_BASE_MATERIALIZATION_SHA256",
    "EXPECTED_BASE_PREFLIGHT_SHA256",
    "EXPECTED_INVALID_WAVE1_CACHE_SHA256",
    "EXPECTED_INVALID_WAVE1_PREFLIGHT_SHA256",
    "EXPECTED_INVALID_WAVE1_RUNTIME_IDENTITY_SHA256",
    "EXPECTED_INVALID_WAVE1_WORK_MANIFEST_SHA256",
    "EXPECTED_PENDING_SOLVER_QUESTION_IDS",
    "EXPECTED_RECOVERY_REPLACEMENTS",
    "INCIDENT_NAME",
    "INVALID_WAVE1_CHECKPOINT_DIR_NAME",
    "INVALID_WAVE1_DIR_NAME",
    "LockedAdaptiveSourceTailError",
    "LockedTailWavePlan",
    "MAX_NEW_PROVIDER_CALLS",
    "MATERIALIZATION_NAME",
    "PREFLIGHT_NAME",
    "REFERENCE_RECOVERY_MATERIALIZATION_SHA256",
    "REFERENCE_RECOVERY_PREFLIGHT_SHA256",
    "REFERENCE_RECOVERY_REPLAY_SHA256",
    "REFERENCE_RECOVERY_WORK_MANIFEST_SHA256",
    "TailDisposition",
    "TailQuestionDecision",
    "TailQuestionWork",
    "TERMINAL_CALL_IDENTITIES",
    "TerminalCallIdentity",
    "WORK_MANIFEST_NAME",
    "build_tail_wave",
    "cache_projection",
    "cap_mapping_plan_new_calls",
    "direct_stream_profile_for_rank",
    "enforce_prompt_recovery_denylist",
    "enforce_runtime_recovery_denylist",
    "enforce_structural_recovery_denylist",
    "load_typed_tail_materialization_root",
    "main",
    "materialization_projection",
    "preflight_projection",
    "recovery_incident_projection",
    "route_lane_order",
    "select_one_tail_candidate",
    "unaffected_selection_projection_sha256",
]
