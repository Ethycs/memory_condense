"""Pinned, provider-free bridge from locked retrieval artifacts to the source gate.

The loader verifies the authoritative query run/replay/runtime plane, the
partition-scan-v2-r96 generation plus eligibility manifest, the guided
run/replay/runtime plane, and the immutable combined-store bytes.  It then
projects only permitted coordinates:

* direct V1 (default): post-selection admitted query spans only;
* direct repack V2 (optional): replayed selected-before-dedup query spans;
* partition: selected-before-dedup spans;
* guided: selected-before-dedup spans.

The V2 profile changes only direct source discovery; direct evidence authority
remains the V1 matched packet. Repeated spans collapse to the first distinct
source *inside each method*.
Cross-method reuse remains a later source-gate mapping-cache operation.  Every
source is joined through ``(namespace_id, source_id)`` to the sealed
``FrozenSourceMembership`` before a candidate or hydration input can exist.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from memory_condense.domain.discourse import quote_sha256
from memory_condense.domain.integrity import file_sha256
from memory_condense.persistence.db import Database
from tools._routed_repair_routing import route_question

from .artifacts import read_sealed_json
from .closure import ELIGIBILITY_FORMAT
from .contracts import (
    ArtifactRef,
    MatchedEvalContractError,
    assert_gold_blind,
    identity_sha256,
    require_sha256,
    require_text,
)
from .partition_scan_v2 import load_partition_scan_v2_generation
from .query_expansion import (
    FrozenSourceMembership,
    FrozenSourceNamespace,
    load_locked_query_expansion_context,
    load_preflighted_query_expansion_population,
)
from .query_expansion_repack_v2 import (
    ROW_FORMAT as QUERY_REPACK_ROW_FORMAT,
    RUN_NAME as QUERY_REPACK_RUN_NAME,
    RUNTIME_LEDGER_NAME as QUERY_REPACK_RUNTIME_LEDGER_NAME,
    replay_query_expansion_repack_v2,
    verify_query_expansion_parent,
)
from .query_fact_adapter import build_query_fact_population
from .query_guided_payload_adapter import (
    build_query_guided_payload_adapter,
    verify_query_guided_construction,
)
from .source_gate_controller import (
    SOURCE_GATE_LANES,
    EligibleFrontierScope,
    QuestionObligation,
    SourceGateActivationReceipt,
    SourceGateCandidate,
    SourceGatePlan,
    SourceGatePolicy,
    SourceGateRound,
    default_source_gate_policy,
)
from .source_history_fact_union import (
    DirectEvidenceRef,
    FactLane,
    ParentIdentity,
    direct_evidence_projection_sha256,
)


FORMAT = "memory-condense-locked-source-gate-adapter-v1"
DIRECT_STREAM_PROFILE_V1 = "query-admitted-delta-v1"
DIRECT_STREAM_PROFILE_REPACK_V2 = (
    "query-repack-selected-before-dedup-v2"
)
DIRECT_STREAM_PROFILES = (
    DIRECT_STREAM_PROFILE_V1,
    DIRECT_STREAM_PROFILE_REPACK_V2,
)
DEFAULT_RETRIEVAL = Path(
    "eval_results/longmemeval-1m-recall-guarded-cumulative-validation-20260822/retrieval.json"
)
DEFAULT_STORE_ROOT = DEFAULT_RETRIEVAL.parent
DEFAULT_CAMPAIGN_ROOT = Path(
    "eval_results/longmemeval-1m-locked-retrieval-mechanism-arms-20260826"
)
DEFAULT_QUERY_ROOT = DEFAULT_CAMPAIGN_ROOT / "matched-eval-spine-v2/s0-plus-query-expansion-v1"
DEFAULT_QUERY_REPACK_ROOT = (
    DEFAULT_CAMPAIGN_ROOT
    / "matched-eval-spine-v2/s0-plus-query-expansion-repack-v2"
)
DEFAULT_PARTITION_GENERATION = DEFAULT_CAMPAIGN_ROOT / "partition-scan-v2-r96/retrieval-generation.json"
DEFAULT_ELIGIBILITY = DEFAULT_CAMPAIGN_ROOT / "independent-closure-v9/eligibility-manifest.json"
DEFAULT_GUIDED_ROOT = DEFAULT_CAMPAIGN_ROOT / "matched-eval-spine-v2/s0-plus-query-guided-scan-v1"

EXPECTED_RETRIEVAL_SHA256 = "e36b54ec6171aa7b40f75682ad85e5822a64d45bc411ffe03bcd9cad0222007f"
EXPECTED_SOURCE_POPULATION_ID = "886e14025a0aedf5a9ba673be8ffc9183acc080b97645adc2b6dd003019438bf"
EXPECTED_QUERY_PREFLIGHT_SHA256 = "dc357e4a4e946c541ca5cb278824c376692ba4e4a97a5947c5b18e8da86c5487"
EXPECTED_QUERY_RUN_SHA256 = "68f7c0c073c405e33cf019c75e69db1ee5be9b9f3dd84f13cd5a427e6508ba07"
EXPECTED_QUERY_RUNTIME_SHA256 = "16d5ceedee9a86d7c719d3d66538a4d8fa23cf8fbee5763097df69f28afc7c94"
EXPECTED_QUERY_REPACK_RUN_SHA256 = "960c8192ff8b97b599f37ac067f79036f4403bd8dfb8cb8532c13b309dea7c47"
EXPECTED_QUERY_REPACK_RUNTIME_SHA256 = "99d4df790f80b95da521fe1ffd5eddb7d7c041f082fc34a386977ee7db9cedd3"
EXPECTED_QUERY_POPULATION_ID = "5030a5ae9ce83be7ae39ad290b492db278c8f090303730766db21edecae33b5e"
EXPECTED_QUERY_PROMPT_POPULATION_SHA256 = "c88a09f1817404d5f29e0cca77fdb260b1479bf004bb8339d543376a3741c02d"
EXPECTED_ELIGIBILITY_SHA256 = "748bd56a7efb8fd70d36bc96f099a53fc506469565577de9635908f6773bdee1"
EXPECTED_PARTITION_GENERATION_SHA256 = "671f0a3418364f544e61897c42569407805e827ae558980760289dae6b5cf388"
EXPECTED_GUIDED_RUN_SHA256 = "a544ae9e6e554fcfc9cfc6167018f06b573fcf6546c9c3f3a6e3feda6ed821ff"
EXPECTED_GUIDED_RUNTIME_SHA256 = "b0edd491ddca674c24728f31cda337226090624db04c63a507eb6188eb802af7"


class LockedSourceGateAdapterError(MatchedEvalContractError):
    """A pinned artifact, store, membership, or activation binding changed."""


def _require(ok: object, message: str) -> None:
    if not ok:
        raise LockedSourceGateAdapterError(message)


def _typed(values: object, cls: type, label: str) -> tuple[Any, ...]:
    _require(type(values) is tuple and all(type(row) is cls for row in values), f"{label} must be an immutable exact-{cls.__name__} tuple")
    return values  # type: ignore[return-value]


def _unique_text(values: tuple[str, ...], label: str) -> tuple[str, ...]:
    _require(type(values) is tuple, f"{label} must be an immutable tuple")
    for value in values:
        require_text(value, label)
    _require(len(values) == len(set(values)), f"{label} must be ordered and unique")
    return values


def _unique_sha(values: tuple[str, ...], label: str) -> tuple[str, ...]:
    _require(type(values) is tuple, f"{label} must be an immutable tuple")
    for value in values:
        require_sha256(value, label)
    _require(len(values) == len(set(values)), f"{label} must be ordered and unique")
    return values


def _seal(kind: str, body: Mapping[str, Any]) -> str:
    value = {"format": f"{FORMAT}-{kind}", **body}
    assert_gold_blind(value, path="locked_source_gate_adapter")
    return identity_sha256(value)


@dataclass(frozen=True, slots=True)
class LockedSourceGatePins:
    retrieval_path: Path = DEFAULT_RETRIEVAL
    store_root: Path = DEFAULT_STORE_ROOT
    query_root: Path = DEFAULT_QUERY_ROOT
    query_repack_root: Path = DEFAULT_QUERY_REPACK_ROOT
    eligibility_path: Path = DEFAULT_ELIGIBILITY
    partition_generation_path: Path = DEFAULT_PARTITION_GENERATION
    guided_root: Path = DEFAULT_GUIDED_ROOT
    retrieval_sha256: str = EXPECTED_RETRIEVAL_SHA256
    source_population_id: str = EXPECTED_SOURCE_POPULATION_ID
    query_preflight_sha256: str = EXPECTED_QUERY_PREFLIGHT_SHA256
    query_run_sha256: str = EXPECTED_QUERY_RUN_SHA256
    query_runtime_sha256: str = EXPECTED_QUERY_RUNTIME_SHA256
    query_repack_run_sha256: str = EXPECTED_QUERY_REPACK_RUN_SHA256
    query_repack_runtime_sha256: str = EXPECTED_QUERY_REPACK_RUNTIME_SHA256
    query_population_id: str = EXPECTED_QUERY_POPULATION_ID
    query_prompt_population_sha256: str = EXPECTED_QUERY_PROMPT_POPULATION_SHA256
    eligibility_sha256: str = EXPECTED_ELIGIBILITY_SHA256
    partition_generation_sha256: str = EXPECTED_PARTITION_GENERATION_SHA256
    guided_run_sha256: str = EXPECTED_GUIDED_RUN_SHA256
    guided_runtime_sha256: str = EXPECTED_GUIDED_RUNTIME_SHA256
    expected_question_count: int = 100
    expected_eligible_count: int = 79

    def __post_init__(self) -> None:
        for value, label in (
            (self.retrieval_sha256, "retrieval"), (self.source_population_id, "source population"),
            (self.query_preflight_sha256, "query preflight"), (self.query_run_sha256, "query run"),
            (self.query_runtime_sha256, "query runtime"), (self.query_population_id, "query population"),
            (self.query_repack_run_sha256, "query repack run"),
            (self.query_repack_runtime_sha256, "query repack runtime"),
            (self.query_prompt_population_sha256, "query prompt population"), (self.eligibility_sha256, "eligibility"),
            (self.partition_generation_sha256, "partition r96 generation"), (self.guided_run_sha256, "guided run"),
            (self.guided_runtime_sha256, "guided runtime"),
        ):
            require_sha256(value, f"pinned {label} SHA-256")
        _require(type(self.expected_question_count) is int and self.expected_question_count > 0, "pinned question count changed")
        _require(type(self.expected_eligible_count) is int and 0 <= self.expected_eligible_count <= self.expected_question_count, "pinned eligible count changed")
        for value in (self.retrieval_path, self.store_root, self.query_root, self.query_repack_root, self.eligibility_path, self.partition_generation_path, self.guided_root):
            _require(isinstance(value, Path), "pinned paths must be Path values")


@dataclass(frozen=True, slots=True)
class LockedSourceGateActivationInput:
    question_id: str
    source_packet_id: str
    map_packet_id: str
    as_of_turn: int
    upstream_question_plan_receipt_sha256: str
    upstream_fact_frontier_receipt_sha256: str
    obligation_ids: tuple[str, ...]
    unresolved_obligations: tuple[QuestionObligation, ...]

    def __post_init__(self) -> None:
        require_text(self.question_id, "activation input question ID")
        require_sha256(self.source_packet_id, "activation input source packet")
        require_sha256(self.map_packet_id, "activation input map packet")
        _require(self.source_packet_id != self.map_packet_id, "activation input collapsed source and map packets")
        _require(type(self.as_of_turn) is int and self.as_of_turn >= 0, "activation input as-of turn changed")
        require_sha256(self.upstream_question_plan_receipt_sha256, "activation input question plan")
        require_sha256(self.upstream_fact_frontier_receipt_sha256, "activation input fact frontier")
        _unique_sha(self.obligation_ids, "activation input obligation IDs")
        _typed(self.unresolved_obligations, QuestionObligation, "activation input unresolved obligations")
        _require(bool(self.unresolved_obligations), "locked source gate is only built for unresolved obligations")
        unresolved = tuple(row.obligation_id for row in self.unresolved_obligations)
        _require(len(set(unresolved)) == len(unresolved) and set(unresolved) <= set(self.obligation_ids), "activation unresolved obligations escaped/repeated upstream IDs")

    def projection(self) -> dict[str, Any]:
        return {
            "as_of_turn": self.as_of_turn,
            "format": f"{FORMAT}-activation-input",
            "map_packet_id": self.map_packet_id,
            "obligation_ids": list(self.obligation_ids),
            "question_id": self.question_id,
            "source_packet_id": self.source_packet_id,
            "unresolved_obligation_ids": [row.obligation_id for row in self.unresolved_obligations],
            "upstream_fact_frontier_receipt_sha256": self.upstream_fact_frontier_receipt_sha256,
            "upstream_question_plan_receipt_sha256": self.upstream_question_plan_receipt_sha256,
        }

    @property
    def receipt_sha256(self) -> str:
        return identity_sha256(self.projection())


def locked_activation_input_from_query_map_adapter(
    row: object,
    *,
    as_of_turn: int,
) -> LockedSourceGateActivationInput:
    """Preserve both packet generations at the post-map/locked-source join."""

    from .query_map_source_gate_adapter import QueryMapSourceGateAdapterRow

    if type(row) is not QueryMapSourceGateAdapterRow:
        raise TypeError("row must be an exact QueryMapSourceGateAdapterRow")
    _require(row.activation is not None, "locked source gate requires an activated map row")
    _require(type(as_of_turn) is int and as_of_turn >= 0, "activation as-of turn changed")
    unresolved = set(row.unresolved_obligation_ids)
    obligations = tuple(
        obligation for obligation in row.obligations if obligation.obligation_id in unresolved
    )
    _require(
        tuple(item.obligation_id for item in obligations) == row.unresolved_obligation_ids,
        "activated map row lost unresolved obligation order",
    )
    return LockedSourceGateActivationInput(
        row.question_id,
        row.source_packet_id,
        row.map_packet_id,
        as_of_turn,
        row.upstream_question_plan_receipt_sha256,
        row.upstream_fact_frontier_receipt_sha256,
        tuple(item.obligation_id for item in row.obligations),
        obligations,
    )


@dataclass(frozen=True, slots=True)
class LockedLaneSourceStream:
    lane: FactLane
    source_ids: tuple[str, ...]
    receipt_sha256: str

    def __post_init__(self) -> None:
        _require(self.lane in SOURCE_GATE_LANES, "locked source stream lane changed")
        _unique_text(self.source_ids, "locked lane source IDs")
        require_sha256(self.receipt_sha256, "locked lane source-stream receipt")


@dataclass(frozen=True, slots=True)
class VerifiedLockedSourceGateRow:
    ordinal: int
    question_id: str
    question_sha256: str
    dated_question: str
    dated_question_sha256: str
    source_packet_id: str
    population_identity_sha256: str
    question_order_sha256: str
    snapshot_id: str
    namespace: FrozenSourceNamespace
    direct_evidence: tuple[DirectEvidenceRef, ...]
    lane_streams: tuple[LockedLaneSourceStream, ...]
    store_dir: Path
    database_sha256: str
    index_sha256: str

    def __post_init__(self) -> None:
        _require(type(self.ordinal) is int and self.ordinal >= 0, "verified row ordinal changed")
        require_text(self.question_id, "verified row question ID")
        require_text(self.dated_question, "verified row dated question")
        for value, label in (
            (self.question_sha256, "question"), (self.dated_question_sha256, "dated question"),
            (self.source_packet_id, "source packet"), (self.population_identity_sha256, "population"),
            (self.question_order_sha256, "question order"), (self.snapshot_id, "snapshot"),
            (self.database_sha256, "store database"), (self.index_sha256, "store index"),
        ):
            require_sha256(value, f"verified row {label}")
        _require(quote_sha256(self.dated_question) == self.dated_question_sha256, "verified row dated question text changed")
        _require(type(self.namespace) is FrozenSourceNamespace, "verified row namespace must be exact")
        _require(self.snapshot_id == self.namespace.snapshot_id, "verified row snapshot/namespace binding changed")
        _typed(self.direct_evidence, DirectEvidenceRef, "verified row direct evidence")
        _typed(self.lane_streams, LockedLaneSourceStream, "verified row lane streams")
        _require(tuple(row.lane for row in self.lane_streams) == SOURCE_GATE_LANES, "verified row lane streams changed canonical order")
        _require(isinstance(self.store_dir, Path) and self.store_dir.is_dir(), "verified row store directory changed")
        membership_ids = {row.source_id for row in self.namespace.sources}
        _require(all(row.namespace_id == self.namespace.namespace_id and row.source_id in membership_ids for row in self.direct_evidence), "direct evidence escaped namespaced membership")
        _require(all(source_id in membership_ids for stream in self.lane_streams for source_id in stream.source_ids), "selected source lacks sealed namespaced membership")


@dataclass(frozen=True, slots=True)
class LockedSourceHydrationInput:
    namespace_id: str
    store_dir: Path
    database_sha256: str
    index_sha256: str
    memberships: tuple[FrozenSourceMembership, ...]

    def __post_init__(self) -> None:
        require_sha256(self.namespace_id, "hydration namespace")
        require_sha256(self.database_sha256, "hydration database")
        require_sha256(self.index_sha256, "hydration index")
        _typed(self.memberships, FrozenSourceMembership, "hydration memberships")
        _require(bool(self.memberships) and len({row.source_id for row in self.memberships}) == len(self.memberships), "hydration memberships are empty or repeated")
        _require(isinstance(self.store_dir, Path) and self.store_dir.is_dir(), "hydration store directory changed")

    @property
    def receipt_sha256(self) -> str:
        return _seal("hydration-input", {
            "database_sha256": self.database_sha256,
            "index_sha256": self.index_sha256,
            "memberships": [row.projection() for row in self.memberships],
            "namespace_id": self.namespace_id,
            "provider_calls": 0,
            "store_dir": str(self.store_dir),
        })

    def revalidate_store_bytes(self) -> None:
        for name, expected, label in (("memory.db", self.database_sha256, "database"), ("hnsw_index.bin", self.index_sha256, "index")):
            path = self.store_dir / name
            _require(path.is_file() and not path.is_symlink() and file_sha256(path) == expected, f"locked hydration store {label} changed")

    def open_read_only_database(self) -> Database:
        self.revalidate_store_bytes()
        return Database(self.store_dir / "memory.db", read_only=True)


@dataclass(frozen=True, slots=True)
class LockedSourceGateQuestion:
    ordinal: int
    plan: SourceGatePlan
    source_packet_id: str
    activation_input_receipt_sha256: str
    namespace: FrozenSourceNamespace
    direct_evidence: tuple[DirectEvidenceRef, ...]
    store_dir: Path
    database_sha256: str
    index_sha256: str

    def __post_init__(self) -> None:
        require_sha256(self.source_packet_id, "locked question source packet")
        require_sha256(self.activation_input_receipt_sha256, "locked question activation input")
        _require(
            self.source_packet_id != self.plan.parent.parent_packet_id
            and self.plan.activation.parent_packet_id == self.plan.parent.parent_packet_id,
            "locked question collapsed or changed source/map packet binding",
        )

    def packet_binding_projection(self) -> dict[str, str]:
        return {
            "activation_input_receipt_sha256": self.activation_input_receipt_sha256,
            "map_packet_id": self.plan.parent.parent_packet_id,
            "source_packet_id": self.source_packet_id,
        }

    def hydration_input(self, round_plan: SourceGateRound) -> LockedSourceHydrationInput:
        _require(type(round_plan) is SourceGateRound and round_plan.gate_plan_receipt_sha256 == self.plan.receipt_sha256, "hydration round escaped locked source-gate plan")
        source_ids = tuple(dict.fromkeys(row.source_id for row in round_plan.selections))
        _require(bool(source_ids), "cannot hydrate an empty source-gate round")
        by_source = {row.source_id: row for row in self.namespace.sources}
        _require(all(value in by_source for value in source_ids), "hydration selection escaped frozen namespace")
        return LockedSourceHydrationInput(self.namespace.namespace_id, self.store_dir, self.database_sha256, self.index_sha256, tuple(by_source[value] for value in source_ids))


@dataclass(frozen=True, slots=True)
class LockedSourceGateAdapterPopulation:
    source_artifacts: tuple[ArtifactRef, ...]
    questions: tuple[LockedSourceGateQuestion, ...]
    direct_stream_profile: str = DIRECT_STREAM_PROFILE_V1

    def __post_init__(self) -> None:
        _typed(self.source_artifacts, ArtifactRef, "locked source-gate artifacts")
        _typed(self.questions, LockedSourceGateQuestion, "locked source-gate questions")
        _require(bool(self.source_artifacts) and bool(self.questions), "locked source-gate adapter requires artifacts and activated questions")
        _require(len({row.role for row in self.source_artifacts}) == len(self.source_artifacts), "locked source-gate artifact roles repeat")
        _require(tuple(row.ordinal for row in self.questions) == tuple(sorted({row.ordinal for row in self.questions})), "activated question order changed")
        _require(
            self.direct_stream_profile in DIRECT_STREAM_PROFILES,
            "locked direct stream profile changed",
        )

    @property
    def direct_stream_profile_receipt_sha256(self) -> str:
        """Identify the selected direct frontier without changing V1 receipts."""

        direct_roles = tuple(
            row.projection()
            for row in self.source_artifacts
            if row.role in {"query_run", "query_repack_v2_run", "query_repack_v2_runtime"}
        )
        return _seal("direct-stream-profile", {
            "direct_stream_profile": self.direct_stream_profile,
            "source_artifacts": list(direct_roles),
        })

    @property
    def receipt_sha256(self) -> str:
        body: dict[str, Any] = {
            "activation_packet_bindings": [row.packet_binding_projection() for row in self.questions],
            "activated_question_count": len(self.questions),
            "gold_loaded": False,
            "provider_calls": 0,
            "question_plan_receipt_sha256s": [row.plan.receipt_sha256 for row in self.questions],
            "retained_transformer_token_state_bytes": 0,
            "source_artifacts": [row.projection() for row in self.source_artifacts],
        }
        if self.direct_stream_profile != DIRECT_STREAM_PROFILE_V1:
            body["direct_stream_profile"] = self.direct_stream_profile
            body["direct_stream_profile_receipt_sha256"] = (
                self.direct_stream_profile_receipt_sha256
            )
        return _seal("population", body)


def _membership_candidates(row: VerifiedLockedSourceGateRow) -> tuple[SourceGateCandidate, ...]:
    by_source = {value.source_id: value for value in row.namespace.sources}
    result: list[SourceGateCandidate] = []
    for stream in row.lane_streams:
        for rank, source_id in enumerate(stream.source_ids):
            membership = by_source.get(source_id)
            _require(membership is not None, "selected source lost its frozen membership")
            result.append(SourceGateCandidate(stream.lane, row.namespace.namespace_id, source_id, rank, identity_sha256(membership.projection()), membership.stream_sha256, stream.receipt_sha256))
    return tuple(result)


def build_locked_source_gate_adapter(
    rows: tuple[VerifiedLockedSourceGateRow, ...],
    activations: tuple[LockedSourceGateActivationInput, ...],
    *,
    source_artifacts: tuple[ArtifactRef, ...],
    policy: SourceGatePolicy | None = None,
    direct_stream_profile: str = DIRECT_STREAM_PROFILE_V1,
) -> LockedSourceGateAdapterPopulation:
    """Build activated plans from an already verified, provider-free source plane."""
    _typed(rows, VerifiedLockedSourceGateRow, "verified locked rows")
    _typed(activations, LockedSourceGateActivationInput, "locked activations")
    _typed(source_artifacts, ArtifactRef, "locked source artifacts")
    _require(
        direct_stream_profile in DIRECT_STREAM_PROFILES,
        "locked direct stream profile changed",
    )
    _require(bool(rows) and bool(activations), "locked source gate requires verified rows and unresolved activations")
    _require(tuple(row.ordinal for row in rows) == tuple(range(len(rows))) and len({row.question_id for row in rows}) == len(rows), "verified locked question population changed")
    activation_by_id = {row.question_id: row for row in activations}
    _require(len(activation_by_id) == len(activations) and set(activation_by_id) <= {row.question_id for row in rows}, "activation question IDs repeat or escape locked population")
    gate_policy = policy or default_source_gate_policy()
    questions: list[LockedSourceGateQuestion] = []
    for row in rows:
        activation_input = activation_by_id.get(row.question_id)
        if activation_input is None:
            continue
        _require(activation_input.source_packet_id == row.source_packet_id, "activation source packet escaped locked source row")
        candidates = _membership_candidates(row)
        activation = SourceGateActivationReceipt(
            row.question_id, row.question_sha256, row.dated_question_sha256,
            activation_input.map_packet_id, activation_input.upstream_question_plan_receipt_sha256,
            activation_input.upstream_fact_frontier_receipt_sha256, activation_input.obligation_ids,
            tuple(value.obligation_id for value in activation_input.unresolved_obligations),
        )
        parent = ParentIdentity(
            row.population_identity_sha256, row.question_order_sha256, row.snapshot_id,
            row.namespace.namespace_id, activation_input.map_packet_id,
            activation_input.upstream_fact_frontier_receipt_sha256,
            direct_evidence_projection_sha256(row.direct_evidence),
        )
        frontier = EligibleFrontierScope(tuple(value.candidate_id for value in candidates), False, _seal("ranked-non-exhaustive-frontier", {
            "lane_stream_receipt_sha256s": [value.receipt_sha256 for value in row.lane_streams],
            "question_id": row.question_id,
        }))
        plan = SourceGatePlan(
            parent, row.question_id, row.question_sha256, row.dated_question,
            row.dated_question_sha256, activation_input.as_of_turn, route_question(row.dated_question),
            source_artifacts + (
                ArtifactRef("store_database", row.database_sha256, str(row.store_dir / "memory.db")),
                ArtifactRef("store_index", row.index_sha256, str(row.store_dir / "hnsw_index.bin")),
            ),
            candidates, activation_input.unresolved_obligations, activation, frontier, gate_policy,
        )
        questions.append(LockedSourceGateQuestion(row.ordinal, plan, row.source_packet_id, activation_input.receipt_sha256, row.namespace, row.direct_evidence, row.store_dir, row.database_sha256, row.index_sha256))
    result = LockedSourceGateAdapterPopulation(
        source_artifacts,
        tuple(questions),
        direct_stream_profile,
    )
    assert_gold_blind({"receipt_sha256": result.receipt_sha256}, path="locked_source_gate_adapter_population")
    return result


def _collapse_source_ids(ids: Sequence[str]) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()
    for value in ids:
        require_text(value, "selected source ID")
        if value not in seen:
            seen.add(value)
            result.append(value)
    return tuple(result)


def project_locked_lane_source_stream(lane: FactLane, source_ids: Sequence[str], *, row_receipt: str, selected_ids: Sequence[str], artifact_sha256: str) -> LockedLaneSourceStream:
    """Seal first-distinct source order while retaining the exact span selection."""
    _require(lane in SOURCE_GATE_LANES, "locked source stream lane changed")
    require_sha256(row_receipt, "locked source stream row receipt")
    require_sha256(artifact_sha256, "locked source stream artifact")
    selected = tuple(selected_ids)
    _unique_text(selected, "locked selected candidate IDs")
    sources = _collapse_source_ids(source_ids)
    return LockedLaneSourceStream(lane, sources, _seal("lane-source-stream", {
        "artifact_sha256": artifact_sha256,
        "lane": lane.value,
        "row_receipt_sha256": row_receipt,
        "selected_candidate_ids": list(selected),
        "source_ids": list(sources),
    }))


def _candidate_sources(raw: Mapping[str, Any], selected_ids: Sequence[str], label: str) -> tuple[str, ...]:
    candidate_ids, candidates = raw.get("candidate_ids"), raw.get("candidates")
    _require(type(candidate_ids) is list and type(candidates) is list and len(candidate_ids) == len(candidates), f"{label} candidate catalog changed")
    _require(all(type(row) is dict for row in candidates), f"{label} candidate projections changed")
    _require(all(type(value) is str and value.strip() for value in candidate_ids), f"{label} candidate IDs changed")
    _require(len(candidate_ids) == len(set(candidate_ids)), f"{label} candidate IDs repeat")
    selected = tuple(selected_ids)
    _unique_text(selected, f"{label} selected candidate IDs")
    pairs = tuple((row.get("evidence_id"), row.get("source_id")) for row in candidates)
    _require(
        all(type(evidence_id) is str and evidence_id.strip() and type(source_id) is str and source_id.strip() for evidence_id, source_id in pairs),
        f"{label} candidate identity/source changed",
    )
    by_id = dict(pairs)
    _require(tuple(by_id) == tuple(candidate_ids) and all(value in by_id for value in selected), f"{label} selected candidates escaped exact catalog")
    return tuple(by_id[value] for value in selected)


def _repack_direct_frontier(
    raw: Mapping[str, Any],
    *,
    ordinal: int,
    prompt: Any,
    parent_row_receipt_sha256: str,
) -> tuple[tuple[str, ...], tuple[str, ...], str]:
    """Read only sealed candidate/source coordinates from a replayed V2 row."""

    _require(type(raw) is dict, "query repack row changed")
    unsigned = dict(raw)
    row_receipt = unsigned.pop("receipt_sha256", None)
    _require(
        require_sha256(row_receipt, "query repack row receipt")
        == identity_sha256(unsigned),
        "query repack row self-seal changed",
    )
    packet = prompt.source.packet
    _require(
        raw.get("format") == QUERY_REPACK_ROW_FORMAT
        and raw.get("ordinal") == ordinal
        and raw.get("question_id") == packet.question_id
        and raw.get("question_sha256") == packet.question_sha256
        and raw.get("dated_question_sha256") == packet.dated_question_sha256
        and raw.get("parent_packet_id") == packet.packet_id
        and raw.get("namespace_id") == prompt.namespace.namespace_id
        and raw.get("parent_row_receipt_sha256")
        == parent_row_receipt_sha256
        and raw.get("provider_calls") == 0
        and raw.get("retrieval_rerun") is False
        and raw.get("retained_transformer_token_state_bytes") == 0,
        "query repack row changed its parent/namespace/provider binding",
    )
    candidate_ids_raw = raw.get("candidate_ids")
    selected_ids_raw = raw.get("selected_before_dedup_candidate_ids")
    metadata = raw.get("candidate_metadata")
    _require(
        type(candidate_ids_raw) is list
        and type(selected_ids_raw) is list
        and type(metadata) is list
        and len(candidate_ids_raw) == len(metadata)
        and all(type(row) is dict for row in metadata),
        "query repack sealed candidate catalog changed",
    )
    candidate_ids = tuple(candidate_ids_raw)
    selected_ids = tuple(selected_ids_raw)
    _unique_sha(candidate_ids, "query repack candidate IDs")
    _unique_sha(selected_ids, "query repack selected-before-dedup IDs")
    _require(
        all(value in set(candidate_ids) for value in selected_ids),
        "query repack selection escaped sealed candidate IDs",
    )
    metadata_ids = tuple(row.get("candidate_id") for row in metadata)
    _require(
        metadata_ids == candidate_ids,
        "query repack candidate metadata order changed",
    )
    source_ids: list[str] = []
    by_id: dict[str, str] = {}
    for candidate_id, candidate in zip(candidate_ids, metadata, strict=True):
        source_id = candidate.get("source_id")
        require_text(source_id, "query repack candidate source ID")
        _require(
            candidate.get("namespace_id") == prompt.namespace.namespace_id,
            "query repack candidate escaped sealed namespace",
        )
        by_id[candidate_id] = source_id
    source_ids.extend(by_id[value] for value in selected_ids)
    collapsed_sources = _collapse_source_ids(source_ids)
    coverage = raw.get("source_membership_coverage")
    _require(type(coverage) is dict, "query repack source coverage changed")
    declared_sources = coverage.get("repack_selected_source_ids")
    _require(
        type(declared_sources) is list
        and tuple(declared_sources) == collapsed_sources
        and coverage.get("repack_selected_source_count")
        == len(collapsed_sources),
        "query repack selected source frontier changed",
    )
    return selected_ids, tuple(source_ids), row_receipt


def _direct_refs(namespace: FrozenSourceNamespace, direct_row: Any) -> tuple[DirectEvidenceRef, ...]:
    evidence = tuple(direct_row.source.packet.protected_evidence) + tuple(direct_row.admitted_delta)
    _require(len({row.evidence_id for row in evidence}) == len(evidence), "direct evidence IDs repeat")
    membership_ids = {row.source_id for row in namespace.sources}
    _require(all(row.source_id in membership_ids for row in evidence), "direct evidence escaped frozen namespace")
    return tuple(DirectEvidenceRef(
        evidence_id=row.evidence_id,
        namespace_id=namespace.namespace_id,
        source_id=row.source_id,
        quote_sha256=quote_sha256(row.text),
        evidence_receipt_sha256=_seal(
            "direct-evidence",
            {
                "evidence_id": row.evidence_id,
                "namespace_id": namespace.namespace_id,
                "source_id": row.source_id,
                "text_sha256": quote_sha256(row.text),
            },
        ),
        text=row.text,
    ) for row in evidence)


def _verify_eligibility(path: Path, expected_sha256: str, source_population: Any, generation: Any, expected_eligible_count: int) -> ArtifactRef:
    artifact = read_sealed_json(path)
    _require(artifact.sha256 == expected_sha256, "pinned eligibility artifact changed")
    payload, rows = artifact.payload, artifact.payload.get("questions")
    assert_gold_blind(payload, path="locked_source_gate_adapter.eligibility")
    body = dict(payload)
    declared = body.pop("manifest_identity_sha256", None)
    _require(require_sha256(declared, "eligibility self-seal") == identity_sha256(body), "eligibility self-seal changed")
    _require(
        payload.get("format") == ELIGIBILITY_FORMAT and payload.get("provider_calls") == 0
        and payload.get("gold_loaded") is False and payload.get("retrieval_sha256") == source_population.retrieval_sha256
        and payload.get("population_identity_sha256") == source_population.snapshot.population_identity_sha256
        and payload.get("question_count") == len(source_population.rows) and payload.get("eligible_question_count") == expected_eligible_count
        and type(rows) is list and len(rows) == len(generation.questions),
        "eligibility envelope changed",
    )
    _require(sum(raw.get("eligible") is True for raw in rows if type(raw) is dict) == expected_eligible_count, "eligibility row count changed")
    for ordinal, (raw, source, partition) in enumerate(zip(rows, source_population.rows, generation.questions, strict=True)):
        _require(type(raw) is dict, "eligibility row changed")
        unsigned = dict(raw)
        row_id = unsigned.pop("row_identity_sha256", None)
        _require(require_sha256(row_id, "eligibility row self-seal") == identity_sha256(unsigned), "eligibility row self-seal changed")
        _require(
            raw.get("ordinal") == ordinal
            and raw.get("question_id") == source.packet.question_id
            and raw.get("question_sha256") == source.packet.question_sha256
            and raw.get("dated_question") == source.packet.dated_question
            and raw.get("dated_question_sha256") == source.packet.dated_question_sha256
            and type(raw.get("eligible")) is bool
            and raw.get("eligible") is partition.eligible,
            "eligibility row binding changed",
        )
    return ArtifactRef("partition_eligibility", artifact.sha256, str(path))


def load_locked_source_gate_adapter(
    activations: tuple[LockedSourceGateActivationInput, ...],
    *,
    pins: LockedSourceGatePins = LockedSourceGatePins(),
    policy: SourceGatePolicy | None = None,
    direct_stream_profile: str = DIRECT_STREAM_PROFILE_V1,
) -> LockedSourceGateAdapterPopulation:
    """Load every pinned plane, revalidate stores, and build activated plans."""
    if type(pins) is not LockedSourceGatePins:
        raise TypeError("pins must be an exact LockedSourceGatePins")
    _typed(activations, LockedSourceGateActivationInput, "locked activations")
    _require(
        direct_stream_profile in DIRECT_STREAM_PROFILES,
        "locked direct stream profile changed",
    )
    population, preflight = load_preflighted_query_expansion_population(
        pins.retrieval_path, output_root=pins.query_root,
        expected_retrieval_sha256=pins.retrieval_sha256, expected_question_count=pins.expected_question_count,
    )
    _require(preflight.sha256 == pins.query_preflight_sha256 and population.source_population.population_id == pins.source_population_id and population.population_id == pins.query_population_id and population.prompt_population.prompt_population_sha256 == pins.query_prompt_population_sha256, "pinned query population changed")
    parent = verify_query_expansion_parent(
        population, parent_output_root=pins.query_root,
        expected_preflight_sha256=pins.query_preflight_sha256, expected_run_sha256=pins.query_run_sha256,
        expected_runtime_ledger_sha256=pins.query_runtime_sha256,
    )
    direct = build_query_fact_population(
        population.source_population, query_preflight=parent.preflight, query_run=parent.run,
        expected_retrieval_sha256=pins.retrieval_sha256, expected_source_population_id=pins.source_population_id,
        expected_query_preflight_sha256=pins.query_preflight_sha256, expected_query_run_sha256=pins.query_run_sha256,
        expected_query_population_id=pins.query_population_id, expected_query_prompt_population_sha256=pins.query_prompt_population_sha256,
    )
    context = load_locked_query_expansion_context(
        pins.retrieval_path, store_root=pins.store_root, expected_retrieval_sha256=pins.retrieval_sha256,
        expected_question_count=pins.expected_question_count, budget=population.budget, include_s0_evidence=population.include_s0_evidence,
    )
    _require(context.population.preflight_projection() == population.preflight_projection(), "sealed namespace population differs from revalidated stores")
    context.revalidate_store_bytes()
    repack_result = None
    if direct_stream_profile == DIRECT_STREAM_PROFILE_REPACK_V2:
        repack_result = replay_query_expansion_repack_v2(
            context,
            parent_output_root=pins.query_root,
            output_root=pins.query_repack_root,
            expected_parent_preflight_sha256=pins.query_preflight_sha256,
            expected_parent_run_sha256=pins.query_run_sha256,
            expected_parent_runtime_ledger_sha256=pins.query_runtime_sha256,
            expected_run_sha256=pins.query_repack_run_sha256,
        )
        _require(
            repack_result.run_artifact.sha256
            == pins.query_repack_run_sha256
            and repack_result.runtime_ledger_artifact.sha256
            == pins.query_repack_runtime_sha256
            and repack_result.physical_provider_calls == 0
            and repack_result.retained_transformer_token_state_bytes == 0,
            "pinned query repack replay/runtime binding changed",
        )
    partition = load_partition_scan_v2_generation(
        str(pins.partition_generation_path), expected_generation_sha256=pins.partition_generation_sha256,
        population=population.source_population, expected_eligibility_manifest_sha256=pins.eligibility_sha256,
    )
    eligibility_ref = _verify_eligibility(pins.eligibility_path, pins.eligibility_sha256, population.source_population, partition, pins.expected_eligible_count)
    guided_construction = verify_query_guided_construction(
        population, query_parent_root=pins.query_root, guided_root=pins.guided_root,
        expected_query_preflight_sha256=pins.query_preflight_sha256, expected_query_run_sha256=pins.query_run_sha256,
        expected_query_runtime_ledger_sha256=pins.query_runtime_sha256, expected_guided_run_sha256=pins.guided_run_sha256,
        expected_guided_runtime_ledger_sha256=pins.guided_runtime_sha256,
    )
    guided = build_query_guided_payload_adapter(population, guided_construction)
    guided_artifact = read_sealed_json(guided_construction.run_path)
    _require(guided_artifact.sha256 == pins.guided_run_sha256, "guided run changed after authoritative verification")
    guided_rows = guided_artifact.payload.get("questions")
    _require(type(guided_rows) is list and len(guided_rows) == pins.expected_question_count, "guided question population changed")
    artifacts = (
        ArtifactRef("sealed_retrieval", pins.retrieval_sha256, str(pins.retrieval_path)),
        ArtifactRef("query_preflight", pins.query_preflight_sha256, str(pins.query_root / "query-expansion-preflight.json")),
        ArtifactRef("query_run", pins.query_run_sha256, str(pins.query_root / "query-expansion-run.json")),
        ArtifactRef("query_runtime", pins.query_runtime_sha256, str(pins.query_root / "runtime-ledger.json")),
        eligibility_ref,
        ArtifactRef("partition_r96_generation", pins.partition_generation_sha256, str(pins.partition_generation_path)),
        ArtifactRef("guided_run", pins.guided_run_sha256, str(guided_construction.run_path)),
        ArtifactRef("guided_runtime", pins.guided_runtime_sha256, str(pins.guided_root / "runtime-ledger.json")),
    )
    repack_rows: tuple[Mapping[str, Any] | None, ...] = (
        (None,) * pins.expected_question_count
    )
    if repack_result is not None:
        raw_repack_rows = repack_result.run_artifact.payload.get("questions")
        _require(
            type(raw_repack_rows) is list
            and len(raw_repack_rows) == pins.expected_question_count
            and all(type(row) is dict for row in raw_repack_rows),
            "query repack question population changed after replay",
        )
        repack_rows = tuple(raw_repack_rows)
        artifacts += (
            ArtifactRef(
                "query_repack_v2_run",
                pins.query_repack_run_sha256,
                str(pins.query_repack_root / QUERY_REPACK_RUN_NAME),
            ),
            ArtifactRef(
                "query_repack_v2_runtime",
                pins.query_repack_runtime_sha256,
                str(
                    pins.query_repack_root
                    / QUERY_REPACK_RUNTIME_LEDGER_NAME
                ),
            ),
        )
    rows: list[VerifiedLockedSourceGateRow] = []
    source = population.source_population
    for ordinal, (prompt, direct_row, partition_row, guided_row, guided_adapter_row, repack_row) in enumerate(zip(population.rows, direct.rows, partition.questions, guided_rows, guided.rows, repack_rows, strict=True)):
        _require(type(guided_row) is dict and guided_row.get("receipt_sha256") == guided_adapter_row.query_row_receipt_sha256, "guided verified row receipt changed")
        namespace = prompt.namespace
        namespace_id = namespace.namespace_id
        _require(
            partition_row.source_database_sha256 == context.database_sha256_by_namespace[namespace_id]
            and partition_row.source_store_receipt_sha256 == namespace.combined_store_receipt_sha256,
            "partition r96 row changed its immutable store binding",
        )
        if direct_stream_profile == DIRECT_STREAM_PROFILE_V1:
            _require(
                repack_row is None,
                "V1 direct profile unexpectedly received a repack row",
            )
            direct_ids = tuple(
                row.evidence_id for row in direct_row.admitted_delta
            )
            direct_source_ids = tuple(
                row.source_id for row in direct_row.admitted_delta
            )
            direct_row_receipt = direct_row.query_row_receipt_sha256
            direct_artifact_sha256 = pins.query_run_sha256
        else:
            _require(
                repack_row is not None,
                "V2 direct profile lost its replayed repack row",
            )
            (
                direct_ids,
                direct_source_ids,
                direct_row_receipt,
            ) = _repack_direct_frontier(
                repack_row,
                ordinal=ordinal,
                prompt=prompt,
                parent_row_receipt_sha256=direct_row.query_row_receipt_sha256,
            )
            direct_artifact_sha256 = pins.query_repack_run_sha256
        direct_stream = project_locked_lane_source_stream(
            FactLane.DIRECT,
            direct_source_ids,
            row_receipt=direct_row_receipt,
            selected_ids=direct_ids,
            artifact_sha256=direct_artifact_sha256,
        )
        partition_ids = partition_row.trace.selected_before_dedup_ids
        partition_by_id = {row.evidence_id: row.source_id for row in partition_row.candidates}
        _require(all(value in partition_by_id for value in partition_ids), "partition selection escaped r96 candidate catalog")
        partition_stream = project_locked_lane_source_stream(FactLane.PARTITION, tuple(partition_by_id[value] for value in partition_ids), row_receipt=partition_row.question_identity_sha256, selected_ids=partition_ids, artifact_sha256=pins.partition_generation_sha256)
        guided_ids = tuple(guided_row.get("selected_before_dedup_candidate_ids", ()))
        _require(guided_ids == guided_adapter_row.selected_before_dedup_ids, "guided selected-before-dedup IDs changed")
        guided_stream = project_locked_lane_source_stream(FactLane.GUIDED, _candidate_sources(guided_row, guided_ids, f"guided row {ordinal}"), row_receipt=guided_adapter_row.query_row_receipt_sha256, selected_ids=guided_ids, artifact_sha256=pins.guided_run_sha256)
        store_dir = context.store_dirs_by_namespace[namespace_id]
        rows.append(VerifiedLockedSourceGateRow(
            ordinal, prompt.source.packet.question_id, prompt.source.packet.question_sha256,
            prompt.source.packet.dated_question, prompt.source.packet.dated_question_sha256,
            prompt.source.packet.packet_id, source.snapshot.population_identity_sha256,
            source.snapshot.question_order_sha256, source.snapshot.snapshot_id, namespace,
            _direct_refs(namespace, direct_row), (direct_stream, partition_stream, guided_stream),
            store_dir, context.database_sha256_by_namespace[namespace_id], context.index_sha256_by_namespace[namespace_id],
        ))
    context.revalidate_store_bytes()
    return build_locked_source_gate_adapter(
        tuple(rows),
        activations,
        source_artifacts=artifacts,
        policy=policy,
        direct_stream_profile=direct_stream_profile,
    )


__all__ = [
    "DEFAULT_CAMPAIGN_ROOT", "DEFAULT_ELIGIBILITY", "DEFAULT_GUIDED_ROOT",
    "DEFAULT_PARTITION_GENERATION", "DEFAULT_QUERY_REPACK_ROOT",
    "DEFAULT_QUERY_ROOT", "DEFAULT_RETRIEVAL", "DEFAULT_STORE_ROOT",
    "DIRECT_STREAM_PROFILE_REPACK_V2", "DIRECT_STREAM_PROFILE_V1",
    "DIRECT_STREAM_PROFILES", "EXPECTED_QUERY_REPACK_RUN_SHA256",
    "EXPECTED_QUERY_REPACK_RUNTIME_SHA256", "LockedLaneSourceStream",
    "LockedSourceGateActivationInput",
    "LockedSourceGateAdapterError", "LockedSourceGateAdapterPopulation",
    "LockedSourceGatePins", "LockedSourceGateQuestion", "LockedSourceHydrationInput",
    "VerifiedLockedSourceGateRow", "build_locked_source_gate_adapter",
    "load_locked_source_gate_adapter", "locked_activation_input_from_query_map_adapter",
    "project_locked_lane_source_stream",
]
