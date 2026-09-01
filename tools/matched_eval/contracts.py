"""Neutral contracts shared by matched retrieval mechanisms.

These types deliberately separate evidence membership, representation, linking,
answer policy, and observations.  A mechanism can therefore add a different
kind of value without pretending that every output is another evidence row.
All runtime objects are gold-blind by construction.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Literal, Mapping, Protocol, TypeAlias

from memory_condense.domain._tokenizer import count_tokens


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_FORBIDDEN_RUNTIME_KEYS = frozenset(
    {
        "answer",
        "answers",
        "baseline_correct",
        "baseline_judge_row_sha256",
        "benchmark_category",
        "category",
        "correct",
        "desired_answer",
        "evidence_topology_class",
        "expected_answer",
        "gold",
        "gold_answer",
        "gold_answer_sha256",
        "ground_truth",
        "ground_truth_answer",
        "judge_row_sha256",
        "judge_verdict_sha256",
        "primary_target_count",
        "primary_target_recalled",
        "question_only_demand_class",
        "reference",
        "reference_answer",
        "reference_answer_sha256",
        "regressed",
        "rescued",
        "target_owner",
        "verdict",
    }
)
_FALSE_GOLD_SENTINELS = frozenset(
    {
        "benchmark_categories_loaded",
        "benchmark_source_labels_loaded",
        "gold_fields_present",
        "gold_loaded",
    }
)


class MatchedEvalContractError(ValueError):
    """Raised when an evaluation-spine invariant is violated."""


def canonical_json_bytes(value: object) -> bytes:
    """Return the stable JSON representation used by the tool-only spine."""

    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def identity_sha256(value: object) -> str:
    # Artifact files include one terminal newline; in-memory behavior identities
    # follow the repository-wide discourse/runtime convention and do not.
    encoded = canonical_json_bytes(value)
    return hashlib.sha256(encoded[:-1]).hexdigest()


def require_sha256(value: str, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise MatchedEvalContractError(f"{label} must be a lowercase SHA-256 digest")
    return value


def require_text(value: str, label: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise MatchedEvalContractError(f"{label} must be non-empty exact text")
    return value


def _require_token_count(text: str, value: int, label: str) -> int:
    if type(value) is not int:
        raise MatchedEvalContractError(f"{label} must be an exact integer")
    expected = count_tokens(text)
    if value != expected:
        raise MatchedEvalContractError(
            f"{label} must equal the tokenizer count ({expected})"
        )
    return value


def _ordered_unique_ids(values: tuple[str, ...], label: str) -> tuple[str, ...]:
    if type(values) is not tuple:
        raise MatchedEvalContractError(f"{label} must be an immutable tuple")
    for value in values:
        require_text(value, label)
    if len(set(values)) != len(values):
        raise MatchedEvalContractError(f"{label} must be ordered and unique")
    return values


def _is_ordered_subsequence(
    values: tuple[str, ...], parent: tuple[str, ...]
) -> bool:
    iterator = iter(parent)
    return all(any(candidate == value for candidate in iterator) for value in values)


def assert_gold_blind(value: object, path: str = "runtime") -> None:
    """Reject gold, verdict, and post-hoc labels from a runtime projection."""

    if isinstance(value, Mapping):
        for key, child in value.items():
            name = str(key)
            normalized = name.casefold()
            child_path = f"{path}.{name}"
            if normalized in _FALSE_GOLD_SENTINELS:
                if child is not False:
                    raise MatchedEvalContractError(
                        f"gold firewall sentinel must be false: {child_path}"
                    )
                continue
            if normalized in _FORBIDDEN_RUNTIME_KEYS:
                raise MatchedEvalContractError(
                    f"gold-bearing field is forbidden at runtime: {child_path}"
                )
            assert_gold_blind(child, child_path)
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            assert_gold_blind(child, f"{path}[{index}]")


@dataclass(frozen=True, slots=True)
class ArtifactRef:
    role: str
    sha256: str
    path: str | None = None

    def __post_init__(self) -> None:
        require_text(self.role, "artifact role")
        require_sha256(self.sha256, f"{self.role} artifact SHA-256")
        if self.path is not None:
            require_text(self.path, f"{self.role} artifact path")

    def projection(self) -> dict[str, str]:
        result = {"role": self.role, "sha256": self.sha256}
        if self.path is not None:
            result["path"] = self.path
        return result


@dataclass(frozen=True, slots=True)
class EvaluationMemorySnapshot:
    """Immutable logical view over existing stores and sealed eval inputs."""

    population_identity_sha256: str
    question_order_sha256: str
    source_artifacts: tuple[ArtifactRef, ...]
    overlay_revisions: tuple[ArtifactRef, ...] = ()
    policy_id: str = "matched_eval_policy_v2"
    renderer_id: str = "matched_typed_slots_v2"
    implementation_id: str = "tools_matched_eval_v2"
    model_ids: tuple[str, ...] = ()
    reheat_memories: bool = False
    learn_consolidation: bool = False

    def __post_init__(self) -> None:
        require_sha256(self.population_identity_sha256, "population identity")
        require_sha256(self.question_order_sha256, "question order")
        require_text(self.policy_id, "policy ID")
        require_text(self.renderer_id, "renderer ID")
        require_text(self.implementation_id, "implementation ID")
        for values, label, expected_type in (
            (self.source_artifacts, "source artifacts", ArtifactRef),
            (self.overlay_revisions, "overlay revisions", ArtifactRef),
            (self.model_ids, "model IDs", str),
        ):
            if type(values) is not tuple or any(
                type(value) is not expected_type for value in values
            ):
                raise MatchedEvalContractError(
                    f"snapshot {label} must be an immutable typed tuple"
                )
        if not self.source_artifacts:
            raise MatchedEvalContractError("snapshot requires a source artifact")
        roles = tuple(row.role for row in self.source_artifacts + self.overlay_revisions)
        if len(set(roles)) != len(roles):
            raise MatchedEvalContractError("snapshot artifact roles must be unique")
        if type(self.reheat_memories) is not bool or type(self.learn_consolidation) is not bool:
            raise MatchedEvalContractError("snapshot learning flags must be exact bools")
        if self.reheat_memories or self.learn_consolidation:
            raise MatchedEvalContractError(
                "evaluation snapshots must disable reheat and consolidation learning"
            )
        for model_id in self.model_ids:
            require_text(model_id, "model ID")

    def projection(self) -> dict[str, object]:
        return {
            "format": "memory-condense-evaluation-memory-snapshot-v2",
            "implementation_id": self.implementation_id,
            "learn_consolidation": self.learn_consolidation,
            "model_ids": list(self.model_ids),
            "overlay_revisions": [row.projection() for row in self.overlay_revisions],
            "policy_id": self.policy_id,
            "population_identity_sha256": self.population_identity_sha256,
            "question_order_sha256": self.question_order_sha256,
            "reheat_memories": self.reheat_memories,
            "renderer_id": self.renderer_id,
            "source_artifacts": [row.projection() for row in self.source_artifacts],
        }

    @property
    def snapshot_id(self) -> str:
        projection = self.projection()
        assert_gold_blind(projection)
        return identity_sha256(projection)


@dataclass(frozen=True, slots=True)
class EvidenceItem:
    evidence_id: str
    source_id: str
    text: str
    token_count: int

    def __post_init__(self) -> None:
        require_text(self.evidence_id, "evidence ID")
        require_text(self.source_id, "evidence source ID")
        if type(self.text) is not str:
            raise MatchedEvalContractError("evidence text must be an exact string")
        _require_token_count(self.text, self.token_count, "evidence token count")


@dataclass(frozen=True, slots=True)
class FactItem:
    fact_id: str
    text: str
    source_evidence_ids: tuple[str, ...]
    token_count: int

    def __post_init__(self) -> None:
        require_text(self.fact_id, "fact ID")
        require_text(self.text, "fact text")
        _ordered_unique_ids(self.source_evidence_ids, "fact citations")
        if not self.source_evidence_ids:
            raise MatchedEvalContractError("facts require cited source evidence")
        _require_token_count(self.text, self.token_count, "fact token count")


@dataclass(frozen=True, slots=True)
class LinkItem:
    link_id: str
    text: str
    source_evidence_ids: tuple[str, ...]
    token_count: int

    def __post_init__(self) -> None:
        require_text(self.link_id, "link ID")
        require_text(self.text, "link text")
        _ordered_unique_ids(self.source_evidence_ids, "link bindings")
        if not self.source_evidence_ids:
            raise MatchedEvalContractError("links require bound source evidence")
        _require_token_count(self.text, self.token_count, "link token count")


class StageDisposition(str, Enum):
    ADDED = "added"
    NO_OP = "no_op"
    OVERFLOW = "overflow"
    INVALID = "invalid"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class StageTrace:
    """Gold-blind candidate-to-admission lifecycle for one stage/question."""

    candidate_ids: tuple[str, ...] = ()
    selected_before_dedup_ids: tuple[str, ...] = ()
    dedup_excluded_ids: tuple[str, ...] = ()
    not_admitted_ids: tuple[str, ...] = ()
    admitted_ids: tuple[str, ...] = ()
    token_cap: int = 0
    tokens_used: int = 0
    provider_prompt_count: int = 0
    disposition: StageDisposition = StageDisposition.NO_OP
    reason: str | None = None

    def __post_init__(self) -> None:
        if (
            type(self.token_cap) is not int
            or type(self.tokens_used) is not int
            or type(self.provider_prompt_count) is not int
            or self.token_cap < 0
            or self.tokens_used < 0
            or self.provider_prompt_count < 0
        ):
            raise MatchedEvalContractError("stage accounting cannot be negative")
        if self.tokens_used > self.token_cap:
            raise MatchedEvalContractError("stage trace exceeds its token cap")
        if type(self.disposition) is not StageDisposition:
            raise MatchedEvalContractError("stage disposition must be canonical")
        candidate = _ordered_unique_ids(self.candidate_ids, "candidate IDs")
        selected = _ordered_unique_ids(
            self.selected_before_dedup_ids, "selected IDs"
        )
        excluded = _ordered_unique_ids(self.dedup_excluded_ids, "dedup IDs")
        not_admitted = _ordered_unique_ids(
            self.not_admitted_ids, "not-admitted IDs"
        )
        admitted = _ordered_unique_ids(self.admitted_ids, "admitted IDs")
        if not _is_ordered_subsequence(selected, candidate):
            raise MatchedEvalContractError(
                "selected IDs must preserve candidate order"
            )
        if any(
            not _is_ordered_subsequence(values, selected)
            for values in (excluded, not_admitted, admitted)
        ):
            raise MatchedEvalContractError(
                "dedup/not-admitted/admitted IDs must preserve selection order"
            )
        partitions = (set(excluded), set(not_admitted), set(admitted))
        if any(
            left & right
            for index, left in enumerate(partitions)
            for right in partitions[index + 1 :]
        ) or set(selected) != set().union(*partitions):
            raise MatchedEvalContractError(
                "selected IDs must partition exactly into dedup, not-admitted, "
                "and admitted IDs"
            )
        if self.disposition is StageDisposition.ADDED and not admitted:
            raise MatchedEvalContractError(
                "an added stage must admit at least one selected item"
            )
        if self.disposition is not StageDisposition.ADDED and admitted:
            raise MatchedEvalContractError("a no-op stage cannot admit IDs")
        if self.reason is not None:
            require_text(self.reason, "stage disposition reason")


@dataclass(frozen=True, slots=True)
class _DeltaBase:
    stage_id: str
    parent_stage_id: str
    trace: StageTrace

    def __post_init__(self) -> None:
        require_text(self.stage_id, "stage ID")
        require_text(self.parent_stage_id, "parent stage ID")
        if self.stage_id == self.parent_stage_id:
            raise MatchedEvalContractError("a stage cannot parent itself")
        if type(self.trace) is not StageTrace:
            raise MatchedEvalContractError("delta trace must be an exact StageTrace")


@dataclass(frozen=True, slots=True)
class MembershipDelta(_DeltaBase):
    kind: Literal["membership"] = field(default="membership", init=False)
    dedup_alias_bindings: tuple[tuple[str, str], ...] = ()
    additions: tuple[EvidenceItem, ...] = ()

    def __post_init__(self) -> None:
        _DeltaBase.__post_init__(self)
        if type(self.dedup_alias_bindings) is not tuple or any(
            type(binding) is not tuple
            or len(binding) != 2
            or any(type(value) is not str for value in binding)
            for binding in self.dedup_alias_bindings
        ):
            raise MatchedEvalContractError(
                "membership dedup aliases must be immutable selected/protected ID pairs"
            )
        alias_ids: list[str] = []
        for selected_id, protected_id in self.dedup_alias_bindings:
            alias_ids.append(require_text(selected_id, "membership selected alias ID"))
            require_text(protected_id, "membership protected alias ID")
        if len(set(alias_ids)) != len(alias_ids):
            raise MatchedEvalContractError(
                "membership selected alias IDs must be unique"
            )
        if not _is_ordered_subsequence(
            tuple(alias_ids), self.trace.dedup_excluded_ids
        ):
            raise MatchedEvalContractError(
                "membership dedup aliases must preserve excluded selection order"
            )
        if type(self.additions) is not tuple or any(
            type(row) is not EvidenceItem for row in self.additions
        ):
            raise MatchedEvalContractError(
                "membership additions must be an immutable EvidenceItem tuple"
            )
        ids = tuple(row.evidence_id for row in self.additions)
        if len(set(ids)) != len(ids):
            raise MatchedEvalContractError("membership additions must be unique")
        if ids != self.trace.admitted_ids:
            raise MatchedEvalContractError("membership additions must match admitted IDs")


@dataclass(frozen=True, slots=True)
class RepresentationDelta(_DeltaBase):
    kind: Literal["representation"] = field(default="representation", init=False)
    dedup_against_evidence_ids: tuple[str, ...] = ()
    bound_evidence_ids: tuple[str, ...] = ()
    facts: tuple[FactItem, ...] = ()

    def __post_init__(self) -> None:
        _DeltaBase.__post_init__(self)
        if type(self.facts) is not tuple or any(
            type(row) is not FactItem for row in self.facts
        ):
            raise MatchedEvalContractError(
                "representation facts must be an immutable FactItem tuple"
            )
        ids = tuple(row.fact_id for row in self.facts)
        if len(set(ids)) != len(ids):
            raise MatchedEvalContractError("fact IDs must be unique")
        _ordered_unique_ids(
            self.dedup_against_evidence_ids,
            "representation dedup-basis evidence IDs",
        )
        bound_ids = _ordered_unique_ids(
            self.bound_evidence_ids, "representation bound evidence IDs"
        )
        if bound_ids != self.trace.admitted_ids:
            raise MatchedEvalContractError(
                "representation trace admissions must be the bound raw evidence IDs"
            )
        bound = set(bound_ids)
        if self.trace.disposition is StageDisposition.ADDED and not self.facts:
            raise MatchedEvalContractError("an added representation requires fact rows")
        if not bound and self.facts:
            raise MatchedEvalContractError("representation requires bound evidence")
        if any(not set(row.source_evidence_ids) <= bound for row in self.facts):
            raise MatchedEvalContractError("fact citations exceed bound evidence")


@dataclass(frozen=True, slots=True)
class LinkingDelta(_DeltaBase):
    kind: Literal["linking"] = field(default="linking", init=False)
    bound_evidence_ids: tuple[str, ...] = ()
    links: tuple[LinkItem, ...] = ()
    evidence_additions: tuple[EvidenceItem, ...] = field(default=(), init=False)

    def __post_init__(self) -> None:
        _DeltaBase.__post_init__(self)
        _ordered_unique_ids(self.bound_evidence_ids, "link bound evidence IDs")
        if type(self.links) is not tuple or any(
            type(row) is not LinkItem for row in self.links
        ):
            raise MatchedEvalContractError(
                "link rows must be an immutable LinkItem tuple"
            )
        ids = tuple(row.link_id for row in self.links)
        if len(set(ids)) != len(ids) or ids != self.trace.admitted_ids:
            raise MatchedEvalContractError("link rows must match unique admitted IDs")
        bound = set(self.bound_evidence_ids)
        if not bound and self.links:
            raise MatchedEvalContractError("linking requires bound evidence")
        if any(not set(row.source_evidence_ids) <= bound for row in self.links):
            raise MatchedEvalContractError("link bindings exceed bound evidence")


@dataclass(frozen=True, slots=True)
class AnswerOperatorDelta(_DeltaBase):
    kind: Literal["answer_operator"] = field(default="answer_operator", init=False)
    operator_id: str | None = None
    instructions: str | None = None

    def __post_init__(self) -> None:
        _DeltaBase.__post_init__(self)
        present = self.operator_id is not None or self.instructions is not None
        if present != (self.trace.disposition is StageDisposition.ADDED):
            raise MatchedEvalContractError("operator payload must match stage disposition")
        if present:
            require_text(self.operator_id or "", "operator ID")
            require_text(self.instructions or "", "operator instructions")
            if self.trace.admitted_ids != (self.operator_id,):
                raise MatchedEvalContractError("operator ID must be the admitted ID")


@dataclass(frozen=True, slots=True)
class ObservationDelta(_DeltaBase):
    kind: Literal["observation"] = field(default="observation", init=False)
    receipt_sha256: str | None = None

    def __post_init__(self) -> None:
        _DeltaBase.__post_init__(self)
        if self.trace.disposition is StageDisposition.ADDED:
            if self.receipt_sha256 is None:
                raise MatchedEvalContractError("an observation requires a receipt")
            require_sha256(self.receipt_sha256, "observation receipt")
        elif self.receipt_sha256 is not None:
            raise MatchedEvalContractError("a no-op observation cannot carry a receipt")


StageDelta: TypeAlias = (
    MembershipDelta
    | RepresentationDelta
    | LinkingDelta
    | AnswerOperatorDelta
    | ObservationDelta
)


@dataclass(frozen=True, slots=True)
class MemoryPacket:
    """Typed context packet rendered exactly once for a final answer."""

    question_id: str
    question_sha256: str
    dated_question: str
    dated_question_sha256: str
    stage_id: str
    protected_evidence: tuple[EvidenceItem, ...] = ()
    admitted_evidence: tuple[EvidenceItem, ...] = ()
    facts: tuple[FactItem, ...] = ()
    links: tuple[LinkItem, ...] = ()
    answer_operators: tuple[tuple[str, str], ...] = ()
    applied_stage_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        require_text(self.question_id, "question ID")
        require_sha256(self.question_sha256, "question SHA-256")
        require_text(self.dated_question, "dated question")
        require_sha256(self.dated_question_sha256, "dated question SHA-256")
        require_text(self.stage_id, "packet stage ID")
        for values, label, expected_type in (
            (self.protected_evidence, "protected evidence", EvidenceItem),
            (self.admitted_evidence, "admitted evidence", EvidenceItem),
            (self.facts, "facts", FactItem),
            (self.links, "links", LinkItem),
        ):
            if type(values) is not tuple or any(
                type(value) is not expected_type for value in values
            ):
                raise MatchedEvalContractError(
                    f"packet {label} must be an immutable typed tuple"
                )
        if type(self.answer_operators) is not tuple or any(
            type(row) is not tuple
            or len(row) != 2
            or any(type(value) is not str for value in row)
            for row in self.answer_operators
        ):
            raise MatchedEvalContractError(
                "packet answer operators must be immutable text pairs"
            )
        for operator_id, instructions in self.answer_operators:
            require_text(operator_id, "packet answer operator ID")
            require_text(instructions, "packet answer operator instructions")
        _ordered_unique_ids(self.applied_stage_ids, "applied stage IDs")
        evidence_ids = tuple(
            row.evidence_id for row in self.protected_evidence + self.admitted_evidence
        )
        if len(set(evidence_ids)) != len(evidence_ids):
            raise MatchedEvalContractError("packet evidence IDs must be unique")
        fact_ids = tuple(row.fact_id for row in self.facts)
        link_ids = tuple(row.link_id for row in self.links)
        operator_ids = tuple(row[0] for row in self.answer_operators)
        if len(set(fact_ids)) != len(fact_ids):
            raise MatchedEvalContractError("packet fact IDs must be unique")
        if len(set(link_ids)) != len(link_ids):
            raise MatchedEvalContractError("packet link IDs must be unique")
        if len(set(operator_ids)) != len(operator_ids):
            raise MatchedEvalContractError("packet operator IDs must be unique")
        if len(set(self.applied_stage_ids)) != len(self.applied_stage_ids):
            raise MatchedEvalContractError("applied stage IDs must be unique")

    @property
    def packet_id(self) -> str:
        projection = asdict(self)
        assert_gold_blind(projection)
        return identity_sha256(projection)


@dataclass(frozen=True, slots=True)
class StageBudget:
    token_cap: int
    provider_prompt_cap: int

    def __post_init__(self) -> None:
        if (
            type(self.token_cap) is not int
            or type(self.provider_prompt_cap) is not int
            or self.token_cap < 0
            or self.provider_prompt_cap < 0
        ):
            raise MatchedEvalContractError("stage budgets cannot be negative")


@dataclass(frozen=True, slots=True)
class StagePlan:
    stage_id: str
    parent_stage_id: str
    mechanism_id: str
    delta_kind: Literal[
        "membership", "representation", "linking", "answer_operator", "observation"
    ]
    budget: StageBudget

    def __post_init__(self) -> None:
        require_text(self.stage_id, "stage plan ID")
        require_text(self.parent_stage_id, "stage parent ID")
        require_text(self.mechanism_id, "mechanism ID")
        if self.delta_kind not in {
            "membership",
            "representation",
            "linking",
            "answer_operator",
            "observation",
        }:
            raise MatchedEvalContractError("stage plan has an unknown delta kind")
        if type(self.budget) is not StageBudget:
            raise MatchedEvalContractError("stage plan budget must be exact")
        if self.stage_id == self.parent_stage_id:
            raise MatchedEvalContractError("a stage cannot parent itself")


class PlanMode(str, Enum):
    ISOLATED = "isolated"
    CUMULATIVE = "cumulative"


@dataclass(frozen=True, slots=True)
class ArmPlan:
    plan_id: str
    mode: PlanMode
    root_stage_id: str
    stages: tuple[StagePlan, ...]
    global_provider_prompt_cap: int
    max_final_prompt_tokens: int = 8_000

    def __post_init__(self) -> None:
        require_text(self.plan_id, "plan ID")
        require_text(self.root_stage_id, "root stage ID")
        if type(self.mode) is not PlanMode:
            raise MatchedEvalContractError("plan mode must be canonical")
        if type(self.stages) is not tuple:
            raise MatchedEvalContractError("plan stages must be an immutable tuple")
        if (
            type(self.global_provider_prompt_cap) is not int
            or self.global_provider_prompt_cap < 0
        ):
            raise MatchedEvalContractError("global provider prompt cap cannot be negative")
        if (
            type(self.max_final_prompt_tokens) is not int
            or self.max_final_prompt_tokens < 1
        ):
            raise MatchedEvalContractError(
                "final prompt token cap must be a positive exact integer"
            )
        stage_ids = tuple(row.stage_id for row in self.stages)
        if len(set(stage_ids)) != len(stage_ids):
            raise MatchedEvalContractError("plan stage IDs must be unique")
        known = {self.root_stage_id}
        previous = self.root_stage_id
        for stage in self.stages:
            if stage.parent_stage_id not in known:
                raise MatchedEvalContractError("plan stages must be topologically ordered")
            if self.mode is PlanMode.ISOLATED and stage.parent_stage_id != self.root_stage_id:
                raise MatchedEvalContractError("isolated stages must start from the root")
            if self.mode is PlanMode.CUMULATIVE and stage.parent_stage_id != previous:
                raise MatchedEvalContractError("cumulative stages must form one ordered chain")
            known.add(stage.stage_id)
            previous = stage.stage_id


class MechanismAdapter(Protocol):
    """Provider-neutral mechanism boundary consumed by the common runner."""

    mechanism_id: str
    delta_kind: str

    def propose(
        self,
        *,
        snapshot: EvaluationMemorySnapshot,
        packet: MemoryPacket,
        stage: StagePlan,
    ) -> StageDelta: ...


def delta_projection(delta: StageDelta) -> dict[str, Any]:
    projection = asdict(delta)
    # The alias field was added after the v2 spine was sealed.  Omitting its
    # empty default keeps every existing delta/artifact identity byte-stable;
    # only mechanisms that need non-identity S0 dedup emit the new field.
    if isinstance(delta, MembershipDelta) and not delta.dedup_alias_bindings:
        projection.pop("dedup_alias_bindings")
    assert_gold_blind(projection)
    return projection
