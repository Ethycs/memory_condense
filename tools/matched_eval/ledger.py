"""Normalized runtime and post-hoc ledgers for matched mechanism runs.

The schema family is intentionally split into two sealed planes joined by an
opaque row ID.  Runtime artifacts stay gold-blind; correctness, ownership, and
topology labels exist only in the post-hoc score plane.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping

from .artifacts import read_sealed_json
from .contracts import (
    MatchedEvalContractError,
    StageDisposition,
    StageTrace,
    assert_gold_blind,
    identity_sha256,
    require_sha256,
    require_text,
)


RUNTIME_LEDGER_FORMAT = "memory-condense-matched-runtime-ledger-v2"
SCORE_LEDGER_FORMAT = "memory-condense-matched-score-ledger-v2"


def _exact_text(value: object, label: str) -> str:
    if type(value) is not str:
        raise MatchedEvalContractError(f"{label} must be exact text")
    return require_text(value, label)


def _exact_sha256(value: object, label: str) -> str:
    if type(value) is not str:
        raise MatchedEvalContractError(f"{label} must be an exact SHA-256 string")
    return require_sha256(value, label)


def _exact_nonnegative_int(value: object, label: str) -> int:
    if type(value) is not int or value < 0:
        raise MatchedEvalContractError(f"{label} must be a non-negative exact integer")
    return value


@dataclass(frozen=True, slots=True)
class RuntimeLedgerEntry:
    event_type: Literal["stage", "answer_observation"]
    ordinal: int
    question_id: str
    question_sha256: str
    arm_label: str
    parent_arm_label: str | None
    stage_id: str
    parent_stage_id: str | None
    mechanism_id: str
    delta_kind: str
    renderer_id: str
    legacy_renderer: bool
    disposition: StageDisposition
    candidate_ids: tuple[str, ...] = ()
    selected_before_dedup_ids: tuple[str, ...] = ()
    dedup_excluded_ids: tuple[str, ...] = ()
    not_admitted_ids: tuple[str, ...] = ()
    admitted_ids: tuple[str, ...] = ()
    token_cap: int = 0
    tokens_used: int = 0
    reported_tokens_used: int = 0
    local_model_calls: int = 0
    provider_calls: int = 0
    provider_prompt_cap: int = 0
    provider_prompt_reserved: int = 0
    global_provider_prompt_cap: int = 0
    historical_provider_calls: int = 0
    max_final_prompt_tokens: int | None = None
    prompt_token_proxy: int | None = None
    parent_packet_sha256: str | None = None
    packet_sha256: str | None = None
    prompt_id: str | None = None
    prompt_messages_sha256: str | None = None
    delta_sha256: str | None = None
    stage_receipt_sha256: str | None = None
    prediction: str | None = None
    prediction_sha256: str | None = None
    changed_from_parent: bool | None = None
    source_row_sha256: str | None = None
    reason: str | None = None

    def __post_init__(self) -> None:
        _exact_nonnegative_int(self.ordinal, "ledger ordinal")
        if type(self.event_type) is not str:
            raise MatchedEvalContractError("runtime event type must be exact text")
        if type(self.legacy_renderer) is not bool:
            raise MatchedEvalContractError("legacy renderer flag must be an exact bool")
        if type(self.disposition) is not StageDisposition:
            raise MatchedEvalContractError(
                "runtime disposition must be an exact StageDisposition"
            )
        for value, label in (
            (self.question_id, "question ID"),
            (self.arm_label, "arm label"),
            (self.stage_id, "stage ID"),
            (self.mechanism_id, "mechanism ID"),
            (self.delta_kind, "delta kind"),
            (self.renderer_id, "renderer ID"),
        ):
            _exact_text(value, label)
        _exact_sha256(self.question_sha256, "question SHA-256")
        for value, label in (
            (self.parent_packet_sha256, "parent packet SHA-256"),
            (self.packet_sha256, "packet SHA-256"),
            (self.prompt_id, "prompt ID"),
            (self.prompt_messages_sha256, "prompt messages SHA-256"),
            (self.delta_sha256, "delta SHA-256"),
            (self.stage_receipt_sha256, "stage receipt SHA-256"),
            (self.prediction_sha256, "prediction SHA-256"),
            (self.source_row_sha256, "source row SHA-256"),
        ):
            if value is not None:
                _exact_sha256(value, label)
        if self.parent_arm_label is not None:
            _exact_text(self.parent_arm_label, "parent arm label")
        if self.parent_stage_id is not None:
            _exact_text(self.parent_stage_id, "parent stage ID")
        if self.prediction is not None:
            _exact_text(self.prediction, "prediction")
        if (self.prediction is None) != (self.prediction_sha256 is None):
            raise MatchedEvalContractError(
                "prediction text and prediction digest must be present together"
            )
        if self.prediction is not None:
            observed_prediction_sha256 = hashlib.sha256(
                self.prediction.encode("utf-8")
            ).hexdigest()
            if observed_prediction_sha256 != self.prediction_sha256:
                raise MatchedEvalContractError(
                    "prediction SHA-256 does not match prediction text"
                )
        if self.changed_from_parent is not None and type(
            self.changed_from_parent
        ) is not bool:
            raise MatchedEvalContractError(
                "changed-from-parent flag must be an exact bool or null"
            )
        if self.reason is not None:
            _exact_text(self.reason, "ledger reason")
        for value, label in (
            (self.token_cap, "token cap"),
            (self.tokens_used, "tokens used"),
            (self.reported_tokens_used, "reported tokens used"),
            (self.local_model_calls, "local-model calls"),
            (self.provider_calls, "provider calls"),
            (self.provider_prompt_cap, "provider prompt cap"),
            (self.provider_prompt_reserved, "provider prompt reservation"),
            (self.global_provider_prompt_cap, "global provider prompt cap"),
            (self.historical_provider_calls, "historical provider calls"),
        ):
            _exact_nonnegative_int(value, label)
        for value, label in (
            (self.max_final_prompt_tokens, "maximum final prompt tokens"),
            (self.prompt_token_proxy, "prompt token proxy"),
        ):
            if value is not None:
                _exact_nonnegative_int(value, label)
        if self.max_final_prompt_tokens == 0:
            raise MatchedEvalContractError(
                "maximum final prompt tokens must be positive when present"
            )
        if (
            self.prompt_token_proxy is not None
            and self.max_final_prompt_tokens is not None
            and self.prompt_token_proxy > self.max_final_prompt_tokens
        ):
            raise MatchedEvalContractError("runtime prompt exceeds its final cap")
        if self.provider_prompt_reserved not in (0, self.provider_prompt_cap):
            raise MatchedEvalContractError(
                "provider prompt reservation must be zero or the full stage cap"
            )
        if self.provider_prompt_reserved > self.global_provider_prompt_cap:
            raise MatchedEvalContractError(
                "stage provider reservation exceeds its global cap"
            )
        for values, label in (
            (self.candidate_ids, "candidate IDs"),
            (self.selected_before_dedup_ids, "selected IDs"),
            (self.dedup_excluded_ids, "dedup-excluded IDs"),
            (self.not_admitted_ids, "not-admitted IDs"),
            (self.admitted_ids, "admitted IDs"),
        ):
            if type(values) is not tuple:
                raise MatchedEvalContractError(f"{label} must be an exact tuple")
            for value in values:
                _exact_text(value, label)
            if len(set(values)) != len(values):
                raise MatchedEvalContractError(f"{label} must be unique")
        if self.event_type == "answer_observation":
            if self.prediction is None:
                raise MatchedEvalContractError(
                    "an answer observation requires a prediction"
                )
            if any(
                (
                    self.candidate_ids,
                    self.selected_before_dedup_ids,
                    self.dedup_excluded_ids,
                    self.not_admitted_ids,
                    self.admitted_ids,
                )
            ):
                raise MatchedEvalContractError(
                    "an answer observation cannot invent a candidate lifecycle"
                )
        elif self.event_type != "stage":
            raise MatchedEvalContractError("unknown runtime ledger event type")
        StageTrace(
            candidate_ids=self.candidate_ids,
            selected_before_dedup_ids=self.selected_before_dedup_ids,
            dedup_excluded_ids=self.dedup_excluded_ids,
            not_admitted_ids=self.not_admitted_ids,
            admitted_ids=self.admitted_ids,
            token_cap=self.token_cap,
            tokens_used=self.tokens_used,
            provider_prompt_count=self.provider_calls,
            disposition=self.disposition,
            reason=self.reason,
        )
        assert_gold_blind(self.projection(include_row_id=False))

    def projection(self, *, include_row_id: bool = True) -> dict[str, Any]:
        result = asdict(self)
        result["disposition"] = self.disposition.value
        for key in (
            "candidate_ids",
            "selected_before_dedup_ids",
            "dedup_excluded_ids",
            "not_admitted_ids",
            "admitted_ids",
        ):
            result[key] = list(result[key])
        if include_row_id:
            result["row_id"] = identity_sha256(result)
        return result

    @property
    def row_id(self) -> str:
        return identity_sha256(self.projection(include_row_id=False))


def runtime_entry_from_stage_run(
    *,
    ordinal: int,
    arm_label: str,
    parent_arm_label: str | None,
    run: object,
    stage_id: str,
    local_model_calls: int = 0,
) -> RuntimeLedgerEntry:
    """Losslessly flatten one common-runner stage into the runtime plane."""

    # Local import keeps the contracts/runner/ledger dependency direction
    # acyclic while retaining an exact public conversion boundary.
    from .runner import ArmRunResult, StageRunResult

    if type(run) is not ArmRunResult:
        raise MatchedEvalContractError("stage ledger conversion requires ArmRunResult")
    try:
        stage = run.stage(stage_id)
    except KeyError as exc:
        raise MatchedEvalContractError(
            f"stage ledger conversion cannot find stage {stage_id!r}"
        ) from exc
    if type(stage) is not StageRunResult:
        raise MatchedEvalContractError("run stage must be an exact StageRunResult")
    receipt = stage.receipt
    trace = receipt.trace
    return RuntimeLedgerEntry(
        event_type="stage",
        ordinal=ordinal,
        question_id=stage.packet.question_id,
        question_sha256=stage.packet.question_sha256,
        arm_label=arm_label,
        parent_arm_label=parent_arm_label,
        stage_id=receipt.stage_id,
        parent_stage_id=receipt.parent_stage_id,
        mechanism_id=receipt.mechanism_id,
        delta_kind=receipt.delta_kind,
        renderer_id=receipt.renderer_id,
        legacy_renderer=False,
        disposition=trace.disposition,
        candidate_ids=trace.candidate_ids,
        selected_before_dedup_ids=trace.selected_before_dedup_ids,
        dedup_excluded_ids=trace.dedup_excluded_ids,
        not_admitted_ids=trace.not_admitted_ids,
        admitted_ids=trace.admitted_ids,
        token_cap=receipt.token_cap,
        tokens_used=trace.tokens_used,
        reported_tokens_used=receipt.reported_tokens_used,
        local_model_calls=local_model_calls,
        provider_calls=receipt.reported_provider_prompt_count,
        provider_prompt_cap=receipt.provider_prompt_cap,
        provider_prompt_reserved=receipt.provider_prompt_reserved,
        global_provider_prompt_cap=run.global_provider_prompt_cap,
        max_final_prompt_tokens=receipt.max_final_prompt_tokens,
        prompt_token_proxy=receipt.output_prompt_token_proxy,
        parent_packet_sha256=receipt.parent_packet_id,
        packet_sha256=receipt.output_packet_id,
        prompt_id=receipt.output_prompt_id,
        prompt_messages_sha256=receipt.output_prompt_messages_sha256,
        delta_sha256=receipt.delta_sha256,
        stage_receipt_sha256=receipt.receipt_sha256,
        reason=trace.reason,
    )


@dataclass(frozen=True, slots=True)
class RuntimeStageRunBinding:
    """One ordered common-runner result projected into the runtime plane."""

    ordinal: int
    arm_label: str
    parent_arm_label: str | None
    run: object
    stage_id: str
    local_model_calls: int = 0

    def __post_init__(self) -> None:
        _exact_nonnegative_int(self.ordinal, "runtime stage binding ordinal")
        _exact_text(self.arm_label, "runtime stage binding arm label")
        _exact_text(self.stage_id, "runtime stage binding stage ID")
        _exact_nonnegative_int(
            self.local_model_calls,
            "runtime stage binding local-model calls",
        )
        if self.parent_arm_label is not None:
            _exact_text(
                self.parent_arm_label,
                "runtime stage binding parent arm label",
            )


@dataclass(frozen=True, slots=True)
class VerifiedRuntimeAnswerPlane:
    """Pinned answer bytes joined to reconstructed common-runner stage rows."""

    answer_run_sha256: str
    runtime_ledger_sha256: str
    runtime_ledger_identity_sha256: str
    snapshot_id: str
    plan_id: str
    renderer_id: str
    answer_run_artifact_role: str
    entries: tuple[RuntimeLedgerEntry, ...]

    def __post_init__(self) -> None:
        for value, label in (
            (self.answer_run_sha256, "verified answer run"),
            (self.runtime_ledger_sha256, "verified runtime ledger"),
            (
                self.runtime_ledger_identity_sha256,
                "verified runtime ledger identity",
            ),
            (self.snapshot_id, "verified runtime snapshot"),
        ):
            _exact_sha256(value, label)
        for value, label in (
            (self.plan_id, "verified runtime plan"),
            (self.renderer_id, "verified runtime renderer"),
            (self.answer_run_artifact_role, "verified answer artifact role"),
        ):
            _exact_text(value, label)
        if type(self.entries) is not tuple or any(
            type(row) is not RuntimeLedgerEntry for row in self.entries
        ):
            raise MatchedEvalContractError(
                "verified runtime entries must be immutable exact values"
            )


def load_verified_runtime_answer_plane(
    *,
    answer_run_path: str | Path,
    answer_run_replay_path: str | Path,
    expected_answer_run_sha256: str,
    runtime_ledger_path: str | Path,
    runtime_ledger_replay_path: str | Path,
    expected_runtime_ledger_sha256: str,
    snapshot_id: str,
    plan_id: str,
    renderer_id: str,
    stage_runs: Iterable[RuntimeStageRunBinding],
    answer_run_artifact_role: str,
    source_artifacts: Iterable[Mapping[str, str]] = (),
    historical_shared_local_model_calls: int = 0,
) -> VerifiedRuntimeAnswerPlane:
    """Verify a sealed answer/runtime plane from canonical stage executions.

    The answer artifact uses only the common population envelope: arm,
    snapshot, renderer, question count, and ordered question identities.  Its
    mechanism-specific prediction fields remain opaque.  Stage rows are not
    trusted from disk.  They are reconstructed from exact ``ArmRunResult``
    values and must reproduce the sealed runtime ledger and its replay
    byte-for-byte.
    """

    expected_answer = _exact_sha256(
        expected_answer_run_sha256, "expected answer run"
    )
    expected_runtime = _exact_sha256(
        expected_runtime_ledger_sha256, "expected runtime ledger"
    )
    expected_snapshot = _exact_sha256(snapshot_id, "expected runtime snapshot")
    expected_plan = _exact_text(plan_id, "expected runtime plan")
    expected_renderer = _exact_text(renderer_id, "expected runtime renderer")
    answer_role = _exact_text(
        answer_run_artifact_role, "answer run artifact role"
    )
    _exact_nonnegative_int(
        historical_shared_local_model_calls,
        "historical shared local-model calls",
    )

    answer = read_sealed_json(answer_run_path)
    answer_replay = read_sealed_json(answer_run_replay_path)
    if (
        answer.sha256 != expected_answer
        or answer_replay.sha256 != expected_answer
        or answer.payload != answer_replay.payload
    ):
        raise MatchedEvalContractError("answer run/replay seals differ")
    if answer.payload.get("gold_loaded") is not False:
        raise MatchedEvalContractError(
            "answer run must explicitly remain gold-blind"
        )
    assert_gold_blind(answer.payload, path="verified_runtime_answer_run")

    bindings = tuple(stage_runs)
    if not bindings or any(
        type(binding) is not RuntimeStageRunBinding for binding in bindings
    ):
        raise MatchedEvalContractError(
            "runtime answer plane requires ordered exact stage bindings"
        )
    entries: list[RuntimeLedgerEntry] = []
    for binding in bindings:
        entry = runtime_entry_from_stage_run(
            ordinal=binding.ordinal,
            arm_label=binding.arm_label,
            parent_arm_label=binding.parent_arm_label,
            run=binding.run,
            stage_id=binding.stage_id,
            local_model_calls=binding.local_model_calls,
        )
        run = binding.run
        if (
            run.snapshot_id != expected_snapshot  # type: ignore[attr-defined]
            or run.plan_id != expected_plan  # type: ignore[attr-defined]
        ):
            raise MatchedEvalContractError(
                "runtime stage run changed snapshot or plan binding"
            )
        if entry.renderer_id != expected_renderer:
            raise MatchedEvalContractError(
                "runtime stage run changed renderer binding"
            )
        stage = run.stage(binding.stage_id)  # type: ignore[attr-defined]
        if (
            entry.stage_receipt_sha256 != stage.receipt.receipt_sha256
            or entry.parent_packet_sha256 != stage.receipt.parent_packet_id
            or entry.packet_sha256 != stage.packet.packet_id
            or entry.packet_sha256 != stage.receipt.output_packet_id
        ):
            raise MatchedEvalContractError(
                "runtime stage receipt or output packet binding changed"
            )
        entries.append(entry)

    ordered_entries = tuple(entries)
    ordinals = tuple(row.ordinal for row in ordered_entries)
    if ordinals != tuple(sorted(set(ordinals))):
        raise MatchedEvalContractError(
            "runtime answer stage bindings must preserve unique ordinal order"
        )
    question_ids = tuple(row.question_id for row in ordered_entries)
    if len(set(question_ids)) != len(question_ids):
        raise MatchedEvalContractError(
            "runtime answer stage bindings must have unique questions"
        )
    arm_labels = {row.arm_label for row in ordered_entries}
    answer_rows = answer.payload.get("questions")
    if (
        len(arm_labels) != 1
        or answer.payload.get("arm_label") != next(iter(arm_labels))
        or answer.payload.get("snapshot_id") != expected_snapshot
        or answer.payload.get("renderer_id") != expected_renderer
        or answer.payload.get("question_count") != len(ordered_entries)
        or type(answer_rows) is not list
        or len(answer_rows) != len(ordered_entries)
        or any(type(row) is not dict for row in answer_rows)
    ):
        raise MatchedEvalContractError(
            "answer run changed its runtime population envelope"
        )
    for entry, row in zip(ordered_entries, answer_rows, strict=True):
        if (
            row.get("ordinal") != entry.ordinal
            or row.get("question_id") != entry.question_id
            or row.get("question_sha256") != entry.question_sha256
        ):
            raise MatchedEvalContractError(
                f"answer run changed question binding at ordinal {entry.ordinal}"
            )

    runtime_sources = (
        *tuple(source_artifacts),
        {"role": answer_role, "sha256": expected_answer},
    )
    expected_ledger = build_runtime_ledger(
        snapshot_id=expected_snapshot,
        plan_id=expected_plan,
        entries=ordered_entries,
        source_artifacts=runtime_sources,
        historical_shared_local_model_calls=(
            historical_shared_local_model_calls
        ),
    )
    ledger = read_sealed_json(runtime_ledger_path)
    ledger_replay = read_sealed_json(runtime_ledger_replay_path)
    if (
        ledger.sha256 != expected_runtime
        or ledger_replay.sha256 != expected_runtime
        or ledger.payload != ledger_replay.payload
    ):
        raise MatchedEvalContractError("runtime ledger/replay seals differ")
    if ledger.payload != expected_ledger:
        raise MatchedEvalContractError(
            "runtime ledger differs from reconstructed stage executions"
        )
    runtime_identity, _answer_row_ids = _validated_runtime_ledger(ledger.payload)
    return VerifiedRuntimeAnswerPlane(
        answer_run_sha256=expected_answer,
        runtime_ledger_sha256=expected_runtime,
        runtime_ledger_identity_sha256=runtime_identity,
        snapshot_id=expected_snapshot,
        plan_id=expected_plan,
        renderer_id=expected_renderer,
        answer_run_artifact_role=answer_role,
        entries=ordered_entries,
    )


@dataclass(frozen=True, slots=True)
class ScoreLedgerEntry:
    runtime_row_id: str
    correct: bool
    baseline_correct: bool | None = None
    changed_from_baseline: bool | None = None
    rescued: bool | None = None
    regressed: bool | None = None
    question_only_demand_class: str | None = None
    evidence_topology_class: str | None = None
    primary_target_count: int | None = None
    primary_target_recalled: int | None = None
    judge_row_sha256: str | None = None
    judge_verdict_sha256: str | None = None
    baseline_judge_row_sha256: str | None = None
    historical_provider_calls: int = 0

    def __post_init__(self) -> None:
        _exact_sha256(self.runtime_row_id, "runtime row ID")
        if type(self.correct) is not bool:
            raise MatchedEvalContractError("correctness must be an exact bool")
        comparison = (
            self.baseline_correct,
            self.changed_from_baseline,
            self.rescued,
            self.regressed,
        )
        if any(value is None for value in comparison):
            if any(value is not None for value in comparison):
                raise MatchedEvalContractError(
                    "baseline correctness/change/rescue/regression must be "
                    "all present or all null"
                )
        else:
            if any(type(value) is not bool for value in comparison):
                raise MatchedEvalContractError(
                    "baseline correctness/change/rescue/regression must be exact bools"
                )
            baseline_correct = self.baseline_correct
            rescued = self.rescued
            regressed = self.regressed
            if rescued != (not baseline_correct and self.correct):
                raise MatchedEvalContractError("score rescue flag is inconsistent")
            if regressed != (baseline_correct and not self.correct):
                raise MatchedEvalContractError("score regression flag is inconsistent")
        if self.question_only_demand_class is not None:
            _exact_text(self.question_only_demand_class, "demand class")
        if self.evidence_topology_class is not None:
            _exact_text(self.evidence_topology_class, "topology class")
        for value, label in (
            (self.judge_row_sha256, "judge row SHA-256"),
            (self.judge_verdict_sha256, "judge verdict SHA-256"),
            (self.baseline_judge_row_sha256, "baseline judge row SHA-256"),
        ):
            if value is not None:
                _exact_sha256(value, label)
        _exact_nonnegative_int(
            self.historical_provider_calls, "historical judge provider calls"
        )
        if (self.primary_target_count is None) != (
            self.primary_target_recalled is None
        ):
            raise MatchedEvalContractError(
                "primary target count and recalled count must be present together"
            )
        for value, label in (
            (self.primary_target_count, "primary target count"),
            (self.primary_target_recalled, "primary target recalled"),
        ):
            if value is not None:
                _exact_nonnegative_int(value, label)
        if (
            self.primary_target_count is not None
            and self.primary_target_recalled is not None
            and self.primary_target_recalled > self.primary_target_count
        ):
            raise MatchedEvalContractError("recalled target count exceeds target count")

    def projection(self) -> dict[str, Any]:
        return asdict(self)


_RUNTIME_LEDGER_KEYS = frozenset(
    {
        "format",
        "gold_loaded",
        "historical_shared_local_model_calls",
        "ledger_identity_sha256",
        "plan_id",
        "question_count",
        "row_count",
        "rows",
        "snapshot_id",
        "source_artifacts",
        "total_historical_local_model_calls",
        "total_historical_provider_calls",
        "total_local_model_calls",
        "total_provider_calls",
    }
)
_RUNTIME_ROW_KEYS = frozenset(
    {row.name for row in fields(RuntimeLedgerEntry)} | {"row_id"}
)
_LIFECYCLE_FIELDS = (
    "candidate_ids",
    "selected_before_dedup_ids",
    "dedup_excluded_ids",
    "not_admitted_ids",
    "admitted_ids",
)
_FORBIDDEN_RUNTIME_ARTIFACT_TOKENS = frozenset({"judge", "score", "gold"})


def _validated_source_artifacts(
    source_artifacts: Iterable[Mapping[str, str]], *, runtime_plane: bool
) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    seen_roles: set[str] = set()
    for index, raw in enumerate(source_artifacts):
        if type(raw) is not dict:
            raise MatchedEvalContractError(
                f"source artifact {index} must be an exact object"
            )
        if set(raw) != {"role", "sha256"}:
            raise MatchedEvalContractError(
                f"source artifact {index} must contain exactly role and sha256"
            )
        role = _exact_text(raw["role"], f"source artifact {index} role")
        sha256 = _exact_sha256(
            raw["sha256"], f"source artifact {index} SHA-256"
        )
        if role in seen_roles:
            raise MatchedEvalContractError("source artifact roles must be unique")
        seen_roles.add(role)
        terminal_kind = role.rpartition(":")[2].casefold()
        terminal_tokens = frozenset(
            token
            for token in re.split(r"[^a-z0-9]+", terminal_kind.replace("_", "-"))
            if token
        )
        if runtime_plane and terminal_tokens & _FORBIDDEN_RUNTIME_ARTIFACT_TOKENS:
            raise MatchedEvalContractError(
                f"score/gold artifact role is forbidden at runtime: {role}"
            )
        result.append({"role": role, "sha256": sha256})
    return result


def _rehydrate_runtime_row(
    raw: object, index: int
) -> tuple[RuntimeLedgerEntry, str]:
    if type(raw) is not dict:
        raise MatchedEvalContractError(f"runtime ledger row {index} must be an object")
    if set(raw) != _RUNTIME_ROW_KEYS:
        raise MatchedEvalContractError(
            f"runtime ledger row {index} has an unexpected schema"
        )
    body = dict(raw)
    row_id = _exact_sha256(body.pop("row_id"), f"runtime row {index} ID")
    for key in _LIFECYCLE_FIELDS:
        values = body[key]
        if type(values) is not list:
            raise MatchedEvalContractError(
                f"runtime row {index} {key} must be an exact array"
            )
        body[key] = tuple(values)
    disposition = body["disposition"]
    if type(disposition) is not str:
        raise MatchedEvalContractError(
            f"runtime row {index} disposition must be exact text"
        )
    try:
        body["disposition"] = StageDisposition(disposition)
        entry = RuntimeLedgerEntry(**body)
    except (TypeError, ValueError) as exc:
        raise MatchedEvalContractError(
            f"runtime ledger row {index} cannot be reconstructed"
        ) from exc
    if entry.row_id != row_id:
        raise MatchedEvalContractError(f"runtime ledger row {index} ID is invalid")
    if entry.projection() != raw:
        raise MatchedEvalContractError(
            f"runtime ledger row {index} is not a canonical projection"
        )
    return entry, row_id


def _validated_runtime_ledger(
    runtime_ledger: Mapping[str, Any],
) -> tuple[str, tuple[str, ...]]:
    if type(runtime_ledger) is not dict:
        raise MatchedEvalContractError("runtime ledger must be an exact object")
    if set(runtime_ledger) != _RUNTIME_LEDGER_KEYS:
        raise MatchedEvalContractError("runtime ledger has an unexpected schema")
    if runtime_ledger["format"] != RUNTIME_LEDGER_FORMAT:
        raise MatchedEvalContractError("runtime ledger format changed")
    if runtime_ledger["gold_loaded"] is not False:
        raise MatchedEvalContractError("runtime ledger must remain gold-blind")
    _exact_text(runtime_ledger["plan_id"], "runtime ledger plan ID")
    _exact_sha256(runtime_ledger["snapshot_id"], "runtime ledger snapshot ID")
    rows = runtime_ledger["rows"]
    if type(rows) is not list:
        raise MatchedEvalContractError("runtime ledger rows must be an exact array")
    hydrated = tuple(
        _rehydrate_runtime_row(raw, index) for index, raw in enumerate(rows)
    )
    entries = tuple(row[0] for row in hydrated)
    row_ids = tuple(row[1] for row in hydrated)
    if len(set(row_ids)) != len(row_ids):
        raise MatchedEvalContractError("runtime ledger row IDs must be unique")

    row_count = _exact_nonnegative_int(
        runtime_ledger["row_count"], "runtime ledger row count"
    )
    question_count = _exact_nonnegative_int(
        runtime_ledger["question_count"], "runtime ledger question count"
    )
    if row_count != len(entries):
        raise MatchedEvalContractError("runtime ledger row count is invalid")
    if question_count != len({row.question_id for row in entries}):
        raise MatchedEvalContractError("runtime ledger question count is invalid")

    source_artifacts = runtime_ledger["source_artifacts"]
    if type(source_artifacts) is not list:
        raise MatchedEvalContractError(
            "runtime source artifacts must be an exact array"
        )
    if (
        _validated_source_artifacts(source_artifacts, runtime_plane=True)
        != source_artifacts
    ):
        raise MatchedEvalContractError("runtime source artifacts are not canonical")

    shared_local_calls = _exact_nonnegative_int(
        runtime_ledger["historical_shared_local_model_calls"],
        "historical shared local-model calls",
    )
    expected_totals = {
        "total_historical_local_model_calls": shared_local_calls,
        "total_historical_provider_calls": sum(
            row.historical_provider_calls for row in entries
        ),
        "total_local_model_calls": sum(row.local_model_calls for row in entries),
        "total_provider_calls": sum(row.provider_calls for row in entries),
    }
    for key, expected in expected_totals.items():
        actual = _exact_nonnegative_int(runtime_ledger[key], key.replace("_", " "))
        if actual != expected:
            raise MatchedEvalContractError(f"runtime ledger {key} is inconsistent")

    answer_rows = tuple(
        (row, row_id)
        for row, row_id in zip(entries, row_ids, strict=True)
        if row.event_type == "answer_observation"
    )
    answer_question_keys = tuple(
        (row.arm_label, row.question_id) for row, _ in answer_rows
    )
    answer_ordinal_keys = tuple((row.arm_label, row.ordinal) for row, _ in answer_rows)
    if len(set(answer_question_keys)) != len(answer_question_keys) or len(
        set(answer_ordinal_keys)
    ) != len(answer_ordinal_keys):
        raise MatchedEvalContractError(
            "runtime ledger answer observations must be unique per "
            "arm/question and arm/ordinal"
        )

    try:
        assert_gold_blind(runtime_ledger)
    except MatchedEvalContractError as exc:
        raise MatchedEvalContractError(
            "runtime ledger contains score/gold data"
        ) from exc
    declared_identity = _exact_sha256(
        runtime_ledger["ledger_identity_sha256"], "runtime ledger identity"
    )
    unsigned = dict(runtime_ledger)
    unsigned.pop("ledger_identity_sha256")
    if identity_sha256(unsigned) != declared_identity:
        raise MatchedEvalContractError("runtime ledger identity seal is invalid")
    return declared_identity, tuple(row_id for _, row_id in answer_rows)


def build_runtime_ledger(
    *,
    snapshot_id: str,
    plan_id: str,
    entries: Iterable[RuntimeLedgerEntry],
    source_artifacts: Iterable[Mapping[str, str]] = (),
    historical_shared_local_model_calls: int = 0,
) -> dict[str, Any]:
    _exact_sha256(snapshot_id, "snapshot ID")
    _exact_text(plan_id, "plan ID")
    _exact_nonnegative_int(
        historical_shared_local_model_calls,
        "historical shared local-model calls",
    )
    ordered = tuple(entries)
    if any(type(row) is not RuntimeLedgerEntry for row in ordered):
        raise MatchedEvalContractError(
            "runtime ledger entries must be exact RuntimeLedgerEntry values"
        )
    row_ids = tuple(row.row_id for row in ordered)
    if len(set(row_ids)) != len(row_ids):
        raise MatchedEvalContractError("runtime ledger row IDs must be unique")
    artifact_rows = _validated_source_artifacts(source_artifacts, runtime_plane=True)
    projection: dict[str, Any] = {
        "format": RUNTIME_LEDGER_FORMAT,
        "gold_loaded": False,
        "plan_id": plan_id,
        "question_count": len({row.question_id for row in ordered}),
        "row_count": len(ordered),
        "rows": [row.projection() for row in ordered],
        "snapshot_id": snapshot_id,
        "source_artifacts": artifact_rows,
        "historical_shared_local_model_calls": historical_shared_local_model_calls,
        "total_local_model_calls": sum(row.local_model_calls for row in ordered),
        "total_provider_calls": sum(row.provider_calls for row in ordered),
        "total_historical_provider_calls": sum(
            row.historical_provider_calls for row in ordered
        ),
        "total_historical_local_model_calls": historical_shared_local_model_calls,
    }
    assert_gold_blind(projection)
    projection["ledger_identity_sha256"] = identity_sha256(projection)
    _validated_runtime_ledger(projection)
    return projection


def build_score_ledger(
    *,
    runtime_ledger: Mapping[str, Any],
    entries: Iterable[ScoreLedgerEntry],
    source_artifacts: Iterable[Mapping[str, str]] = (),
) -> dict[str, Any]:
    runtime_ledger_identity_sha256, answer_row_ids = _validated_runtime_ledger(
        runtime_ledger
    )
    ordered = tuple(entries)
    if any(type(row) is not ScoreLedgerEntry for row in ordered):
        raise MatchedEvalContractError(
            "score ledger entries must be exact ScoreLedgerEntry values"
        )
    row_ids = tuple(row.runtime_row_id for row in ordered)
    if row_ids != answer_row_ids:
        raise MatchedEvalContractError(
            "score rows must cover every runtime answer observation in exact order"
        )
    comparison_presence = tuple(row.baseline_correct is not None for row in ordered)
    if comparison_presence and not (
        all(comparison_presence) or not any(comparison_presence)
    ):
        raise MatchedEvalContractError(
            "score ledger baseline comparisons must be present for every row or none"
        )
    scored = sum(1 for row in ordered if row.correct)
    baseline_rows = tuple(row for row in ordered if row.baseline_correct is not None)
    rescued = sum(1 for row in ordered if row.rescued)
    regressed = sum(1 for row in ordered if row.regressed)
    artifact_rows = _validated_source_artifacts(source_artifacts, runtime_plane=False)
    projection: dict[str, Any] = {
        "aggregate": {
            "baseline_correct": (
                sum(1 for row in baseline_rows if row.baseline_correct)
                if baseline_rows
                else None
            ),
            "candidate_correct": scored,
            "net_marginal": rescued - regressed if baseline_rows else None,
            "regressed": regressed if baseline_rows else None,
            "rescued": rescued if baseline_rows else None,
        },
        "format": SCORE_LEDGER_FORMAT,
        "row_count": len(ordered),
        "rows": [row.projection() for row in ordered],
        "runtime_ledger_identity_sha256": runtime_ledger_identity_sha256,
        "source_artifacts": artifact_rows,
        "total_historical_provider_calls": sum(
            row.historical_provider_calls for row in ordered
        ),
    }
    projection["ledger_identity_sha256"] = identity_sha256(projection)
    return projection
