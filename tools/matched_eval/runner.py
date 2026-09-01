"""Synchronous, provider-neutral execution for matched-evaluation arm plans.

The runner owns composition and accounting, not mechanism internals.  Adapters
receive an immutable snapshot, the packet at their declared parent, and their
own non-borrowable stage plan.  A bad adapter can therefore make only its stage
a no-op; it cannot mutate a parent packet or consume another stage's allowance.
"""

from __future__ import annotations

import inspect
from dataclasses import asdict, dataclass, replace
from typing import Mapping

from memory_condense.domain._tokenizer import count_tokens

from .contracts import (
    AnswerOperatorDelta,
    ArmPlan,
    EvaluationMemorySnapshot,
    LinkingDelta,
    MatchedEvalContractError,
    MechanismAdapter,
    MembershipDelta,
    MemoryPacket,
    ObservationDelta,
    PlanMode,
    RepresentationDelta,
    StageDelta,
    StageDisposition,
    StagePlan,
    StageTrace,
    assert_gold_blind,
    delta_projection,
    identity_sha256,
    require_sha256,
    require_text,
)
from .renderer import (
    RenderedPrompt,
    render_memory_packet_for_id,
)


_DELTA_TYPES = (
    MembershipDelta,
    RepresentationDelta,
    LinkingDelta,
    AnswerOperatorDelta,
    ObservationDelta,
)
_DELTA_TYPE_BY_KIND = {
    "membership": MembershipDelta,
    "representation": RepresentationDelta,
    "linking": LinkingDelta,
    "answer_operator": AnswerOperatorDelta,
    "observation": ObservationDelta,
}


@dataclass(frozen=True, slots=True)
class StageRunReceipt:
    """Normalized, gold-blind receipt for one logical stage execution."""

    snapshot_id: str
    plan_id: str
    question_id: str
    stage_id: str
    parent_stage_id: str
    mechanism_id: str
    delta_kind: str
    parent_packet_id: str
    output_packet_id: str
    renderer_id: str
    output_prompt_id: str
    output_prompt_messages_sha256: str
    output_prompt_token_proxy: int
    max_final_prompt_tokens: int
    token_cap: int
    reported_tokens_used: int
    provider_prompt_cap: int
    provider_prompt_reserved: int
    reported_provider_prompt_count: int
    trace: StageTrace
    delta_sha256: str | None = None

    def __post_init__(self) -> None:
        require_sha256(self.snapshot_id, "receipt snapshot ID")
        require_text(self.plan_id, "receipt plan ID")
        require_text(self.question_id, "receipt question ID")
        require_text(self.stage_id, "receipt stage ID")
        require_text(self.parent_stage_id, "receipt parent stage ID")
        require_text(self.mechanism_id, "receipt mechanism ID")
        require_text(self.delta_kind, "receipt delta kind")
        require_sha256(self.parent_packet_id, "receipt parent packet ID")
        require_sha256(self.output_packet_id, "receipt output packet ID")
        require_text(self.renderer_id, "receipt renderer ID")
        require_sha256(self.output_prompt_id, "receipt output prompt ID")
        require_sha256(
            self.output_prompt_messages_sha256,
            "receipt output prompt messages SHA-256",
        )
        if self.delta_sha256 is not None:
            require_sha256(self.delta_sha256, "receipt delta SHA-256")
        if (
            self.token_cap < 0
            or self.reported_tokens_used < 0
            or self.provider_prompt_cap < 0
            or self.provider_prompt_reserved < 0
            or self.reported_provider_prompt_count < 0
            or self.output_prompt_token_proxy < 0
            or self.max_final_prompt_tokens < 1
        ):
            raise MatchedEvalContractError("receipt accounting cannot be negative")
        if self.output_prompt_token_proxy > self.max_final_prompt_tokens:
            raise MatchedEvalContractError("receipt output prompt exceeds its final cap")
        if self.provider_prompt_reserved not in (0, self.provider_prompt_cap):
            raise MatchedEvalContractError(
                "a stage must reserve either zero or its complete provider cap"
            )
        if self.trace.token_cap != self.token_cap:
            raise MatchedEvalContractError("normalized trace must bind the stage token cap")
        if self.trace.tokens_used != min(self.reported_tokens_used, self.token_cap):
            raise MatchedEvalContractError(
                "normalized trace must bind reported stage token use"
            )
        if self.trace.provider_prompt_count != self.reported_provider_prompt_count:
            raise MatchedEvalContractError(
                "normalized trace must bind reported provider prompts"
            )
        if self.trace.disposition is StageDisposition.ADDED:
            if self.delta_sha256 is None:
                raise MatchedEvalContractError("an added stage must bind its delta")
            if self.output_packet_id == self.parent_packet_id:
                raise MatchedEvalContractError("an added stage must advance the packet")
        elif self.output_packet_id != self.parent_packet_id:
            raise MatchedEvalContractError("a no-op stage must preserve its parent packet")
        assert_gold_blind(self.projection())

    @property
    def provider_prompt_cap_compliant(self) -> bool:
        return self.reported_provider_prompt_count <= self.provider_prompt_reserved

    def projection(self) -> dict[str, object]:
        trace = asdict(self.trace)
        trace["disposition"] = self.trace.disposition.value
        return {
            "format": "memory-condense-matched-stage-run-receipt-v2",
            "delta_kind": self.delta_kind,
            "delta_sha256": self.delta_sha256,
            "mechanism_id": self.mechanism_id,
            "max_final_prompt_tokens": self.max_final_prompt_tokens,
            "output_packet_id": self.output_packet_id,
            "output_prompt_id": self.output_prompt_id,
            "output_prompt_messages_sha256": self.output_prompt_messages_sha256,
            "output_prompt_token_proxy": self.output_prompt_token_proxy,
            "parent_packet_id": self.parent_packet_id,
            "parent_stage_id": self.parent_stage_id,
            "plan_id": self.plan_id,
            "provider_prompt_cap": self.provider_prompt_cap,
            "provider_prompt_reserved": self.provider_prompt_reserved,
            "question_id": self.question_id,
            "reported_provider_prompt_count": self.reported_provider_prompt_count,
            "reported_tokens_used": self.reported_tokens_used,
            "renderer_id": self.renderer_id,
            "snapshot_id": self.snapshot_id,
            "stage_id": self.stage_id,
            "token_cap": self.token_cap,
            "trace": trace,
        }

    @property
    def receipt_sha256(self) -> str:
        return identity_sha256(self.projection())


@dataclass(frozen=True, slots=True)
class StageRunResult:
    """The immutable packet and normalized receipt produced for one stage."""

    packet: MemoryPacket
    rendered_prompt: RenderedPrompt
    receipt: StageRunReceipt

    def __post_init__(self) -> None:
        if self.packet.packet_id != self.receipt.output_packet_id:
            raise MatchedEvalContractError("stage packet does not match its receipt")
        if self.rendered_prompt.packet_id != self.packet.packet_id:
            raise MatchedEvalContractError("stage prompt does not match its packet")
        if (
            self.rendered_prompt.renderer_id != self.receipt.renderer_id
            or self.rendered_prompt.prompt_id != self.receipt.output_prompt_id
            or self.rendered_prompt.messages_sha256
            != self.receipt.output_prompt_messages_sha256
            or self.rendered_prompt.total_prompt_token_proxy
            != self.receipt.output_prompt_token_proxy
        ):
            raise MatchedEvalContractError("stage prompt does not match its receipt")

    @property
    def stage_id(self) -> str:
        return self.receipt.stage_id

    @property
    def trace(self) -> StageTrace:
        return self.receipt.trace

    @property
    def provider_prompt_count(self) -> int:
        return self.receipt.reported_provider_prompt_count


@dataclass(frozen=True, slots=True)
class ArmRunResult:
    """Immutable result of one isolated-star or cumulative-chain plan."""

    snapshot_id: str
    plan_id: str
    mode: PlanMode
    root_stage_id: str
    root_packet: MemoryPacket
    root_prompt: RenderedPrompt
    stages: tuple[StageRunResult, ...]
    global_provider_prompt_cap: int
    max_final_prompt_tokens: int
    provider_prompt_reserved: int
    provider_prompt_count: int

    def __post_init__(self) -> None:
        require_sha256(self.snapshot_id, "run snapshot ID")
        require_text(self.plan_id, "run plan ID")
        require_text(self.root_stage_id, "run root stage ID")
        if type(self.mode) is not PlanMode:
            raise MatchedEvalContractError("run mode must be canonical")
        if self.root_packet.stage_id != self.root_stage_id:
            raise MatchedEvalContractError("run root packet has the wrong stage ID")
        if self.root_prompt.packet_id != self.root_packet.packet_id:
            raise MatchedEvalContractError("run root prompt has the wrong packet ID")
        require_text(self.root_prompt.renderer_id, "run root prompt renderer")
        if (
            type(self.max_final_prompt_tokens) is not int
            or self.max_final_prompt_tokens < 1
            or self.root_prompt.total_prompt_token_proxy > self.max_final_prompt_tokens
        ):
            raise MatchedEvalContractError("run root prompt exceeds its final cap")
        if self.global_provider_prompt_cap < 0:
            raise MatchedEvalContractError("run provider prompt cap cannot be negative")
        if not 0 <= self.provider_prompt_reserved <= self.global_provider_prompt_cap:
            raise MatchedEvalContractError("run provider reservation exceeds the global cap")
        if self.provider_prompt_count < 0:
            raise MatchedEvalContractError("run provider prompt count cannot be negative")
        if self.provider_prompt_reserved != sum(
            row.receipt.provider_prompt_reserved for row in self.stages
        ):
            raise MatchedEvalContractError("run provider reservation does not reconcile")
        if self.provider_prompt_count != sum(
            row.receipt.reported_provider_prompt_count for row in self.stages
        ):
            raise MatchedEvalContractError("run provider prompt count does not reconcile")
        stage_ids = tuple(row.stage_id for row in self.stages)
        if len(set(stage_ids)) != len(stage_ids):
            raise MatchedEvalContractError("run stage IDs must be unique")
        previous_stage_id = self.root_stage_id
        previous_packet = self.root_packet
        for row in self.stages:
            receipt = row.receipt
            expected_parent_stage_id = (
                self.root_stage_id
                if self.mode is PlanMode.ISOLATED
                else previous_stage_id
            )
            expected_parent_packet = (
                self.root_packet if self.mode is PlanMode.ISOLATED else previous_packet
            )
            if (
                receipt.snapshot_id != self.snapshot_id
                or receipt.plan_id != self.plan_id
                or receipt.question_id != self.root_packet.question_id
            ):
                raise MatchedEvalContractError(
                    "run stage receipt changed snapshot, plan, or question binding"
                )
            if (
                receipt.parent_stage_id != expected_parent_stage_id
                or receipt.parent_packet_id != expected_parent_packet.packet_id
            ):
                raise MatchedEvalContractError("run stage lineage changed")
            if (
                row.packet.question_id != self.root_packet.question_id
                or row.packet.question_sha256 != self.root_packet.question_sha256
                or row.packet.dated_question_sha256
                != self.root_packet.dated_question_sha256
            ):
                raise MatchedEvalContractError("run stage changed question identity")
            if (
                receipt.max_final_prompt_tokens != self.max_final_prompt_tokens
                or receipt.output_prompt_token_proxy > self.max_final_prompt_tokens
            ):
                raise MatchedEvalContractError("run stage changed final prompt cap")
            if receipt.renderer_id != self.root_prompt.renderer_id:
                raise MatchedEvalContractError("run stage changed renderer identity")
            previous_stage_id = row.stage_id
            previous_packet = row.packet
        assert_gold_blind(self.projection())

    @property
    def provider_prompt_cap_compliant(self) -> bool:
        return (
            self.provider_prompt_count <= self.global_provider_prompt_cap
            and all(
                row.receipt.provider_prompt_cap_compliant for row in self.stages
            )
        )

    @property
    def packets(self) -> tuple[MemoryPacket, ...]:
        return (self.root_packet,) + tuple(row.packet for row in self.stages)

    @property
    def traces(self) -> tuple[StageTrace, ...]:
        return tuple(row.trace for row in self.stages)

    def stage(self, stage_id: str) -> StageRunResult:
        for row in self.stages:
            if row.stage_id == stage_id:
                return row
        raise KeyError(stage_id)

    def packet_for(self, stage_id: str) -> MemoryPacket:
        if stage_id == self.root_stage_id:
            return self.root_packet
        return self.stage(stage_id).packet

    def projection(self) -> dict[str, object]:
        return {
            "format": "memory-condense-matched-arm-run-result-v2",
            "global_provider_prompt_cap": self.global_provider_prompt_cap,
            "max_final_prompt_tokens": self.max_final_prompt_tokens,
            "mode": self.mode.value,
            "plan_id": self.plan_id,
            "provider_prompt_cap_compliant": self.provider_prompt_cap_compliant,
            "provider_prompt_count": self.provider_prompt_count,
            "provider_prompt_reserved": self.provider_prompt_reserved,
            "root_packet_id": self.root_packet.packet_id,
            "root_prompt_id": self.root_prompt.prompt_id,
            "root_prompt_messages_sha256": self.root_prompt.messages_sha256,
            "root_prompt_token_proxy": self.root_prompt.total_prompt_token_proxy,
            "root_stage_id": self.root_stage_id,
            "snapshot_id": self.snapshot_id,
            "stages": [
                {
                    "output_packet_id": row.packet.packet_id,
                    "receipt_sha256": row.receipt.receipt_sha256,
                    "stage_id": row.stage_id,
                }
                for row in self.stages
            ],
        }

    @property
    def result_sha256(self) -> str:
        return identity_sha256(self.projection())


class MatchedEvalRunner:
    """Execute registered mechanism adapters without owning a provider client."""

    __slots__ = ("_adapters",)

    def __init__(self, adapters: Mapping[str, MechanismAdapter]) -> None:
        registry: dict[str, MechanismAdapter] = {}
        for mechanism_id, adapter in adapters.items():
            require_text(mechanism_id, "adapter registry mechanism ID")
            if getattr(adapter, "mechanism_id", None) != mechanism_id:
                raise MatchedEvalContractError(
                    "adapter registry key must match adapter mechanism_id"
                )
            delta_kind = getattr(adapter, "delta_kind", None)
            if delta_kind not in _DELTA_TYPE_BY_KIND:
                raise MatchedEvalContractError("adapter declares an unknown delta kind")
            propose = getattr(adapter, "propose", None)
            if not callable(propose):
                raise MatchedEvalContractError("adapter must define synchronous propose")
            if inspect.iscoroutinefunction(propose):
                raise MatchedEvalContractError("adapter propose must be synchronous")
            registry[mechanism_id] = adapter
        self._adapters = registry

    def run(
        self,
        *,
        snapshot: EvaluationMemorySnapshot,
        root_packet: MemoryPacket,
        plan: ArmPlan,
    ) -> ArmRunResult:
        snapshot_id = snapshot.snapshot_id
        if root_packet.stage_id != plan.root_stage_id:
            raise MatchedEvalContractError("root packet does not match the plan root")
        # Computing the ID also applies the recursive gold firewall.
        root_packet.packet_id
        root_prompt = render_memory_packet_for_id(
            root_packet,
            renderer_id=snapshot.renderer_id,
        )
        if root_prompt.total_prompt_token_proxy > plan.max_final_prompt_tokens:
            raise MatchedEvalContractError("root packet exceeds the final prompt token cap")

        packets_by_stage: dict[str, MemoryPacket] = {
            plan.root_stage_id: root_packet
        }
        prompts_by_stage: dict[str, RenderedPrompt] = {
            plan.root_stage_id: root_prompt
        }
        stage_results: list[StageRunResult] = []
        provider_prompt_reserved = 0
        provider_prompt_count = 0

        for stage in plan.stages:
            parent = packets_by_stage[stage.parent_stage_id]
            parent_prompt = prompts_by_stage[stage.parent_stage_id]
            adapter = self._adapters.get(stage.mechanism_id)
            if adapter is None:
                result = self._terminal_result(
                    snapshot_id=snapshot_id,
                    plan=plan,
                    stage=stage,
                    parent=parent,
                    rendered_prompt=parent_prompt,
                    disposition=StageDisposition.INVALID,
                    reason="adapter_not_registered",
                )
            elif adapter.delta_kind != stage.delta_kind:
                result = self._terminal_result(
                    snapshot_id=snapshot_id,
                    plan=plan,
                    stage=stage,
                    parent=parent,
                    rendered_prompt=parent_prompt,
                    disposition=StageDisposition.INVALID,
                    reason="adapter_delta_kind_mismatch",
                )
            elif (
                provider_prompt_reserved + stage.budget.provider_prompt_cap
                > plan.global_provider_prompt_cap
            ):
                result = self._terminal_result(
                    snapshot_id=snapshot_id,
                    plan=plan,
                    stage=stage,
                    parent=parent,
                    rendered_prompt=parent_prompt,
                    disposition=StageDisposition.OVERFLOW,
                    reason="global_provider_prompt_cap",
                )
            else:
                reserved = stage.budget.provider_prompt_cap
                provider_prompt_reserved += reserved
                result = self._invoke_stage(
                    snapshot=snapshot,
                    snapshot_id=snapshot_id,
                    plan=plan,
                    stage=stage,
                    parent=parent,
                    parent_prompt=parent_prompt,
                    adapter=adapter,
                    provider_prompt_reserved=reserved,
                )

            packets_by_stage[stage.stage_id] = result.packet
            prompts_by_stage[stage.stage_id] = result.rendered_prompt
            stage_results.append(result)
            provider_prompt_count += result.provider_prompt_count

        return ArmRunResult(
            snapshot_id=snapshot_id,
            plan_id=plan.plan_id,
            mode=plan.mode,
            root_stage_id=plan.root_stage_id,
            root_packet=root_packet,
            root_prompt=root_prompt,
            stages=tuple(stage_results),
            global_provider_prompt_cap=plan.global_provider_prompt_cap,
            max_final_prompt_tokens=plan.max_final_prompt_tokens,
            provider_prompt_reserved=provider_prompt_reserved,
            provider_prompt_count=provider_prompt_count,
        )

    def _invoke_stage(
        self,
        *,
        snapshot: EvaluationMemorySnapshot,
        snapshot_id: str,
        plan: ArmPlan,
        stage: StagePlan,
        parent: MemoryPacket,
        parent_prompt: RenderedPrompt,
        adapter: MechanismAdapter,
        provider_prompt_reserved: int,
    ) -> StageRunResult:
        try:
            proposed = adapter.propose(snapshot=snapshot, packet=parent, stage=stage)
        except Exception:
            return self._terminal_result(
                snapshot_id=snapshot_id,
                plan=plan,
                stage=stage,
                parent=parent,
                rendered_prompt=parent_prompt,
                disposition=StageDisposition.FAILED,
                reason="adapter_exception",
                provider_prompt_reserved=provider_prompt_reserved,
            )

        if inspect.isawaitable(proposed):
            close = getattr(proposed, "close", None)
            if callable(close):
                close()
            return self._terminal_result(
                snapshot_id=snapshot_id,
                plan=plan,
                stage=stage,
                parent=parent,
                rendered_prompt=parent_prompt,
                disposition=StageDisposition.INVALID,
                reason="adapter_returned_awaitable",
                provider_prompt_reserved=provider_prompt_reserved,
            )
        if not isinstance(proposed, _DELTA_TYPES):
            return self._terminal_result(
                snapshot_id=snapshot_id,
                plan=plan,
                stage=stage,
                parent=parent,
                rendered_prompt=parent_prompt,
                disposition=StageDisposition.INVALID,
                reason="adapter_returned_invalid_delta",
                provider_prompt_reserved=provider_prompt_reserved,
            )

        trace = proposed.trace
        try:
            projection = delta_projection(proposed)
            delta_sha256 = identity_sha256(projection)
        except (MatchedEvalContractError, TypeError, ValueError):
            return self._terminal_result(
                snapshot_id=snapshot_id,
                plan=plan,
                stage=stage,
                parent=parent,
                rendered_prompt=parent_prompt,
                disposition=StageDisposition.INVALID,
                reason="delta_not_gold_blind",
                source_trace=trace,
                provider_prompt_reserved=provider_prompt_reserved,
            )

        expected_type = _DELTA_TYPE_BY_KIND[stage.delta_kind]
        if not isinstance(proposed, expected_type):
            return self._terminal_result(
                snapshot_id=snapshot_id,
                plan=plan,
                stage=stage,
                parent=parent,
                rendered_prompt=parent_prompt,
                disposition=StageDisposition.INVALID,
                reason="delta_kind_mismatch",
                source_trace=trace,
                provider_prompt_reserved=provider_prompt_reserved,
                delta_sha256=delta_sha256,
            )
        if proposed.stage_id != stage.stage_id:
            return self._terminal_result(
                snapshot_id=snapshot_id,
                plan=plan,
                stage=stage,
                parent=parent,
                rendered_prompt=parent_prompt,
                disposition=StageDisposition.INVALID,
                reason="delta_stage_mismatch",
                source_trace=trace,
                provider_prompt_reserved=provider_prompt_reserved,
                delta_sha256=delta_sha256,
            )
        if proposed.parent_stage_id != stage.parent_stage_id:
            return self._terminal_result(
                snapshot_id=snapshot_id,
                plan=plan,
                stage=stage,
                parent=parent,
                rendered_prompt=parent_prompt,
                disposition=StageDisposition.INVALID,
                reason="delta_parent_mismatch",
                source_trace=trace,
                provider_prompt_reserved=provider_prompt_reserved,
                delta_sha256=delta_sha256,
            )
        if trace.tokens_used > stage.budget.token_cap:
            return self._terminal_result(
                snapshot_id=snapshot_id,
                plan=plan,
                stage=stage,
                parent=parent,
                rendered_prompt=parent_prompt,
                disposition=StageDisposition.OVERFLOW,
                reason="stage_token_cap",
                source_trace=trace,
                provider_prompt_reserved=provider_prompt_reserved,
                delta_sha256=delta_sha256,
            )
        if trace.provider_prompt_count > stage.budget.provider_prompt_cap:
            return self._terminal_result(
                snapshot_id=snapshot_id,
                plan=plan,
                stage=stage,
                parent=parent,
                rendered_prompt=parent_prompt,
                disposition=StageDisposition.OVERFLOW,
                reason="stage_provider_prompt_cap",
                source_trace=trace,
                provider_prompt_reserved=provider_prompt_reserved,
                delta_sha256=delta_sha256,
            )
        if trace.token_cap != stage.budget.token_cap:
            return self._terminal_result(
                snapshot_id=snapshot_id,
                plan=plan,
                stage=stage,
                parent=parent,
                rendered_prompt=parent_prompt,
                disposition=StageDisposition.INVALID,
                reason="stage_token_cap_mismatch",
                source_trace=trace,
                provider_prompt_reserved=provider_prompt_reserved,
                delta_sha256=delta_sha256,
            )
        measured_tokens_used = _delta_content_token_count(proposed)
        if measured_tokens_used > stage.budget.token_cap:
            return self._terminal_result(
                snapshot_id=snapshot_id,
                plan=plan,
                stage=stage,
                parent=parent,
                rendered_prompt=parent_prompt,
                disposition=StageDisposition.OVERFLOW,
                reason="stage_token_cap",
                source_trace=trace,
                provider_prompt_reserved=provider_prompt_reserved,
                delta_sha256=delta_sha256,
            )
        if trace.tokens_used != measured_tokens_used:
            return self._terminal_result(
                snapshot_id=snapshot_id,
                plan=plan,
                stage=stage,
                parent=parent,
                rendered_prompt=parent_prompt,
                disposition=StageDisposition.INVALID,
                reason="stage_token_accounting_mismatch",
                source_trace=trace,
                provider_prompt_reserved=provider_prompt_reserved,
                delta_sha256=delta_sha256,
            )
        if isinstance(proposed, MembershipDelta) and proposed.dedup_alias_bindings:
            existing = {
                row.evidence_id
                for row in parent.protected_evidence + parent.admitted_evidence
            }
            try:
                _require_post_selection_dedup(
                    trace,
                    existing,
                    alias_bindings=proposed.dedup_alias_bindings,
                )
            except MatchedEvalContractError:
                return self._terminal_result(
                    snapshot_id=snapshot_id,
                    plan=plan,
                    stage=stage,
                    parent=parent,
                    rendered_prompt=parent_prompt,
                    disposition=StageDisposition.INVALID,
                    reason="packet_invariant",
                    source_trace=trace,
                    provider_prompt_reserved=provider_prompt_reserved,
                    delta_sha256=delta_sha256,
                )
        if trace.disposition is not StageDisposition.ADDED:
            return self._terminal_result(
                snapshot_id=snapshot_id,
                plan=plan,
                stage=stage,
                parent=parent,
                rendered_prompt=parent_prompt,
                disposition=trace.disposition,
                reason=f"adapter_{trace.disposition.value}",
                source_trace=trace,
                provider_prompt_reserved=provider_prompt_reserved,
                delta_sha256=delta_sha256,
            )

        try:
            packet = _apply_delta(parent=parent, stage=stage, delta=proposed)
        except (MatchedEvalContractError, TypeError, ValueError):
            return self._terminal_result(
                snapshot_id=snapshot_id,
                plan=plan,
                stage=stage,
                parent=parent,
                rendered_prompt=parent_prompt,
                disposition=StageDisposition.INVALID,
                reason="packet_invariant",
                source_trace=trace,
                provider_prompt_reserved=provider_prompt_reserved,
                delta_sha256=delta_sha256,
            )

        try:
            rendered_prompt = render_memory_packet_for_id(
                packet,
                renderer_id=snapshot.renderer_id,
            )
        except (MatchedEvalContractError, TypeError, ValueError):
            return self._terminal_result(
                snapshot_id=snapshot_id,
                plan=plan,
                stage=stage,
                parent=parent,
                rendered_prompt=parent_prompt,
                disposition=StageDisposition.INVALID,
                reason="renderer_invariant",
                source_trace=trace,
                provider_prompt_reserved=provider_prompt_reserved,
                delta_sha256=delta_sha256,
            )
        if rendered_prompt.total_prompt_token_proxy > plan.max_final_prompt_tokens:
            return self._terminal_result(
                snapshot_id=snapshot_id,
                plan=plan,
                stage=stage,
                parent=parent,
                rendered_prompt=parent_prompt,
                disposition=StageDisposition.OVERFLOW,
                reason="final_prompt_token_cap",
                source_trace=trace,
                provider_prompt_reserved=provider_prompt_reserved,
                delta_sha256=delta_sha256,
            )

        normalized_trace = replace(trace, reason=None)
        receipt = _receipt(
            snapshot_id=snapshot_id,
            plan=plan,
            stage=stage,
            parent=parent,
            packet=packet,
            rendered_prompt=rendered_prompt,
            trace=normalized_trace,
            provider_prompt_reserved=provider_prompt_reserved,
            reported_tokens_used=trace.tokens_used,
            reported_provider_prompt_count=trace.provider_prompt_count,
            delta_sha256=delta_sha256,
        )
        return StageRunResult(
            packet=packet, rendered_prompt=rendered_prompt, receipt=receipt
        )

    @staticmethod
    def _terminal_result(
        *,
        snapshot_id: str,
        plan: ArmPlan,
        stage: StagePlan,
        parent: MemoryPacket,
        rendered_prompt: RenderedPrompt,
        disposition: StageDisposition,
        reason: str,
        source_trace: StageTrace | None = None,
        provider_prompt_reserved: int = 0,
        delta_sha256: str | None = None,
    ) -> StageRunResult:
        trace = _normalized_terminal_trace(
            stage=stage,
            disposition=disposition,
            reason=reason,
            source=source_trace,
        )
        reported_tokens_used = source_trace.tokens_used if source_trace is not None else 0
        reported_provider_prompt_count = (
            source_trace.provider_prompt_count if source_trace is not None else 0
        )
        receipt = _receipt(
            snapshot_id=snapshot_id,
            plan=plan,
            stage=stage,
            parent=parent,
            packet=parent,
            rendered_prompt=rendered_prompt,
            trace=trace,
            provider_prompt_reserved=provider_prompt_reserved,
            reported_tokens_used=reported_tokens_used,
            reported_provider_prompt_count=reported_provider_prompt_count,
            delta_sha256=delta_sha256,
        )
        return StageRunResult(
            packet=parent, rendered_prompt=rendered_prompt, receipt=receipt
        )


def _normalized_terminal_trace(
    *,
    stage: StagePlan,
    disposition: StageDisposition,
    reason: str,
    source: StageTrace | None,
) -> StageTrace:
    if disposition is StageDisposition.ADDED:
        raise MatchedEvalContractError("terminal traces cannot add packet content")
    return StageTrace(
        candidate_ids=source.candidate_ids if source is not None else (),
        selected_before_dedup_ids=(
            source.selected_before_dedup_ids if source is not None else ()
        ),
        dedup_excluded_ids=(
            source.dedup_excluded_ids if source is not None else ()
        ),
        not_admitted_ids=(
            tuple(
                evidence_id
                for evidence_id in source.selected_before_dedup_ids
                if evidence_id not in set(source.dedup_excluded_ids)
            )
            if source is not None
            else ()
        ),
        admitted_ids=(),
        token_cap=stage.budget.token_cap,
        tokens_used=min(source.tokens_used, stage.budget.token_cap) if source else 0,
        provider_prompt_count=source.provider_prompt_count if source else 0,
        disposition=disposition,
        reason=reason,
    )


def _receipt(
    *,
    snapshot_id: str,
    plan: ArmPlan,
    stage: StagePlan,
    parent: MemoryPacket,
    packet: MemoryPacket,
    rendered_prompt: RenderedPrompt,
    trace: StageTrace,
    provider_prompt_reserved: int,
    reported_tokens_used: int,
    reported_provider_prompt_count: int,
    delta_sha256: str | None,
) -> StageRunReceipt:
    return StageRunReceipt(
        snapshot_id=snapshot_id,
        plan_id=plan.plan_id,
        question_id=parent.question_id,
        stage_id=stage.stage_id,
        parent_stage_id=stage.parent_stage_id,
        mechanism_id=stage.mechanism_id,
        delta_kind=stage.delta_kind,
        parent_packet_id=parent.packet_id,
        output_packet_id=packet.packet_id,
        renderer_id=rendered_prompt.renderer_id,
        output_prompt_id=rendered_prompt.prompt_id,
        output_prompt_messages_sha256=rendered_prompt.messages_sha256,
        output_prompt_token_proxy=rendered_prompt.total_prompt_token_proxy,
        max_final_prompt_tokens=plan.max_final_prompt_tokens,
        token_cap=stage.budget.token_cap,
        reported_tokens_used=reported_tokens_used,
        provider_prompt_cap=stage.budget.provider_prompt_cap,
        provider_prompt_reserved=provider_prompt_reserved,
        reported_provider_prompt_count=reported_provider_prompt_count,
        trace=trace,
        delta_sha256=delta_sha256,
    )


def _delta_content_token_count(delta: StageDelta) -> int:
    if isinstance(delta, MembershipDelta):
        return sum(row.token_count for row in delta.additions)
    if isinstance(delta, RepresentationDelta):
        return sum(row.token_count for row in delta.facts)
    if isinstance(delta, LinkingDelta):
        return sum(row.token_count for row in delta.links)
    if isinstance(delta, AnswerOperatorDelta):
        return count_tokens(delta.instructions) if delta.instructions is not None else 0
    if isinstance(delta, ObservationDelta):
        return 0
    raise MatchedEvalContractError("unsupported stage delta")


def _apply_delta(
    *, parent: MemoryPacket, stage: StagePlan, delta: StageDelta
) -> MemoryPacket:
    applied_stage_ids = parent.applied_stage_ids + (stage.stage_id,)

    if isinstance(delta, MembershipDelta):
        existing = {
            row.evidence_id
            for row in parent.protected_evidence + parent.admitted_evidence
        }
        _require_post_selection_dedup(
            delta.trace,
            existing,
            alias_bindings=delta.dedup_alias_bindings,
        )
        if any(row.evidence_id in existing for row in delta.additions):
            raise MatchedEvalContractError(
                "membership additions cannot duplicate parent evidence"
            )
        return replace(
            parent,
            stage_id=stage.stage_id,
            admitted_evidence=parent.admitted_evidence + delta.additions,
            applied_stage_ids=applied_stage_ids,
        )

    evidence_ids = {
        row.evidence_id for row in parent.protected_evidence + parent.admitted_evidence
    }
    if isinstance(delta, RepresentationDelta):
        dedup_basis = set(delta.dedup_against_evidence_ids) or {
            row.evidence_id for row in parent.protected_evidence
        }
        if not dedup_basis <= evidence_ids:
            raise MatchedEvalContractError(
                "representation dedup basis must belong to the parent packet"
            )
        _require_post_selection_dedup(delta.trace, dedup_basis)
        if len(set(delta.bound_evidence_ids)) != len(delta.bound_evidence_ids):
            raise MatchedEvalContractError(
                "representation bound evidence IDs must be unique"
            )
        existing = {row.fact_id for row in parent.facts}
        if any(row.fact_id in existing for row in delta.facts):
            raise MatchedEvalContractError("representation cannot duplicate parent facts")
        return replace(
            parent,
            stage_id=stage.stage_id,
            facts=parent.facts + delta.facts,
            applied_stage_ids=applied_stage_ids,
        )

    if isinstance(delta, LinkingDelta):
        if len(set(delta.bound_evidence_ids)) != len(delta.bound_evidence_ids):
            raise MatchedEvalContractError("link bound evidence IDs must be unique")
        if not set(delta.bound_evidence_ids) <= evidence_ids:
            raise MatchedEvalContractError(
                "linking cannot bind evidence outside its parent packet"
            )
        existing = {row.link_id for row in parent.links}
        if any(row.link_id in existing for row in delta.links):
            raise MatchedEvalContractError("linking cannot duplicate parent links")
        return replace(
            parent,
            stage_id=stage.stage_id,
            links=parent.links + delta.links,
            applied_stage_ids=applied_stage_ids,
        )

    if isinstance(delta, AnswerOperatorDelta):
        if delta.operator_id is None or delta.instructions is None:
            raise MatchedEvalContractError("added answer operator is incomplete")
        if delta.operator_id in {row[0] for row in parent.answer_operators}:
            raise MatchedEvalContractError(
                "answer operator cannot duplicate a parent operator"
            )
        return replace(
            parent,
            stage_id=stage.stage_id,
            answer_operators=parent.answer_operators
            + ((delta.operator_id, delta.instructions),),
            applied_stage_ids=applied_stage_ids,
        )

    if isinstance(delta, ObservationDelta):
        # Observation receipts remain outside the answer packet.  Only logical
        # stage lineage advances; no evidence, facts, links, or operator enters.
        return replace(
            parent,
            stage_id=stage.stage_id,
            applied_stage_ids=applied_stage_ids,
        )

    raise MatchedEvalContractError("unsupported stage delta")


def _require_post_selection_dedup(
    trace: StageTrace,
    dedup_evidence_ids: set[str],
    *,
    alias_bindings: tuple[tuple[str, str], ...] = (),
) -> None:
    """Bind membership/EM exclusion to its declared post-selection basis."""

    alias_by_selected = dict(alias_bindings)
    if len(alias_by_selected) != len(alias_bindings) or any(
        protected_id not in dedup_evidence_ids
        for protected_id in alias_by_selected.values()
    ):
        raise MatchedEvalContractError(
            "membership dedup aliases must bind selected items to parent evidence"
        )
    expected_excluded = tuple(
        evidence_id
        for evidence_id in trace.selected_before_dedup_ids
        if evidence_id in dedup_evidence_ids or evidence_id in alias_by_selected
    )
    expected_remaining = tuple(
        evidence_id
        for evidence_id in trace.selected_before_dedup_ids
        if evidence_id not in set(expected_excluded)
    )
    if (
        trace.dedup_excluded_ids != expected_excluded
        or set(trace.not_admitted_ids) | set(trace.admitted_ids)
        != set(expected_remaining)
        or len(trace.not_admitted_ids) + len(trace.admitted_ids)
        != len(expected_remaining)
    ):
        raise MatchedEvalContractError(
            "membership/representation must apply its dedup basis only after "
            "selection and preserve the complete selected-item partition"
        )


def run_arm(
    *,
    snapshot: EvaluationMemorySnapshot,
    root_packet: MemoryPacket,
    plan: ArmPlan,
    adapters: Mapping[str, MechanismAdapter],
) -> ArmRunResult:
    """Convenience entry point for one synchronous arm run."""

    return MatchedEvalRunner(adapters).run(
        snapshot=snapshot,
        root_packet=root_packet,
        plan=plan,
    )


__all__ = [
    "ArmRunResult",
    "MatchedEvalRunner",
    "StageRunReceipt",
    "StageRunResult",
    "run_arm",
]
