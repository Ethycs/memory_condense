#!/usr/bin/env python3
"""Provider-free base-cap sweep for the locked adaptive source gate.

Selection is frozen from the sealed query plan, terminal V2 evidence map, and
pinned direct/partition/guided source streams before this program optionally
opens the posthoc registered-source plan.  The target plan can annotate recall;
it cannot activate a question, choose a route, or change a source prefix.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:
    repository = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(repository / "src"), str(repository)]

from tools import run_locked_adaptive_source_map as source_cli  # noqa: E402
from tools import run_locked_query_evidence_map_solver_v2 as map_cli  # noqa: E402
from tools import run_locked_query_payload_answers as payload_cli  # noqa: E402
from tools.analyze_locked_query_answer_joint_failures import (  # noqa: E402
    DEFAULT_TARGET_PLAN,
    EXPECTED_TARGET_PLAN_SHA256,
)
from tools.matched_eval.artifacts import (  # noqa: E402
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
    load_locked_source_gate_adapter,
)
from tools.matched_eval.query_evidence_map_solver_v2_live import (  # noqa: E402
    replay_evidence_map,
)
from tools.matched_eval.query_map_source_gate_adapter import (  # noqa: E402
    CONSOLIDATED_OBLIGATION_MODE,
    OBLIGATION_MODES,
    STATE_CHAIN_DIRECT_AUTHORITY_PROFILE,
    STATE_CHAIN_PROFILES,
)
from tools.matched_eval.source_gate_controller import (  # noqa: E402
    LaneSourceBudget,
    SourceGatePolicy,
)
from tools.matched_eval.source_history_fact_union import FactLane  # noqa: E402
from tools.matched_eval.source_history_fact_union import (  # noqa: E402
    DEFAULT_HISTORY_WINDOW_TOKEN_CAP,
)
from tools.matched_eval.source_history_mapper_live import (  # noqa: E402
    HARD_CONTEXT_TOKEN_CAP,
)


FORMAT = "memory-condense-locked-source-gate-base-policy-sweep-v1"
STRUCTURAL_FORMAT = f"{FORMAT}-structural"
DEFAULT_OUTPUT = (
    source_cli.DEFAULT_OUTPUT / "source-gate-base-policy-sweep.json"
)
EXPECTED_QUESTION_COUNT = 100
EXPECTED_SOURCE_TARGET_COUNT = 188


class SourceGatePolicySweepError(MatchedEvalContractError):
    """A sealed parent, fixed prefix, physical-call count, or target plan changed."""


def _require(ok: object, message: str) -> None:
    if not ok:
        raise SourceGatePolicySweepError(message)


def _unique_text(values: tuple[str, ...], label: str) -> tuple[str, ...]:
    _require(type(values) is tuple, f"{label} must be an immutable tuple")
    for value in values:
        require_text(value, label)
    _require(len(values) == len(set(values)), f"{label} must be ordered and unique")
    return values


def _fraction(numerator: int, denominator: int) -> dict[str, int]:
    _require(
        type(numerator) is int
        and type(denominator) is int
        and numerator >= 0
        and denominator > 0,
        "invalid exact fraction",
    )
    value = Fraction(numerator, denominator)
    return {
        "denominator": denominator,
        "numerator": numerator,
        "reduced_denominator": value.denominator,
        "reduced_numerator": value.numerator,
    }


@dataclass(frozen=True, slots=True)
class BaseCapPolicy:
    policy_id: str
    direct_base_source_cap: int
    guided_base_source_cap: int
    partition_base_source_cap: int = 0

    def __post_init__(self) -> None:
        require_text(self.policy_id, "base-cap policy ID")
        for value, label in (
            (self.direct_base_source_cap, "direct base cap"),
            (self.guided_base_source_cap, "guided base cap"),
            (self.partition_base_source_cap, "partition base cap"),
        ):
            _require(type(value) is int and value >= 0, f"{label} changed")
    def projection(self) -> dict[str, Any]:
        return {
            "direct_base_source_cap": self.direct_base_source_cap,
            "guided_base_source_cap": self.guided_base_source_cap,
            "partition_base_source_cap": self.partition_base_source_cap,
            "policy_id": self.policy_id,
        }


BASE_CAP_POLICIES = (
    BaseCapPolicy("D1/G0", 1, 0),
    BaseCapPolicy("D0/P1/G0", 0, 0, 1),
    BaseCapPolicy("D0/G1", 0, 1),
    BaseCapPolicy("D1/G1", 1, 1),
    BaseCapPolicy("D1/P1/G1", 1, 1, 1),
    BaseCapPolicy("D2/G1", 2, 1),
    BaseCapPolicy("D3/G1", 3, 1),
    BaseCapPolicy("D3/G2", 3, 2),
    BaseCapPolicy("D5/G2", 5, 2),
)


@dataclass(frozen=True, slots=True)
class SweepQuestion:
    """Gold-blind source prefixes and exact physical window multiplicities."""

    ordinal: int
    question_id: str
    route_id: str
    namespace_id: str | None
    activated: bool
    baseline_source_ids: tuple[str, ...]
    direct_source_ids: tuple[str, ...]
    partition_source_ids: tuple[str, ...]
    guided_source_ids: tuple[str, ...]
    physical_window_calls_by_source: tuple[tuple[str, int], ...]
    history_window_token_cap: int = 0
    maximum_combined_prompt_token_proxy: int = 0

    def __post_init__(self) -> None:
        _require(type(self.ordinal) is int and self.ordinal >= 0, "sweep ordinal changed")
        for value, label in (
            (self.question_id, "sweep question ID"),
            (self.route_id, "sweep route ID"),
        ):
            require_text(value, label)
        _require(type(self.activated) is bool, "sweep activation flag changed")
        if self.activated:
            require_sha256(self.namespace_id, "sweep namespace")  # type: ignore[arg-type]
        else:
            _require(
                self.namespace_id is None,
                "a map-satisfied row must not invent an unused source namespace",
            )
        for values, label in (
            (self.baseline_source_ids, "baseline source IDs"),
            (self.direct_source_ids, "direct source IDs"),
            (self.partition_source_ids, "partition source IDs"),
            (self.guided_source_ids, "guided source IDs"),
        ):
            _unique_text(values, label)
        _require(
            type(self.physical_window_calls_by_source) is tuple,
            "physical source-call rows must be immutable",
        )
        physical_sources: list[str] = []
        for source_id, count in self.physical_window_calls_by_source:
            require_text(source_id, "physical source ID")
            _require(type(count) is int and count > 0, "physical window count changed")
            physical_sources.append(source_id)
        _require(
            len(physical_sources) == len(set(physical_sources)),
            "physical source-call rows repeat",
        )
        if not self.activated:
            _require(
                not self.direct_source_ids
                and not self.partition_source_ids
                and not self.guided_source_ids
                and not self.physical_window_calls_by_source,
                "a map-satisfied row cannot acquire source-gate work",
            )
            _require(
                self.history_window_token_cap == 0
                and self.maximum_combined_prompt_token_proxy == 0,
                "a map-satisfied row cannot acquire mapper prompt metadata",
            )
        else:
            _require(
                type(self.history_window_token_cap) is int
                and self.history_window_token_cap >= 0
                and type(self.maximum_combined_prompt_token_proxy) is int
                and 0 <= self.maximum_combined_prompt_token_proxy <= HARD_CONTEXT_TOKEN_CAP,
                "activated mapper prompt metadata changed",
            )


def _histogram(values: Sequence[int]) -> dict[str, int]:
    return {
        str(value): count
        for value, count in sorted(Counter(values).items())
    }


def _selected(
    row: SweepQuestion, policy: BaseCapPolicy
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    if not row.activated:
        return (), (), (), ()
    direct = row.direct_source_ids[: policy.direct_base_source_cap]
    partition = row.partition_source_ids[: policy.partition_base_source_cap]
    guided = row.guided_source_ids[: policy.guided_base_source_cap]
    # Logical lane credit survives cross-method physical deduplication.
    unique = tuple(dict.fromkeys((*direct, *partition, *guided)))
    return direct, partition, guided, unique


def sweep_base_policies(
    rows: tuple[SweepQuestion, ...],
    *,
    policies: tuple[BaseCapPolicy, ...] = BASE_CAP_POLICIES,
    source_parent_receipt_sha256: str,
    map_adapter_receipt_sha256: str,
    obligation_mode: str = CONSOLIDATED_OBLIGATION_MODE,
    state_chain_profile: str = STATE_CHAIN_DIRECT_AUTHORITY_PROFILE,
) -> dict[str, Any]:
    """Freeze policy prefixes and exact provider-free call accounting."""

    _require(
        type(rows) is tuple
        and bool(rows)
        and all(type(row) is SweepQuestion for row in rows),
        "sweep rows must be an immutable exact population",
    )
    _require(
        tuple(row.ordinal for row in rows) == tuple(range(len(rows)))
        and len({row.question_id for row in rows}) == len(rows),
        "sweep question order changed",
    )
    _require(
        type(policies) is tuple
        and bool(policies)
        and all(type(row) is BaseCapPolicy for row in policies)
        and len({row.policy_id for row in policies}) == len(policies),
        "base-cap policies changed type or repeat",
    )
    require_sha256(source_parent_receipt_sha256, "source parent receipt")
    require_sha256(map_adapter_receipt_sha256, "map adapter receipt")
    _require(obligation_mode in OBLIGATION_MODES, "obligation mode changed")
    _require(state_chain_profile in STATE_CHAIN_PROFILES, "state-chain profile changed")
    activated = tuple(row for row in rows if row.activated)
    policy_rows: list[dict[str, Any]] = []
    for policy in policies:
        per_question: list[dict[str, Any]] = []
        routes: dict[str, Counter[str]] = defaultdict(Counter)
        logical_distribution: list[int] = []
        unique_distribution: list[int] = []
        window_distribution: list[int] = []
        for row in rows:
            direct, partition, guided, unique = _selected(row, policy)
            logical = len(direct) + len(partition) + len(guided)
            window_counts = dict(row.physical_window_calls_by_source)
            _require(
                all(source_id in window_counts for source_id in unique),
                "a swept prefix escaped the physically verified D5/G2 base",
            )
            windows = sum(window_counts[source_id] for source_id in unique)
            per_question.append(
                {
                    "activated": row.activated,
                    "guided_source_ids": list(guided),
                    "history_window_token_cap": row.history_window_token_cap,
                    "logical_selection_count": logical,
                    "ordinal": row.ordinal,
                    "physical_unique_source_call_count": len(unique),
                    "physical_window_call_count": windows,
                    "partition_source_ids": list(partition),
                    "question_id": row.question_id,
                    "route_id": row.route_id,
                    "direct_source_ids": list(direct),
                    "maximum_combined_prompt_token_proxy": (
                        row.maximum_combined_prompt_token_proxy
                    ),
                }
            )
            if row.activated:
                logical_distribution.append(logical)
                unique_distribution.append(len(unique))
                window_distribution.append(windows)
                routes[row.route_id].update(
                    activated_question_count=1,
                    logical_selection_count=logical,
                    physical_unique_source_call_count=len(unique),
                    physical_window_call_count=windows,
                )
        logical_total = sum(logical_distribution)
        unique_total = sum(unique_distribution)
        window_total = sum(window_distribution)
        policy_rows.append(
            {
                **policy.projection(),
                "activated_question_count": len(activated),
                "logical_selection_count": logical_total,
                "mean_logical_selections_per_activated_question": _fraction(
                    logical_total, len(activated)
                ),
                "mean_physical_unique_source_calls_per_activated_question": _fraction(
                    unique_total, len(activated)
                ),
                "mean_physical_window_calls_per_activated_question": _fraction(
                    window_total, len(activated)
                ),
                "no_op_question_count": len(rows) - len(activated),
                "per_activated_question_distributions": {
                    "logical_selection_count": _histogram(logical_distribution),
                    "physical_unique_source_call_count": _histogram(unique_distribution),
                    "physical_window_call_count": _histogram(window_distribution),
                },
                "per_question": per_question,
                "physical_unique_source_call_count": unique_total,
                "physical_window_call_count": window_total,
                "route_totals": {
                    route_id: dict(sorted(counts.items()))
                    for route_id, counts in sorted(routes.items())
                },
            }
        )
    body = {
        "activated_question_count": len(activated),
        "base_only": True,
        "format": STRUCTURAL_FORMAT,
        "gold_loaded": False,
        "hard_caps_and_tail_steps_unchanged": {
            "direct": {"hard_source_cap": 12, "tail_step_source_cap": 2},
            "guided": {"hard_source_cap": 8, "tail_step_source_cap": 2},
            "partition": {"hard_source_cap": 10, "tail_step_source_cap": 2},
        },
        "mechanism_defaults_changed": False,
        "partition_one_is_analysis_only_isolated_control": True,
        "history_windowing": {
            "analysis_only_exact_prompt_safe_slew": True,
            "default_max_history_window_token_cap": DEFAULT_HISTORY_WINDOW_TOKEN_CAP,
            "hard_context_token_cap": HARD_CONTEXT_TOKEN_CAP,
            "maximum_combined_prompt_token_proxy": max(
                row.maximum_combined_prompt_token_proxy for row in rows
            ),
            "selected_window_token_cap_distribution": _histogram(
                [row.history_window_token_cap for row in activated]
            ),
        },
        "map_adapter_receipt_sha256": map_adapter_receipt_sha256,
        "obligation_compilation_mode": obligation_mode,
        "no_op_question_count": len(rows) - len(activated),
        "policies": policy_rows,
        "provider_calls": 0,
        "question_count": len(rows),
        "retained_transformer_token_state_bytes": 0,
        "source_parent_receipt_sha256": source_parent_receipt_sha256,
        "state_chain_profile": state_chain_profile,
        "target_plan_loaded": False,
    }
    assert_gold_blind(body, path="source_gate_policy_sweep.structural")
    return {**body, "structural_selection_sha256": identity_sha256(body)}


def _validate_target_plan(
    payload: Mapping[str, Any],
    *,
    artifact_sha256: str,
    expected_artifact_sha256: str | None,
    rows: tuple[SweepQuestion, ...],
) -> tuple[tuple[int, str, str], ...]:
    require_sha256(artifact_sha256, "target-plan artifact")
    if expected_artifact_sha256 is not None:
        _require(
            artifact_sha256
            == require_sha256(expected_artifact_sha256, "expected target-plan artifact"),
            "posthoc target-plan artifact changed",
        )
    _require(
        payload.get("format") == "memory-condense-retrieval-target-owner-plan-v1"
        and payload.get("gold_target_tags_posthoc_only") is True
        and payload.get("runtime_use_forbidden") is True
        and payload.get("provider_calls") == 0,
        "target plan is not sealed posthoc-only analysis input",
    )
    unsigned = dict(payload)
    declared = unsigned.pop("plan_sha256", None)
    _require(
        require_sha256(declared, "target-plan self seal") == identity_sha256(unsigned),
        "target-plan self seal changed",
    )
    order = payload.get("ordered_question_keys")
    _require(
        type(order) is list
        and len(order) == len(rows)
        and all(type(item) is dict for item in order)
        and tuple((item.get("ordinal"), item.get("question_id")) for item in order)
        == tuple((row.ordinal, row.question_id) for row in rows),
        "target-plan question order changed",
    )
    targets = payload.get("desired_targets")
    _require(type(targets) is list, "target-plan desired targets changed")
    source_targets: list[tuple[int, str, str]] = []
    for item in targets:
        _require(type(item) is dict, "target-plan row changed")
        if item.get("target_kind") != "source_id":
            continue
        ordinal = item.get("ordinal")
        question_id = item.get("question_id")
        source_id = item.get("target_id")
        _require(
            type(ordinal) is int
            and 0 <= ordinal < len(rows)
            and question_id == rows[ordinal].question_id,
            "source target escaped sealed question order",
        )
        require_text(source_id, "registered source target")
        require_sha256(item.get("target_sha256"), "registered source target seal")
        source_targets.append((ordinal, question_id, source_id))
    _require(
        len(source_targets) == payload.get("desired_source_target_count")
        and len(source_targets) == len(set(source_targets)),
        "registered source-target count changed or repeats",
    )
    return tuple(source_targets)


def attach_posthoc_source_coverage(
    structural: Mapping[str, Any],
    rows: tuple[SweepQuestion, ...],
    target_plan: Mapping[str, Any],
    *,
    target_plan_artifact_sha256: str,
    expected_target_plan_artifact_sha256: str | None = None,
) -> dict[str, Any]:
    """Annotate immutable selections; target data never enters gate decisions."""

    _require(
        structural.get("format") == STRUCTURAL_FORMAT
        and structural.get("target_plan_loaded") is False,
        "posthoc coverage requires a frozen gold-blind structural sweep",
    )
    unsigned = dict(structural)
    declared = unsigned.pop("structural_selection_sha256", None)
    structural_sha = require_sha256(declared, "structural selection")
    _require(
        structural_sha == identity_sha256(unsigned),
        "structural selection changed before posthoc annotation",
    )
    targets = _validate_target_plan(
        target_plan,
        artifact_sha256=target_plan_artifact_sha256,
        expected_artifact_sha256=expected_target_plan_artifact_sha256,
        rows=rows,
    )
    baseline = {row.ordinal: set(row.baseline_source_ids) for row in rows}
    def source_hit(ordinal: int, question_id: str, source_id: str, reached: Mapping[int, set[str]]) -> bool:
        values = reached[ordinal]
        return source_id in values or f"{question_id}::{source_id}" in values

    baseline_covered = sum(
        source_hit(ordinal, question_id, source_id, baseline)
        for ordinal, question_id, source_id in targets
    )
    structural_policies = structural.get("policies")
    _require(type(structural_policies) is list, "structural policies changed")
    coverage: list[dict[str, Any]] = []
    for policy in structural_policies:
        _require(type(policy) is dict, "structural policy row changed")
        per_question = policy.get("per_question")
        _require(
            type(per_question) is list and len(per_question) == len(rows),
            "structural per-question selection changed",
        )
        reached = {ordinal: set(values) for ordinal, values in baseline.items()}
        for expected, selected in zip(rows, per_question, strict=True):
            _require(
                type(selected) is dict
                and selected.get("ordinal") == expected.ordinal
                and selected.get("question_id") == expected.question_id
                and selected.get("route_id") == expected.route_id
                and selected.get("activated") is expected.activated,
                "posthoc coverage row changed structural routing",
            )
            direct = selected.get("direct_source_ids")
            partition = selected.get("partition_source_ids")
            guided = selected.get("guided_source_ids")
            _require(
                type(direct) is list
                and type(partition) is list
                and type(guided) is list
                and all(type(value) is str for value in (*direct, *partition, *guided)),
                "posthoc coverage source prefix changed",
            )
            reached[expected.ordinal].update((*direct, *partition, *guided))
        covered = sum(
            source_hit(ordinal, question_id, source_id, reached)
            for ordinal, question_id, source_id in targets
        )
        coverage.append(
            {
                "baseline_covered_source_target_count": baseline_covered,
                "covered_source_target_count": covered,
                "incremental_covered_source_target_count": covered - baseline_covered,
                "physical_unique_source_call_count": policy[
                    "physical_unique_source_call_count"
                ],
                "physical_window_call_count": policy["physical_window_call_count"],
                "policy_id": policy["policy_id"],
                "source_target_coverage": _fraction(covered, len(targets)),
                "uncovered_source_target_count": len(targets) - covered,
            }
        )
    for cost_field, pareto_field in (
        ("physical_unique_source_call_count", "pareto_on_unique_source_calls"),
        ("physical_window_call_count", "pareto_on_physical_window_calls"),
    ):
        for candidate in coverage:
            dominated = any(
                other["covered_source_target_count"]
                >= candidate["covered_source_target_count"]
                and other[cost_field] <= candidate[cost_field]
                and (
                    other["covered_source_target_count"]
                    > candidate["covered_source_target_count"]
                    or other[cost_field] < candidate[cost_field]
                )
                for other in coverage
            )
            candidate[pareto_field] = not dominated
    body = {
        "coverage_call_pareto": coverage,
        "format": FORMAT,
        "posthoc_analysis_only": True,
        "provider_calls": 0,
        "registered_source_target_count": len(targets),
        "runtime_use_forbidden": True,
        "selection_and_routing_frozen_before_target_plan_load": True,
        "source_target_match_rule": "exact_or_question_id_double_colon_qualified_v1",
        "structural_selection": dict(structural),
        "structural_selection_sha256": structural_sha,
        "target_plan_artifact_sha256": target_plan_artifact_sha256,
        "target_plan_loaded_after_structural_selection": True,
        "target_plan_self_sha256": target_plan["plan_sha256"],
    }
    return {**body, "analysis_sha256": identity_sha256(body)}


def build_real_analysis(
    *,
    max_concurrency: int,
    gateway_url: str,
    obligation_mode: str = CONSOLIDATED_OBLIGATION_MODE,
    state_chain_profile: str = STATE_CHAIN_DIRECT_AUTHORITY_PROFILE,
) -> dict[str, Any]:
    """Replay sealed parents once, then sweep a verified D5/P1/G2 superset."""

    query_run, map_plan, map_plane, adapter = source_cli.load_locked_query_map(
        max_concurrency=max_concurrency,
        gateway_url=gateway_url,
        obligation_mode=obligation_mode,
        state_chain_profile=state_chain_profile,
    )
    activations = source_cli.activation_inputs_from_query_map(adapter)
    superset_policy = SourceGatePolicy(
        "analysis-only-direct5-partition1-guided2-base-superset-v1",
        (
            LaneSourceBudget(FactLane.DIRECT, 5, 12, 2),
            LaneSourceBudget(FactLane.PARTITION, 1, 10, 2),
            LaneSourceBudget(FactLane.GUIDED, 2, 8, 2),
        ),
        global_unique_source_cap=24,
        max_physical_map_calls=48,
        max_rounds=16,
    )
    population = load_locked_source_gate_adapter(
        activations,
        policy=superset_policy,
    )
    base = source_cli.build_locked_base_round(
        population,
        query_adapter=adapter,
    )
    base_by_ordinal = {row.ordinal: row for row in base.questions}
    gate_by_ordinal = {row.ordinal: row for row in population.questions}
    _require(
        set(gate_by_ordinal) == set(base_by_ordinal)
        == {row.ordinal for row in adapter.activated_rows},
        "locked base and terminal-map activations differ",
    )
    sweep_rows: list[SweepQuestion] = []
    for planned in map_plan.rows:
        packet = planned.direct_plan_row.adapter.source.packet
        evidence = (
            *packet.protected_evidence,
            *planned.retained_query_delta,
        )
        baseline_sources = tuple(dict.fromkeys(row.source_id for row in evidence))
        gate_question = gate_by_ordinal.get(planned.ordinal)
        if gate_question is None:
            sweep_rows.append(
                SweepQuestion(
                    planned.ordinal,
                    packet.question_id,
                    planned.route.style.value,
                    None,
                    False,
                    baseline_sources,
                    (),
                    (),
                    (),
                    (),
                )
            )
            continue
        gate = gate_question.plan
        base_row = base_by_ordinal[planned.ordinal]
        round_plan = base_row.gate_round
        mapping_plan = base_row.mapping_plan
        window_cap = base_row.hydration_plan.max_window_tokens
        maximum_combined = base_row.mapper_preflight.maximum_combined_token_proxy
        direct = tuple(row.source_id for row in gate.candidates_for(FactLane.DIRECT))
        partition = tuple(
            row.source_id for row in gate.candidates_for(FactLane.PARTITION)
        )
        guided = tuple(row.source_id for row in gate.candidates_for(FactLane.GUIDED))
        physical = Counter(
            row.source_id for row in mapping_plan.work_items
        )
        max_prefix = tuple(
            dict.fromkeys((*direct[:5], *partition[:1], *guided[:2]))
        )
        _require(
            set(physical) == set(max_prefix)
            and len(round_plan.selections)
            == min(5, len(direct)) + min(1, len(partition)) + min(2, len(guided)),
            "verified D5/P1/G2 physical work differs from swept prefix",
        )
        sweep_rows.append(
            SweepQuestion(
                planned.ordinal,
                packet.question_id,
                planned.route.style.value,
                gate.parent.namespace_id,
                True,
                baseline_sources,
                direct,
                partition,
                guided,
                tuple((source_id, physical[source_id]) for source_id in max_prefix),
                window_cap,
                maximum_combined,
            )
        )
    rows = tuple(sweep_rows)
    structural = sweep_base_policies(
        rows,
        source_parent_receipt_sha256=population.receipt_sha256,
        map_adapter_receipt_sha256=adapter.receipt_sha256,
        obligation_mode=obligation_mode,
        state_chain_profile=state_chain_profile,
    )
    # The registered target plan is deliberately opened only after the
    # structural selection hash above exists.
    target = read_sealed_json(DEFAULT_TARGET_PLAN)
    return attach_posthoc_source_coverage(
        structural,
        rows,
        target.payload,
        target_plan_artifact_sha256=target.sha256,
        expected_target_plan_artifact_sha256=EXPECTED_TARGET_PLAN_SHA256,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-concurrency", type=int, default=4)
    parser.add_argument("--gateway-url", default=source_cli.live.DEFAULT_GATEWAY_URL)
    parser.add_argument(
        "--obligation-mode",
        choices=sorted(OBLIGATION_MODES),
        default=CONSOLIDATED_OBLIGATION_MODE,
    )
    parser.add_argument(
        "--state-chain-profile",
        choices=sorted(STATE_CHAIN_PROFILES),
        default=STATE_CHAIN_DIRECT_AUTHORITY_PROFILE,
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    result = build_real_analysis(
        max_concurrency=args.max_concurrency,
        gateway_url=str(args.gateway_url),
        obligation_mode=args.obligation_mode,
        state_chain_profile=args.state_chain_profile,
    )
    artifact = publish_sealed_json(args.output, result)
    summary = {
        "analysis_sha256": result["analysis_sha256"],
        "artifact": artifact.path.as_posix(),
        "artifact_sha256": artifact.sha256,
        "coverage_call_pareto": result["coverage_call_pareto"],
        "gold_loaded_for_runtime": False,
        "posthoc_analysis_only": True,
        "provider_calls": 0,
        "structural_selection_sha256": result["structural_selection_sha256"],
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
