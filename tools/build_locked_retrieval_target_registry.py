#!/usr/bin/env python3
"""Build the posthoc desired-target registry for the locked retrieval matrix.

The registry is deliberately built from the benchmark's labeled source
sessions, not from any retrieval arm's candidates.  All answer runs and their
zero-call replays are verified before the gold-bearing topology ledger is
opened.  The resulting tags are analysis-only and must never be read by a
retrieval or answer runtime.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    _ROOT = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(_ROOT / "src"), str(_ROOT)]

from memory_condense.domain.discourse import identity_sha256
from memory_condense.eval import run_fast_1m_em_facts as em_runner
from tools._locked_em_repair_adapter import _read_canonical_artifact
from tools import score_locked_retrieval_target_ownership as scorer


FORMAT = scorer.REGISTRY_FORMAT
PLAN_FORMAT = "memory-condense-retrieval-target-owner-plan-v1"
POLICY_FORMAT = "memory-condense-retrieval-target-owner-policy-v1"
STYLE_FORMAT = "memory-condense-locked-retrieval-style-ledger-v1"
REQUIRED_METHODS = ("s0", "em", "representative", "global", "hebbian", "cav")
RELATIONAL_OPERATORS = frozenset(
    {
        "state_update",
        "temporal_interval",
        "temporal_order_select",
        "numeric_aggregate_compare",
        "set_or_list_join",
        "preference_synthesis",
    }
)


class RegistryBuildError(ValueError):
    pass


def _require(ok: Any, message: str) -> None:
    if not ok:
        raise RegistryBuildError(message)


def _sha(value: object, label: str) -> str:
    _require(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value),
        f"invalid {label}",
    )
    return str(value)


def _source_owner(row: Mapping[str, Any]) -> str:
    topology = row.get("retrieval_topology")
    operator = row.get("answer_operator")
    if topology == "point":
        return "hebbian" if operator == "preference_synthesis" else "s0"
    if topology in {"local_pair", "local_fanout"}:
        return "em"
    if topology == "dispersed_join":
        mapping = {
            "direct_lookup": "s0",
            "insufficient_evidence": "s0",
            "state_update": "s0",
            "numeric_aggregate_compare": "em",
            "set_or_list_join": "em",
            "temporal_interval": "representative",
            "temporal_order_select": "global",
            "preference_synthesis": "hebbian",
        }
        _require(operator in mapping, f"unowned dispersed operator: {operator}")
        return mapping[str(operator)]
    raise RegistryBuildError(f"unowned retrieval topology: {topology}")


def _target(
    *,
    ordinal: int,
    question_id: str,
    target_kind: str,
    target_id: str,
    primary_owner: str,
    basis: Mapping[str, Any],
) -> dict[str, Any]:
    body = {
        "ordinal": ordinal,
        "question_id": question_id,
        "target_kind": target_kind,
        "target_id": target_id,
        "primary_owner": primary_owner,
    }
    return body | {
        "target_sha256": identity_sha256(body),
        "assignment_basis": dict(basis),
        "assignment_basis_sha256": identity_sha256(dict(basis)),
    }


def _policy() -> dict[str, Any]:
    value = {
        "format": POLICY_FORMAT,
        "target_universe": (
            "all benchmark-labeled answer-session source targets, plus one "
            "operator-relation target for each relational question and one "
            "coverage-check target for each insufficient-evidence question"
        ),
        "source_assignment": {
            "point/default": "s0",
            "point/preference_synthesis": "hebbian",
            "local_pair_or_fanout": "em",
            "dispersed/direct_lookup_or_state_update_or_insufficient": "s0",
            "dispersed/numeric_or_set_join": "em",
            "dispersed/temporal_interval": "representative",
            "dispersed/temporal_order_select": "global",
            "dispersed/preference_synthesis": "hebbian",
        },
        "relation_assignment": "cav",
        "coverage_check_assignment": "s0",
        "owner_route_bindings": {
            owner: list(routes)
            for owner, routes in scorer.DEFAULT_OWNER_ROUTE_BINDINGS.items()
            if owner in REQUIRED_METHODS
        },
        "shared_reachability": (
            "measured later from every arm's pre-dedup discovery ledger; it "
            "does not change primary responsibility"
        ),
        "runtime_use_forbidden": True,
        "gold_posthoc_only": True,
    }
    return value | {"policy_sha256": identity_sha256(value)}


def _load_style(path: Path, expected_sha256: str) -> tuple[dict[str, Any], str]:
    value, observed = _read_canonical_artifact(
        path, expected_sha256=_sha(expected_sha256, "style-ledger SHA-256")
    )
    _require(value.get("format") == STYLE_FORMAT, "style-ledger format changed")
    _require(value.get("provider_calls") == 0, "style ledger made provider calls")
    _require(isinstance(value.get("questions"), list), "style-ledger questions missing")
    return value, observed


def _plan_from_style(
    style: Mapping[str, Any], observed_style_sha: str
) -> dict[str, Any]:
    """Build the immutable target/owner projection without answer outcomes."""

    bindings = style.get("bindings")
    rows = style.get("questions")
    _require(isinstance(bindings, Mapping), "style-ledger bindings missing")
    _require(isinstance(rows, list), "style-ledger rows missing")
    population_sha = _sha(
        bindings.get("population_identity_sha256"),
        "style population identity SHA-256",
    )
    ordered_question_keys: list[dict[str, Any]] = []
    targets: list[dict[str, Any]] = []
    for ordinal, row in enumerate(rows):
        _require(isinstance(row, Mapping), f"style row {ordinal} is invalid")
        question_id = row.get("question_id")
        _require(
            row.get("ordinal") == ordinal
            and isinstance(question_id, str)
            and question_id,
            f"style question order changed at {ordinal}",
        )
        ordered_question_keys.append(
            {"ordinal": ordinal, "question_id": question_id}
        )
        source_ids = row.get("expected_source_ids")
        positions = row.get("expected_source_positions")
        _require(
            isinstance(source_ids, list)
            and source_ids
            and len(source_ids) == row.get("expected_source_count")
            and len(source_ids) == len(set(source_ids))
            and isinstance(positions, list)
            and len(positions) == len(source_ids),
            f"desired source universe changed at {ordinal}",
        )
        owner = _source_owner(row)
        scope = {
            "question_id": question_id,
            "expected_source_ids": source_ids,
            "expected_source_positions": positions,
        }
        scope_sha = identity_sha256(scope)
        for source_id, position in zip(source_ids, positions, strict=True):
            _require(isinstance(source_id, str) and source_id, "source target ID is empty")
            basis = {
                "answer_operator": row.get("answer_operator"),
                "benchmark_category": row.get("benchmark_category"),
                "retrieval_topology": row.get("retrieval_topology"),
                "source_position": position,
                "source_scope_sha256": scope_sha,
                "rule": "deterministic_source_responsibility",
            }
            targets.append(
                _target(
                    ordinal=ordinal,
                    question_id=str(question_id),
                    target_kind="source_id",
                    target_id=source_id,
                    primary_owner=owner,
                    basis=basis,
                )
            )

        operator = row.get("answer_operator")
        if operator in RELATIONAL_OPERATORS:
            relation_identity = {
                "question_id": question_id,
                "answer_operator": operator,
                "expected_source_ids": source_ids,
                "source_scope_sha256": scope_sha,
            }
            targets.append(
                _target(
                    ordinal=ordinal,
                    question_id=str(question_id),
                    target_kind="relation",
                    target_id=identity_sha256(relation_identity),
                    primary_owner="cav",
                    basis=relation_identity | {"rule": "operator_relation"},
                )
            )
        if operator == "insufficient_evidence":
            coverage_identity = {
                "question_id": question_id,
                "expected_source_ids": source_ids,
                "source_scope_sha256": scope_sha,
            }
            targets.append(
                _target(
                    ordinal=ordinal,
                    question_id=str(question_id),
                    target_kind="coverage_check",
                    target_id=identity_sha256(coverage_identity),
                    primary_owner="s0",
                    basis=coverage_identity | {"rule": "unsupported_conclusion_check"},
                )
            )

    target_keys = [
        (row["ordinal"], row["target_kind"], row["target_id"]) for row in targets
    ]
    _require(len(target_keys) == len(set(target_keys)), "target universe contains duplicates")
    owner_sets = {
        method: {row["target_sha256"] for row in targets if row["primary_owner"] == method}
        for method in REQUIRED_METHODS
    }
    for index, method in enumerate(REQUIRED_METHODS):
        for other in REQUIRED_METHODS[index + 1 :]:
            _require(not owner_sets[method] & owner_sets[other], "primary-owner sets overlap")
    declared_universe = {row["target_sha256"] for row in targets}
    _require(
        set().union(*owner_sets.values()) == declared_universe,
        "primary-owner union does not equal desired target universe",
    )

    policy = _policy()
    owner_counts = Counter(row["primary_owner"] for row in targets)
    plan: dict[str, Any] = {
        "format": PLAN_FORMAT,
        "population_identity_sha256": population_sha,
        "style_ledger_sha256": observed_style_sha,
        "policy": policy,
        "policy_sha256": policy["policy_sha256"],
        "question_count": len(ordered_question_keys),
        "ordered_question_keys": ordered_question_keys,
        "desired_target_count": len(targets),
        "desired_source_target_count": sum(
            row["target_kind"] == "source_id" for row in targets
        ),
        "desired_relation_target_count": sum(
            row["target_kind"] == "relation" for row in targets
        ),
        "desired_coverage_check_target_count": sum(
            row["target_kind"] == "coverage_check" for row in targets
        ),
        "primary_owner_counts": {
            method: owner_counts[method] for method in REQUIRED_METHODS
        },
        "desired_target_universe_sha256": identity_sha256(
            [row["target_sha256"] for row in targets]
        ),
        "desired_targets": targets,
        "answer_run_or_judge_inputs_loaded": False,
        "gold_target_tags_posthoc_only": True,
        "runtime_use_forbidden": True,
        "every_target_has_exactly_one_primary_owner": True,
        "primary_owner_sets_pairwise_disjoint": True,
        "primary_owner_union_equals_declared_target_universe": True,
        "unassigned_primary_owner_count": 0,
        "provider_calls": 0,
    }
    plan["plan_sha256"] = scorer._self_sha(plan, "plan_sha256")
    return plan


def _validate_plan(value: Mapping[str, Any]) -> dict[str, Any]:
    _require(value.get("format") == PLAN_FORMAT, "target-plan format changed")
    _require(
        value.get("plan_sha256") == scorer._self_sha(value, "plan_sha256"),
        "target-plan self-hash changed",
    )
    _require(
        value.get("answer_run_or_judge_inputs_loaded") is False
        and value.get("runtime_use_forbidden") is True
        and value.get("provider_calls") == 0,
        "target plan crossed the pre-arm firewall",
    )
    policy = value.get("policy")
    questions = value.get("ordered_question_keys")
    targets = value.get("desired_targets")
    _require(
        isinstance(policy, Mapping)
        and policy.get("policy_sha256") == identity_sha256(
            {key: child for key, child in policy.items() if key != "policy_sha256"}
        )
        and value.get("policy_sha256") == policy.get("policy_sha256"),
        "target-plan policy changed",
    )
    _require(
        isinstance(questions, list)
        and value.get("question_count") == len(questions)
        and isinstance(targets, list)
        and value.get("desired_target_count") == len(targets),
        "target-plan population changed",
    )
    for ordinal, question in enumerate(questions):
        _require(
            question == {
                "ordinal": ordinal,
                "question_id": question.get("question_id"),
            }
            and isinstance(question.get("question_id"), str)
            and question["question_id"],
            f"target-plan question order changed at {ordinal}",
        )
    keys: set[tuple[int, str, str]] = set()
    owner_counts: Counter[str] = Counter()
    for target in targets:
        _require(isinstance(target, Mapping), "target-plan target is invalid")
        ordinal = target.get("ordinal")
        _require(
            type(ordinal) is int
            and 0 <= ordinal < len(questions)
            and target.get("question_id") == questions[ordinal]["question_id"],
            "target-plan target question binding changed",
        )
        body = {
            "ordinal": ordinal,
            "question_id": target.get("question_id"),
            "target_kind": target.get("target_kind"),
            "target_id": target.get("target_id"),
            "primary_owner": target.get("primary_owner"),
        }
        _require(
            body["primary_owner"] in REQUIRED_METHODS
            and target.get("target_sha256") == identity_sha256(body)
            and target.get("assignment_basis_sha256")
            == identity_sha256(target.get("assignment_basis")),
            "target-plan target identity changed",
        )
        key = (ordinal, str(body["target_kind"]), str(body["target_id"]))
        _require(key not in keys, "target-plan target is duplicated")
        keys.add(key)
        owner_counts[str(body["primary_owner"])] += 1
    expected_counts = {
        method: owner_counts[method] for method in REQUIRED_METHODS
    }
    _require(
        value.get("primary_owner_counts") == expected_counts
        and value.get("desired_target_universe_sha256")
        == identity_sha256([target["target_sha256"] for target in targets])
        and value.get("unassigned_primary_owner_count") == 0
        and value.get("every_target_has_exactly_one_primary_owner") is True
        and value.get("primary_owner_sets_pairwise_disjoint") is True
        and value.get("primary_owner_union_equals_declared_target_universe")
        is True,
        "target-plan owner union changed",
    )
    return dict(value)


def build_target_plan(
    *,
    style_ledger_path: Path,
    style_ledger_sha256: str,
    output_path: Path,
) -> tuple[dict[str, Any], str]:
    style, observed_style_sha = _load_style(
        style_ledger_path, style_ledger_sha256
    )
    plan = _validate_plan(_plan_from_style(style, observed_style_sha))
    return plan, em_runner._publish(output_path, plan)


def build_registry(
    *,
    answer_run_specs: Sequence[str],
    target_plan_path: Path,
    target_plan_sha256: str,
    output_path: Path,
) -> tuple[dict[str, Any], str]:
    # Run/replay seals are verified before the gold-bearing plan is opened.
    answer_runs = scorer._verify_answer_runs(answer_run_specs)
    _require(
        tuple(answer_runs) == REQUIRED_METHODS,
        "answer-run methods/order must be s0,em,representative,global,hebbian,cav",
    )
    raw_plan, observed_plan_sha = _read_canonical_artifact(
        target_plan_path,
        expected_sha256=_sha(target_plan_sha256, "target-plan SHA-256"),
    )
    plan = _validate_plan(raw_plan)
    first = next(iter(answer_runs.values()))
    _require(
        plan["population_identity_sha256"]
        == first["population_identity_sha256"]
        and plan["question_count"] == first["question_count"],
        "target plan belongs to another answer population",
    )
    ordered_questions = []
    for ordinal, (planned, run_row) in enumerate(
        zip(plan["ordered_question_keys"], first["questions"], strict=True)
    ):
        _require(
            planned
            == {"ordinal": ordinal, "question_id": run_row.get("question_id")},
            f"target plan/run question order changed at {ordinal}",
        )
        ordered_questions.append(
            {
                "ordinal": ordinal,
                "question_id": run_row.get("question_id"),
                "question_sha256": run_row.get("question_sha256"),
                "dated_question_sha256": run_row.get("dated_question_sha256"),
            }
        )
    registry: dict[str, Any] = {
        "format": FORMAT,
        "population_identity_sha256": plan["population_identity_sha256"],
        "answer_run_bindings": [
            {
                "discovering_method": method,
                "arm_label": run["arm_label"],
                "run_sha256": run["run_sha256"],
                "run_replay_sha256": run["run_replay_sha256"],
            }
            for method, run in answer_runs.items()
        ],
        "target_plan_file_sha256": observed_plan_sha,
        "target_plan_identity_sha256": plan["plan_sha256"],
        "target_plan": plan,
        "style_ledger_sha256": plan["style_ledger_sha256"],
        "policy": plan["policy"],
        "policy_sha256": plan["policy_sha256"],
        "question_count": plan["question_count"],
        "ordered_questions": ordered_questions,
        "desired_target_count": plan["desired_target_count"],
        "desired_source_target_count": plan["desired_source_target_count"],
        "desired_relation_target_count": plan["desired_relation_target_count"],
        "desired_coverage_check_target_count": plan[
            "desired_coverage_check_target_count"
        ],
        "primary_owner_counts": plan["primary_owner_counts"],
        "desired_target_universe_sha256": plan[
            "desired_target_universe_sha256"
        ],
        "desired_targets": plan["desired_targets"],
        "constructed_after_all_answer_run_seals": True,
        "answer_runs_verified_before_gold_target_plan_load": True,
        "gold_target_tags_posthoc_only": True,
        "runtime_use_forbidden": True,
        "immutable_target_plan_reproduced_byte_for_byte": True,
        "every_target_has_exactly_one_primary_owner": True,
        "primary_owner_sets_pairwise_disjoint": True,
        "primary_owner_union_equals_declared_target_universe": True,
        "unassigned_primary_owner_count": 0,
        "provider_calls": 0,
    }
    registry["registry_sha256"] = scorer._self_sha(registry, "registry_sha256")
    return registry, em_runner._publish(output_path, registry)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("plan", "registry"))
    parser.add_argument(
        "--answer-run", action="append", default=[], metavar="METHOD=PATH=SHA256"
    )
    parser.add_argument("--style-ledger", type=Path)
    parser.add_argument("--expected-style-ledger-sha256")
    parser.add_argument("--target-plan", type=Path)
    parser.add_argument("--expected-target-plan-sha256")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.phase == "plan":
        _require(
            args.style_ledger is not None
            and args.expected_style_ledger_sha256 is not None
            and not args.answer_run
            and args.target_plan is None
            and args.expected_target_plan_sha256 is None,
            "plan phase requires only the hash-pinned style ledger",
        )
        plan, digest = build_target_plan(
            style_ledger_path=args.style_ledger,
            style_ledger_sha256=args.expected_style_ledger_sha256,
            output_path=args.output,
        )
        counts = plan["primary_owner_counts"]
        print(
            f"Target plan {digest}: targets={plan['desired_target_count']}; "
            f"owners={json.dumps(counts, sort_keys=True, separators=(',', ':'))}; "
            "answer_inputs=0; provider_calls=0"
        )
        return 0
    _require(
        args.target_plan is not None
        and args.expected_target_plan_sha256 is not None
        and args.style_ledger is None
        and args.expected_style_ledger_sha256 is None,
        "registry phase requires the immutable hash-pinned target plan",
    )
    registry, digest = build_registry(
        answer_run_specs=args.answer_run,
        target_plan_path=args.target_plan,
        target_plan_sha256=args.expected_target_plan_sha256,
        output_path=args.output,
    )
    counts = registry["primary_owner_counts"]
    print(
        f"Target registry {digest}: targets={registry['desired_target_count']}; "
        f"owners={json.dumps(counts, sort_keys=True, separators=(',', ':'))}; "
        "unassigned=0; provider_calls=0"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "FORMAT",
    "PLAN_FORMAT",
    "POLICY_FORMAT",
    "RELATIONAL_OPERATORS",
    "REQUIRED_METHODS",
    "RegistryBuildError",
    "build_parser",
    "build_registry",
    "build_target_plan",
    "main",
]
