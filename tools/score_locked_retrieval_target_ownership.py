#!/usr/bin/env python3
"""Provider-free target-owner recall over sealed structural arm ledgers.

All answer run/replay pairs and gold-blind structural ledgers are verified
before the posthoc desired-target registry is opened. Discovery is measured
from each ledger's declared pre-dedup projection; admission is measured from
its declared post-dedup projection.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import re
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    _ROOT = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(_ROOT / "src"), str(_ROOT)]

from memory_condense.domain.discourse import identity_sha256
from memory_condense.eval import run_fast_1m_em_facts as em_runner
from memory_condense.eval._artifact_json import canonical_json_bytes
from tools._locked_em_repair_adapter import _read_canonical_artifact


REGISTRY_FORMAT = "memory-condense-retrieval-target-owner-registry-v1"
LEDGER_FORMAT = "memory-condense-structural-target-ledger-v1"
SCORE_FORMAT = "memory-condense-retrieval-target-owner-score-v1"
DEFAULT_OWNER_ROUTE_BINDINGS: dict[str, tuple[str, ...]] = {
    "s0": ("s0", "causal_graph_coverage_predecessor"),
    "em": (
        "em",
        "direct_episode_additions",
        "post_selection_em_fact_conversion_v2",
    ),
    "representative": ("representative", "S0_PLUS_REPRESENTATIVE_BRIDGE"),
    "global": ("global", "S0_PLUS_ARTIFACT_GLOBAL"),
    "hebbian": ("hebbian", "h1", "causal_hebbian_h1"),
    "cav": ("cav", "genuine_cav_v2_two_pass"),
}
_LEDGER_FORBIDDEN = frozenset(
    {
        "answer_session_ids",
        "benchmark_category",
        "gold",
        "gold_answer",
        "primary_owner",
        "reference",
        "reference_answer",
    }
)
_LOADER = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*:[A-Za-z_][A-Za-z0-9_]*$")


class TargetCoverageError(ValueError):
    pass


def _require(ok: Any, message: str) -> None:
    if not ok:
        raise TargetCoverageError(message)


def _sha(value: object, label: str) -> str:
    _require(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value),
        f"invalid {label}",
    )
    return str(value)


def _self_sha(value: Mapping[str, Any], field: str) -> str:
    return identity_sha256(
        {key: child for key, child in value.items() if key != field}
    )


def _read(path: Path, expected: str) -> tuple[dict[str, Any], str]:
    return _read_canonical_artifact(
        path, expected_sha256=_sha(expected, f"{path} SHA-256")
    )


def _parse_file_spec(value: str, label: str) -> tuple[str, Path, str]:
    try:
        name, remainder = value.split("=", 1)
        path, digest = remainder.rsplit("=", 1)
    except ValueError as exc:
        raise TargetCoverageError(f"{label} must be NAME=PATH=SHA256") from exc
    _require(name and path, f"invalid {label}")
    return name, Path(path), _sha(digest, f"{label} SHA-256")


def _contains_key(value: object, forbidden: frozenset[str]) -> bool:
    if isinstance(value, Mapping):
        return any(
            str(key).casefold() in forbidden or _contains_key(child, forbidden)
            for key, child in value.items()
        )
    return isinstance(value, Sequence) and not isinstance(
        value, (str, bytes)
    ) and any(_contains_key(child, forbidden) for child in value)


def _verify_answer_runs(specs: Sequence[str]) -> dict[str, dict[str, Any]]:
    """Verify every answer run/replay before any gold-bearing file is read."""

    _require(specs, "at least one --answer-run is required")
    result: dict[str, dict[str, Any]] = {}
    for spec in specs:
        method, path, expected = _parse_file_spec(spec, "answer-run")
        _require(method not in result, f"duplicate answer-run method: {method}")
        run, digest = _read(path, expected)
        replay, replay_digest = _read(path.with_name("run-replay.json"), expected)
        _require(
            canonical_json_bytes(run) == canonical_json_bytes(replay)
            and digest == replay_digest,
            f"answer run/replay differ for {method}",
        )
        _require(
            run.get("gold_loaded") is False
            and isinstance(run.get("arm_label"), str)
            and isinstance(run.get("questions"), list),
            f"answer run is not sealed and gold-blind for {method}",
        )
        result[method] = {
            "discovering_method": method,
            "arm_label": run["arm_label"],
            "run_sha256": digest,
            "run_replay_sha256": replay_digest,
            "run_path": path,
            "population_identity_sha256": run.get("population_identity_sha256"),
            "question_count": run.get("question_count"),
            "questions": run["questions"],
        }
    first = next(iter(result.values()))
    first_order = [
        (
            row.get("ordinal"),
            row.get("question_id"),
            row.get("question_sha256"),
            row.get("dated_question_sha256"),
        )
        for row in first["questions"]
    ]
    for method, run in result.items():
        order = [
            (
                row.get("ordinal"),
                row.get("question_id"),
                row.get("question_sha256"),
                row.get("dated_question_sha256"),
            )
            for row in run["questions"]
        ]
        _require(
            run["population_identity_sha256"]
            == first["population_identity_sha256"]
            and run["question_count"] == first["question_count"]
            and order == first_order,
            f"answer-run population differs for {method}",
        )
    return result


def _ordered_questions(
    answer_runs: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    first = next(iter(answer_runs.values()))
    return [
        {
            "ordinal": ordinal,
            "question_id": row.get("question_id"),
            "question_sha256": row.get("question_sha256"),
            "dated_question_sha256": row.get("dated_question_sha256"),
        }
        for ordinal, row in enumerate(first["questions"])
    ]


def _validate_registry(
    value: Mapping[str, Any],
    answer_runs: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    _require(
        value.get("format") == REGISTRY_FORMAT, "target registry format changed"
    )
    _require(
        value.get("constructed_after_all_answer_run_seals") is True,
        "registry predates answer seals",
    )
    _require(
        value.get("gold_target_tags_posthoc_only") is True,
        "registry does not mark gold tags posthoc",
    )
    _require(
        value.get("registry_sha256") == _self_sha(value, "registry_sha256"),
        "registry self-hash changed",
    )
    plan = value.get("target_plan")
    if plan is not None:
        _require(
            isinstance(plan, Mapping)
            and plan.get("format")
            == "memory-condense-retrieval-target-owner-plan-v1"
            and plan.get("plan_sha256") == _self_sha(plan, "plan_sha256")
            and value.get("target_plan_identity_sha256")
            == plan.get("plan_sha256")
            and value.get("target_plan_file_sha256")
            == hashlib.sha256(canonical_json_bytes(plan)).hexdigest()
            and value.get("immutable_target_plan_reproduced_byte_for_byte")
            is True,
            "immutable target-plan binding changed",
        )
        copied_fields = (
            "population_identity_sha256",
            "style_ledger_sha256",
            "policy",
            "policy_sha256",
            "question_count",
            "desired_target_count",
            "desired_source_target_count",
            "desired_relation_target_count",
            "desired_coverage_check_target_count",
            "primary_owner_counts",
            "desired_target_universe_sha256",
            "desired_targets",
        )
        _require(
            all(value.get(field) == plan.get(field) for field in copied_fields),
            "registry does not reproduce its immutable target plan",
        )
    bindings = value.get("answer_run_bindings")
    questions = value.get("ordered_questions")
    targets = value.get("desired_targets")
    _require(
        isinstance(bindings, list)
        and isinstance(questions, list)
        and isinstance(targets, list),
        "registry populations are missing",
    )
    expected_bindings = [
        {
            "discovering_method": method,
            "arm_label": row["arm_label"],
            "run_sha256": row["run_sha256"],
            "run_replay_sha256": row["run_replay_sha256"],
        }
        for method, row in answer_runs.items()
    ]
    _require(bindings == expected_bindings, "registry answer-run bindings changed")
    expected_questions = _ordered_questions(answer_runs)
    first = next(iter(answer_runs.values()))
    _require(
        value.get("population_identity_sha256")
        == first["population_identity_sha256"]
        and value.get("question_count") == first["question_count"] == len(questions)
        and questions == expected_questions,
        "registry/run population binding changed",
    )

    owners = set(answer_runs)
    seen: set[tuple[int, str, str]] = set()
    normalized: list[dict[str, Any]] = []
    for target in targets:
        _require(isinstance(target, Mapping), "desired target must be an object")
        ordinal = target.get("ordinal")
        kind = target.get("target_kind")
        target_id = target.get("target_id")
        owner = target.get("primary_owner")
        _require(
            type(ordinal) is int and 0 <= ordinal < len(questions),
            "target ordinal changed",
        )
        _require(
            target.get("question_id") == questions[ordinal]["question_id"],
            "target question binding changed",
        )
        _require(
            isinstance(kind, str)
            and kind
            and isinstance(target_id, str)
            and target_id,
            "target identity is empty",
        )
        _require(
            isinstance(owner, str) and owner in owners,
            "target has no exact primary owner",
        )
        key = (ordinal, kind, target_id)
        _require(
            key not in seen,
            "desired target has more than one primary-owner row",
        )
        seen.add(key)
        body = {
            "ordinal": ordinal,
            "question_id": target["question_id"],
            "target_kind": kind,
            "target_id": target_id,
            "primary_owner": owner,
        }
        _require(
            target.get("target_sha256") == identity_sha256(body),
            "desired target binding changed",
        )
        basis = target.get("assignment_basis")
        if basis is not None:
            _require(
                isinstance(basis, Mapping)
                and target.get("assignment_basis_sha256")
                == identity_sha256(dict(basis)),
                "target assignment basis changed",
            )
        normalized.append(
            body
            | {
                "target_sha256": target["target_sha256"],
                "assignment_basis": None if basis is None else dict(basis),
            }
        )
    _require(
        value.get("desired_target_count") == len(normalized),
        "desired target count changed",
    )
    if "primary_owner_counts" in value:
        counts = {
            owner: sum(row["primary_owner"] == owner for row in normalized)
            for owner in answer_runs
        }
        _require(
            value["primary_owner_counts"] == counts,
            "primary-owner counts changed",
        )
    if "desired_target_universe_sha256" in value:
        _require(
            value["desired_target_universe_sha256"]
            == identity_sha256([row["target_sha256"] for row in normalized]),
            "desired target universe changed",
        )
    for flag in (
        "every_target_has_exactly_one_primary_owner",
        "primary_owner_sets_pairwise_disjoint",
        "primary_owner_union_equals_declared_target_universe",
    ):
        if flag in value:
            _require(value[flag] is True, f"registry invariant failed: {flag}")
    _require(
        value.get("unassigned_primary_owner_count", 0) == 0,
        "registry has unassigned targets",
    )
    return [dict(row) for row in questions], normalized


def _event(value: object, method: str | None = None) -> dict[str, Any]:
    _require(isinstance(value, Mapping), "target event must be an object")
    _require(
        not _contains_key(value, _LEDGER_FORBIDDEN),
        "structural ledger may not contain gold or assign target ownership",
    )
    required = {
        "target_id",
        "target_kind",
        "discovering_method",
        "disposition",
        "route_local_receipt_sha256",
    }
    _require(required <= set(value), "target event schema changed")
    _require(
        all(isinstance(value[key], str) and value[key] for key in required),
        "target event value changed",
    )
    _sha(value["route_local_receipt_sha256"], "route-local receipt SHA-256")
    if method is not None:
        _require(
            value["discovering_method"] == method,
            "target event discovering route changed",
        )
    aliases: list[str] = []
    for key in ("source_target_ids", "cited_source_target_ids"):
        items = value.get(key, ())
        _require(
            isinstance(items, Sequence) and not isinstance(items, (str, bytes)),
            f"{key} changed",
        )
        aliases.extend(str(item) for item in items)
    for key in ("source_target_id", "evidence_target_id"):
        item = value.get(key)
        if item is not None:
            _require(isinstance(item, str) and item, f"{key} changed")
            aliases.append(item)
    _require(
        aliases and all(item for item in aliases),
        "target event has no source identity",
    )
    return dict(value) | {"_source_target_ids": list(dict.fromkeys(aliases))}


def _projection_fields(value: Mapping[str, Any], key: str) -> tuple[str, ...]:
    projection = value.get(key)
    _require(isinstance(projection, str) and projection, f"ledger {key} changed")
    fields = tuple(projection.split("+"))
    _require(
        all(field and field.strip() == field for field in fields)
        and len(fields) == len(set(fields)),
        f"ledger {key} changed",
    )
    return fields


def _event_identity(value: Mapping[str, Any]) -> tuple[str, str, str]:
    return (
        str(value["target_kind"]),
        str(value["target_id"]),
        str(value["discovering_method"]),
    )


def _load_ledgers(
    specs: Sequence[str],
    answer_runs: Mapping[str, Mapping[str, Any]],
    questions: Sequence[Mapping[str, Any]],
    *,
    loader_specs: Sequence[str] = (),
    require_loaders: bool = False,
    max_concurrency: int = 4,
) -> list[dict[str, Any]]:
    _require(specs, "target scoring requires at least one --ledger")
    loaders: dict[str, str] = {}
    for spec in loader_specs:
        try:
            name, loader_spec = spec.split("=", 1)
        except ValueError as exc:
            raise TargetCoverageError(
                "ledger-loader must be METHOD=MODULE:FUNCTION"
            ) from exc
        _require(
            name not in loaders
            and _LOADER.fullmatch(loader_spec) is not None,
            "invalid or duplicate ledger-loader",
        )
        loaders[name] = loader_spec
    if require_loaders:
        _require(
            set(loaders) == set(answer_runs),
            "strict target scoring requires one loader per structural ledger",
        )
    ledgers: list[dict[str, Any]] = []
    covered: set[str] = set()
    for index, spec in enumerate(specs):
        name, path, expected = _parse_file_spec(spec, "ledger")
        _require(
            name in answer_runs and name not in covered,
            f"ledger {index} method binding changed",
        )
        value, digest = _read(path, expected)
        loader_spec = loaders.get(name)
        if loader_spec is not None:
            module_name, function_name = loader_spec.split(":", 1)
            loader = getattr(
                importlib.import_module(module_name), function_name, None
            )
            _require(callable(loader), f"invalid structural loader: {loader_spec}")
            s0 = answer_runs.get("s0")
            _require(s0 is not None, "structural replay requires the sealed s0 run")
            verified, verified_sha = loader(
                path,
                expected_ledger_sha256=expected,
                run_path=Path(answer_runs[name]["run_path"]),
                expected_run_sha256=answer_runs[name]["run_sha256"],
                s0_run_path=Path(s0["run_path"]),
                expected_s0_run_sha256=s0["run_sha256"],
                max_concurrency=max_concurrency,
                expected_question_count=len(questions),
            )
            _require(
                verified_sha == digest
                and canonical_json_bytes(verified) == canonical_json_bytes(value),
                f"structural loader returned another ledger for {name}",
            )
        _require(
            value.get("format") == LEDGER_FORMAT
            and value.get("ledger_sha256") == _self_sha(value, "ledger_sha256"),
            f"ledger {index} identity changed",
        )
        _require(
            not _contains_key(value, _LEDGER_FORBIDDEN)
            and value.get("ownership_policy")
            == "join-primary-owner-from-posthoc-desired-target-registry",
            f"ledger {index} crossed the gold/ownership firewall",
        )
        run = answer_runs[name]
        source_run = value.get("source_run_sha256", value.get("arm_run_sha256"))
        rows = value.get("questions")
        _require(
            value.get("arm_label") == run["arm_label"]
            and source_run == run["run_sha256"],
            f"ledger {index} answer binding changed",
        )
        _require(
            value.get("population_identity_sha256")
            == run["population_identity_sha256"]
            and value.get("question_count") == len(questions)
            and isinstance(rows, list)
            and len(rows) == len(questions),
            f"ledger {index} population changed",
        )
        discovery_fields = _projection_fields(value, "discovery_projection")
        admission_fields = _projection_fields(value, "admission_projection")
        normalized_rows = []
        for ordinal, (row, question) in enumerate(
            zip(rows, questions, strict=True)
        ):
            _require(
                isinstance(row, Mapping)
                and row.get("ordinal") == ordinal
                and row.get("question_id") == question["question_id"],
                f"ledger {index} order changed at {ordinal}",
            )
            if "ledger_row_sha256" in row:
                _require(
                    row["ledger_row_sha256"]
                    == _self_sha(row, "ledger_row_sha256"),
                    f"ledger {index} row identity changed at {ordinal}",
                )
            _require(
                all(
                    isinstance(row.get(field), list)
                    for field in discovery_fields + admission_fields
                ),
                f"ledger {index} projection field changed at {ordinal}",
            )
            before = [
                _event(event)
                for field in discovery_fields
                for event in row[field]
            ]
            after = [
                _event(event)
                for field in admission_fields
                for event in row[field]
            ]
            before_ids = {_event_identity(event) for event in before}
            _require(
                all(_event_identity(event) in before_ids for event in after),
                f"ledger {index} admitted an undiscovered target at {ordinal}",
            )
            normalized_rows.append({"before": before, "after": after})
        covered.add(name)
        ledgers.append(
            {
                "name": name,
                "sha256": digest,
                "identity_sha256": value["ledger_sha256"],
                "source_run_sha256": source_run,
                "loader": loader_spec,
                "journal_replay_verified": loader_spec is not None,
                "rows": normalized_rows,
            }
        )
    _require(
        covered == set(answer_runs),
        "target ledgers do not cover every registered method",
    )
    return ledgers


def _source_aliases(event: Mapping[str, Any], question_id: str) -> set[str]:
    prefix = f"{question_id}::"
    result: set[str] = set()
    for item in event["_source_target_ids"]:
        result.add(item)
        if item.startswith(prefix):
            result.add(item[len(prefix) :])
    return result


def _target_reached(
    target: Mapping[str, Any], events: Sequence[Mapping[str, Any]]
) -> bool:
    if any(
        event["target_kind"] == target["target_kind"]
        and event["target_id"] == target["target_id"]
        for event in events
    ):
        return True
    question_id = str(target["question_id"])
    if target["target_kind"] == "source_id":
        return any(
            target["target_id"] in _source_aliases(event, question_id)
            for event in events
        )
    basis = target.get("assignment_basis")
    expected_sources = (
        set(basis.get("expected_source_ids", ()))
        if isinstance(basis, Mapping)
        else set()
    )
    if not expected_sources:
        return False
    if target["target_kind"] == "relation":
        relation_events = [
            event for event in events if "relation" in str(event["target_kind"])
        ]
        reached = (
            set().union(
                *(
                    _source_aliases(event, question_id)
                    for event in relation_events
                )
            )
            if relation_events
            else set()
        )
        return expected_sources <= reached
    if target["target_kind"] == "coverage_check":
        reached = (
            set().union(
                *(_source_aliases(event, question_id) for event in events)
            )
            if events
            else set()
        )
        return expected_sources <= reached
    return False


def _owner_routes(
    registry: Mapping[str, Any], owners: Sequence[str]
) -> dict[str, set[str]]:
    policy = registry.get("policy")
    configured = (
        policy.get("owner_route_bindings") if isinstance(policy, Mapping) else None
    )
    result: dict[str, set[str]] = {}
    for owner in owners:
        routes = configured.get(owner) if isinstance(configured, Mapping) else None
        if routes is None:
            routes = DEFAULT_OWNER_ROUTE_BINDINGS.get(owner, (owner,))
        _require(
            isinstance(routes, Sequence)
            and not isinstance(routes, (str, bytes))
            and routes
            and all(isinstance(route, str) and route for route in routes),
            f"owner route binding changed for {owner}",
        )
        result[owner] = set(routes)
    flattened = [route for routes in result.values() for route in routes]
    _require(
        len(flattened) == len(set(flattened)),
        "discovering route belongs to more than one primary owner",
    )
    return result


def score(
    *,
    answer_run_specs: Sequence[str],
    registry_path: Path,
    registry_sha256: str,
    ledger_specs: Sequence[str],
    output_path: Path,
    ledger_loader_specs: Sequence[str] = (),
    require_ledger_loaders: bool = False,
    max_concurrency: int = 4,
) -> tuple[dict[str, Any], str]:
    answer_runs = _verify_answer_runs(answer_run_specs)
    questions = _ordered_questions(answer_runs)
    # Structural ledgers are verified before the gold-bearing registry opens.
    ledgers = _load_ledgers(
        ledger_specs,
        answer_runs,
        questions,
        loader_specs=ledger_loader_specs,
        require_loaders=require_ledger_loaders,
        max_concurrency=max_concurrency,
    )
    registry, observed_registry_sha = _read(registry_path, registry_sha256)
    registry_questions, targets = _validate_registry(registry, answer_runs)
    _require(
        registry_questions == questions,
        "registry/ledger question order changed",
    )
    owner_routes = _owner_routes(registry, tuple(answer_runs))

    outcomes = []
    for target in targets:
        before_by_route: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        after_by_route: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for ledger in ledgers:
            row = ledger["rows"][target["ordinal"]]
            for event in row["before"]:
                before_by_route[event["discovering_method"]].append(event)
            for event in row["after"]:
                after_by_route[event["discovering_method"]].append(event)
        discovered = {
            route
            for route, events in before_by_route.items()
            if _target_reached(target, events)
        }
        admitted = {
            route
            for route, events in after_by_route.items()
            if _target_reached(target, events)
        }
        owner = target["primary_owner"]
        primary_routes = owner_routes[owner]
        outcomes.append(
            {
                "ordinal": target["ordinal"],
                "question_id": target["question_id"],
                "target_sha256": target["target_sha256"],
                "target_kind": target["target_kind"],
                "primary_owner": owner,
                "primary_discovered": bool(discovered & primary_routes),
                "primary_admitted": bool(admitted & primary_routes),
                "primary_discovering_routes": sorted(discovered & primary_routes),
                "primary_admitting_routes": sorted(admitted & primary_routes),
                "alternate_discovering_routes": sorted(discovered - primary_routes),
                "alternate_admitting_routes": sorted(admitted - primary_routes),
                "union_discovered": bool(discovered),
                "union_admitted": bool(admitted),
            }
        )

    per_owner = []
    alternates = []
    for owner in answer_runs:
        selected = [row for row in outcomes if row["primary_owner"] == owner]
        count = len(selected)
        discovered = sum(row["primary_discovered"] for row in selected)
        admitted = sum(row["primary_admitted"] for row in selected)
        alternate = sum(
            bool(row["alternate_discovering_routes"]) for row in selected
        )
        alternate_only = sum(
            not row["primary_discovered"]
            and bool(row["alternate_discovering_routes"])
            for row in selected
        )
        per_owner.append(
            {
                "primary_owner": owner,
                "desired_targets": count,
                "primary_discovered": discovered,
                "primary_discovery_recall": (
                    None if not count else discovered / count
                ),
                "primary_admitted": admitted,
                "primary_admission_recall": None if not count else admitted / count,
                "alternate_reachable": alternate,
                "alternate_only_reachable": alternate_only,
                "union_discovered": sum(
                    row["union_discovered"] for row in selected
                ),
                "union_admitted": sum(row["union_admitted"] for row in selected),
            }
        )
        for alternate_owner, routes in owner_routes.items():
            if alternate_owner == owner:
                continue
            reached = sum(
                bool(set(row["alternate_discovering_routes"]) & routes)
                for row in selected
            )
            alternates.append(
                {
                    "primary_owner": owner,
                    "alternate_owner": alternate_owner,
                    "desired_targets": count,
                    "reached": reached,
                    "reachability": None if not count else reached / count,
                }
            )

    total = len(outcomes)
    union_discovered = sum(row["union_discovered"] for row in outcomes)
    union_admitted = sum(row["union_admitted"] for row in outcomes)
    result = {
        "format": SCORE_FORMAT,
        "registry_sha256": observed_registry_sha,
        "answer_runs_and_structural_ledgers_verified_before_registry_load": True,
        "answer_run_bindings": registry["answer_run_bindings"],
        "structural_ledger_bindings": [
            {
                key: ledger[key]
                for key in (
                    "name",
                    "sha256",
                    "identity_sha256",
                    "source_run_sha256",
                    "loader",
                    "journal_replay_verified",
                )
            }
            for ledger in ledgers
        ],
        "owner_route_bindings": {
            owner: sorted(routes) for owner, routes in owner_routes.items()
        },
        "population_identity_sha256": registry["population_identity_sha256"],
        "question_count": len(questions),
        "aggregate": {
            "desired_targets": total,
            "assigned_primary_owner_targets": total,
            "unassigned_primary_owner_count": 0,
            "union_discovered": union_discovered,
            "union_discovery_recall": (
                None if not total else union_discovered / total
            ),
            "union_admitted": union_admitted,
            "union_admission_recall": None if not total else union_admitted / total,
            "union_unreached_count": total - union_discovered,
            "discovered_then_deduped_count": sum(
                row["union_discovered"] and not row["union_admitted"]
                for row in outcomes
            ),
        },
        "per_owner": per_owner,
        "alternate_owner_reachability": alternates,
        "targets": outcomes,
        "discovery_credited_from_pre_dedup_projection": True,
        "admission_measured_from_post_dedup_projection": True,
        "all_structural_ledgers_journal_replayed": all(
            ledger["journal_replay_verified"] for ledger in ledgers
        ),
        "provider_calls": 0,
    }
    return result, em_runner._publish(output_path, result)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--answer-run", action="append", default=[], metavar="METHOD=PATH=SHA256"
    )
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--expected-registry-sha256", required=True)
    parser.add_argument(
        "--ledger", action="append", default=[], metavar="METHOD=PATH=SHA256"
    )
    parser.add_argument(
        "--ledger-loader",
        action="append",
        default=[],
        metavar="METHOD=MODULE:FUNCTION",
    )
    parser.add_argument("--max-concurrency", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result, digest = score(
        answer_run_specs=args.answer_run,
        registry_path=args.registry,
        registry_sha256=args.expected_registry_sha256,
        ledger_specs=args.ledger,
        output_path=args.output,
        ledger_loader_specs=args.ledger_loader,
        require_ledger_loaders=True,
        max_concurrency=args.max_concurrency,
    )
    print(
        f"Target-owner score {digest}: "
        f"union={result['aggregate']['union_discovered']}/"
        f"{result['aggregate']['desired_targets']}; provider_calls=0"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_OWNER_ROUTE_BINDINGS",
    "LEDGER_FORMAT",
    "REGISTRY_FORMAT",
    "SCORE_FORMAT",
    "build_parser",
    "main",
    "score",
]
