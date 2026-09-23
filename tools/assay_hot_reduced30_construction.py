#!/usr/bin/env python3
"""Build and audit the exact locked reduced-30 provider-free construction.

This harness deliberately owns only the *population slice* and its telemetry.
It does not own a retrieval policy.  A policy can be supplied in either of two
ways:

* ``construct`` imports an explicit compatible assay module and asks it to
  compose only the thirty locked rows from the sealed ingest/store artifacts;
* ``slice`` accepts any sealed, gold-free selection and extracts the same
  thirty rows without assuming the selection's format or historical digest.

The question IDs and dated-question hashes below are the immutable lock.  They
contain no answers, references, benchmark labels, or judge observations.  No
provider is called.  The resulting artifact preserves the provider packets and
adds construction telemetry for required slots, lane budgets and rejections,
fact rendering, and context occupancy.

A module used by ``construct`` may expose ``build_reduced_candidate_universe``
and ``compose_reduced_arm`` hooks.  Without those hooks, the harness uses the
user-led envelope cache/index and ``_compose_arm`` interface implemented by the
v6 assay family.  This makes successor modules selectable without pinning this
harness to an old selection format or implementation identity.
"""

from __future__ import annotations

import argparse
import copy
import importlib
import json
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    _ROOT = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(_ROOT / "src"), str(_ROOT)]

from memory_condense.domain.discourse import quote_sha256  # noqa: E402
from memory_condense.domain.integrity import file_sha256  # noqa: E402
from tools.matched_eval.artifacts import (  # noqa: E402
    publish_sealed_json,
    read_sealed_json,
)
from tools.matched_eval.contracts import (  # noqa: E402
    assert_gold_blind,
    identity_sha256,
)
from tools.matched_eval.hot_v3_user_led_envelope_shadow import (  # noqa: E402
    build_user_led_envelope_shadow_index,
)
from tools.matched_eval.query_expansion import (  # noqa: E402
    load_locked_query_expansion_context,
)
from tools.matched_eval.typed_operator_spec import (  # noqa: E402
    compile_typed_operator_spec,
)


FORMAT = "memory-condense-hot-reduced30-construction-v1"
ROW_FORMAT = f"{FORMAT}-row-v1"
SELECTION_NAME = "selection.json"
DEFAULT_ASSAY_MODULE = "tools.assay_hot_v6_spine_episode_fact_ledger_full100"
DEFAULT_INPUT_SELECTION = Path(
    "eval_results/longmemeval-1m-hot-v6-spine-episode-fact-ledger-"
    "full100-20260907-r3/selection.json"
)
DEFAULT_OUTPUT_ROOT = Path(
    "eval_results/longmemeval-1m-hot-reduced30-construction-20260908-r1"
)

# Zero-based locked-population ordinal, opaque question ID, dated-question SHA.
# This is the complete r3 miss population and carries no answer-side material.
LOCKED_QUESTIONS: tuple[tuple[int, str, str], ...] = (
    (5, "06878be2", "488d098964e1c68638411404eb2126164877d443ba18a64dd995820663777319"),
    (6, "gpt4_fa19884d", "9d6c446ed5db97bd900af61feea6a5d462514a71b9c4b4d556c8fb39b2e5f376"),
    (14, "d23cf73b", "843167d47f684d7f3ff2c2a2b27660eadc7f2c782444a3738d6dbd815abd0102"),
    (15, "15745da0", "c2ecdb09fe74b304af67e0d49829ea6c8beeac0217cac34e9b53fddbd200586d"),
    (17, "gpt4_65aabe59", "65d7e85c696683e0595096e6e0c3adfa1812c401fdde48f33caf973e483d1439"),
    (25, "993da5e2", "f34ce51619a39beac7a0184ee9fd4d0be3e2479bd4196ee7d04f0a95407ce9a3"),
    (36, "32260d93", "0a80ec0e0502f6f83e594db8b110e0baf14db69fc56b6756e62283370eb9961f"),
    (40, "9d25d4e0", "4a17156a1a6540f55decaf65895eb36c2f02679e8a92195273f835798b9c3c10"),
    (42, "a96c20ee_abs", "bb7f6ae97337bd39be463cf98dce4d5386c74809636e3f809e4847b0dce3b7ff"),
    (43, "gpt4_1e4a8aec", "8263b8e7e0b2ccafe1387f074677b7c320a525b7ded1468cbe54090d2181e657"),
    (48, "gpt4_0a05b494", "57dbfdccde421350ed8f66e61a65410f2f1fa3a5156997e36851b3020acd9f4f"),
    (49, "a89d7624", "2912cee862186c95bf7defc0d2a1e31846b7dc0b41a1fddd8296fd2104382e9e"),
    (51, "41698283", "ca8ebe3579ed7a7e94a3f81a9b1d0accdf06d9d2c4c5bd654b7f507d19344df8"),
    (52, "3d86fd0a", "ae415c7d43d6910efeef2ad721b5634d2b0b8ae033ceca6dfbb269428aaa14ae"),
    (53, "3a704032", "b887bed864d509a002d6ed57f3299538558a4ce31aadf8466af14b2efb839a17"),
    (59, "b29f3365", "376e78f9ff68597272386302af573fdde9e3d0d9e18cdfcdd013e57fd783af86"),
    (61, "gpt4_15e38248", "10b808075dbbf9ef413324898c5f223492b9a00f64449aecf3d09a12a1000f44"),
    (66, "00ca467f", "6e07a651745081207e39fab18b232ad17a53200fd06f3b3c7ce7952213472071"),
    (67, "80ec1f4f", "e434a57f3d0eb2165e4435228fe1cd7d6f88eeaa21d5ae6d9d1d369830026596"),
    (69, "0a995998", "fe4c33a64692e3069d7eaf2bad68e4f6530653b298244d73de520eb3c89b1fb5"),
    (75, "2318644b", "4f36c3f4bdfe7a678cb0b1a93d3e200436009fab7c590f8792d40f43864ccbbc"),
    (77, "0bc8ad92", "9e1af8ac4577ab2c313a78929a44e4d957156348eb3c2883926211835da6609f"),
    (79, "7527f7e2", "998ffefa4308d7fe30ddad8f62af7eb8ce2017ec3314825ee4ccd846a8fb19d2"),
    (81, "1a1907b4", "7bc462bb60a3057d208baf9e29ffb525ec727129d7fee2ebaf48a2fab56ae9d5"),
    (82, "1d4e3b97", "8a97414ef3d450afa235d6011eff058c253aa5ec01e89f8b250fd3b6a586b0ec"),
    (83, "c14c00dd", "a53b45e375dbb0c2232f429fb4efd314872fd436e4865dcd1c2c07a1c8007f33"),
    (86, "gpt4_7f6b06db", "6f3d2689d504f7a4ee5ec41aab846d64d16c08a05000d6fefb738ced9e7e6c48"),
    (87, "2788b940", "fa329bb0fb02bee91aee758c9208f39ebd57837821f802035a04d96e861a7aca"),
    (94, "9a707b81", "c8cf08c321365ad95e677cfdb89a754857c38aa2dcbe584fc40315c96fe4643e"),
    (97, "7405e8b1", "5cca8b64afd501e037d246e1fced0b782378ce7f3aa05546712f448720aa5974"),
)
QUESTION_COUNT = len(LOCKED_QUESTIONS)
LOCKED_IDENTITY_SHA256 = identity_sha256(
    [
        {
            "ordinal": ordinal,
            "prompt_question_sha256": question_sha,
            "question_id": question_id,
        }
        for ordinal, question_id, question_sha in LOCKED_QUESTIONS
    ]
)


class Reduced30ConstructionError(ValueError):
    """A locked identity, sealed artifact, or construction invariant changed."""


def _require(ok: object, message: str) -> None:
    if not ok:
        raise Reduced30ConstructionError(message)


def _mapping(value: object, label: str) -> dict[str, Any]:
    _require(type(value) is dict, f"{label} must be an exact object")
    return value  # type: ignore[return-value]


def _list(value: object, label: str) -> list[Any]:
    _require(type(value) is list, f"{label} must be an exact list")
    return value  # type: ignore[return-value]


def _resolve_path(value: object, path: str) -> object:
    current = value
    for segment in path.split("."):
        current = _mapping(current, f"arm path {path}").get(segment)
    return current


def _arm_candidates(
    value: Mapping[str, Any],
    *,
    path: str = "",
    depth: int = 0,
) -> list[tuple[int, str, dict[str, Any]]]:
    """Find provider arms without assuming a selection row format."""

    if depth > 7:
        return []
    result: list[tuple[int, str, dict[str, Any]]] = []
    if type(value.get("provider_messages")) is list:
        score = (
            32 * int(type(value.get("episode_selection")) is dict)
            + 16 * int(type(value.get("fact_ledger")) is dict)
            + 8 * int("rendered_raw_chunk_ids" in value)
            + 4 * int("packed_chunk_ids" in value)
            + 2 * int("mode" in value)
            + int(type(value.get("context_token_proxy")) is int)
        )
        result.append((score, path, dict(value)))
    for key, child in value.items():
        if type(child) is not dict:
            continue
        child_path = f"{path}.{key}" if path else str(key)
        result.extend(_arm_candidates(child, path=child_path, depth=depth + 1))
    return result


def find_provider_arm(
    row: Mapping[str, Any], explicit_path: str | None = None
) -> tuple[str, dict[str, Any]]:
    """Return the diagnostic/provider arm in a format-neutral row."""

    if explicit_path:
        arm = _mapping(_resolve_path(row, explicit_path), "explicit provider arm")
        _require(type(arm.get("provider_messages")) is list, "explicit arm has no provider messages")
        return explicit_path, arm
    candidates = sorted(_arm_candidates(row), key=lambda item: (item[0], -len(item[1])), reverse=True)
    _require(bool(candidates), "selection row contains no provider arm")
    top_score = candidates[0][0]
    winners = [candidate for candidate in candidates if candidate[0] == top_score]
    _require(
        len(winners) == 1,
        "selection row has ambiguous provider arms; pass --arm-path",
    )
    return winners[0][1], winners[0][2]


def _dated_question(module: Any, arm: Mapping[str, Any]) -> str:
    typed_module = getattr(module, "typed", None)
    extractor = getattr(typed_module, "_extract_dated_question", None)
    if extractor is None:
        from tools import assay_hot_v3_typed_operator_full100 as typed_module

        extractor = typed_module._extract_dated_question  # noqa: SLF001
    question = extractor(arm)
    _require(type(question) is str and bool(question), "dated question is missing")
    return question


def _locked_rows(rows_value: object) -> list[tuple[int, dict[str, Any]]]:
    rows = _list(rows_value, "selection questions")
    by_id: dict[str, dict[str, Any]] = {}
    for position, value in enumerate(rows):
        row = _mapping(value, f"selection row {position}")
        question_id = row.get("question_id")
        _require(type(question_id) is str and question_id not in by_id, "question IDs must be unique")
        by_id[question_id] = row
    selected: list[tuple[int, dict[str, Any]]] = []
    for ordinal, question_id, expected_sha in LOCKED_QUESTIONS:
        row = by_id.get(question_id)
        _require(row is not None, f"locked question {ordinal} / {question_id} is missing")
        _require(
            row.get("prompt_question_sha256") == expected_sha,
            f"locked question hash changed at {ordinal}",
        )
        declared_ordinal = row.get("ordinal")
        if len(rows) >= max(item[0] for item in LOCKED_QUESTIONS):
            _require(declared_ordinal == ordinal, f"global ordinal changed at {ordinal}")
        selected.append((ordinal, row))
    _require(len(selected) == QUESTION_COUNT, "reduced population changed")
    return selected


def _counter_projection(counter: Counter[str]) -> dict[str, int]:
    return {key: counter[key] for key in sorted(counter)}


def _question_report(
    *,
    module: Any,
    global_ordinal: int,
    row: Mapping[str, Any],
    arm_path: str | None,
) -> dict[str, Any]:
    located_path, arm = find_provider_arm(row, arm_path)
    question = _dated_question(module, arm)
    expected_sha = next(
        question_sha
        for ordinal, _question_id, question_sha in LOCKED_QUESTIONS
        if ordinal == global_ordinal
    )
    _require(quote_sha256(question) == expected_sha, f"provider question changed at {global_ordinal}")
    spec = compile_typed_operator_spec(question)
    slot_ids = [slot.slot_id for slot in spec.required_slots]
    episode = arm.get("episode_selection")
    groups = episode.get("selected_groups", []) if type(episode) is dict else []
    admitted_envelopes = {
        str(manifest.get("envelope_id"))
        for manifest in arm.get("episode_manifests", [])
        if type(manifest) is dict
    }

    def episode_slots(*, admitted_only: bool) -> set[str]:
        covered: set[str] = set()
        for group in groups:
            if type(group) is not dict:
                continue
            if admitted_only and str(group.get("envelope_id")) not in admitted_envelopes:
                continue
            audit = group.get("rank_audit")
            scores = audit.get("slot_lane_scores", {}) if type(audit) is dict else {}
            if type(scores) is dict:
                covered.update(
                    str(slot_id)
                    for slot_id, score in scores.items()
                    if type(score) is int and score > 0
                )
        return covered & set(slot_ids)

    selected_episode_slots = episode_slots(admitted_only=False)
    rendered_episode_slots = episode_slots(admitted_only=True)
    fact_ledger = arm.get("fact_ledger")
    fact = fact_ledger if type(fact_ledger) is dict else {}
    unresolved = {
        str(value) for value in fact.get("unresolved_slot_ids", [])
    } & set(slot_ids)
    compiled_fact_slots = set(slot_ids) - unresolved if fact else set()
    rendered_fact_slots = {
        str(slot_id)
        for value in fact.get("selected_facts", [])
        if type(value) is dict
        for slot_id in value.get("bound_slot_ids", [])
    } & set(slot_ids)
    lane_decisions = episode.get("lane_decisions", []) if type(episode) is dict else []
    decisions: dict[str, Counter[str]] = {}
    for value in lane_decisions:
        if type(value) is not dict:
            continue
        lane = str(value.get("lane", "unknown"))
        decision = str(value.get("decision", "unknown"))
        decisions.setdefault(lane, Counter())[decision] += 1
    decision_projection = {
        lane: _counter_projection(values) for lane, values in sorted(decisions.items())
    }
    rejection_count = sum(
        count
        for values in decisions.values()
        for decision, count in values.items()
        if decision.startswith("rejected_")
    )
    slots = []
    for slot in spec.required_slots:
        slots.append(
            {
                "compiled_fact_match": slot.slot_id in compiled_fact_slots,
                "kind": slot.kind.value,
                "label": slot.label,
                "rendered_episode_match": slot.slot_id in rendered_episode_slots,
                "rendered_fact_match": slot.slot_id in rendered_fact_slots,
                "selected_episode_match": slot.slot_id in selected_episode_slots,
                "slot_id": slot.slot_id,
            }
        )
    report = {
        "answer_shape": spec.answer_shape.value,
        "arm_path": located_path,
        "compiled_fact_count": int(fact.get("compiled_fact_count", 0)),
        "context_token_proxy": int(arm.get("context_token_proxy", 0)),
        "fact_selection_status": fact.get("selection_status"),
        "fallback_reason": arm.get("fallback_reason"),
        "global_ordinal": global_ordinal,
        "lane_budgets": copy.deepcopy(episode.get("lane_budgets", {})) if type(episode) is dict else {},
        "lane_decision_counts": decision_projection,
        "lane_proposal_count": int(episode.get("lane_proposal_count", 0)) if type(episode) is dict else 0,
        "lane_rejection_count": rejection_count,
        "lane_used_chunks": copy.deepcopy(episode.get("lane_used_chunks", {})) if type(episode) is dict else {},
        "lane_used_tokens": copy.deepcopy(episode.get("lane_used_tokens", {})) if type(episode) is dict else {},
        "mode": arm.get("mode"),
        "prompt_workspace_token_proxy": int(arm.get("prompt_workspace_token_proxy", 0)),
        "question_id": row.get("question_id"),
        "rendered_episode_raw_chunk_count": len(arm.get("rendered_raw_chunk_ids", [])),
        "rendered_fact_count": len(arm.get("rendered_fact_ids", [])),
        "required_slot_count": len(slot_ids),
        "required_slot_coverage": slots,
        "selected_episode_count": int(episode.get("selected_episode_count", 0)) if type(episode) is dict else 0,
        "typed_operation": spec.operation,
    }
    report["receipt_sha256"] = identity_sha256(report)
    return report


def _aggregate(reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    contexts = [int(row["context_token_proxy"]) for row in reports]
    workspaces = [int(row["prompt_workspace_token_proxy"]) for row in reports]
    aggregate_decisions: dict[str, Counter[str]] = {}
    used_chunks: Counter[str] = Counter()
    used_tokens: Counter[str] = Counter()
    slot_totals: Counter[str] = Counter()
    full_episode = full_compiled = full_rendered_fact = 0
    questions_with_slots = 0
    for row in reports:
        for lane, values in row["lane_decision_counts"].items():
            target = aggregate_decisions.setdefault(str(lane), Counter())
            target.update({str(key): int(value) for key, value in values.items()})
        used_chunks.update({str(key): int(value) for key, value in row["lane_used_chunks"].items()})
        used_tokens.update({str(key): int(value) for key, value in row["lane_used_tokens"].items()})
        slots = row["required_slot_coverage"]
        questions_with_slots += int(bool(slots))
        selected_episode = sum(bool(slot["rendered_episode_match"]) for slot in slots)
        compiled = sum(bool(slot["compiled_fact_match"]) for slot in slots)
        rendered_fact = sum(bool(slot["rendered_fact_match"]) for slot in slots)
        slot_totals["required"] += len(slots)
        slot_totals["rendered_episode_match"] += selected_episode
        slot_totals["compiled_fact_match"] += compiled
        slot_totals["rendered_fact_match"] += rendered_fact
        full_episode += int(bool(slots) and selected_episode == len(slots))
        full_compiled += int(bool(slots) and compiled == len(slots))
        full_rendered_fact += int(bool(slots) and rendered_fact == len(slots))
    return {
        "compiled_fact_count": sum(int(row["compiled_fact_count"]) for row in reports),
        "context_token_max": max(contexts),
        "context_token_mean": round(sum(contexts) / len(contexts), 3),
        "context_token_min": min(contexts),
        "fact_rendered_count": sum(int(row["rendered_fact_count"]) for row in reports),
        "lane_decision_counts": {
            lane: _counter_projection(values)
            for lane, values in sorted(aggregate_decisions.items())
        },
        "lane_proposal_count": sum(int(row["lane_proposal_count"]) for row in reports),
        "lane_rejection_count": sum(int(row["lane_rejection_count"]) for row in reports),
        "lane_used_chunks": _counter_projection(used_chunks),
        "lane_used_tokens": _counter_projection(used_tokens),
        "prompt_workspace_token_max": max(workspaces),
        "provider_calls": 0,
        "questions_all_compiled_fact_slots_matched": full_compiled,
        "questions_all_rendered_episode_slots_matched": full_episode,
        "questions_all_rendered_fact_slots_matched": full_rendered_fact,
        "questions_with_required_slots": questions_with_slots,
        "questions_without_required_slots": len(reports) - questions_with_slots,
        "question_count": len(reports),
        "rendered_episode_raw_chunk_count": sum(
            int(row["rendered_episode_raw_chunk_count"]) for row in reports
        ),
        "required_slot_totals": _counter_projection(slot_totals),
    }


def _wrap_rows(
    *,
    module: Any,
    rows: Sequence[tuple[int, Mapping[str, Any]]],
    arm_path: str | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    wrapped: list[dict[str, Any]] = []
    reports: list[dict[str, Any]] = []
    for reduced_ordinal, (global_ordinal, source_row) in enumerate(rows):
        report = _question_report(
            module=module,
            global_ordinal=global_ordinal,
            row=source_row,
            arm_path=arm_path,
        )
        body = {
            "format": ROW_FORMAT,
            "global_ordinal": global_ordinal,
            "prompt_question_sha256": source_row["prompt_question_sha256"],
            "question_id": source_row["question_id"],
            "reduced_ordinal": reduced_ordinal,
            "source_row": copy.deepcopy(dict(source_row)),
            "telemetry": report,
        }
        wrapped.append({**body, "row_receipt_sha256": identity_sha256(body)})
        reports.append(report)
    return wrapped, reports


def _artifact(
    *,
    module: Any,
    rows: Sequence[tuple[int, Mapping[str, Any]]],
    arm_path: str | None,
    input_binding: Mapping[str, Any],
) -> dict[str, Any]:
    wrapped, reports = _wrap_rows(module=module, rows=rows, arm_path=arm_path)
    body = {
        "aggregate": _aggregate(reports),
        "format": FORMAT,
        "gold_fields_present": False,
        "input_binding": copy.deepcopy(dict(input_binding)),
        "locked_question_identity_sha256": LOCKED_IDENTITY_SHA256,
        "provider_calls": 0,
        "question_count": QUESTION_COUNT,
        "questions": wrapped,
        "status": "sealed_provider_free_reduced30_construction",
    }
    selection = {**body, "receipt_sha256": identity_sha256(body)}
    assert_gold_blind(selection, path="hot_reduced30_construction")
    return selection


def build_from_selection(
    *,
    input_selection: Path,
    arm_path: str | None = None,
) -> dict[str, Any]:
    source = read_sealed_json(input_selection)
    _require(source.payload.get("provider_calls") == 0, "source selection performed provider calls")
    assert_gold_blind(source.payload, path="hot_reduced30_source_selection")
    rows = _locked_rows(source.payload.get("questions"))
    module = importlib.import_module(DEFAULT_ASSAY_MODULE)
    binding = {
        "input_mode": "sealed_selection_slice",
        "selection_format": source.payload.get("format"),
        "selection_sha256": source.sha256,
    }
    return _artifact(module=module, rows=rows, arm_path=arm_path, input_binding=binding)


def _assay_component(module: Any, name: str, required_method: str) -> Any:
    """Resolve a direct or one-level legacy assay dependency, fail closed."""

    component = getattr(module, name, None)
    location = f"module.{name}"
    if component is None:
        legacy = getattr(module, "legacy", None)
        component = getattr(legacy, name, None) if legacy is not None else None
        location = f"module.legacy.{name}"
    _require(component is not None, f"assay module has no {name} dependency")
    _require(
        callable(getattr(component, required_method, None)),
        f"{location} has no callable {required_method}",
    )
    return component


def _source_loader(module: Any) -> Any:
    hook = getattr(module, "load_reduced_source", None)
    if hook is not None:
        _require(callable(hook), "assay module load_reduced_source hook is not callable")
        return hook
    source_assay = _assay_component(module, "source_assay", "_load_source")
    return source_assay._load_source  # noqa: SLF001


def _candidate_universe(module: Any, context: Any, namespace: Any) -> tuple[Any, dict[str, Any]]:
    hook = getattr(module, "build_reduced_candidate_universe", None)
    if hook is not None:
        _require(callable(hook), "candidate-universe hook is not callable")
        index, binding = hook(context=context, namespace=namespace)
        return index, _mapping(binding, "candidate-universe binding")
    shadow = _assay_component(module, "shadow", "_build_cache")
    cache, timing = shadow._build_cache(context, namespace)  # noqa: SLF001
    index = build_user_led_envelope_shadow_index(cache)
    binding = {
        "cache": cache.projection(),
        "cache_build_binding": {
            key: value for key, value in timing.items() if key != "cache_build_ns"
        },
        "index": index.projection(),
        "namespace_id": str(namespace.namespace_id),
        "store": {
            "database_sha256": cache.source_database_sha256,
            "physical_store_row_count": cache.physical_store_row_count,
            "store_receipt_sha256": cache.source_store_receipt_sha256,
        },
    }
    return index, binding


def _compose(module: Any, **kwargs: Any) -> dict[str, Any]:
    compose = getattr(module, "compose_reduced_arm", None) or getattr(module, "_compose_arm", None)
    _require(compose is not None, "assay module has no compose_reduced_arm or _compose_arm")
    return _mapping(compose(**kwargs), "composed arm")


def build_from_module(
    *,
    assay_module: str,
    source_root: Path | None,
    retrieval_path: Path | None,
    store_root: Path | None,
    arm_path: str | None = "arms.a3_protected_union",
) -> dict[str, Any]:
    module = importlib.import_module(assay_module)
    actual_source_root = (source_root or Path(module.DEFAULT_SOURCE_ROOT)).resolve()
    actual_store_root = (store_root or Path(module.DEFAULT_STORE_ROOT)).resolve()
    actual_retrieval = (retrieval_path or Path(module.DEFAULT_RETRIEVAL)).resolve()
    loader = _source_loader(module)
    construction, construction_sha, runtime_sha, replay_sha = loader(actual_source_root)
    source_rows = _list(construction.get("questions"), "source construction questions")
    locked_source_rows = _locked_rows(source_rows)
    expected_retrieval_sha = file_sha256(actual_retrieval)
    context = load_locked_query_expansion_context(
        actual_retrieval,
        store_root=actual_store_root,
        expected_retrieval_sha256=expected_retrieval_sha,
        expected_question_count=len(source_rows),
    )
    population_by_question = {
        str(row.source.packet.question_id): row for row in context.population.rows
    }
    universes: dict[str, tuple[Any, dict[str, Any]]] = {}
    output: list[tuple[int, Mapping[str, Any]]] = []
    for global_ordinal, source_row in locked_source_rows:
        unsigned = dict(source_row)
        declared_receipt = unsigned.pop("row_receipt_sha256", None)
        if declared_receipt is not None:
            _require(declared_receipt == identity_sha256(unsigned), f"source row receipt changed at {global_ordinal}")
        question_id = str(source_row["question_id"])
        population_row = population_by_question.get(question_id)
        _require(population_row is not None, f"population row missing at {global_ordinal}")
        namespace = population_row.namespace
        namespace_id = str(namespace.namespace_id)
        if namespace_id not in universes:
            universes[namespace_id] = _candidate_universe(module, context, namespace)
        index, candidate_binding = universes[namespace_id]
        source_arm = _mapping(source_row.get("effective_arm"), "source effective arm")
        question = _dated_question(module, source_arm)
        _require(
            quote_sha256(question) == source_row["prompt_question_sha256"],
            f"source prompt binding changed at {global_ordinal}",
        )
        arm = _compose(
            module,
            source_arm=source_arm,
            dated_question=question,
            index=index,
            candidate_binding=candidate_binding,
        )
        row_body = {
            "arms": {"a3_protected_union": arm},
            "format": f"{FORMAT}-module-source-row-v1",
            "ordinal": global_ordinal,
            "prompt_question_sha256": source_row["prompt_question_sha256"],
            "question_id": question_id,
            "source_row_receipt_sha256": declared_receipt,
        }
        output.append((global_ordinal, {**row_body, "row_receipt_sha256": identity_sha256(row_body)}))
    binding = {
        "assay_module": assay_module,
        "assay_module_sha256": file_sha256(Path(module.__file__).resolve()),
        "input_mode": "direct_reduced_module_construction",
        "retrieval_sha256": expected_retrieval_sha,
        "source_construction_sha256": construction_sha,
        "source_replay_sha256": replay_sha,
        "source_runtime_sha256": runtime_sha,
    }
    return _artifact(module=module, rows=output, arm_path=arm_path, input_binding=binding)


def validate_selection(selection: Mapping[str, Any]) -> None:
    _require(
        selection.get("format") == FORMAT
        and selection.get("status") == "sealed_provider_free_reduced30_construction"
        and selection.get("gold_fields_present") is False
        and selection.get("provider_calls") == 0
        and selection.get("question_count") == QUESTION_COUNT
        and selection.get("locked_question_identity_sha256") == LOCKED_IDENTITY_SHA256,
        "reduced construction header changed",
    )
    unsigned = dict(selection)
    receipt = unsigned.pop("receipt_sha256", None)
    _require(receipt == identity_sha256(unsigned), "reduced construction receipt changed")
    rows = _list(selection.get("questions"), "reduced construction rows")
    _require(len(rows) == QUESTION_COUNT, "reduced construction row count changed")
    for reduced_ordinal, (row_value, locked) in enumerate(zip(rows, LOCKED_QUESTIONS, strict=True)):
        row = _mapping(row_value, f"reduced row {reduced_ordinal}")
        body = dict(row)
        row_receipt = body.pop("row_receipt_sha256", None)
        ordinal, question_id, question_sha = locked
        _require(
            row_receipt == identity_sha256(body)
            and row.get("reduced_ordinal") == reduced_ordinal
            and row.get("global_ordinal") == ordinal
            and row.get("question_id") == question_id
            and row.get("prompt_question_sha256") == question_sha,
            f"reduced row identity changed at {ordinal}",
        )
    assert_gold_blind(selection, path="loaded_hot_reduced30_construction")


def _publish(output_root: Path, payload: dict[str, Any]) -> dict[str, Any]:
    validate_selection(payload)
    artifact, created = publish_sealed_json(output_root / SELECTION_NAME, payload)
    aggregate = payload["aggregate"]
    return {
        "compiled_fact_count": aggregate["compiled_fact_count"],
        "context_token_max": aggregate["context_token_max"],
        "created": created,
        "fact_rendered_count": aggregate["fact_rendered_count"],
        "lane_rejection_count": aggregate["lane_rejection_count"],
        "new_provider_calls": 0,
        "question_count": QUESTION_COUNT,
        "selection_sha256": artifact.sha256,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    sliced = commands.add_parser("slice")
    sliced.add_argument("--input-selection", type=Path, default=DEFAULT_INPUT_SELECTION)
    sliced.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    sliced.add_argument("--arm-path")
    construct = commands.add_parser("construct")
    construct.add_argument("--assay-module", default=DEFAULT_ASSAY_MODULE)
    construct.add_argument("--source-root", type=Path)
    construct.add_argument("--retrieval", type=Path)
    construct.add_argument("--store-root", type=Path)
    construct.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    construct.add_argument("--arm-path", default="arms.a3_protected_union")
    verify = commands.add_parser("verify")
    verify.add_argument("--selection", type=Path, default=DEFAULT_OUTPUT_ROOT / SELECTION_NAME)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "slice":
        payload = build_from_selection(
            input_selection=args.input_selection.resolve(),
            arm_path=args.arm_path,
        )
        result = _publish(args.output_root.resolve(), payload)
    elif args.command == "construct":
        payload = build_from_module(
            assay_module=args.assay_module,
            source_root=args.source_root,
            retrieval_path=args.retrieval,
            store_root=args.store_root,
            arm_path=args.arm_path,
        )
        result = _publish(args.output_root.resolve(), payload)
    else:
        artifact = read_sealed_json(args.selection.resolve())
        validate_selection(artifact.payload)
        result = {
            "new_provider_calls": 0,
            "question_count": QUESTION_COUNT,
            "selection_sha256": artifact.sha256,
            "verified": True,
        }
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_ASSAY_MODULE",
    "DEFAULT_INPUT_SELECTION",
    "DEFAULT_OUTPUT_ROOT",
    "FORMAT",
    "LOCKED_IDENTITY_SHA256",
    "LOCKED_QUESTIONS",
    "QUESTION_COUNT",
    "SELECTION_NAME",
    "build_from_module",
    "build_from_selection",
    "find_provider_arm",
    "main",
    "validate_selection",
]
