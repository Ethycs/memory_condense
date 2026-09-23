from copy import deepcopy
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from memory_condense.domain._tokenizer import count_tokens, tokenizer_proxy_identity
from tools.native_spine_namespace import NativeNamespace
from tools import native_spine_joint_population as admission


def artifact(payload, sha="source"):
    return SimpleNamespace(payload=payload, sha256=sha)


def manifests():
    sources = artifact({"question_inputs": False, "gold_inputs": False, "raw_inputs_to_qwen": False,
        "tokenizer": tokenizer_proxy_identity(), "body_count": 120})
    store = artifact({"sources_sha256": "source", "complete_source_compilation": True,
        "all_prepared_bodies_admitted": True, "body_count": 120, "prepared_body_count": 120})
    hierarchy = artifact({"complete_source_compilation": True, "complete_available_body_hierarchies": True,
        "complete_native_hierarchies": True, "body_count": 120, "prepared_body_count": 120,
        "raw_inputs_to_qwen": False})
    return sources, store, hierarchy


@pytest.mark.parametrize("defect", ["partial_store", "partial_trees", "missing_body", "tokenizer", "raw_qwen"])
def test_compilation_gate_rejects_partial_or_changed_inputs(defect):
    sources, store, hierarchy = manifests()
    if defect == "partial_store":
        store.payload["complete_source_compilation"] = False
    elif defect == "partial_trees":
        hierarchy.payload["complete_native_hierarchies"] = False
    elif defect == "missing_body":
        store.payload["body_count"] -= 1
    elif defect == "tokenizer":
        sources.payload["tokenizer"] = {"different": True}
    elif defect == "raw_qwen":
        hierarchy.payload["raw_inputs_to_qwen"] = True
    with pytest.raises(ValueError):
        admission.require_compilation(sources, store, hierarchy)


def case(ordinal=0):
    return {"ordinal": ordinal, "question_id": f"q{ordinal}", "namespace_id": f"n{ordinal}",
        "namespace_sha256": f"ns{ordinal}", "question": "What did the user choose?",
        "question_date": "2026/09/12 (Saturday) 00:00"}


@pytest.mark.parametrize("defect", ["pilot", "repeated_namespace", "wrong_history", "gold"])
def test_cases_require_exact_full100_population_and_namespace_binding(defect):
    rows = [case(i) for i in range(100)]
    bindings = {r["namespace_id"]: {"sha256": r["namespace_sha256"]} for r in rows}
    cases = artifact({"cases": rows, "sources_sha256": "source",
        "gold_answer_text_included": False, "ingest_use_permitted": False})
    if defect == "pilot":
        cases.payload["cases"] = rows[:10]
    elif defect == "repeated_namespace":
        rows[-1]["namespace_id"] = rows[0]["namespace_id"]
    elif defect == "wrong_history":
        rows[-1]["namespace_sha256"] = "different"
    elif defect == "gold":
        cases.payload["gold_answer_text_included"] = True
    with pytest.raises(ValueError):
        admission.validate_cases(cases, artifact({}), bindings)


@pytest.fixture(scope="module")
def million_token_body():
    text = "a " * 1_000_000
    assert count_tokens(text) >= 1_000_000
    return text


def namespace(past, future=""):
    turns = {"past": SimpleNamespace(text=past, created_at=datetime(2026, 9, 12, tzinfo=timezone.utc))}
    if future:
        turns["future"] = SimpleNamespace(text=future, created_at=datetime(2026, 9, 13, tzinfo=timezone.utc))
    audit = {"complete_namespace": True, "partial_use_explicit": False,
        "missing_summary_occurrence_ids": [], "missing_hierarchy_occurrence_ids": [],
        "namespace_id": "n0", "namespace_source_sha256": "ns0",
        "body_tokens": sum(count_tokens(t.text) for t in turns.values())}
    return NativeNamespace(SimpleNamespace(turns=turns),
        SimpleNamespace(sections=[SimpleNamespace(summary="summary")], receipt_sha256="atomic"),
        SimpleNamespace(receipt_sha256="hierarchy"), audit)


def test_complete_gate_counts_actual_body_text_and_accepts_question_day_evidence(million_token_body):
    admission.require_compilation(*manifests())
    rows = [case(i) for i in range(100)]
    cases = artifact({"cases": rows, "sources_sha256": "source",
        "gold_answer_text_included": False, "ingest_use_permitted": False})
    assert admission.validate_cases(cases, artifact({}),
        {r["namespace_id"]: {"sha256": r["namespace_sha256"]} for r in rows}) == rows
    result = admission.namespace_receipt(namespace(million_token_body), case(),
        SimpleNamespace(values={"summary": object()}))
    assert result["through_question_day_body_tokens"] == count_tokens(million_token_body)


@pytest.mark.parametrize("defect", ["future_only", "inflated_total", "missing_vector", "partial"])
def test_namespace_gate_rejects_future_tokens_metadata_inflation_or_missing_assets(million_token_body, defect):
    ns = namespace("Short prior memory.", million_token_body) if defect == "future_only" else namespace(million_token_body)
    vectors = SimpleNamespace(values={"summary": object()})
    if defect == "inflated_total":
        ns.audit["body_tokens"] += 1_000_000
    elif defect == "missing_vector":
        vectors.values = {}
    elif defect == "partial":
        ns.audit["partial_use_explicit"] = True
    with pytest.raises(ValueError):
        admission.namespace_receipt(ns, case(), vectors)
