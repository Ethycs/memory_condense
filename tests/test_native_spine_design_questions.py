from copy import deepcopy
from types import SimpleNamespace

import pytest

from tools.evaluate_native_spine_design_questions import load_questions
from tools.matched_eval.artifacts import publish_sealed_json


def question_set():
    case = {"namespace_id": "one-cached-history", "namespace_sha256": "a"*64,
            "question_date": "2023/05/30 (Tue) 23:18"}
    scope = SimpleNamespace(sha256="b"*64, payload={"case": case})
    payload = {"scope_sha256": scope.sha256, "history_count": 1, "development_set": True,
        "general_accuracy_claim_permitted": False,
        "questions": [{**case, "question_id": str(i)} for i in range(6)],
        # An unavailable reference must not be opened during question admission.
        "references": {"path": "must-not-read-reference-yet", "sha256": "c"*64}}
    return scope, payload


def test_six_dated_questions_admit_without_loading_golds(tmp_path):
    scope, payload = question_set()
    artifact, _ = publish_sealed_json(tmp_path/"questions.json", payload)
    assert load_questions(artifact.path, scope).sha256 == artifact.sha256


@pytest.mark.parametrize("corruption", ["population", "namespace", "date", "duplicate", "scope", "claim"])
def test_design_admission_prevents_accidental_broadening_or_false_claims(tmp_path, corruption):
    scope, payload = question_set()
    payload = deepcopy(payload)
    if corruption == "population":
        payload["questions"] = [{**payload["questions"][0], "question_id": str(i)} for i in range(100)]
    elif corruption == "namespace":
        payload["questions"][1]["namespace_id"] = "another-history"
    elif corruption == "date":
        payload["questions"][1]["question_date"] = "2023/05/01 (Mon) 23:18"
    elif corruption == "duplicate":
        payload["questions"][1]["question_id"] = payload["questions"][0]["question_id"]
    elif corruption == "scope":
        payload["scope_sha256"] = "d"*64
    else:
        payload["general_accuracy_claim_permitted"] = True
    artifact, _ = publish_sealed_json(tmp_path/"questions.json", payload)
    with pytest.raises(ValueError, match="same dated cached history"):
        load_questions(artifact.path, scope)


def test_configurable_builder_preserves_previous_default_packets():
    from memory_condense.application.native_spine_context_retrieval import ResidentNativeSpineContextMemory
    from memory_condense.application.native_spine_retrieval import ResidentNativeSpineMemory
    from tests.test_native_spine_routing import DATED, QUERY, fixture
    from tools.evaluate_native_spine_context_pilot import build
    from tools.evaluate_native_spine_design_questions import build_packet
    h, semantic, hierarchy, encoder = fixture()
    memory = ResidentNativeSpineMemory(semantic, hierarchy, encoder=encoder, load_turn=h.get_turn)
    contextual = ResidentNativeSpineContextMemory(semantic, hierarchy, encoder=encoder, load_turn=h.get_turn)
    question = {"retrieval_query": QUERY, "prompt_question": DATED}
    for arm in ("flat", "user_first", "parent_context"):
        assert build_packet(memory, contextual, question, arm, 3072) == build(memory, contextual, question, arm)
    assert encoder.calls == [QUERY]*6


def test_reader_change_preserves_exact_raw_prompt_and_does_not_mutate_control():
    from tools.evaluate_native_spine_design_questions import reader_messages
    original = [{"role": "system", "content": "Original policy."},
                {"role": "user", "content": "Exact raw transcript.\n  Original whitespace.\nQuestion: what changed?"}]
    before = deepcopy(original)
    changed = reader_messages(original, "v5")
    assert original == before and changed[1:] == original[1:]
    assert changed[0]["content"] != original[0]["content"]
    assert reader_messages(original, "v2") == original
