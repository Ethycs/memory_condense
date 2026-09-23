from dataclasses import asdict
from pathlib import Path

import pytest

from memory_condense.application.section_retrieval import hydrate_section_plan
from memory_condense.domain._discourse_identity import canonical_json, identity_sha256
from memory_condense.search.episodes.user_spine_hierarchy import UserSpineExchange
from memory_condense.search.native_spine_memory import materialize_history
from memory_condense.search.native_spine_merges import NeutralMergeCache, neutral_key, neutral_messages
from memory_condense.search.native_spine_summary import body_identity, fragment_body
from memory_condense.search.section_routing import SectionSummaryIndex
from memory_condense.search.section_summary import SectionSummary
from memory_condense.search.spine_merge_batch import PendingMerge
from memory_condense.search.spine_summary import SpineSummaryFragment, SpineSummaryRequest
from tools import compile_native_spine_exchanges as compiler
from tools.matched_eval.artifacts import publish_sealed_json, read_sealed_json


def request(day="2026-01-02", *, role="user", literal="2025-04-05", cap=128):
    return SpineSummaryRequest("user_spine" if role == "user" else "attached_context", (
        SpineSummaryFragment(role, day, f"The speaker plans a visit on {literal}."),
        SpineSummaryFragment(role, day + " through " + day, "The speaker has not booked it."),
    ), max_output_tokens=cap)


def test_identical_model_inputs_reuse_across_dates_without_resolving_literal_dates():
    one, two = request(), request("2026-09-12")
    assert one.prompt_sha256 != two.prompt_sha256
    assert neutral_key(one) == neutral_key(two)
    wire = canonical_json(neutral_messages(one))
    assert "2026-01-02" not in wire and "transcript_date" not in wire
    assert "2025-04-05" in wire and "has not booked" in wire
    cache = NeutralMergeCache()
    with pytest.raises(PendingMerge):
        cache(one)
    cache.accept(one, canonical_json({"summary": "User plans the visit but has not booked it."}))
    assert cache(one) == cache(two)
    with pytest.raises(ValueError, match="accepted date-neutral merge changed"):
        cache.accept(two, canonical_json({"summary": "A conflicting generation."}))


@pytest.mark.parametrize("changed", ["role", "literal", "budget", "user_spine"])
def test_cache_key_retains_semantic_and_budget_inputs(changed):
    base = request(role="assistant")
    other = {
        "role": request(), "literal": request(role="assistant", literal="2025-07-08"),
        "budget": request(role="assistant", cap=64),
        "user_spine": SpineSummaryRequest("attached_context", base.fragments, user_spine="User wants a different trip.", max_output_tokens=128),
    }[changed]
    assert neutral_key(base) != neutral_key(other)


def test_neutralization_rejects_cross_occurrence_merges_and_untyped_inputs():
    one = request()
    changed = SpineSummaryRequest("user_spine", (
        one.fragments[0], SpineSummaryFragment("user", "2026-09-12", "Another event."),
    ))
    with pytest.raises(ValueError, match="cross source occurrence"):
        neutral_messages(changed)
    changed = SpineSummaryRequest("user_spine", (
        SpineSummaryFragment("user", "2026-01-02 through 2026-09-12", "Events at different dates."),
    ))
    with pytest.raises(ValueError, match="cross source occurrence"):
        neutral_messages(changed)
    with pytest.raises(TypeError):
        neutral_messages("RAW_TRANSCRIPT")


class Backend:
    max_batch_size = 4
    identity = {"fixture": True, "raw_inputs_to_qwen": False}
    identity_sha256 = identity_sha256(identity)

    def __init__(self, *, fail=False, invalid_first=False):
        self.calls = 0
        self.fail, self.invalid_first = fail, invalid_first

    def generate(self, jobs, attempt):
        self.calls += 1
        if self.fail:
            raise RuntimeError("simulated stopped local process")
        rows = []
        for job in jobs:
            wire = canonical_json(neutral_messages(job, attempt))
            assert "RAW_CANARY" not in wire and "2026-01-02" not in wire
            summary = "User discussed furniture." if job.kind == "user_spine" else "Assistant suggested a furniture shop."
            response = "not JSON" if self.invalid_first and attempt == 0 else canonical_json({"summary": summary})
            rows.append({"merge_key": neutral_key(job), "response": response, "stopped": True})
        return {"backend_sha256": self.identity_sha256, "rows": rows, "elapsed_s": 0,
                "raw_inputs_to_qwen": False, "remote_provider_calls": 0,
                "timestamp_metadata_in_model_inputs": False}


@pytest.fixture
def prepared(tmp_path):
    body = {"turns": [
        {"role": "user", "text": "RAW_CANARY " + "chairs and desks " * 35},
        {"role": "assistant", "text": "An independent furniture shop may have those items."},
    ]}
    fs = fragment_body(body, token_cap=30)
    atoms = [{"pointer": f.pointer(), "summary": ("User wants blue chairs and a large wooden desk. " * 8)
              if f.role == "user" else "Assistant suggests a shop."} for f in fs]
    source = {"original_session_ordinal": 0, "session_id": "fixture", "created_at": "2026-01-02T00:00:00+00:00",
              "metadata_text": "source boundary", "body_sha256": body_identity(body), "dataset_origin": "M"}
    source["occurrence_id"] = identity_sha256(source)
    history = materialize_history([source], load_body=lambda sha: body, load_summaries=lambda sha: atoms,
                                  compiler_identity="fixture-store")
    row, _ = publish_sealed_json(tmp_path / "bodies" / "fixture.json", {
        "source": source, "summary_body_store_sha256": "fixture-store",
        "atoms": [asdict(a) for a in history.atoms], "raw_text_included": False,
    })
    publish_sealed_json(tmp_path / "inputs.json", {
        "implementation": compiler.implementation(), "raw_text_included": False, "question_or_gold_inputs": False,
        "summary_body_store_sha256": "fixture-store", "body_count": 1, "atom_count": len(atoms),
        "complete_source_compilation": False,
        "bodies": [{"path": str(row.path.relative_to(tmp_path)), "sha256": row.sha256, "body_sha256": body_identity(body)}],
    })
    return tmp_path, history


def test_compiled_exchanges_replay_and_hydrate_exact_raw_under_original_dates(prepared):
    root, history = prepared
    backend = Backend(invalid_first=True)
    partial = compiler.execute(root, backend, 0)
    assert not partial.payload["complete_available_body_exchanges"] and backend.calls == 0
    result = compiler.execute(root, backend, 128)
    assert result.payload["complete_available_body_exchanges"] is True
    assert result.payload["complete_source_compilation"] is False
    assert result.payload["full100_target_passed"] is False
    assert backend.calls > 1
    saved_calls = backend.calls
    assert compiler.execute(root, backend, 0).sha256 == result.sha256
    assert backend.calls == saved_calls
    artifact = read_sealed_json(root / result.payload["compiled_bodies"][0]["path"])
    exchanges = [UserSpineExchange(**dict(e, section=SectionSummary.from_dict(e["section"])))
                 for e in artifact.payload["exchanges"]]
    assert tuple(s for e in exchanges for s in e.section.spans) == tuple(s for a in history.atoms for s in a.spans)
    plan = SectionSummaryIndex([e.section for e in exchanges]).route("furniture shop blue chairs", max_sections=len(exchanges))
    packet = hydrate_section_plan(plan, load_turn=history.get_turn, max_raw_spans=128, max_context_tokens=4096)
    assert not packet.diagnostics
    actual = [e for s in packet.sections for e in s.evidence]
    assert len(actual) == len(history.atoms)
    for e in actual:
        turn = history.get_turn(e.span.turn_id)
        assert e.text == turn.text[e.span.start_char:e.span.end_char]
        assert e.span.created_at == "2026-01-02T00:00:00+00:00"


def test_interrupted_local_generation_cannot_be_implicitly_retried(prepared):
    root, _ = prepared
    backend = Backend(fail=True)
    with pytest.raises(RuntimeError, match="simulated stopped"):
        compiler.execute(root, backend, 128)
    with pytest.raises(ValueError, match="refusing an implicit retry"):
        compiler.execute(root, backend, 128)
    assert backend.calls == 1
    assert not (root / "result.json").exists()
