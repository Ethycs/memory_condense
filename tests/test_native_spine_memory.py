import copy
import hashlib
import json
import sqlite3

import pytest

from memory_condense.application.section_retrieval import hydrate_section_plan
from memory_condense.domain._discourse_identity import canonical_json, identity_sha256
from memory_condense.search.native_spine_memory import materialize_history, validate_body_summaries
from memory_condense.search.native_spine_summary import body_identity, fragment_body
from memory_condense.search.section_routing import SectionSummaryIndex
from tools import assemble_native_spine_summaries as assembler
from tools.matched_eval.artifacts import publish_sealed_json


def body_and_atoms():
    body = {"turns": [
        {"role": "user", "text": "I plan to visit the café tomorrow. 🌍 I have not booked it."},
        {"role": "assistant", "text": "You could also visit the nearby museum."},
    ]}
    atoms = [{"pointer": f.pointer(), "summary": "ROUTING_ADDRESS describes a proposed visit."}
             for f in fragment_body(body, token_cap=9)]
    return body, atoms


def source(body, ordinal, day):
    row = {"original_session_ordinal": ordinal, "session_id": "repeated-session",
           "created_at": day + "T00:00:00+00:00", "metadata_text": "source boundary",
           "body_sha256": body_identity(body), "dataset_origin": "M"}
    return row | {"occurrence_id": identity_sha256(row)}


def test_cached_body_hydrates_at_two_exact_dates_without_merging_occurrences():
    body, atoms = body_and_atoms()
    sessions = [source(body, 0, "2026-01-01"), source(body, 1, "2026-02-02")]
    loads = []

    def load_body(sha):
        loads.append(sha)
        return body

    history = materialize_history(sessions, load_body=load_body,
                                  load_summaries=lambda sha: atoms, compiler_identity="fixture")
    assert loads == [body_identity(body)]
    assert len(history.atoms) == len(atoms) * 2
    assert len(history.turns) == 4
    assert len(set(history.occurrence_ids)) == 2
    index = SectionSummaryIndex(history.atoms)
    plan = index.route("ROUTING_ADDRESS", max_sections=len(history.atoms))
    packet = hydrate_section_plan(plan, load_turn=history.get_turn,
                                  max_raw_spans=128, max_context_tokens=4096)
    assert not packet.diagnostics
    assert len(packet.sections) == len(history.atoms)
    for section in packet.sections:
        for evidence in section.evidence:
            turn = history.get_turn(evidence.span.turn_id)
            assert evidence.text == turn.text[evidence.span.start_char:evidence.span.end_char]
            assert evidence.span.created_at == turn.created_at.isoformat()
    assert "ROUTING_ADDRESS" not in packet.render_context()
    assert {t.created_at.isoformat() for t in history.turns.values()} == {s["created_at"] for s in sessions}
    with pytest.raises(TypeError):
        history.turns["foreign"] = next(iter(history.turns.values()))


@pytest.mark.parametrize("defect", ["missing_last", "missing_first", "reordered", "overlap",
                                   "wrong_role", "wrong_hash", "wrong_body", "bool_coordinate"])
def test_incomplete_or_mutated_source_bodies_cannot_materialize(defect):
    body, atoms = body_and_atoms()
    atoms = copy.deepcopy(atoms)
    if defect == "missing_last":
        atoms.pop()
    elif defect == "missing_first":
        atoms.pop(0)
    elif defect == "reordered":
        atoms.reverse()
    elif defect == "overlap":
        atoms[1]["pointer"]["start_char"] -= 1
    elif defect == "wrong_role":
        atoms[0]["pointer"]["role"] = "assistant"
    elif defect == "wrong_hash":
        atoms[0]["pointer"]["turn_text_sha256"] = "0" * 64
    elif defect == "wrong_body":
        body["turns"][0]["text"] += " Changed."
    else:
        atoms[0]["pointer"]["turn_ordinal"] = False
    with pytest.raises(ValueError):
        validate_body_summaries(body, atoms)


@pytest.mark.parametrize("defect", ["duplicate", "timestamp_changed", "question_field", "wrong_loader"])
def test_namespace_materialization_rejects_ambiguous_or_foreign_sources(defect):
    body, atoms = body_and_atoms()
    sessions = [source(body, 0, "2026-01-01")]
    if defect == "duplicate":
        sessions.append(sessions[0])
    elif defect == "timestamp_changed":
        sessions[0]["created_at"] = "2026-01-02T00:00:00+00:00"
    elif defect == "question_field":
        sessions[0]["question"] = "A benchmark question"
    elif defect == "wrong_loader":
        body = {"turns": [{"role": "user", "text": "A different history."}]}
    with pytest.raises(ValueError):
        materialize_history(sessions, load_body=lambda sha: body,
                            load_summaries=lambda sha: atoms, compiler_identity="fixture")


def test_summary_store_has_no_raw_text_and_rejects_database_mutation(tmp_path):
    body, atoms = body_and_atoms()
    path = tmp_path / "summary-bodies.sqlite"
    with sqlite3.connect(path) as database:
        database.execute("CREATE TABLE bodies(body_sha256 TEXT PRIMARY KEY, summaries_json TEXT, summary_sha256 TEXT)")
        database.execute("INSERT INTO bodies VALUES (?,?,?)",
                         (body_identity(body), canonical_json(atoms), identity_sha256(atoms)))
        database.commit()
    database.close()
    publish_sealed_json(tmp_path / "summary-bodies.json", {
        "database_sha256": assembler.digest(path),
        "implementation": {name: assembler.digest(name) for name in assembler.FILES},
    })
    store = assembler.SummaryBodies(tmp_path)
    try:
        assert store.load(body_identity(body)) == atoms
        assert body["turns"][0]["text"] not in json.dumps(store.load(body_identity(body)))
        with pytest.raises(KeyError):
            store.load("0" * 64)
    finally:
        store.close()
    with path.open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(ValueError, match="database changed"):
        assembler.SummaryBodies(tmp_path)
