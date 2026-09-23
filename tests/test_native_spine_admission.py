import copy
import hashlib
import json
import sqlite3
from types import SimpleNamespace

import pytest

from memory_condense.domain._discourse_identity import canonical_json, identity_sha256
from memory_condense.domain._tokenizer import count_chat_prompt_token_proxy
from memory_condense.search.native_spine_batch import messages
from memory_condense.search.native_spine_memory import validate_body_summaries
from memory_condense.search.native_spine_repair import partition
from memory_condense.search.native_spine_resegmentation import subdivide
from memory_condense.search.native_spine_summary import body_identity, fragment_body
from tools import assemble_native_spine_admitted as admission
from tools import compile_native_spine as compiler
from tools import finish_native_spine_section_repairs as refinement
from tools import repair_native_spine_batches as prior
from tools import resegment_native_spine_repairs as stage
from tools import run_native_spine_batches as runner
from tools.matched_eval.artifacts import publish_sealed_json, read_sealed_json


class Client:
    max_retries = 0

    def __init__(self, invalid=False):
        self.calls, self.invalid = 0, invalid
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def with_options(self, **kwargs):
        assert kwargs["max_retries"] == 0
        return self

    def close(self):
        pass

    def create(self, **kwargs):
        self.calls += 1
        wire = json.loads(kwargs["messages"][1]["content"])
        atoms = [{"label": f["label"], "summary": "Over budget. " * 100 if self.invalid
                  else f["speaker"] + " discusses a furniture purchase."}
                 for t in wire["transcripts"] for f in t["fragments"]]
        return SimpleNamespace(id="fixture", model=admission.MODEL, usage=None, choices=[
            SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content=canonical_json({"atoms": atoms})))
        ])


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    bodies = sorted([
        {"turns": [{"role": "user", "text": "I plan to buy two blue chairs and a wooden table next week."},
                   {"role": "assistant", "text": "You could try an independent furniture store."}]},
        {"turns": [{"role": "user", "text": "I bought a red desk and a green lamp yesterday morning."},
                   {"role": "assistant", "text": "Your new furniture sounds useful."}]},
    ], key=body_identity)
    raw_root, source = tmp_path / "raw", tmp_path / "source"
    raw_root.mkdir()
    bank = raw_root / "bodies.sqlite"
    db = sqlite3.connect(bank)
    db.execute("CREATE TABLE bodies(body_sha256 TEXT PRIMARY KEY, body_json TEXT)")
    db.executemany("INSERT INTO bodies VALUES (?,?)", [(body_identity(b), canonical_json(b)) for b in bodies])
    db.commit()
    db.close()
    sources, _ = publish_sealed_json(raw_root / "sources.json", {"body_bank_path": bank.name, "body_count": 2})
    fs = tuple(f for b in bodies for f in fragment_body(b))
    bindings = []
    for ordinal, group in enumerate((fs[:1], fs[1:3], fs[3:])):
        prompt = messages(group)
        request, _ = publish_sealed_json(source / "requests" / f"{ordinal:06}.json", {
            "ordinal": ordinal, "sources_sha256": sources.sha256, "messages": prompt,
            "messages_sha256": identity_sha256(prompt), "pointers": [f.pointer() for f in group],
            "prompt_token_proxy": count_chat_prompt_token_proxy(prompt),
        })
        bindings.append({"path": str(request.path.relative_to(source)), "sha256": request.sha256, "atoms": len(group)})
    digest = hashlib.sha256()
    for fragment in fs:
        compiler.add_pointer(digest, fragment)
    plan, _ = publish_sealed_json(source / "preflight.json", {
        "implementation": compiler.implementation(), "models": [admission.MODEL],
        "gateway": compiler.GATEWAY, "retries": 0, "raw_inputs_to_qwen": False,
        "question_or_gold_inputs": False, "mode": "full", "sources_sha256": sources.sha256,
        "sources_root": str(raw_root.resolve()), "body_bank_sha256": admission.digest(bank),
        "requests": bindings, "max_atoms": 24, "max_prompt_tokens": 7000, "max_new_tokens": 4096,
        "fragment_count": len(fs), "body_count": 2, "ordered_pointer_sha256": digest.hexdigest(),
        "timeout_s": 240, "concurrency": 1,
    })
    good, bad = Client(), Client(invalid=True)
    monkeypatch.setattr(runner, "_completion_client", lambda *args: bad)
    runner.run_one(source, plan, admission.MODEL, bindings[0], True)
    monkeypatch.setattr(runner, "_completion_client", lambda *args: good)
    runner.run_one(source, plan, admission.MODEL, bindings[1], True)
    return SimpleNamespace(source=source, plan=plan, bindings=bindings, bodies=bodies,
                           fragments=fs, good=good, bad=bad, bank=bank)


def repair_chain(tmp_path, monkeypatch, corpus):
    first, second, third = (tmp_path / name for name in ("wording", "sections", "refinement"))
    prior.prepare(corpus.source, first, [0])
    monkeypatch.setattr(prior, "_completion_client", lambda *args: corpus.bad)
    with pytest.raises(ValueError):
        prior.execute(first, True)
    stage.prepare(first, second)
    monkeypatch.setattr(stage, "_completion_client", lambda *args: corpus.good)
    result = stage.execute(second, True)
    assert result.payload["complete_repair_snapshot"] is True
    refinement.prepare(second, third)
    refinement.execute(third, False)
    return third


def forbid_provider(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("admission must never create a provider")
    for module in (runner, prior, stage, refinement):
        monkeypatch.setattr(module, "_completion_client", forbidden)


def test_repairs_admit_complete_bodies_and_freeze_partial_snapshot(tmp_path, monkeypatch, corpus):
    repair = repair_chain(tmp_path, monkeypatch, corpus)
    unchanged = [read_sealed_json(f).sha256 for f in sorted((corpus.source / "validated").glob("*.json"))]
    calls = corpus.good.calls + corpus.bad.calls
    forbid_provider(monkeypatch)
    root = tmp_path / "partial"
    with pytest.raises(ValueError, match="every original batch"):
        admission.assemble(corpus.source, root, repair_roots=[repair])
    assert not (root / "summary-bodies.json").exists()
    result = admission.assemble(corpus.source, root, repair_roots=[repair], allow_partial=True)
    assert result.payload["body_count"] == 1
    assert result.payload["original_fragments_covered"] == 2
    assert result.payload["atom_count"] > 2
    assert result.payload["additional_sections"] == result.payload["atom_count"] - 2
    assert result.payload["complete_source_compilation"] is False
    assert result.payload["pending_batches"] == 1 and result.payload["unrepaired_batches"] == 0
    store = admission.AdmittedSummaryBodies(root)
    try:
        atoms = store.load(body_identity(corpus.bodies[0]))
        assert validate_body_summaries(corpus.bodies[0], atoms) == body_identity(corpus.bodies[0])
        with pytest.raises(KeyError):
            store.load(body_identity(corpus.bodies[1]))
    finally:
        store.close()
    assert corpus.good.calls + corpus.bad.calls == calls
    assert unchanged == [read_sealed_json(f).sha256 for f in sorted((corpus.source / "validated").glob("*.json"))]

    monkeypatch.setattr(runner, "_completion_client", lambda *args: corpus.good)
    runner.run_one(corpus.source, corpus.plan, admission.MODEL, corpus.bindings[2], True)
    forbid_provider(monkeypatch)
    # Progress in the live producer cannot silently promote an old snapshot.
    assert admission.assemble(corpus.source, root, repair_roots=[repair], allow_partial=True).sha256 == result.sha256
    with pytest.raises(ValueError, match="partial snapshot"):
        admission.assemble(corpus.source, root, repair_roots=[repair])
    complete = admission.assemble(corpus.source, tmp_path / "complete", repair_roots=[repair])
    assert complete.payload["complete_source_compilation"] is True
    assert complete.payload["body_count"] == 2 and complete.payload["original_fragments_covered"] == 4
    assert complete.payload["atom_count"] > 4
    assert complete.payload["full100_target_passed"] is False
    assert admission.assemble(corpus.source, tmp_path / "complete", repair_roots=[repair]).sha256 == complete.sha256


def test_unrepaired_and_pending_fragments_cannot_leak_incomplete_bodies(tmp_path, monkeypatch, corpus):
    forbid_provider(monkeypatch)
    result = admission.assemble(corpus.source, tmp_path / "store", allow_partial=True)
    assert result.payload["body_count"] == result.payload["atom_count"] == 0
    assert result.payload["unrepaired_batches"] == result.payload["pending_batches"] == 1
    assert result.payload["complete_source_compilation"] is False


def test_conflicting_repair_inputs_rejected(tmp_path, monkeypatch, corpus):
    repair = repair_chain(tmp_path, monkeypatch, corpus)
    forbid_provider(monkeypatch)
    with pytest.raises(ValueError, match="duplicate repaired batch"):
        admission.verified_repairs([repair, repair], corpus.plan, admission.MODEL)
    wrong = SimpleNamespace(payload=corpus.plan.payload, sha256="wrong-compilation")
    with pytest.raises(ValueError, match="original compilation"):
        admission.verified_repairs([repair], wrong, admission.MODEL)


@pytest.mark.parametrize("defect", ["hole", "role", "raw_hash", "valid_text", "extra", "budget"])
def test_replacement_coverage_rechecked_before_cache_insertion(corpus, defect):
    fs = corpus.fragments[:2]
    response = canonical_json({"atoms": [
        {"label": "T0", "summary": "Too long. " * 100},
        {"label": "T1", "summary": "Assistant suggests a shop."},
    ]})
    valid, bad = partition(response, fs)
    atoms = [{"pointer": f.pointer(), "summary": "User describes a purchase."} for f in subdivide(fs[0])]
    atoms.append(valid[1])
    row = {"summaries": copy.deepcopy(atoms), "original_atom_count": 2,
           "unchanged_valid_atom_indices": list(valid), "replaced_original_atom_indices": list(bad)}
    assert admission.repaired_atoms(response, fs, row) == atoms
    if defect == "hole":
        row["summaries"].pop(0)
    elif defect == "role":
        row["summaries"][0]["pointer"]["role"] = "assistant"
    elif defect == "raw_hash":
        row["summaries"][0]["pointer"]["span_text_sha256"] = "wrong"
    elif defect == "valid_text":
        row["summaries"][-1]["summary"] = "Changed a valid summary."
    elif defect == "extra":
        row["summaries"].append(copy.deepcopy(atoms[-1]))
    else:
        row["summaries"][0]["summary"] = "Too long. " * 100
    with pytest.raises(ValueError):
        admission.repaired_atoms(response, fs, row)


def test_storage_rejects_database_changes(tmp_path, monkeypatch, corpus):
    forbid_provider(monkeypatch)
    root = tmp_path / "store"
    admission.assemble(corpus.source, root, allow_partial=True)
    with (root / "summary-bodies.sqlite").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="database changed"):
        admission.AdmittedSummaryBodies(root)
