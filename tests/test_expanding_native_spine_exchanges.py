from pathlib import Path

import pytest

from memory_condense.domain._discourse_identity import identity_sha256
from tests.test_native_spine_exchanges import Backend, prepared
from tests.test_reused_native_spine_exchanges import completed, copy_inputs
from tools import compile_native_spine_exchanges as original
from tools import compile_reused_native_spine_exchanges as reused
from tools import compile_recovered_native_spine_exchanges as recovered
from tools import compile_expanding_native_spine_exchanges as expanding
from tools.matched_eval.artifacts import read_sealed_json


@pytest.mark.parametrize("producer", ["original", "reused", "recovered", "expanding"])
def test_expansion_reuses_complete_producers_and_preserves_their_records(prepared, monkeypatch, producer):
    root, legacy, backend, prior_result = completed(prepared)
    source = legacy
    if producer != "original":
        source = copy_inputs(legacy, root / producer)
        if producer == "recovered":
            journal = original.NeutralJournal(legacy, read_sealed_json(legacy / "preflight.json"), backend, 0)
            journal.replay()
            cache = dict(journal.cache.values)
            # Exercise recovered compilation and reuse dispatch with a fixture
            # seed. The real recovery's input/projection validation has separate
            # tests and is also exercised by actual zero-call cache admission.
            def seed(recovery_root, selected_backend, inputs_sha):
                assert selected_backend.identity_sha256 == backend.identity_sha256
                assert inputs_sha == read_sealed_json(source / "inputs.json").sha256
                return dict(cache), {"root": str(root / "fixture-recovery"),
                                     "merge_cache_sha256": identity_sha256(cache)}
            monkeypatch.setattr(recovered.recovery_seed, "load", seed)
            prior_result = recovered.execute(source, backend, 0, recovery_root=root / "fixture-recovery")
        else:
            module = reused if producer == "reused" else expanding
            prior_result = module.execute(source, backend, 0, reuse_roots=[legacy])
    frozen = {path: path.read_bytes() for path in source.rglob("*.json")}
    calls = backend.calls
    target = copy_inputs(source, root / "target")
    result = expanding.execute(target, backend, 0, reuse_roots=[source])
    assert result.payload["complete_available_body_exchanges"] and backend.calls == calls
    assert not result.payload["complete_source_compilation"]
    assert not list((target / "responses").glob("*.json"))
    assert not list((target / "requests").glob("*.json"))
    plan = read_sealed_json(target / "preflight.json")
    assert plan.payload["producer_format"] == expanding.FORMAT
    assert plan.payload["reuse_roots"][0]["result_sha256"] == prior_result.sha256
    assert plan.payload["reused_merge_keys"] > 0
    assert expanding.execute(target, backend, 0).sha256 == result.sha256
    old = read_sealed_json(source / prior_result.payload["compiled_bodies"][0]["path"])
    new = read_sealed_json(target / result.payload["compiled_bodies"][0]["path"])
    assert old.payload["raw_span_population_sha256"] == new.payload["raw_span_population_sha256"]
    assert [(e["user_spine"], e["attached_context"], e["section"]["spans"]) for e in old.payload["exchanges"]] == [
        (e["user_spine"], e["attached_context"], e["section"]["spans"]) for e in new.payload["exchanges"]]
    assert all(path.read_bytes() == data for path, data in frozen.items())


@pytest.mark.parametrize("defect", ["sources", "backend", "duplicate", "cycle", "response"])
def test_expansion_refuses_invalid_reuse_before_generation(prepared, defect):
    root, source, backend, _ = completed(prepared)
    target = copy_inputs(source, root / "target", sources="different" if defect == "sources" else "fixture-sources")
    if defect == "backend":
        backend.identity_sha256 = "different-model"
    if defect == "response":
        path = next((source / "responses").glob("*.json"))
        path.write_bytes(path.read_bytes() + b" ")
    sources = [source, source] if defect == "duplicate" else [target] if defect == "cycle" else [source]
    calls = backend.calls
    with pytest.raises(ValueError):
        expanding.execute(target, backend, 128, reuse_roots=sources)
    assert backend.calls == calls and not (target / "preflight.json").exists()


def test_expansion_refuses_partial_or_interrupted_sources(prepared):
    root, _ = prepared
    partial = copy_inputs(root, root / "partial")
    backend = Backend()
    original.execute(partial, backend, 0)
    target = copy_inputs(partial, root / "target")
    with pytest.raises(ValueError, match="completed source exchanges"):
        expanding.execute(target, backend, 128, reuse_roots=[partial])
    assert backend.calls == 0
    failed = Backend(fail=True)
    with pytest.raises(RuntimeError, match="simulated stopped"):
        original.execute(partial, failed, 128)
    with pytest.raises(ValueError, match="implicit retry"):
        expanding.execute(target, backend, 128, reuse_roots=[partial])
    assert backend.calls == 0


def test_new_generation_rescans_accepted_work_at_the_budget_boundary(prepared):
    root, _ = prepared
    target = copy_inputs(root, root / "target")
    backend = Backend()
    initial = expanding.execute(target, backend, 0)
    assert not initial.payload["complete_available_body_exchanges"]
    pending = read_sealed_json(next((target / "requests").glob("*.json")))
    result = expanding.execute(target, backend, len(pending.payload["jobs"]))
    assert result.payload["complete_available_body_exchanges"] and backend.calls > 0
    calls = backend.calls
    assert expanding.execute(target, backend, 0).sha256 == result.sha256
    assert backend.calls == calls
