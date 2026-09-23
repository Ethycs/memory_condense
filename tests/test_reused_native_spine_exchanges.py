from pathlib import Path

import pytest

from tests.test_native_spine_exchanges import Backend, prepared
from tools import compile_native_spine_exchanges as original
from tools import compile_reused_native_spine_exchanges as reused
from tools.matched_eval.artifacts import publish_sealed_json, read_sealed_json


def copy_inputs(source, target, *, sources="fixture-sources"):
    source = Path(source)
    inputs = read_sealed_json(source/"inputs.json")
    for binding in inputs.payload["bodies"]:
        body = read_sealed_json(source/binding["path"])
        publish_sealed_json(target/binding["path"], body.payload)
    publish_sealed_json(target/"inputs.json", dict(inputs.payload, sources_sha256=sources))
    return target


def completed(prepared):
    root, history = prepared
    legacy = copy_inputs(root, root/"legacy")
    backend = Backend()
    result = original.execute(legacy, backend, 128)
    assert result.payload["complete_available_body_exchanges"] and backend.calls > 0
    return root, legacy, backend, result


def test_successor_reuses_actual_accepted_merges_without_forging_new_responses_and_replays(prepared):
    root, legacy, backend, old_result = completed(prepared)
    calls = backend.calls
    frozen = {str(p.relative_to(legacy)): p.read_bytes() for p in legacy.rglob("*.json")}
    new = copy_inputs(legacy, root/"new")
    result = reused.execute(new, backend, 0, reuse_roots=[legacy])
    assert result.payload["complete_available_body_exchanges"] and backend.calls == calls
    assert not list((new/"responses").glob("*.json")) and not list((new/"requests").glob("*.json"))
    plan = read_sealed_json(new/"preflight.json")
    assert plan.payload["producer_format"] == reused.FORMAT and plan.payload["reused_merge_keys"] > 0
    assert plan.payload["reuse_roots"][0]["result_sha256"] == old_result.sha256
    assert reused.execute(new, backend, 0).sha256 == result.sha256 and backend.calls == calls
    old_body = read_sealed_json(legacy/old_result.payload["compiled_bodies"][0]["path"]).payload
    new_body = read_sealed_json(new/result.payload["compiled_bodies"][0]["path"]).payload
    assert old_body["raw_span_population_sha256"] == new_body["raw_span_population_sha256"]
    assert [(e["user_spine"], e["attached_context"], e["section"]["spans"]) for e in old_body["exchanges"]] == [
        (e["user_spine"], e["attached_context"], e["section"]["spans"]) for e in new_body["exchanges"]]
    assert all((legacy/path).read_bytes() == raw for path, raw in frozen.items())


def test_successor_can_reuse_a_completed_successor_with_its_ancestor_cache(prepared):
    root, legacy, backend, _ = completed(prepared)
    first = copy_inputs(legacy, root/"first")
    reused.execute(first, backend, 0, reuse_roots=[legacy])
    calls = backend.calls
    second = copy_inputs(legacy, root/"second")
    result = reused.execute(second, backend, 0, reuse_roots=[first])
    assert result.payload["complete_available_body_exchanges"] and backend.calls == calls
    assert read_sealed_json(second/"preflight.json").payload["reused_merge_keys"] > 0


@pytest.mark.parametrize("defect", ["sources", "duplicate", "cycle"])
def test_reuse_rejects_foreign_corpus_duplicate_or_cyclic_sources_before_generation(prepared, defect):
    root, legacy, backend, _ = completed(prepared)
    target = copy_inputs(legacy, root/"target", sources="different" if defect == "sources" else "fixture-sources")
    sources = [legacy, legacy] if defect == "duplicate" else [target] if defect == "cycle" else [legacy]
    calls = backend.calls
    with pytest.raises(ValueError):
        reused.execute(target, backend, 128, reuse_roots=sources)
    assert backend.calls == calls and not (target/"result.json").exists()


def test_corrupted_source_response_cannot_be_reused_or_implicitly_regenerated(prepared):
    root, legacy, backend, _ = completed(prepared)
    response = next((legacy/"responses").glob("*.json"))
    response.write_bytes(response.read_bytes() + b" ")
    target = copy_inputs(legacy, root/"target")
    calls = backend.calls
    with pytest.raises(ValueError):
        reused.execute(target, backend, 128, reuse_roots=[legacy])
    assert backend.calls == calls and not (target/"preflight.json").exists()
