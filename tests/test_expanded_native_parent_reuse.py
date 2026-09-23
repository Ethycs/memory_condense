from pathlib import Path

import pytest

from memory_condense.search.native_spine_merges import neutral_key
from tests.test_native_spine_exchanges import Backend, request
from tools import compile_expanded_native_spine_hierarchy as expanded
from tools import compile_native_spine_parent_budgets as parent
from tools.compile_native_spine_exchanges import NeutralJournal
from tools.matched_eval.artifacts import publish_sealed_json, read_sealed_json


def source_cache(root, backend, *, ancestors=None):
    exchanges = root/"source-exchanges"
    publish_sealed_json(exchanges/"inputs.json", {"sources_sha256": "source-corpus"})
    payload = dict(format="native-spine-parent-budgeted-hierarchy-v1", implementation=parent.implementation(),
        backend_sha256=backend.identity_sha256, attention_method_sha256="fixed-method",
        max_exchange_channel_tokens=128, max_parent_channel_tokens=512,
        raw_inputs_to_qwen=False, exchange_root=str(exchanges.resolve()))
    if ancestors is not None:
        payload.update(producer_format=expanded.FORMAT, producer_implementation=expanded.implementation(),
                       reuse_parent_roots=ancestors)
    plan, _ = publish_sealed_json(root/"preflight.json", payload)
    if ancestors is None:
        job = request(cap=512)
        journal = NeutralJournal(root, plan, backend, 128)
        assert journal.resolve({neutral_key(job): job})
    publish_sealed_json(root/"result.json", {"preflight_sha256": plan.sha256,
        "complete_available_body_hierarchies": True})
    return root


def test_parent_cache_reuses_authenticated_summary_outputs_without_generation(tmp_path):
    source = source_cache(tmp_path/"source", Backend())
    backend = Backend(fail=True)
    values, receipts = expanded.reusable_parents([source], backend, "source-corpus", "fixed-method")
    assert values == {neutral_key(request(cap=512)): "User discussed furniture."}
    assert backend.calls == 0 and receipts[0]["accepted_merge_count"] == 1
    assert receipts[0]["result_sha256"] == read_sealed_json(source/"result.json").sha256


@pytest.mark.parametrize("corpus,method", [("other-corpus", "fixed-method"), ("source-corpus", "different-method")])
def test_parent_reuse_rejects_changed_corpus_or_attention_method(tmp_path, corpus, method):
    source = source_cache(tmp_path/"source", Backend())
    backend = Backend(fail=True)
    with pytest.raises(ValueError, match="source or method changed"):
        expanded.reusable_parents([source], backend, corpus, method)
    assert backend.calls == 0


def test_expanded_parent_source_retains_exact_ancestor_cache_provenance(tmp_path):
    source = source_cache(tmp_path/"source", Backend())
    backend = Backend(fail=True)
    expected, receipts = expanded.reusable_parents([source], backend, "source-corpus", "fixed-method")
    second = source_cache(tmp_path/"second", backend, ancestors=receipts)
    actual, links = expanded.reusable_parents([second], backend, "source-corpus", "fixed-method")
    assert actual == expected and links[0]["accepted_merge_count"] == 1 and backend.calls == 0
    assert not list((second/"responses").glob("*.json"))


def test_changed_ancestor_receipt_is_rejected_even_when_its_cached_summary_is_valid(tmp_path):
    source = source_cache(tmp_path/"source", Backend())
    backend = Backend(fail=True)
    _, receipts = expanded.reusable_parents([source], backend, "source-corpus", "fixed-method")
    changed = [dict(receipts[0], merge_cache_sha256="changed")]
    second = source_cache(tmp_path/"second", backend, ancestors=changed)
    with pytest.raises(ValueError, match="ancestry changed"):
        expanded.reusable_parents([second], backend, "source-corpus", "fixed-method")
    assert backend.calls == 0


def test_corrupt_parent_response_cannot_be_reused_or_regenerated(tmp_path):
    source = source_cache(tmp_path/"source", Backend())
    response = next((source/"responses").glob("*.json"))
    response.write_bytes(response.read_bytes()+b" ")
    backend = Backend(fail=True)
    with pytest.raises(ValueError):
        expanded.reusable_parents([source], backend, "source-corpus", "fixed-method")
    assert backend.calls == 0
