import pytest

from tests.test_native_spine_exchanges import Backend, prepared
from tests.test_reused_native_spine_exchanges import completed, copy_inputs
from tools import compile_native_spine_exchanges as original
from tools import compile_reused_native_spine_exchanges as reused
from tools import native_spine_exchange_journal_cache as cache
from tools.matched_eval.artifacts import read_sealed_json


def forbidden(*args, **kwargs):
    raise AssertionError('journal reuse must not reconstruct bodies or call a model')


def test_reuses_exact_ancestor_cache_without_body_compilation(prepared, monkeypatch):
    root, legacy, backend, _ = completed(prepared)
    successor = copy_inputs(legacy, root/'successor')
    reused.execute(successor, backend, 0, reuse_roots=[legacy])
    expected = read_sealed_json(successor/'preflight.json').payload
    monkeypatch.setattr(original, 'execute', forbidden)
    monkeypatch.setattr(reused, '_execute', forbidden)
    monkeypatch.setattr(backend, 'generate', forbidden)
    reader = cache.NativeExchangeJournalCache(backend, 'fixture-sources')
    values, attempts, receipt = reader.load(successor)
    assert values and attempts
    assert receipt['accepted_merge_count'] == expected['reused_merge_keys']
    assert receipt['merge_cache_sha256'] == expected['reused_merge_cache_sha256']
    # An already authenticated root is not replayed again during this process.
    monkeypatch.setattr(cache.bounded.BoundedJournal, 'replay', forbidden)
    again = reader.load(successor)
    assert again == (values, attempts, receipt)
    values.clear()
    assert reader.load(successor)[0]


def test_rejects_foreign_source_before_model_calls(prepared, monkeypatch):
    _, legacy, backend, _ = completed(prepared)
    monkeypatch.setattr(backend, 'generate', forbidden)
    with pytest.raises(ValueError, match='source or implementation'):
        cache.NativeExchangeJournalCache(backend, 'different-source').load(legacy)


def test_corrupted_response_is_not_a_reusable_cache(prepared, monkeypatch):
    _, legacy, backend, _ = completed(prepared)
    response = next((legacy/'responses').glob('*.json'))
    response.write_bytes(response.read_bytes()+b' ')
    monkeypatch.setattr(backend, 'generate', forbidden)
    with pytest.raises(ValueError):
        cache.NativeExchangeJournalCache(backend, 'fixture-sources').load(legacy)


def test_partial_journal_retains_no_implicit_retry_rule(prepared, monkeypatch):
    root, _ = prepared
    source = copy_inputs(root, root/'interrupted')
    backend = Backend(fail=True)
    with pytest.raises(RuntimeError, match='simulated stopped'):
        original.execute(source, backend, 128)
    monkeypatch.setattr(backend, 'generate', forbidden)
    with pytest.raises(ValueError, match='refusing an implicit retry'):
        cache.NativeExchangeJournalCache(backend, 'fixture-sources').load(source, complete=False)


def test_duplicate_and_cyclic_ancestry_are_rejected(prepared):
    _, legacy, backend, _ = completed(prepared)
    reader = cache.NativeExchangeJournalCache(backend, 'fixture-sources')
    _, _, receipt = reader.load(legacy)
    with pytest.raises(ValueError, match='duplicate'):
        reader.ancestors([receipt, receipt])
    reader.active.add(legacy.resolve())
    with pytest.raises(ValueError, match='cyclic'):
        reader.load(legacy)
