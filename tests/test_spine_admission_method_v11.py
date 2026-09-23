import pytest

from tools import admit_spine_corpus_v7 as admission
from tools import verify_spine_admission_method_v11 as verification
from tools.matched_eval.artifacts import read_sealed_json
from tests import test_spine_admission_method_v10 as previous_tests
from tests import test_spine_admission_support_escaping as escaping_tests


@pytest.mark.parametrize('invalid', [False, True])
def test_new_admission_replays_complete_original_and_recovered_compactions(tmp_path, monkeypatch, invalid):
    prior_admission = previous_tests.admission
    monkeypatch.setattr(admission, 'prepare', lambda *args: prior_admission.prepare(*args))
    monkeypatch.setattr(previous_tests, 'admission', admission)
    monkeypatch.setattr(previous_tests, 'verification', verification)
    path, repair_root, lineage_calls = previous_tests.content_fixture(tmp_path, monkeypatch, invalid=invalid)
    before = path.read_bytes()
    result = verification.verify(path, repair_root)
    assert result.payload['atom_count'] == result.payload['budget_compacted_atoms'] == 9
    assert result.payload['compaction_provider_attempts'] == 2 + int(invalid)
    assert result.payload['new_provider_calls'] == 0 and path.read_bytes() == before
    assert verification.load_verified_method(path, read_sealed_json(path))[1] == result.sha256
    assert len(lineage_calls) == 2
    policy = read_sealed_json(path.parent / 'source-binding-policy-v10-prefix-0001.json')
    assert result.payload['method'] == verification.conditional_method(policy.payload)


def test_legacy_admission_and_compaction_transport_retain_common_extended_method(tmp_path, monkeypatch):
    monkeypatch.setattr(previous_tests, 'verification', verification)
    previous_tests.test_prior_compaction_transport_and_regular_admission_keep_one_method(tmp_path, monkeypatch)


def test_embedded_quote_repair_replays_native_content_without_calls(tmp_path, monkeypatch):
    escaping_tests.test_complete_raw_replay_preserves_summary_text_and_records_support_escaping(tmp_path, monkeypatch)
    path = tmp_path / 'original/offset-000/source-bound-atoms-prefix-0001.json'
    monkeypatch.setattr(verification, 'prepare', admission.prepare)
    monkeypatch.setattr(verification, 'verify_transport_lineage', lambda *args:
        {'additional_compaction_attempts':0, 'additional_raw_attempts':0})
    before = path.read_bytes()
    result = verification.verify(path)
    assert result.payload['atom_count'] == 2 and result.payload['compaction_provider_attempts'] == 0
    assert path.read_bytes() == before
    assert verification.load_verified_method(path, read_sealed_json(path))[1] == result.sha256
    policy = read_sealed_json(path.parent / 'source-binding-policy-v10-prefix-0001.json').payload
    for change in ({'syntax_repair':'rewrite summaries'}, {'implementation':{}},
                   {'invalid_slot_recovery':'unbounded'}, {'summary_use':'answer evidence'}):
        with pytest.raises(ValueError):
            verification.conditional_method({**policy, **change})


@pytest.mark.parametrize('failure', ['omitted_repairs', 'missing_raw_response', 'unverified_lineage'])
def test_extended_admission_requires_full_content_and_transport(tmp_path, monkeypatch, failure):
    prior_admission = previous_tests.admission
    monkeypatch.setattr(admission, 'prepare', lambda *args: prior_admission.prepare(*args))
    monkeypatch.setattr(previous_tests, 'admission', admission)
    monkeypatch.setattr(previous_tests, 'verification', verification)
    path, repairs, _ = previous_tests.content_fixture(tmp_path, monkeypatch)
    if failure == 'omitted_repairs':
        repairs = None
    elif failure == 'missing_raw_response':
        next((path.parent / 'raw-checkpoints').glob('*/*.response.json')).unlink()
    else:
        def reject(*args):
            raise ValueError('unverified transport')
        monkeypatch.setattr(verification, 'verify_transport_lineage', reject)
    with pytest.raises((ValueError, RuntimeError, FileNotFoundError)):
        verification.verify(path, repairs)
    assert not path.with_name('conditional-method-v11-prefix-0001.json').exists()
