import json

import pytest

from tools import admit_spine_corpus_v7 as admission
from tools import repair_spine_summary_budget_v4 as compact
from tools import recover_spine_summary_budget_v2 as recovery
from tools import build_spine_corpus_hierarchy_resilient as resilient
from tools.matched_eval.artifacts import read_sealed_json
from tests import test_spine_admission_method_verification as original
from tests import test_spine_compaction_batch_completion as completed
from tests import test_spine_summary_budget_multibatch as batches


def test_complete_raw_replay_preserves_summary_text_and_records_support_escaping(tmp_path, monkeypatch):
    monkeypatch.setattr(original, 'admission', admission)
    record = original.record
    def malformed_support(content, **kwargs):
        body = json.loads(content)
        body['atoms'][0]['support'] = ['“A "quoted" support string.”']
        broken = json.dumps(body, ensure_ascii=False).replace('\\"quoted\\"', '"quoted"')
        return record(broken, **kwargs)
    monkeypatch.setattr(original, 'record', malformed_support)
    path, _ = original.fixture(tmp_path, monkeypatch)
    before = path.read_bytes()
    atoms = read_sealed_json(path).payload
    assert atoms['complete_namespace'] and len(atoms['atoms']) == 2
    assert [a['summary'] for a in atoms['atoms']] == ['User asks for a plan.', 'Assistant suggests a plan.']
    assert atoms['summary_texts_unchanged'] and atoms['new_provider_calls'] == 0
    repair = atoms['admission_audits'][0]['support_json_syntax_repair']
    assert len(repair['original_insertion_offsets']) == 2 and repair['original_characters_preserved']
    admission.admit(path.parent.parent, 0, 1)
    assert path.read_bytes() == before


@pytest.mark.parametrize('invalid', [False, True])
def test_existing_bounded_compaction_branches_replay_without_new_calls(tmp_path, monkeypatch, invalid):
    corpus, parent, preflight, second, _ = completed.fixture(tmp_path, monkeypatch, invalid=invalid)
    monkeypatch.setattr(admission, 'prepare', completed.admission.prepare)
    batches.record(batches.output(second), **batches.runtime_kwargs(parent, preflight, second))
    repair_root = parent
    if invalid:
        repair_root = tmp_path / 'recovery'
        recovery.prepare(repair_root, parent)
        monkeypatch.setattr(resilient, '_completion_client', lambda *args:
            original.Client(json.dumps({'summary':'User reports item 0.'})))
        recovery.run(repair_root, True, 2)
    else:
        compact.run(parent, False)
    monkeypatch.setattr(compact, '_completion_client', lambda *args: pytest.fail('admission called Qwen'))
    monkeypatch.setattr(resilient, '_completion_client', lambda *args: pytest.fail('admission called Qwen'))
    admission.admit(corpus, 0, 1, repair_root)
    path = corpus / 'offset-000/source-bound-atoms-prefix-0001.json'
    before = path.read_bytes()
    atoms = read_sealed_json(path).payload
    assert atoms['budget_compacted_atoms'] == 9 and len(atoms['atoms']) == 9
    assert [a['summary'] for a in atoms['atoms']] == [f'User reports item {i}.' for i in range(9)]
    assert atoms['complete_namespace'] and atoms['new_provider_calls'] == 0
    admission.admit(corpus, 0, 1, repair_root)
    assert path.read_bytes() == before
