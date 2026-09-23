import json

from tools import native_spine_engineering_dedup_resume as dedup


def test_repeated_raw_text_is_compiled_once_but_both_spans_survive(tmp_path, monkeypatch):
    text = 'The cache occurrence regression was observed in this tool result.'
    rows = [{'turn_id': name, 'role': 'system', 'text': text} for name in ('first', 'second')]
    calls = []

    class Gateway:
        def call(self, kind, messages, **kwargs):
            fragments = json.loads(messages[1]['content'])['fragments']
            calls.append(fragments)
            return {'content': json.dumps({'atoms': [
                {'label': f['label'], 'summary': 'System recorded a cache regression.',
                 'support': ['cache occurrence regression']} for f in fragments]})}

    monkeypatch.setattr(dedup.s.components.raw_summary, 'pack_batches', dedup.pack_unique_raw_batches)
    atoms = dedup.s.SummaryCompiler(tmp_path, Gateway()).raw(rows)
    assert [len(c) for c in calls] == [1]
    assert len(list((tmp_path / 'atomic-summaries').glob('*.json'))) == 1
    assert len(atoms) == 2
    assert {a.spans[0].turn_id for a in atoms} == {'first', 'second'}
    assert len({a.section_id for a in atoms}) == 2


def test_different_roles_do_not_share_a_pending_summary():
    rows = [{'role': role, 'text': 'An identical sentence.'} for role in ('user', 'assistant')]
    fragments = dedup.s.components.fragments_for(rows)
    batches = dedup.pack_unique_raw_batches(fragments)
    assert [f.role for batch in batches for f in batch] == ['user', 'assistant']
