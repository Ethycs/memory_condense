from copy import deepcopy
from types import SimpleNamespace

import pytest

from memory_condense.domain._discourse_identity import identity_sha256, quote_sha256
from memory_condense.eval.spine_reader_policy_v5 import SPINE_READER_SYSTEM_PROMPT_V5
from tools import evaluate_native_spine_reader100 as evaluation
from tools.matched_eval.artifacts import publish_sealed_json


def test_reader_change_preserves_raw_evidence_question_and_baseline():
    evidence = '<S1>Do not use onions. I chose shallots.</S1>\nQuestion: What did I choose?'
    original = [{'role': 'system', 'content': SPINE_READER_SYSTEM_PROMPT_V5},
                {'role': 'user', 'content': evidence + '\nShort answer:'}]
    before = deepcopy(original)
    changed = evaluation.reader.complete_reader_messages(original)
    assert original == before
    assert changed[-1] == {'role': 'user', 'content': evidence + '\nAnswer:'}
    assert changed[0]['content'] != original[0]['content']


def test_existing_journal_authentication_accepts_reader_evidence_and_rejects_changes(tmp_path):
    messages = [{'role': 'user', 'content': 'Which item?'}]
    hydration, routing = {'exact': 'raw'}, {'raw_reads_during_routing': 0}
    evidence, _ = publish_sealed_json(tmp_path / 'evidence/000.json', {
        'messages': {a: messages for a in evaluation.ARMS},
        'hydration': {'parent_context': hydration}, 'routing': {'parent_context': routing}})
    calls = [{'call_index': i, 'question': {'ordinal': 0}, 'arm': arm, 'messages': messages,
              'messages_sha256': identity_sha256(messages), 'evidence_sha256': evidence.sha256}
             for i, arm in enumerate(evaluation.ARMS)]
    plan, _ = publish_sealed_json(tmp_path / 'preflight.json', {'calls': calls})
    for call in calls:
        prefix = tmp_path / 'journal' / f'{call["call_index"]:03d}'
        req, _ = publish_sealed_json(prefix.with_suffix('.request.json'), {'preflight_sha256': plan.sha256, 'call': call})
        publish_sealed_json(prefix.with_suffix('.response.json'), {'request_sha256': req.sha256,
            'messages': messages, 'hydration': hydration if call['arm'] == 'parent_context' else None,
            'routing': routing if call['arm'] == 'parent_context' else None,
            'measurement': {'messages_sha256': identity_sha256(messages), 'model': evaluation.frozen.MODEL,
                'max_tokens': 256, 'prediction': 'item', 'prediction_sha256': quote_sha256('item')}})
    assert len(evaluation.frozen.recorded(tmp_path, plan)) == 2
    with pytest.raises(ValueError, match='all 100 questions'):
        evaluation.seal_answers(tmp_path, plan)
    altered = deepcopy(plan.payload)
    altered['calls'][0]['messages'][0]['content'] = 'Different question'
    with pytest.raises(ValueError, match='binding'):
        evaluation.frozen.recorded(tmp_path, SimpleNamespace(payload=altered, sha256=plan.sha256))


def test_reader_rejects_unexpected_source_policy():
    with pytest.raises(ValueError, match='unchanged v5'):
        evaluation.reader.complete_reader_messages([{'role': 'system', 'content': 'unknown'}])
