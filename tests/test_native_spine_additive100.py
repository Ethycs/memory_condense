from copy import deepcopy
from types import SimpleNamespace

import pytest

from memory_condense.application.native_spine_context_retrieval import ResidentNativeSpineContextMemory
from memory_condense.application.threaded_section_context import TranscriptOrder
from memory_condense.domain._discourse_identity import identity_sha256
from tests.test_native_spine_application_reader100 import instructions
from tests.test_native_spine_context_routing import parent_fixture
from tests.test_native_spine_routing import QUERY
from tools import evaluate_native_spine_additive100 as evaluation
from tools.matched_eval.artifacts import publish_sealed_json


def candidate(tmp_path, ordinal=0):
    history, semantic, hierarchy, encoder = parent_fixture(future=True)
    resident = ResidentNativeSpineContextMemory(semantic, hierarchy,
        encoder=encoder, load_turn=history.get_turn)
    options = []

    def retrieve(query, dated, **kwargs):
        options.append(kwargs)
        return resident.retrieve(query, dated, **kwargs)

    case = {'ordinal': ordinal, 'question_id': 'test-question', 'namespace_id': 'test-history',
        'question': QUERY, 'question_date': '2026/09/12 (Saturday) 23:00'}
    question = evaluation.frozen.question(case)
    config = {**evaluation.context_policy.DENSE_PARENT_2048, 'max_direct': 8}
    messages, hydration, routing, rendered = evaluation.build(SimpleNamespace(retrieve=retrieve),
        question, config, TranscriptOrder(history.turns.values()), instructions())
    assert options == [config]
    assert encoder.calls == [QUERY]
    assert routing['baseline']['max_sections'] == 8
    assert all(e['span']['created_at'] < '2026-09-13' for s in hydration['sections'] for e in s['evidence'])
    evidence, _ = publish_sealed_json(tmp_path / 'evidence.json', {
        'messages': {a: messages for a in evaluation.ARMS},
        'hydration': {'parent_context': hydration}, 'routing': {'parent_context': routing},
        'rendered': {'parent_context': rendered}})
    arms = evaluation.ARMS if ordinal % 2 == 0 else evaluation.ARMS[::-1]
    calls = [{'call_index': ordinal * 2 + i, 'question': question, 'arm': a,
        'messages': deepcopy(messages), 'messages_sha256': identity_sha256(messages),
        'evidence_sha256': evidence.sha256} for i, a in enumerate(arms)]
    return case, evidence, calls


@pytest.mark.parametrize('ordinal', [0, 1])
def test_narrow_policy_live_retrieval_and_alternating_matched_controls(tmp_path, ordinal):
    case, evidence, calls = candidate(tmp_path, ordinal)
    evaluation.validate_pair(calls, case, evidence, instructions())


@pytest.mark.parametrize('defect', ['question', 'evidence', 'schedule', 'prompt', 'reader'])
def test_sealed_schedule_rejects_contamination_or_unmatched_control(tmp_path, defect):
    case, evidence, calls = candidate(tmp_path)
    calls = deepcopy(calls)
    if defect == 'question':
        calls[0]['question']['retrieval_query'] = 'Different question with a reference answer'
    elif defect == 'evidence':
        calls[0]['evidence_sha256'] = '0' * 64
    elif defect == 'schedule':
        calls.reverse()
    else:
        # Updating the message digest must not make changed evidence/instructions admissible.
        calls[0]['messages'][0 if defect == 'reader' else -1]['content'] += '\nInjected answer'
        calls[0]['messages_sha256'] = identity_sha256(calls[0]['messages'])
    with pytest.raises(ValueError, match='candidate'):
        evaluation.validate_pair(calls, case, evidence, instructions())


def test_partial_answers_cannot_open_grading(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluation.frozen, 'recorded', lambda *_: [None] * 199)
    with pytest.raises(ValueError, match='all 100 questions and controls'):
        evaluation.seal_answers(tmp_path, SimpleNamespace())


@pytest.mark.parametrize('defect', [None, 'context', 'raw', 'rendered'])
def test_preservation_requires_original_context_and_audited_exact_packet(tmp_path, defect):
    _, evidence, _ = candidate(tmp_path)
    checked_payload = {k: evidence.payload[k]['parent_context']
                       for k in ('hydration', 'routing', 'rendered')}
    checked = SimpleNamespace(payload=deepcopy(checked_payload))
    original = SimpleNamespace(payload=deepcopy(evidence.payload))
    if defect is None:
        evaluation.validate_preservation(evidence, original, checked)
        return
    if defect == 'context':
        original.payload['routing']['parent_context']['consulted_chunk_ids'] = ['foreign']
    elif defect == 'raw':
        original.payload['hydration']['parent_context']['sections'][0]['evidence'][0]['text'] += ' foreign'
    else:
        checked.payload['rendered']['text'] += ' foreign'
    with pytest.raises(ValueError, match='prior evidence'):
        evaluation.validate_preservation(evidence, original, checked)
