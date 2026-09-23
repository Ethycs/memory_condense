from copy import deepcopy
from types import SimpleNamespace

import pytest

from memory_condense.application.native_spine_context_retrieval import ResidentNativeSpineContextMemory
from memory_condense.application.threaded_section_context import TranscriptOrder
from memory_condense.domain._discourse_identity import identity_sha256, quote_sha256
from tests.test_native_spine_context_routing import parent_fixture
from tests.test_native_spine_parent_users100 import candidate
from tests.test_native_spine_routing import QUERY
from tools import evaluate_native_spine_extractive100 as evaluation
from tools.matched_eval.artifacts import publish_sealed_json


def instructions():
    return {'format': 'native-spine-reader-instructions-v1', 'name': 'user-extraction-v8',
        'baseline_system_sha256': quote_sha256(evaluation.reader.SPINE_READER_SYSTEM_PROMPT_V6),
        'system_prompt': evaluation.reader.SPINE_READER_SYSTEM_PROMPT_V8}


def changed_candidate(tmp_path, ordinal=0):
    case, original, calls = candidate(tmp_path / 'original', ordinal)
    history, semantic, hierarchy, encoder = parent_fixture(future=True)
    resident = ResidentNativeSpineContextMemory(semantic, hierarchy,
        encoder=encoder, load_turn=history.get_turn)
    policy = {**evaluation.context_policy.DENSE_PARENT_2048, 'max_direct': 8}
    result = evaluation.build(resident, evaluation.frozen.question(case), policy,
        TranscriptOrder(history.turns.values()), instructions())
    messages, hydration, routing, rendered = result
    assert encoder.calls == [QUERY]
    payload = {'messages': {a: messages for a in evaluation.ARMS},
        'hydration': {'parent_context': hydration}, 'routing': {'parent_context': routing},
        'rendered': {'parent_context': rendered}}
    evidence, _ = publish_sealed_json(tmp_path / 'changed.json', payload)
    for c in calls:
        c.update(messages=deepcopy(messages), messages_sha256=identity_sha256(messages),
                 evidence_sha256=evidence.sha256)
    return case, original, evidence, calls


@pytest.mark.parametrize('ordinal', [0, 1])
def test_fresh_retrieval_changes_only_system_instructions(tmp_path, ordinal):
    case, original, evidence, calls = changed_candidate(tmp_path, ordinal)
    evaluation.validate_pair(calls, case, evidence, instructions())
    evaluation.validate_preservation(evidence, original)
    assert evidence.payload['messages']['parent_context'][0] != original.payload['messages']['parent_context'][0]


@pytest.mark.parametrize('defect', ['routing', 'hydration', 'rendered', 'question'])
def test_reader_cannot_change_evidence_even_with_new_message_hashes(tmp_path, defect):
    _, original, evidence, _ = changed_candidate(tmp_path)
    payload = deepcopy(evidence.payload)
    if defect == 'question':
        for messages in payload['messages'].values():
            messages[-1]['content'] += '\nUnapproved hint'
    else:
        payload[defect]['parent_context']['unapproved_hint'] = 'changed'
    with pytest.raises(ValueError, match='reader comparison changed'):
        evaluation.validate_preservation(SimpleNamespace(payload=payload), original)


@pytest.mark.parametrize('defect', ['question', 'evidence', 'schedule', 'prompt', 'reader'])
def test_candidate_and_api_control_keep_same_question_and_prompt(tmp_path, defect):
    case, _, evidence, calls = changed_candidate(tmp_path)
    if defect == 'question':
        calls[0]['question']['retrieval_query'] = 'Altered question'
    elif defect == 'evidence':
        calls[0]['evidence_sha256'] = '0' * 64
    elif defect == 'schedule':
        calls.reverse()
    else:
        calls[0]['messages'][0 if defect == 'reader' else -1]['content'] += '\nAltered content'
        calls[0]['messages_sha256'] = identity_sha256(calls[0]['messages'])
    with pytest.raises(ValueError, match='candidate'):
        evaluation.validate_pair(calls, case, evidence, instructions())


@pytest.mark.parametrize('field', ['name', 'system_prompt'])
def test_policy_is_fixed_before_execution(field):
    policy = instructions()
    policy[field] += ' changed'
    with pytest.raises(ValueError, match='fixed extractive'):
        evaluation.validate_reader_policy(policy)


def test_partial_population_cannot_be_graded(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluation.frozen, 'recorded', lambda *_: [None] * 199)
    with pytest.raises(ValueError, match='all 100 questions and controls'):
        evaluation.seal_answers(tmp_path, SimpleNamespace())
