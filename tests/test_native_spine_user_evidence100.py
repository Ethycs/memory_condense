from copy import deepcopy
from types import SimpleNamespace

import pytest

from memory_condense.domain._discourse_identity import identity_sha256
from tests.test_native_spine_application_reader100 import instructions
from tests.test_native_spine_context_routing import parent_fixture
from tests.test_native_spine_parent_users100 import candidate
from tools import evaluate_native_spine_user_evidence100 as evaluation
from tools.matched_eval.artifacts import publish_sealed_json


def projected_candidate(tmp_path, ordinal=0):
    case, original, _ = candidate(tmp_path, ordinal)
    history, _, _, _ = parent_fixture(future=True)
    order = evaluation.presentation.renderer.TranscriptOrder(history.turns.values())
    hydrated = evaluation.presentation.hydration_from_payload(original.payload['hydration']['parent_context'])
    rendered = evaluation.presentation.renderer.render_user_spine_sections(hydrated, order)
    question = evaluation.frozen.question(case)
    messages = evaluation.reader.apply_reader(evaluation.context_policy.messages(question,
        SimpleNamespace(render_context=lambda: rendered.text)), evaluation.validate_reader_policy(instructions()))
    payload = deepcopy(original.payload)
    payload['rendered']['parent_context'] = rendered.identity_payload()
    payload['messages'] = {a: messages for a in evaluation.ARMS}
    evidence, _ = publish_sealed_json(tmp_path / 'projected.json', payload)
    arms = evaluation.ARMS if ordinal % 2 == 0 else evaluation.ARMS[::-1]
    calls = [{'call_index': ordinal * 2 + i, 'question': question, 'arm': a,
              'messages': deepcopy(messages), 'messages_sha256': identity_sha256(messages),
              'evidence_sha256': evidence.sha256} for i, a in enumerate(arms)]
    return case, original, evidence, calls


@pytest.mark.parametrize('ordinal', [0, 1])
def test_projection_keeps_same_model_reader_raw_hydration_and_matched_question(tmp_path, ordinal):
    case, original, evidence, calls = projected_candidate(tmp_path, ordinal)
    evaluation.validate_preservation(evidence, original)
    evaluation.validate_pair(calls, case, evidence, instructions())
    assert evaluation.MODEL == evaluation.previous.MODEL
    assert evaluation.validate_reader_policy is evaluation.previous.validate_reader_policy


@pytest.mark.parametrize('field', ['hydration', 'routing'])
def test_projection_cannot_change_retrieval_or_hydration(tmp_path, field):
    _, original, evidence, _ = projected_candidate(tmp_path)
    payload = deepcopy(evidence.payload)
    payload[field]['parent_context']['injected'] = True
    with pytest.raises(ValueError, match='baseline hydration or routing'):
        evaluation.validate_preservation(SimpleNamespace(payload=payload), original)


@pytest.mark.parametrize('defect', ['omitted', 'missing_user', 'duplicate', 'changed_text', 'receipt'])
def test_projection_cannot_lose_or_rewrite_user_evidence(tmp_path, defect):
    _, original, evidence, _ = projected_candidate(tmp_path)
    payload = deepcopy(evidence.payload)
    rendered = payload['rendered']['parent_context']
    if defect == 'omitted':
        rendered['omitted_section_ids'].append('arbitrary-section')
    elif defect == 'missing_user':
        rendered['placements'].pop()
    elif defect == 'duplicate':
        rendered['placements'].append(rendered['placements'][0])
    elif defect == 'changed_text':
        start = rendered['placements'][0]['start_char']
        rendered['text'] = rendered['text'][:start] + '!' + rendered['text'][start + 1:]
    else:
        rendered['source_hydration_sha256'] = '0' * 64
    with pytest.raises(ValueError, match='exact whole-section evidence'):
        evaluation.validate_preservation(SimpleNamespace(payload=payload), original)


@pytest.mark.parametrize('defect', ['reader', 'question', 'control'])
def test_same_reader_and_identical_candidate_control_prompts_are_required(tmp_path, defect):
    case, _, evidence, calls = projected_candidate(tmp_path)
    calls = deepcopy(calls)
    if defect == 'question':
        calls[0]['question']['retrieval_query'] = 'Injected reference answer'
    elif defect == 'reader':
        calls[0]['messages'][0]['content'] += '\nDifferent reader'
        calls[0]['messages_sha256'] = identity_sha256(calls[0]['messages'])
    else:
        calls.reverse()
    with pytest.raises(ValueError, match='candidate'):
        evaluation.validate_pair(calls, case, evidence, instructions())
