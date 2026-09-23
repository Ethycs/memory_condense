from copy import deepcopy
from types import SimpleNamespace

import pytest

from memory_condense.domain._discourse_identity import quote_sha256
from tests.test_native_spine_application100 import candidate
from tools import evaluate_native_spine_application_model100 as evaluation
from tools.matched_eval.artifacts import publish_sealed_json


MODEL = 'codex_sdk/gpt-5.6-sol'


def journal_fixture(tmp_path, call_index=0):
    case, evidence, calls = candidate(tmp_path / 'evidence' / 'fixture')
    # The packet file is addressed by question ordinal in the response validator.
    publish_sealed_json(tmp_path / 'evidence' / '000.json', evidence.payload)
    plan, _ = publish_sealed_json(tmp_path / 'preflight.json', {'model': MODEL, 'calls': calls})
    call = calls[call_index]
    prefix = tmp_path / 'journal' / f'{call_index:03d}'
    request, _ = publish_sealed_json(prefix.with_suffix('.request.json'),
        {'preflight_sha256': plan.sha256, 'call': call})
    payload = {'request_sha256': request.sha256, 'messages': call['messages'],
        'measurement': {'model': MODEL, 'messages_sha256': call['messages_sha256'],
            'max_tokens': 256, 'prediction': 'Test answer', 'prediction_sha256': quote_sha256('Test answer')},
        **{k: evidence.payload[k].get(call['arm']) for k in ('hydration','routing','rendered')}}
    response, _ = publish_sealed_json(prefix.with_suffix('.response.json'), payload)
    return call, request, response, evidence, plan


@pytest.mark.parametrize('call_index', [0, 1])
def test_candidate_and_control_bind_selected_model_and_exact_packet(tmp_path, call_index):
    evaluation.validate_response(*journal_fixture(tmp_path, call_index))


@pytest.mark.parametrize('defect', ['model', 'prediction', 'rendered', 'request', 'output_cap'])
def test_response_cannot_be_relabelled_or_changed(tmp_path, defect):
    call, request, response, evidence, plan = journal_fixture(tmp_path)
    changed = deepcopy(response.payload)
    if defect == 'model':
        changed['measurement']['model'] = 'codex_sdk/gpt-5.6-terra'
    elif defect == 'prediction':
        changed['measurement']['prediction'] = 'A changed answer'
    elif defect == 'rendered':
        changed['rendered'] = None
    elif defect == 'request':
        changed['request_sha256'] = '0' * 64
    else:
        changed['measurement']['max_tokens'] = 512
    with pytest.raises(ValueError, match='selected model'):
        evaluation.validate_response(call, request, SimpleNamespace(payload=changed), evidence, plan)


def test_response_journal_rejects_gap_and_unacknowledged_stream(tmp_path):
    _, _, _, _, plan = journal_fixture(tmp_path, call_index=1)
    with pytest.raises(ValueError, match='gap'):
        evaluation.recorded(tmp_path, plan)
    (tmp_path / 'journal' / '000.reserved').write_text(plan.sha256, encoding='utf-8')
    with pytest.raises(ValueError, match='no implicit retry'):
        evaluation.recorded(tmp_path, plan)


def test_incomplete_answers_cannot_open_grading(tmp_path):
    _, _, _, _, plan = journal_fixture(tmp_path)
    with pytest.raises(ValueError, match='all 100 questions and controls'):
        evaluation.seal_answers(tmp_path, plan)


def test_raw_answer_model_requires_inventory_and_never_accepts_qwen():
    inventory = SimpleNamespace(payload={'model_ids': [MODEL, 'Qwen3-8B'], 'gateway': evaluation.frozen.GATEWAY})
    assert evaluation.validate_model(MODEL, inventory) == MODEL
    for model in ('Qwen3-8B', 'codex_sdk/gpt-5.6-terra', None):
        with pytest.raises(ValueError, match='admitted raw-answer model'):
            evaluation.validate_model(model, inventory)
    inventory.payload['gateway'] = 'https://different.invalid/v1'
    with pytest.raises(ValueError, match='authorized gateway'):
        evaluation.validate_model(MODEL, inventory)
