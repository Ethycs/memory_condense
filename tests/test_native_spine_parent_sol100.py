from copy import deepcopy
from types import SimpleNamespace

import pytest

from tests.test_native_spine_application_reader100 import instructions
from tests.test_native_spine_parent_users100 import candidate
from tools import evaluate_native_spine_parent_sol100 as evaluation


@pytest.mark.parametrize('ordinal', [0, 1])
def test_model_comparison_preserves_complete_prompt_and_packet(tmp_path, ordinal):
    case, original, calls = candidate(tmp_path, ordinal)
    same = SimpleNamespace(payload=deepcopy(original.payload))
    evaluation.validate_pair(calls, case, original, instructions())
    evaluation.validate_preservation(same, original)
    assert evaluation.MODEL == 'codex_sdk/gpt-5.6-sol'
    assert evaluation.seal_answers is evaluation.model_io.seal_answers


@pytest.mark.parametrize('field', ['messages', 'routing', 'hydration', 'rendered'])
def test_model_change_cannot_also_change_reader_or_evidence(tmp_path, field):
    _, original, _ = candidate(tmp_path)
    changed = deepcopy(original.payload)
    if field == 'messages':
        for messages in changed[field].values():
            messages[0]['content'] += '\nChanged reader instructions'
    else:
        changed[field]['parent_context']['unapproved_change'] = True
    with pytest.raises(ValueError, match='baseline evidence or prompt'):
        evaluation.validate_preservation(SimpleNamespace(payload=changed), original)
