from pathlib import Path

import pytest

from tools import native_spine_build_replay as replay
from tools import native_spine_build_replay_memory as memory


def test_memory_prefix_contains_only_completed_replay_responses(tmp_path):
    seed = [{'role': 'user', 'text': 'Earlier request', 'turn_id': 'seed-0'}]
    prompts = [{'text': f'Original prompt {i}', 'original_turn_index': i + 100} for i in range(8)]
    replay.publish(tmp_path / 'replay-plan.json', {'implementation_sha256': replay.evaluation.digest(replay.__file__),
        'seed': seed, 'prompts': prompts})
    replay.publish(tmp_path / 'steps/00/complete.json', {'message': 'New generated answer zero'})
    replay.publish(tmp_path / 'steps/01/complete.json', {'message': 'Future answer must not enter step one'})
    _, rows, _ = replay.state(tmp_path, 1)
    assert [r['text'] for r in rows] == ['Earlier request', 'Original prompt 0', 'New generated answer zero']
    assert all('Future' not in r['text'] for r in rows)


@pytest.mark.parametrize('path', ['../outside.py', r'C:\outside.py', '.git/config', '.env', 'src/../../outside.py'])
def test_model_file_access_cannot_escape_checkout(tmp_path, path):
    (tmp_path / 'workspace').mkdir()
    with pytest.raises(ValueError):
        replay.safe_path(tmp_path, path)


@pytest.mark.parametrize('raw', ['raw conversation', {'role': 'user', 'text': 'raw conversation'}])
def test_qwen_merge_rejects_untyped_raw_inputs_before_any_provider(tmp_path, raw):
    with pytest.raises(TypeError):
        memory.JournaledQwen(tmp_path)(raw)


def test_summary_cache_does_not_reuse_an_assistant_fragment_as_user():
    user, = memory.fragments_for([{'role': 'user', 'text': 'Use turns only.'}])
    assistant, = memory.fragments_for([{'role': 'assistant', 'text': 'Use turns only.'}])
    assert memory.fragment_key(user) != memory.fragment_key(assistant)
