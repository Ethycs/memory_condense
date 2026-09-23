import pytest

from tests.test_native_spine_exchanges import prepared
from tests.test_pending_native_spine_exchanges import stopped
from tools import compile_pending_native_spine_exchanges as pending
from tools import prepare_native_spine_frozen_corpus as preparation
from tools.matched_eval.artifacts import publish_sealed_json
from tools.prepare_native_spine_design_slice import bound


def ready(prepared):
    root, _, backend, report, cache = stopped(prepared)
    plan = pending.prepare(root/'pending', report.path, cache.path)
    result = pending.execute(root/'pending', backend, 128)
    row = result.payload['compiled_bodies'][0]
    body = {'body_sha256': row['body_sha256'], 'atoms': row['atoms'], 'exchange': row['artifact'],
        'exchange_preflight_sha256': plan.sha256, 'parent': None, 'parent_preflight_sha256': None}
    # A completed parent deliberately has no exchange files in this fixture.
    # It must never enter attention preparation.
    scope, _ = publish_sealed_json(root/'scope.json', {
        'format': preparation.FORMAT, 'implementation_sha256': preparation.digest(preparation.__file__),
        'body_count': 2, 'missing_parent_count': 1,
        'bodies': [body, {'body_sha256': 'f'*64, 'parent': {'completed': True}}]})
    return root, scope, body


def test_only_missing_parent_attention_is_prepared_and_resume_does_not_reload_bodies(prepared, monkeypatch):
    root, scope, _ = ready(prepared)
    first = preparation.prepare_attention(scope.path, root/'attention', root/'attention-cache')
    assert len(first.payload['bodies']) == 1
    assert first.payload['raw_inputs_to_qwen'] is False
    assert first.payload['query_or_gold_inputs'] is False
    assert first.payload['existing_parent_bodies_recompiled'] == 0
    assert 'RAW_CANARY' not in str(first.payload['jobs'])
    assert '2026-01-02' not in str(first.payload['jobs'])
    assert first.payload['jobs']
    def forbidden(*args, **kwargs):
        raise AssertionError('resumption must reuse completed preparation checkpoints')
    monkeypatch.setattr(preparation, 'load_body', forbidden)
    assert preparation.prepare_attention(scope.path, root/'attention', root/'attention-cache').sha256 == first.sha256


def test_changed_exchange_rejects_before_attention_preflight(prepared):
    root, scope, body = ready(prepared)
    artifact = bound(body['exchange'])
    artifact.path.write_bytes(artifact.path.read_bytes()+b' ')
    with pytest.raises(ValueError):
        preparation.prepare_attention(scope.path, root/'attention', root/'attention-cache')
    assert not (root/'attention/preflight.json').exists()


def test_changed_preparation_checkpoint_cannot_be_resumed(prepared):
    root, scope, _ = ready(prepared)
    plan = preparation.prepare_attention(scope.path, root/'attention', root/'attention-cache')
    checkpoint = bound(plan.payload['preparation_checkpoints'][0])
    checkpoint.path.write_bytes(checkpoint.path.read_bytes()+b' ')
    with pytest.raises(ValueError):
        preparation.prepare_attention(scope.path, root/'attention', root/'attention-cache')
