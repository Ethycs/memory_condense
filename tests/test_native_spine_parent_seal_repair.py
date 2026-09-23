import pytest

from tools import repair_native_spine_parent_seal as repair
from tools.matched_eval.artifacts import publish_sealed_json, read_sealed_json


def test_exact_reconstruction_restores_only_missing_checksum(tmp_path):
    rebuilt, _ = publish_sealed_json(tmp_path/'rebuilt.json', {'body': 'exact'})
    target = tmp_path/'original.json'
    target.write_bytes(rebuilt.path.read_bytes())
    before = target.stat().st_mtime_ns
    restored = repair.restore_missing_seal(target, rebuilt.path)
    assert restored.sha256 == rebuilt.sha256
    assert target.stat().st_mtime_ns == before
    assert read_sealed_json(target).payload == {'body': 'exact'}


def test_changed_body_cannot_be_resealed(tmp_path):
    rebuilt, _ = publish_sealed_json(tmp_path/'rebuilt.json', {'body': 'exact'})
    target = tmp_path/'original.json'
    target.write_bytes(b'{"body":"changed"}\n')
    with pytest.raises(ValueError, match='differs from exact'):
        repair.restore_missing_seal(target, rebuilt.path)
    assert not target.with_suffix('.json.sha256').exists()
    assert target.read_bytes() == b'{"body":"changed"}\n'


def test_existing_invalid_checksum_is_preserved_and_rejected(tmp_path):
    rebuilt, _ = publish_sealed_json(tmp_path/'rebuilt.json', {'body': 'exact'})
    target = tmp_path/'original.json'
    target.write_bytes(rebuilt.path.read_bytes())
    sidecar = target.with_suffix('.json.sha256')
    sidecar.write_bytes(b'invalid')
    with pytest.raises(ValueError, match='no checksum'):
        repair.restore_missing_seal(target, rebuilt.path)
    assert sidecar.read_bytes() == b'invalid'


def test_reconstruction_cannot_issue_a_missing_model_job():
    with pytest.raises(ValueError, match='saved summaries are insufficient'):
        repair.ReplayOnly(object()).resolve({'missing': object()})
