import pytest

from memory_condense.domain._discourse_identity import identity_sha256
from memory_condense.search.episodes.qwen_episode_signal import QwenAttentionHeadSurpriseScorer
from memory_condense.search.section_routing import SectionSummaryIndex
from tests.test_attention_summary_sections import SummaryLinker
from tests.test_native_spine_exchanges import Backend
from tests.test_native_spine_hierarchy import group
from tools import compile_remaining_native_spine_hierarchies as compiler
from tools.matched_eval.artifacts import publish_sealed_json, read_sealed_json


def fixture(root, monkeypatch):
    ready, missing = identity_sha256('ready'), identity_sha256('missing')
    groups = {ready: group('ready', 1), missing: group('missing', 12)}
    rows = [{'body_sha256': sha, 'exchange': {'sha256': g[0].sha256},
        'atoms': {'sha256': g[1].sha256}} for sha, g in groups.items()]
    plan, _ = publish_sealed_json(root/'preflight.json', {'fixture': True})
    backend = Backend()
    scorer = QwenAttentionHeadSurpriseScorer(SummaryLinker(), max_spans=8, span_token_cap=128)
    monkeypatch.setattr(compiler.preparation, 'load_body', lambda row: groups[row['body_sha256']])
    return ready, missing, groups, rows, plan, backend, scorer


def test_completed_parent_batch_is_reused_without_recompilation_or_generation(tmp_path, monkeypatch):
    ready, missing, groups, rows, plan, backend, scorer = fixture(tmp_path, monkeypatch)
    first = compiler.bounded.BoundedJournal(tmp_path, plan, backend, 0)
    partial = compiler.compile_batches(tmp_path, plan, rows, scorer, first, batch_size=1)
    assert set(partial) == {ready} and backend.calls == 0
    assert (tmp_path/'batches/0000.json').exists()
    assert not (tmp_path/'batches/0001.json').exists()
    def only_pending(row):
        assert row['body_sha256'] == missing, 'completed bodies must not be reconstructed'
        return groups[missing]
    monkeypatch.setattr(compiler.preparation, 'load_body', only_pending)
    resumed = compiler.bounded.BoundedJournal(tmp_path, plan, backend, 128)
    resumed.replay()
    done = compiler.compile_batches(tmp_path, plan, rows, scorer, resumed, batch_size=1)
    assert set(done) == {ready, missing} and backend.calls > 0
    for sha, row in done.items():
        body = read_sealed_json(tmp_path/row['path'])
        tree = SectionSummaryIndex.from_json(body.payload['index_json'])
        atoms = SectionSummaryIndex.from_json(body.payload['atomic_index_json'])
        root = next(s for s in tree.sections if s.section_id == body.payload['root_section_ids'][0])
        assert atoms.sections == SectionSummaryIndex(groups[sha][2]).sections
        assert root.spans == tuple(s for atom in groups[sha][2] for s in atom.spans)
    def forbidden(*args, **kwargs):
        raise AssertionError('completed parent batches cannot compile or generate')
    monkeypatch.setattr(compiler.preparation, 'load_body', forbidden)
    monkeypatch.setattr(backend, 'generate', forbidden)
    replay = compiler.bounded.BoundedJournal(tmp_path, plan, backend, 0)
    replay.replay()
    assert compiler.compile_batches(tmp_path, plan, rows, scorer, replay, batch_size=1) == done


def test_changed_completed_parent_is_rejected_on_batch_reuse(tmp_path, monkeypatch):
    ready, _, _, rows, plan, backend, scorer = fixture(tmp_path, monkeypatch)
    journal = compiler.bounded.BoundedJournal(tmp_path, plan, backend, 0)
    result = compiler.compile_batches(tmp_path, plan, rows, scorer, journal, batch_size=1)
    path = tmp_path/result[ready]['path']
    path.write_bytes(path.read_bytes()+b' ')
    with pytest.raises(ValueError):
        compiler.compile_batches(tmp_path, plan, rows, scorer, journal, batch_size=1)
    assert backend.calls == 0


def test_parent_checkpoint_cannot_be_applied_to_different_atomic_inputs(tmp_path, monkeypatch):
    _, _, _, rows, plan, backend, scorer = fixture(tmp_path, monkeypatch)
    journal = compiler.bounded.BoundedJournal(tmp_path, plan, backend, 0)
    compiler.compile_batches(tmp_path, plan, rows, scorer, journal, batch_size=1)
    changed = [dict(rows[0], atoms={'sha256': 'foreign-atoms'}), rows[1]]
    with pytest.raises(ValueError, match='source population'):
        compiler.compile_batches(tmp_path, plan, changed, scorer, journal, batch_size=1)
    assert backend.calls == 0


def test_interrupted_parent_generation_is_not_retried_on_resume(tmp_path, monkeypatch):
    _, _, _, rows, plan, backend, scorer = fixture(tmp_path, monkeypatch)
    backend.fail = True
    journal = compiler.bounded.BoundedJournal(tmp_path, plan, backend, 128)
    with pytest.raises(RuntimeError, match='simulated stopped'):
        compiler.compile_batches(tmp_path, plan, rows, scorer, journal, batch_size=1)
    calls = backend.calls
    with pytest.raises(ValueError, match='refusing an implicit retry'):
        compiler.bounded.BoundedJournal(tmp_path, plan, backend, 128).replay()
    assert backend.calls == calls
