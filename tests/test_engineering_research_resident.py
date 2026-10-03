"""Resident publication agrees with disk reconstruction and survives failed sync."""
from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor
import json
import shutil
import threading

import numpy as np
import pytest

from memory_condense.application.chat_session import ChatEvent
from memory_condense.persistence import native_spine_store, native_spine_parent_store
from memory_condense.persistence import native_spine_incremental_store as incremental
from memory_condense.search.native_spine_parent_users import project_parent_users
from memory_condense.application.condenser import MemoryCondenser
from memory_condense.search.episodes.qwen_episode_signal import QwenAttentionHeadSurpriseScorer
from tools import engineering_research_memory as memory
from tools import engineering_research_resident as resident
from tests.test_attention_summary_sections import SummaryLinker
from tests.test_engineering_research_seed import fixture


STAMP = '2026-09-29T00:00:00+00:00'


def test_streamed_groups_preserve_sealed_roots_and_reopen_exactly(tmp_path,monkeypatch):
    backend,events,_,actor=setup(tmp_path,monkeypatch)
    backend.sync(events)
    backend.start_stream()
    initial=backend.last_reopen
    additions=[(ChatEvent(f'stream-u{i}','user',f'Plan deployment {i}.',STAMP),
                ChatEvent(f'stream-a{i}','assistant',f'Keep migration {i} pending.',STAMP)) for i in range(10)]
    try:
        with ThreadPoolExecutor(max_workers=3) as pool:
            ready=list(pool.map(backend.prepare_exchange,additions))
        assert backend.last_reopen==initial
        assert backend.app.transcript.get_turn('stream-u0') is None
        for i,item in enumerate(ready):
            events.extend(additions[i])
            backend.sync_prepared(events,(item,))
            if i==7:
                sealed=dict(backend._stream_groups)
        assert len(backend._stream_groups)==1
        assert backend._stream_groups==sealed
        backend.finalize_stream(events)
        assert len(backend._stream_groups)==2
        observed=backend.recall_published('Plan deployment 0?')
        assert observed['published_events']==len(events)
        assert backend.app.transcript.get_turn('stream-u0').text=='Plan deployment 0.'
    finally:
        backend.close()
    reopened=resident.ResidentNativeBackend(tmp_path,tmp_path/'live',actor)
    try:
        reopened.sync(events)
        assert reopened.recall_published('Plan deployment 0?')==observed
        reopened.start_stream()
        extra=(ChatEvent('after-reopen','user','Record a new plan.',STAMP),)
        prepared=reopened.prepare_exchange(extra)
        events.extend(extra)
        reopened.sync_prepared(events,(prepared,))
        reopened.finalize_stream(events)
        assert reopened.last_reopen['snapshot']['turn_count']==len(events)
    finally:
        reopened.close()


def test_preparation_does_not_write_raw_or_publish_and_final_sync_matches_reopen(tmp_path, monkeypatch):
    backend, events, _, actor=setup(tmp_path,monkeypatch)
    backend.sync(events)
    old=backend.last_reopen
    events.extend([ChatEvent('prepare-u','user','Keep the deployment planned.',STAMP),
                   ChatEvent('prepare-a','assistant','No deployment was executed.',STAMP)])
    try:
        backend.prepare(events)
        assert backend.app.transcript.get_turn('prepare-u') is None
        assert backend.last_reopen==old
        assert backend.recall_published('Deployment?')['snapshot']==old['snapshot']
        events.append(ChatEvent('prepare-later','user','The rollback tag is orchid-927.',STAMP))
        backend.sync(events)
        observed=backend.recall_published('Deployment?')
        assert observed['published_events']==len(events)
        assert backend.app.transcript.get_turn('prepare-u').text==events[-3].text
    finally:
        backend.close()
    reopened=resident.ResidentNativeBackend(tmp_path,tmp_path/'live',actor)
    try:
        reopened.sync(events)
        assert reopened.recall_published('Deployment?')==observed
    finally:
        reopened.close()


def test_compile_overlaps_raw_indexing_and_failed_compile_keeps_published_prefix(tmp_path, monkeypatch):
    backend, events, _, _ = setup(tmp_path, monkeypatch)
    backend.sync(events)
    old_count=len(events)
    indexed=threading.Event()
    compile_original=backend._compile
    ingest_original=backend.app.capture_native_many
    def ingest(*args,**kwargs):
        result=ingest_original(*args,**kwargs)
        indexed.set()
        return result
    def fail_after_index(live):
        assert indexed.wait(5), 'Raw indexing must overlap compilation'
        raise RuntimeError('simulated compiler failure')
    monkeypatch.setattr(backend.app,'capture_native_many',ingest)
    monkeypatch.setattr(backend,'_compile',fail_after_index)
    events.append(ChatEvent('overlap-new','user','Deployment remains a plan.',STAMP))
    try:
        with pytest.raises(RuntimeError,match='compiler failure'):
            backend.sync(events)
        assert backend.app.transcript.get_turn('overlap-new') is not None
        assert backend.recall_published('Deployment?')['published_events']==old_count
        monkeypatch.setattr(backend,'_compile',compile_original)
        backend.sync(events)
        assert backend.recall_published('Deployment?')['published_events']==old_count+1
        assert backend.app.pending_ingest_count()==0
    finally:
        backend.close()


def test_published_recall_during_raw_append_and_atomic_switch(tmp_path, monkeypatch):
    backend, events, _, _ = setup(tmp_path, monkeypatch)
    backend.sync(events)
    old_count = len(events)
    entered, release = threading.Event(), threading.Event()
    publish = backend.app.install_native_spine_incremental
    def held_publish(*args, **kwargs):
        entered.set()
        assert release.wait(10)
        return publish(*args, **kwargs)
    monkeypatch.setattr(backend.app, 'install_native_spine_incremental', held_publish)
    events.append(ChatEvent('latest', 'user', 'New fact: cluster cobalt-731.', STAMP))
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            writer = pool.submit(backend.sync, events)
            try:
                assert entered.wait(5)
                assert backend.app.transcript.get_turn('latest') is not None
                before = pool.submit(backend.recall_published, 'New fact?').result(timeout=3)
                assert not writer.done()
                assert before['published_events'] == old_count
                assert all(r['span']['turn_id'] != 'latest' for r in before['references'])
            finally:
                release.set()
            writer.result(timeout=5)
        after = backend.recall_published('New fact?')
        assert after['published_events'] == old_count+1
        assert after['snapshot'] != before['snapshot']
        assert 'latest' in {r['span']['turn_id'] for r in after['references']}
    finally:
        backend.close()


def test_published_hydration_uses_authenticated_snapshot_without_disk_reads(tmp_path, monkeypatch):
    backend, events, _, _ = setup(tmp_path, monkeypatch)
    backend.sync(events)
    try:
        expected=backend.recall_published('Deployment?')
        def unexpected(*args, **kwargs):
            raise AssertionError('Published recall must not access the mutable raw store')
        monkeypatch.setattr(backend.app.transcript, 'get_turn', unexpected)
        monkeypatch.setattr(resident.sqlite3, 'connect', unexpected)
        assert backend.recall_published('Deployment?')==expected
        assert expected['references']
        with pytest.raises(TypeError):
            backend._published[3]['foreign']=None
    finally:
        backend.close()


def test_native_continuation_never_calls_raw_embedder_or_reloads_all_turns(tmp_path, monkeypatch):
    from memory_condense.application.chat_session import RecallPacket
    backend, events, _, actor = setup(tmp_path, monkeypatch)
    backend.sync(events)
    try:
        packet = backend.recall_published('Deployment?')
        def forbidden(*args, **kwargs):
            pytest.fail('Native continuation must not maintain the legacy raw index or reload its prefix')
        monkeypatch.setattr(backend.encoder, 'embed_chunks', forbidden)
        monkeypatch.setattr(backend.app, 'ingest_many', forbidden)
        monkeypatch.setattr(backend.app.transcript, 'get_all', forbidden)
        additions = (ChatEvent('native-u', 'user', 'Continue the deployment plan.', STAMP),
            ChatEvent('_chat:recall:native-p', 'system', packet['text'], STAMP,
                {'_chat': dict(kind='recall',packet_id='native-p',references=packet['references'])}),
            ChatEvent('native-a', 'assistant', 'The deployment remains planned.', STAMP))
        backend.start_stream()
        prepared = backend.prepare_exchange(additions)
        events.extend(additions)
        backend.sync_prepared(events, (prepared,))
        recall = RecallPacket('native-p', 'Deployment?', packet['text'], tuple(packet['references']), 'native-u')
        first = backend.learn(recall, access_event_id='native-feedback')
        backend.learn(recall, access_event_id='native-feedback')
        assert first['learning'] is not None
        rows = backend.app._db.execute("SELECT embedding FROM chunks WHERE turn_id IN ('native-u','native-a')").fetchall()
        assert rows and all(row[0] is None for row in rows)
        expected = backend.last_reopen['snapshot']
    finally:
        backend.close()
    reopened = resident.ResidentNativeBackend(tmp_path,tmp_path/'live',actor)
    try:
        reopened.sync(events)
        assert reopened.last_reopen['snapshot'] == expected
    finally:
        reopened.close()


def setup(tmp_path, monkeypatch):
    binding, rows, encoder = fixture(tmp_path)
    calls = []
    encoder.embed_queries = lambda texts: np.asarray([[1., 0.] for _ in texts], dtype=np.float32)
    encoder.close = lambda: None
    encoder.park = lambda: None
    monkeypatch.setattr(resident, 'StagedEmbedding', lambda **kwargs: encoder)
    monkeypatch.setattr(memory, 'StagedEmbedding', lambda **kwargs: encoder)

    class Gateway:
        def __init__(self, root):
            pass
        def call(self, kind, messages, **kwargs):
            if kind == 'merge':
                return dict(request_sha256='0'*64, content=json.dumps({'summary': 'New deployment context.'}))
            assert kind == 'raw'
            fragments = json.loads(messages[1]['content'])['fragments']
            calls.extend(f['fragment'] for f in fragments)
            return dict(request_sha256='0'*64, content=json.dumps({'atoms': [
                dict(label=f['label'], summary='New deployment fact.', support=[f['fragment']]) for f in fragments]}))

    class Attention:
        max_spans, span_token_cap = 8, 128
        def __init__(self, *args):
            linker = SummaryLinker()
            linker.max_candidates = 8
            self.delegate = QwenAttentionHeadSurpriseScorer(linker, max_spans=8, span_token_cap=128)
        def score_sequence(self, texts):
            return self.delegate.score_sequence(texts)
        def park(self):
            pass

    monkeypatch.setattr(memory, 'Gateway', Gateway)
    monkeypatch.setattr(memory, 'LocalAttention', Attention)
    monkeypatch.setattr(resident, 'ResidentAttention', Attention)
    events = [ChatEvent(r['turn_id'], r['role'], r['text'], r['created_at'], {'source_id': r['source_id']}) for r in rows]
    actor = dict(case_id='test', source=dict(family='live', export_timestamp=STAMP), native_seed=binding)
    shutil.copytree(binding['directory'], tmp_path/'live/store/memory')
    backend = resident.ResidentNativeBackend(tmp_path, tmp_path/'live', actor)
    return backend, events, calls, actor


def test_resident_fast_forward_matches_clean_compile_and_cold_reopen(tmp_path, monkeypatch):
    backend, events, calls, actor = setup(tmp_path, monkeypatch)
    backend.sync(events)
    app, encoder = backend.app, backend.encoder
    original_load = native_spine_store.load
    original_parent_load = native_spine_parent_store.load
    monkeypatch.setattr(native_spine_store, 'load', lambda *a, **k: pytest.fail('Warm sync must not reconstruct disk indexes'))
    monkeypatch.setattr(native_spine_parent_store, 'load', lambda *a, **k: pytest.fail('Warm sync must not reconstruct parents'))
    additions = [ChatEvent('live-input', 'user', 'Deploy to cobalt-731.', STAMP),
                 ChatEvent('live-output', 'assistant', 'Migration ID is migration-482.', STAMP),
                 ChatEvent('live-tool', 'tool', 'Schema checks passed: 3.', STAMP)]
    for event in additions:
        events.append(event)
        backend.sync(events)
        assert backend.app is app and backend.encoder is encoder
        assert app.native_spine_receipt()['turn_count'] == len(events)
    backend.sync(events)
    assert len(backend.timings) == 4  # Cold admission and three real refreshes.
    before = backend.recall('Latest deployment?')
    native, parents = app.native_spine_receipt(), app.native_parent_user_receipt()
    with pytest.raises(ValueError, match='fast-forward'):
        backend.sync(events[:-1])
    with pytest.raises(ValueError, match='fast-forward'):
        backend.sync([replace(events[0], text='rewritten'), *events[1:]])
    backend.close()
    monkeypatch.setattr(native_spine_store, 'load', original_load)
    monkeypatch.setattr(native_spine_parent_store, 'load', original_parent_load)
    reopened = resident.ResidentNativeBackend(tmp_path, tmp_path/'live', actor)
    try:
        reopened.sync(events)
        after = reopened.recall('Latest deployment?')
        assert before == after
    finally:
        reopened.close()
    clean = tmp_path/'clean'
    clean.mkdir()
    memory.install(tmp_path, dict(rows=[e.row('live') for e in events], source_id='live',
        storage_timestamp=STAMP, case_root=str(clean), output=str(clean/'installed.json'),
        scope='control', native_seed=actor['native_seed']))
    report = memory.read(clean/'installed.json')
    assert report['snapshot'] == native
    # The new parent format uses stable root IDs. Compare against a fresh
    # section-store build of the independently compiled clean hierarchy.
    with MemoryCondenser(clean/'memory', embedder=encoder, auto_extract=False) as app:
        base = app._load_native_spine()[1]
        projection = project_parent_users(base.hierarchy, stable_ids=True)
        prior = app._load_native_spine()[2].router.parent_semantic
        by_root = {json.loads(s.summarizer_identity)['original_root_sha256']: v
                   for s, v in zip(prior.sections, prior._dense._matrix, strict=True)}
        seed = resident.load_seed(actor['native_seed'], [e.row('live') for e in events])
        parent_matrix = np.asarray([seed.vectors[s.section_id][1] if s.section_id in seed.vectors else
                                    by_root[json.loads(s.summarizer_identity)['original_root_sha256']]
                                    for s in projection.sections], dtype=np.float32)
        app.install_native_spine_incremental(base.semantic.hierarchy, base.hierarchy,
            base.semantic._dense._matrix, embedding_identity=base.semantic.embedding_identity,
            projection=projection, parent_matrix=parent_matrix)
        assert app.native_parent_user_receipt() == parents
    assert calls == []  # Short entries keep exact text without raw generation.


def test_failed_publication_never_acknowledges_or_recalls_stale_snapshot(tmp_path, monkeypatch):
    backend, events, calls, actor = setup(tmp_path, monkeypatch)
    backend.sync(events)
    previous = backend.last_reopen
    paths = [backend.app.database_path.parent / name for name in ('native-spine.sqlite', native_spine_parent_store.FILENAME)]
    original_bytes = [p.read_bytes() for p in paths]
    events.append(ChatEvent('new', 'user', 'New fact: cluster cobalt-731.', STAMP))
    publish = incremental.publish
    def fail(*args, **kwargs):
        def abort():
            raise RuntimeError('simulated publication failure')
        kwargs['before_commit'] = abort
        return publish(*args, **kwargs)
    monkeypatch.setattr(incremental, 'publish', fail)
    try:
        with pytest.raises(RuntimeError, match='publication failure'):
            backend.sync(events)
        log = [json.loads(line) for line in (backend.arm_root/'ingestion-timings.jsonl').read_text().splitlines()]
        assert log[-1]['operation'] == 'sync_failed' and log[-1]['phase'] == 'publication'
        assert 'raw_ingest_s' in log[-1]
        assert backend.last_reopen == previous
        assert [p.read_bytes() for p in paths] == original_bytes
        with pytest.raises(ValueError, match='raw transcript advanced'):
            backend.recall('New fact?')
        # Batched readers explicitly use the prior prefix, never the partially
        # ingested raw suffix. Eager callers retain their freshness barrier.
        prior = backend.recall_published('New fact?')
        assert prior['published_events'] == len(events)-1
        assert prior['snapshot'] == previous['snapshot']
        assert all(r['span']['turn_id'] != 'new' for r in prior['references'])
        monkeypatch.setattr(incremental, 'publish', publish)
        # Restart after the raw write but before native publication. The old
        # pair is valid for its prefix, and the journal still owns the suffix.
        backend.close()
        backend = resident.ResidentNativeBackend(tmp_path, tmp_path/'live', actor)
        backend.sync(events)
        assert backend.app.native_spine_receipt()['turn_count'] == len(events)
        assert calls == []
    finally:
        backend.close()


def test_resident_cache_preserves_old_method_and_binds_selected_placement(tmp_path, monkeypatch):
    backend, events, _, actor = setup(tmp_path, monkeypatch)
    cache = tmp_path/'cache/attention'
    old = resident.cache_method(cache)
    old_bytes = (cache/'method.json').read_bytes()
    base = resident.ResidentAttention

    class PreparedAttention(base):
        placement = True
        def prepare(self):
            self.host_embeddings = self.placement

    monkeypatch.setattr(resident, 'ResidentAttention', PreparedAttention)
    try:
        backend.sync(events)
        first = backend.attention.preflight
        assert first.payload['host_embeddings'] is True
    finally:
        backend.close()
    PreparedAttention.placement = False
    backend = resident.ResidentNativeBackend(tmp_path, tmp_path/'live', actor)
    try:
        backend.sync(events)
        second = backend.attention.preflight
        assert second.payload['host_embeddings'] is False
        assert len({old.sha256, first.sha256, second.sha256}) == 3
        assert (cache/'method.json').read_bytes() == old_bytes
        assert len(list(cache.glob('method-*.json'))) == 3
    finally:
        backend.close()
