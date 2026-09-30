"""Full native publication after append agrees with clean-store ingestion."""
from dataclasses import asdict
import hashlib
import json
import shutil

import numpy as np
import pytest

from memory_condense.application.condenser import MemoryCondenser
from memory_condense.persistence.db import Database
from memory_condense.persistence.transcript_store import TranscriptStore
from memory_condense.search.episodes.qwen_episode_signal import QwenAttentionHeadSurpriseScorer
from tools import engineering_research_memory as memory
from tools.engineering_research_seed import load_seed
from tests.test_attention_summary_sections import SummaryLinker
from tests.test_native_spine_default_policy import persisted


def fixture(tmp_path):
    directory = tmp_path/'seed'
    encoder, parents = persisted(directory)
    with MemoryCondenser(directory, embedder=encoder, auto_extract=False, read_only=True) as app:
        native = app.native_spine_receipt()
        rows = [dict(turn_id=t.turn_id, source_id=t.source_id, role=t.role, text=t.text,
                     created_at=t.created_at.isoformat()) for t in app.transcript.get_all()]
    receipt = tmp_path/'seed.json'
    memory.save(receipt, dict(snapshot=native, parent_snapshot=parents,
        application_files={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.iterdir() if p.is_file()}))
    return dict(directory=str(directory), receipt=str(receipt)), rows, encoder


def test_seed_rejects_changed_provenance_and_reuses_exact_descriptors(tmp_path):
    binding, rows, _ = fixture(tmp_path)
    seed = load_seed(binding, rows)
    assert len(seed.turns) == len(rows)
    assert {s.spans[0].turn_id for s in seed.atoms} == {r['turn_id'] for r in rows}
    with pytest.raises(ValueError, match='provenance'):
        load_seed(binding, [dict(rows[0], text='Changed'), *rows[1:]])
    with pytest.raises(ValueError, match='complete native seed'):
        load_seed(binding, rows[:-1])
    with pytest.raises(ValueError, match='live source identity'):
        load_seed(binding, [*rows, dict(rows[-1], turn_id='new')])


def test_native_append_and_clean_ingestion_publish_identical_combined_index(tmp_path, monkeypatch):
    binding, rows, encoder = fixture(tmp_path)
    encoder.embed_queries = lambda texts: np.asarray([[1., 0.] for _ in texts], dtype=np.float32)
    encoder.close = lambda: None
    monkeypatch.setattr(memory, 'StagedEmbedding', lambda **kwargs: encoder)
    raw_calls = []
    class Gateway:
        def __init__(self, root):
            pass
        def call(self, kind, messages, **kwargs):
            assert kind == 'raw'
            fragments = json.loads(messages[1]['content'])['fragments']
            raw_calls.extend(f['fragment'] for f in fragments)
            return dict(request_sha256='0'*64, content=json.dumps({'atoms': [
                dict(label=f['label'], summary='New deployment fact.', support=[f['fragment']]) for f in fragments]}))
    monkeypatch.setattr(memory, 'Gateway', Gateway)
    class Attention:
        max_spans, span_token_cap = 8, 128
        def __init__(self, *args):
            linker = SummaryLinker()
            linker.max_candidates = 8
            self.delegate = QwenAttentionHeadSurpriseScorer(linker, max_spans=8, span_token_cap=128)
        def score_sequence(self, texts):
            return self.delegate.score_sequence(texts)
    monkeypatch.setattr(memory, 'LocalAttention', Attention)
    stamp = '2026-09-29T00:00:00+00:00'
    rows += [dict(turn_id='live-input', source_id='live', role='user', text='Deploy to cobalt-731.', created_at=stamp),
             dict(turn_id='live-output', source_id='live', role='assistant', text='Migration ID is migration-482.', created_at=stamp)]
    shutil.copytree(binding['directory'], tmp_path/'append/memory')
    snapshots = []
    for mode in ('append', 'clean'):
        folder = tmp_path/mode
        folder.mkdir(exist_ok=True)
        request = dict(rows=rows, source_id='live', storage_timestamp=stamp, case_root=str(folder),
                       output=str(folder/'installed.json'), scope=mode, native_seed=binding)
        memory.install(tmp_path, request)
        report = memory.read(folder/'installed.json')
        assert report['new_turns'] == (2 if mode == 'append' else len(rows))
        with MemoryCondenser(folder/'memory', embedder=encoder, auto_extract=False, read_only=True) as app:
            assert app.native_spine_receipt() == report['snapshot']
            assert app.native_parent_user_receipt() == report['parent_snapshot']
            assert [(t.turn_id, t.text) for t in app.transcript.get_all()] == [(r['turn_id'], r['text']) for r in rows]
        snapshots.append((report['snapshot'], report['parent_snapshot']))
    assert snapshots[0] == snapshots[1]
    assert raw_calls == []  # Both new turns qualify for exact short-text reuse.
