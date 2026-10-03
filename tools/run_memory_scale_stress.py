"""One combined memory per scale, 100 questions per nominal million tokens.

Reuse authenticated source compilation, never old answers. All sources compete
in one index. Original dates, questions, raw spans and source IDs are preserved.
This measures cached ingestion and live IO, not cold source summarization.
"""
from __future__ import annotations

import argparse
from contextlib import closing
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import time

import numpy as np

from memory_condense.application.condenser import MemoryCondenser
from memory_condense.domain._discourse_identity import quote_sha256
from memory_condense.domain._tokenizer import count_tokens
from memory_condense.persistence.db import Database
from memory_condense.persistence.transcript_store import TranscriptStore
from memory_condense.persistence import native_spine_store, native_spine_parent_store
from memory_condense.search.native_spine_parent_users import project_parent_users
from memory_condense.search.section_routing import SectionSummaryIndex
from tools.engineering_research_gateway import emit, read, save
from tools.run_inline_validation_battery import SOURCE, command


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def align_atomic_vectors(index, sections, matrix):
    """Bind cached rows to section identities after the union index sorts them."""
    matrix = np.asarray(matrix)
    if matrix.dtype != np.float32 or matrix.ndim != 2 or len(sections) != len(matrix):
        raise ValueError('Invalid source section vector matrix')
    rows = {}
    for section, vector in zip(sections, matrix, strict=True):
        if section.section_id in rows:
            raise ValueError('Duplicate source section identity')
        rows[section.section_id] = (section.receipt_sha256, vector)
    if set(rows) != {s.section_id for s in index.sections}:
        raise ValueError('Combined section population changed')
    for section in index.sections:
        if rows[section.section_id][0] != section.receipt_sha256:
            raise ValueError('Combined source section changed')
    return np.asarray([rows[s.section_id][1] for s in index.sections], dtype=np.float32)


class CachedSourceEmbedding:
    """Exact source chunk cache; a cache miss fails instead of inventing a vector."""
    dim = 1024

    def __init__(self):
        self.chunks = {}
        self.hits = 0

    def add(self, directory):
        with closing(sqlite3.connect((directory/'memory.db').as_uri()+'?mode=ro', uri=True)) as db:
            for tid, start, end, text, vector, lexical in db.execute(
                    'SELECT turn_id,start_char,end_char,text,embedding,lexical_weights FROM chunks'):
                values = np.frombuffer(vector, dtype=np.float32)
                if values.shape != (self.dim,) or not np.isfinite(values).all():
                    raise ValueError('Invalid cached source chunk embedding')
                key = (tid, start, end, quote_sha256(text))
                if key in self.chunks:
                    raise ValueError('Duplicate source occurrence')
                self.chunks[key] = (values, json.loads(lexical) if lexical else None)

    def embed_chunks(self, chunks):
        result = []
        for chunk in chunks:
            key = (chunk.turn_id, chunk.start_char, chunk.end_char, quote_sha256(chunk.text))
            vector, lexical = self.chunks[key]
            result.append(chunk.model_copy(update=dict(embedding=vector.tolist(), lexical_weights=lexical)))
            self.hits += 1
        return result

    def embed_query(self, _query):
        raise RuntimeError('Source-cache ingestion must not issue a query')


def build(root, million):
    """Materialize one larger application via real ingest and native install APIs."""
    if root.exists():
        raise ValueError('Source assembly requires a fresh directory')
    root.mkdir(parents=True)
    started = time.perf_counter()
    sources = [SOURCE.resolve()/f'history-{i:02}' for i in range(1, million+1)]
    turns, atoms, hierarchy, matrices, parents = [], [], [], [], {}
    encoder = CachedSourceEmbedding()
    bindings, identity = [], None
    for source in sources:
        receipt = read(source/'ingest-complete.json')
        for name, sha in receipt['application_files'].items():
            if digest(source/'application'/name) != sha:
                raise ValueError('Original source store changed')
        with Database(source/'application/memory.db', read_only=True) as db:
            part_turns = TranscriptStore(db).get_all()
        snapshot = native_spine_store.load(source/'application/native-spine.sqlite', turns=part_turns)
        parent, parent_receipt = native_spine_parent_store.load(
            source/'application'/native_spine_parent_store.FILENAME,
            hierarchy=snapshot.hierarchy, native_receipt=snapshot.receipt)
        if snapshot.receipt != receipt['snapshot'] or parent_receipt != receipt['parent_snapshot']:
            raise ValueError('Source receipt mismatch')
        if identity is not None and snapshot.semantic.embedding_identity != identity:
            raise ValueError('Cannot combine different summary embedding identities')
        identity = snapshot.semantic.embedding_identity
        turns.extend(part_turns)
        atoms.extend(snapshot.semantic.sections)
        hierarchy.extend(snapshot.hierarchy.sections)
        matrices.append(snapshot.semantic._dense._matrix)
        for section, vector in zip(parent.sections, parent._dense._matrix, strict=True):
            root_sha = json.loads(section.summarizer_identity)['original_root_sha256']
            if root_sha in parents:
                raise ValueError('Duplicate parent occurrence')
            parents[root_sha] = (section.summary, vector)
        encoder.add(source/'application')
        bindings.append(dict(source=str(source), receipt_sha256=digest(source/'ingest-complete.json'),
                             snapshot=snapshot.receipt))
        emit(phase='source_loaded', sources=len(bindings), target_sources=million, turns=len(turns))
    if len({t.turn_id for t in turns}) != len(turns):
        raise ValueError('Combined history contains duplicate turn IDs')
    # Chronological order is independent of questions and references.
    turns.sort(key=lambda t: (t.created_at, t.source_id))
    atomic_index, hierarchy_index = SectionSummaryIndex(atoms), SectionSummaryIndex(hierarchy)
    projection = project_parent_users(hierarchy_index)
    parent_vectors = []
    for section in projection.sections:
        summary, vector = parents[json.loads(section.summarizer_identity)['original_root_sha256']]
        if summary != section.summary:
            raise ValueError('Parent summary changed during assembly')
        parent_vectors.append(vector)
    save(root/'ingest-plan.json', dict(nominal_tokens=million*1_000_000,
        sources=bindings, entrypoint='MemoryCondenser.ingest_many',
        compiled_summary_cache_reused=True, exact_chunk_vector_cache_reused=True,
        questions_or_references_loaded=False, source_filter_at_recall=False))
    with MemoryCondenser(root/'application', embedder=encoder, auto_extract=False) as app:
        for offset in range(0, len(turns), 128):
            app.ingest_many([(t.role, t.text, t.source_id, t.created_at, t.turn_id)
                             for t in turns[offset:offset+128]])
            if offset % 1024 == 0:
                emit(phase='ingesting', turns=min(offset+128,len(turns)), total=len(turns))
        if app.pending_ingest_count() or app.transcript.count() != len(turns):
            raise ValueError('Combined ingestion incomplete')
        native = app.install_native_spine(atomic_index, hierarchy_index,
            align_atomic_vectors(atomic_index, atoms, np.concatenate(matrices)),
            embedding_identity=identity)
    parent_receipt = native_spine_parent_store.publish(root/'application'/native_spine_parent_store.FILENAME,
        hierarchy=hierarchy_index, matrix=np.asarray(parent_vectors,dtype=np.float32), native_receipt=native)
    files = {p.name:digest(p) for p in (root/'application').iterdir() if p.name in
             ('memory.db','hnsw_index.bin','native-spine.sqlite',native_spine_parent_store.FILENAME)}
    if len(files) != 4 or native['body_tokens'] < million*1_000_000:
        raise ValueError('Combined source size or persistence incomplete')
    save(root/'ingest-complete.json', dict(snapshot=native, parent_snapshot=parent_receipt,
        application_files=files, closed=True, elapsed_s=time.perf_counter()-started,
        new_qwen_calls=0, questions_or_references_loaded=False, cached_chunk_hits=encoder.hits))
    # Freeze evaluation population only after source ingestion is complete.
    question_parts = [read(p/'questions/questions.json')['questions'] for p in sources]
    questions = [part[i] for i in range(100) for part in question_parts]
    references = [r for p in sources for r in read(p/'questions/references.json')['references']]
    if len(questions) != million*100 or len({q['question_id'] for q in questions}) != len(questions):
        raise ValueError('Wrong question population')
    days = {datetime.strptime(q['question_date'][:10],'%Y/%m/%d').date() for q in questions}
    dated_tokens = [(t.created_at.date(),count_tokens(t.text)) for t in turns]
    eligible = {str(day):sum(n for stamp,n in dated_tokens if stamp<=day) for day in days}
    unique = {(t.role,quote_sha256(t.text)):t for t in turns}
    save(root/'scope.json',dict(actual_body_tokens=native['body_tokens'],nominal_body_tokens=million*1_000_000,
        through_question_day_body_tokens=min(eligible.values()),eligible_tokens_by_question_day=eligible,
        unique_role_text_tokens=sum(count_tokens(t.text) for t in unique.values()),
        history_count=1,question_count=len(questions),source_histories=million,
        source_assembly='Chronological union of existing benchmark archives; one combined routing index',
        repeated_content='Occurrence identities preserved; unique role/text token count reported separately',
        source_summarization='Authenticated precompiled caches; cold summarization is not measured'))
    save(root/'questions/questions.json',dict(questions=questions,question_count=len(questions)))
    save(root/'questions/references.json',dict(references=references,evaluation_only=True,ingest_use_permitted=False))
    emit(phase='combined_source_complete',body_tokens=native['body_tokens'],questions=len(questions),
         elapsed_s=time.perf_counter()-started)


def repair_vectors(root, source):
    """Repair a cached union in a fresh directory without raw reingestion."""
    if root.exists():
        raise ValueError('Repair requires a fresh directory')
    started = time.perf_counter()
    old = read(source/'ingest-complete.json')
    plan = read(source/'ingest-plan.json')
    for name, sha in old['application_files'].items():
        if digest(source/'application'/name) != sha:
            raise ValueError('Combined source changed')
    sections, matrices = [], []
    for binding in plan['sources']:
        original = Path(binding['source'])
        if digest(original/'ingest-complete.json') != binding['receipt_sha256']:
            raise ValueError('Original source receipt changed')
        receipt = read(original/'ingest-complete.json')
        for name, sha in receipt['application_files'].items():
            if digest(original/'application'/name) != sha:
                raise ValueError('Original source changed')
        with Database(original/'application/memory.db', read_only=True) as db:
            turns = TranscriptStore(db).get_all()
        snapshot = native_spine_store.load(original/'application/native-spine.sqlite', turns=turns)
        if snapshot.receipt != binding['snapshot']:
            raise ValueError('Original snapshot changed')
        sections.extend(snapshot.semantic.sections)
        matrices.append(snapshot.semantic._dense._matrix)
        emit(phase='repair_source_verified', source=str(original))
    with Database(source/'application/memory.db', read_only=True) as db:
        turns = TranscriptStore(db).get_all()
    snapshot = native_spine_store.load(source/'application/native-spine.sqlite', turns=turns)
    parents, parent_receipt = native_spine_parent_store.load(
        source/'application'/native_spine_parent_store.FILENAME,
        hierarchy=snapshot.hierarchy, native_receipt=snapshot.receipt)
    if snapshot.receipt != old['snapshot'] or parent_receipt != old['parent_snapshot']:
        raise ValueError('Combined snapshot receipt changed')
    matrix = align_atomic_vectors(snapshot.semantic.hierarchy, sections, np.concatenate(matrices))
    mismatches = sum(a.tobytes() != b.tobytes()
                     for a, b in zip(matrix, snapshot.semantic._dense._matrix, strict=True))
    if not mismatches:
        raise ValueError('No vector alignment defect to repair')
    (root/'application').mkdir(parents=True)
    for name in ('memory.db', 'hnsw_index.bin'):
        shutil.copy2(source/'application'/name, root/'application'/name)
    native = native_spine_store.publish(root/'application/native-spine.sqlite',
        atomic_index=snapshot.semantic.hierarchy, hierarchy=snapshot.hierarchy, matrix=matrix,
        embedding_identity=snapshot.semantic.embedding_identity, turns=turns)
    parent = native_spine_parent_store.publish(root/'application'/native_spine_parent_store.FILENAME,
        hierarchy=snapshot.hierarchy, matrix=parents._dense._matrix, native_receipt=native)
    files = {p.name:digest(p) for p in (root/'application').iterdir()}
    if any(files[n] != old['application_files'][n] for n in ('memory.db', 'hnsw_index.bin')):
        raise ValueError('Repair changed the raw application')
    save(root/'ingest-plan.json', dict(plan, repair_source=str(source),
        repair='Identity-aligned atomic vectors; raw application copied byte-for-byte',
        repair_implementation_sha256=digest(__file__)))
    save(root/'ingest-complete.json', dict(old, snapshot=native, parent_snapshot=parent,
        application_files=files, elapsed_s=time.perf_counter()-started,
        original_ingest_elapsed_s=old['elapsed_s'], raw_history_reingestions=0,
        corrected_vector_rows=mismatches))
    shutil.copytree(source/'questions', root/'questions')
    save(root/'scope.json', read(source/'scope.json'))
    # Independent persisted-row comparison, using the original identity bindings.
    with closing(sqlite3.connect((root/'application/native-spine.sqlite').as_uri()+'?mode=ro', uri=True)) as db:
        payload, raw = db.execute('SELECT payload,vectors FROM snapshot WHERE id=1').fetchone()
    p = json.loads(payload)
    width = 4*p['matrix_shape'][1]
    expected = {s.section_id:v.tobytes() for s,v in zip(sections,np.concatenate(matrices),strict=True)}
    for i, section in enumerate(p['atomic_sections']):
        if raw[i*width:(i+1)*width] != expected[section['section_id']]:
            raise ValueError('Persisted vector alignment differs from original sources')
    save(root/'vector-alignment-audit.json', dict(sections=len(sections),
        original_mismatches=mismatches, repaired_mismatches=0,
        raw_application_unchanged=True, questions_unchanged=True, references_unchanged=True,
        model_calls=0, raw_history_reingestions=0))
    emit(phase='vector_repair_complete', corrected_rows=mismatches, elapsed_s=time.perf_counter()-started)


def gate(report):
    n=report['question_count']
    for condition, reason in (
        (report['invalid_grades']/n > .02, 'Invalid grades exceed 2%'),
        (report['support_complete']/n < .90, 'Exact support coverage below 90%'),
        (report['correct']/n < .75, 'Automated accuracy below 75%'),
        (report['inline_memory']['accepted']/n < .80, 'Inline acceptance below 80%')):
        if condition:
            return reason
    return None


def run(root, runtime_root):
    root.mkdir(parents=True,exist_ok=False)
    save(root/'campaign.json',dict(scales=[2,5,10],questions=[200,500,1000],total_questions=1700,
        actor='codex_sdk/gpt-5.6-sol',judge='codex_sdk/gpt-5.6-sol',reasoning_effort='none',
        history_count=3,raw_content_to_attention=False,source_cache_reuse=True,
        new_source_summarization=False,source_filter_at_recall=False,
        question_order='Round robin across source archives; originals unchanged',
        implementation_sha256=digest(__file__)))
    reports=[]
    try:
        for million in (2,5,10):
            name=f'{million}m'
            source,case=root/'sources'/name,root/'qa'/name
            command(root,name+'-ingest',[__file__,'build','--root',str(source),'--million',str(million)])
            command(root,name+'-prepare',['tools/evaluate_chat_io_local100.py','prepare','--root',str(case),
                '--runtime-root',str(runtime_root/name),'--source',str(source),'--question-count',str(million*100),
                '--reader-gateway','https://central-dev.zt:4000/v1','--reader-model','codex_sdk/gpt-5.6-sol',
                '--judge-model','codex_sdk/gpt-5.6-sol','--inline-memory','--stop-on-obvious-problems',
                '--empty-response-retries','2'])
            command(root,name+'-live',['tools/evaluate_chat_io_local100.py','live','--root',str(case)])
            command(root,name+'-audit',['tools/evaluate_chat_io_local100.py','audit','--root',str(case)])
            report=read(case/'report.json')
            summary={k:report[k] for k in ('question_count','body_tokens','correct','support_complete',
                                         'invalid_grades','answer_latency','reader_latency','inline_memory')}
            summary.update(scale=name,scope=read(source/'scope.json'),audit=read(case/'reopen-audit.json'),
                           cycle_s=report['cycle']['total_cycle_s'],final_drain_s=report['cycle']['final_drain_s'])
            reports.append(summary)
            save(root/f'{name}-complete.json',summary)
            reason=gate(report)
            if reason:
                raise RuntimeError(name+': '+reason)
        save(root/'complete.json',dict(results=reports,questions=1700))
    except Exception as exc:
        save(root/'stopped.json',dict(error_type=type(exc).__name__,reason=str(exc),results=reports))
        emit(phase='campaign_stopped',reason=str(exc))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('build','run','repair-vectors'))
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--runtime-root',type=Path)
    parser.add_argument('--million',type=int,choices=(2,5,10))
    parser.add_argument('--source',type=Path)
    args=parser.parse_args()
    if args.phase=='build':
        if args.million is None: parser.error('build requires --million')
        build(args.root.resolve(),args.million)
    elif args.phase=='repair-vectors':
        if args.source is None: parser.error('repair-vectors requires --source')
        repair_vectors(args.root.resolve(),args.source.resolve())
    else:
        if args.runtime_root is None: parser.error('run requires --runtime-root')
        run(args.root.resolve(),args.runtime_root.resolve())
