"""One frozen 1M history, 100 fresh answers through ChatIO, durable live co-access.

Historical native retrieval is pinned to the original dated source snapshot.
Today's evaluation I/O is normally ingested into one writable clone and cannot
contaminate historical evidence. This does NOT measure live native hierarchy
refresh; that distinction is sealed in the plan and report.
"""
from __future__ import annotations

import argparse
from contextlib import closing
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sqlite3
import statistics
import time
from types import SimpleNamespace

from memory_condense.application.chat_io import ChatIO
from memory_condense.application.chat_native import learn_native_packet
from memory_condense.application.chat_session import ChatEvent, ChatSession
from memory_condense.application.condenser import MemoryCondenser
from tools import run_native_spine_user_completion_answers as old
from tools.matched_eval.artifacts import read_sealed_json

ROOT = Path('eval_results/chat-io-single100-20260929-r1')
SOURCE = old.CAMPAIGN / 'history-01'
BASELINE = Path('eval_results/native-spine-completion-single100-20260923-r1/history-01/report.json')
save, read = old.publish, read_sealed_json


def implementation():
    paths = [Path(__file__), *Path('src/memory_condense/application').glob('chat_*.py')]
    return {str(p): old.digest(p) for p in paths}


def prepare(root):
    if root.exists():
        raise ValueError('Use a new run directory')
    ingestion = read(SOURCE/'ingest-complete.json')
    scope = read(SOURCE/'scope.json')
    if scope.payload['through_question_day_body_tokens'] < 1_000_000:
        raise ValueError('Not a million-token history')
    for name, sha in ingestion.payload['application_files'].items():
        if old.digest(SOURCE/'application'/name) != sha:
            raise ValueError('Original application changed')
    questions = read(SOURCE/'questions/questions.json')
    population = [old.current.frozen.question(q) for q in questions.payload['questions']]
    if len(population) != 100 or len({q['question_id'] for q in population}) != 100:
        raise ValueError('Requires 100 distinct questions')
    campaign = read(old.CAMPAIGN/'campaign.json')
    plan = dict(implementation=implementation(), source=old.binding(ingestion), questions=population,
        question_binding=old.binding(questions), baseline=old.binding(read(BASELINE)),
        policy=read(old.POLICY).payload, reader=old.rebased(campaign.payload['reader_policy']).payload,
        model=old.MODEL, history_count=1, question_count=100,
        body_tokens=scope.payload['actual_body_tokens'], max_tokens=256, retries=0,
        answer_calls=100, judge_calls=100, live_native_hierarchy_refresh_exercised=False,
        lifecycle='Historical snapshot reopened once; current ChatIO and ordinary application ingestion on one writable clone; no test-answer feedback into historical evidence',
        source_ingestion_reused=True, new_history_rebuilds=0, references_opened=False)
    save(root/'run.json', plan)
    shutil.copytree(SOURCE/'application', root/'application')
    old.emit(phase='prepared', questions=100, body_tokens=plan['body_tokens'], new_history_rebuilds=0)


class HistoricalBackend:
    def __init__(self, root, plan):
        self.root, self.plan = root, plan
        self.live = self.history = self.encoder = None
        self.events = ()
        self.packet = None
        self.questions = {q['retrieval_query']: q for q in plan['questions']}

    def open(self):
        if self.live is not None:
            return
        self.encoder = old.current.frozen.EmbeddingService(device='cuda', batch_size=8)
        self.history = MemoryCondenser(SOURCE/'application', embedder=self.encoder, auto_extract=False, read_only=True)
        self.live = MemoryCondenser(self.root/'application', embedder=self.encoder, auto_extract=False)
        original = read(SOURCE/'ingest-complete.json').payload
        if self.history.native_spine_receipt() != original['snapshot'] or self.history.native_parent_user_receipt() != original['parent_snapshot']:
            raise ValueError('Historical source snapshot differs')
        self.order = old.current.presentation.renderer.TranscriptOrder(self.history.transcript.get_all())
        self.encoder.embed_query('Warm one historical memory session.')

    def sync(self, events):
        self.open()
        persisted = self.live.transcript.get_all()
        if len(persisted) > len(events):
            raise ValueError('Live store is ahead of the journal')
        for turn, event in zip(persisted, events):
            role = event.role if event.role in ('user', 'assistant', 'system') else 'system'
            if (turn.turn_id, turn.role, turn.text) != (event.event_id, role, event.text):
                raise ValueError('Journal and application prefixes differ')
        new = [(e.role if e.role in ('user', 'assistant', 'system') else 'system', e.text,
                e.metadata.get('source_id', 'chat-io-evaluation'), datetime.fromisoformat(e.created_at), e.event_id)
               for e in events[len(persisted):]]
        if new:
            self.live.ingest_many(new)
        self.events = events

    def recall(self, query):
        q = self.questions[query]
        messages, hydration, routing, rendered = old.policy_tool.build(
            SimpleNamespace(retrieve=self.history.retrieve_native_spine), q, self.plan['policy'], self.order)
        messages = old.current.reader.apply_reader(messages, old.current.validate_reader_policy(self.plan['reader']))
        delivered = {p['span_sha256'] for p in rendered['placements']}
        references = [dict(section_id=s['section']['section_id'], span=e['span'], independent=True)
                      for s in hydration['sections'] for e in s['evidence'] if e['span']['receipt_sha256'] in delivered]
        if routing['raw_reads_during_routing'] or routing['query_qwen_passes']:
            raise ValueError('Summary-only routing violated')
        self.packet = dict(messages=messages, hydration=hydration, routing=routing, rendered=rendered)
        return dict(text=rendered['text'], references=references)

    def learn(self, packet, *, access_event_id):
        return learn_native_packet(self.live, packet, self.events, access_event_id=access_event_id)

    def close(self):
        for resource in (self.live, self.history, self.encoder):
            if resource is not None:
                resource.close()


def source_events():
    from memory_condense.persistence.db import Database
    from memory_condense.persistence.transcript_store import TranscriptStore
    with Database(SOURCE/'application/memory.db', read_only=True) as db:
        return tuple(ChatEvent(t.turn_id, t.role, t.text, t.created_at.isoformat(), {'source_id': t.source_id})
                     for t in TranscriptStore(db).get_all())


def run(root):
    old.require_idle()
    plan = read(root/'run.json')
    if plan.payload['implementation'] != implementation():
        raise ValueError('Frozen test implementation changed')
    started = time.perf_counter()
    backend = HistoricalBackend(root, plan.payload)
    with ChatSession(root/'chat', 'single-1m-100', backend) as chat:
        chat.ingest_many(source_events())
        chat.flush()
        save(root/'initial-reopen.json', dict(cold_setup_s=time.perf_counter()-started,
            snapshot=backend.history.native_spine_receipt(), events=chat.status(), worker_pid=os.getpid()))
        io = ChatIO(chat)
        with closing(old.current.frozen._completion_client('LITELLM_KEY', old.current.frozen.GATEWAY).with_options(max_retries=0)) as client:
            for ordinal, q in enumerate(plan.payload['questions']):
                prefix = root/'answers'/f'{ordinal:03d}'
                request = save(prefix.with_suffix('.request.json'), dict(plan_sha256=plan.sha256, question=q))
                if prefix.with_suffix('.response.json').exists():
                    continue
                with prefix.with_suffix('.reserved').open('x', encoding='utf-8') as handle:
                    handle.write(request.sha256 + '\n')
                before = time.perf_counter()
                def reader(packet):
                    measured = old.current.frozen.measure_streaming_answer(client=client, model=plan.payload['model'],
                        prepare_prompt=lambda: backend.packet['messages'], max_tokens=256)
                    return dict(content=measured['prediction'], measurement=measured)
                result = io.exchange(ChatEvent(f'q{ordinal:03d}', 'user', q['retrieval_query']),
                                     request_id=f'answer-{ordinal:03d}', reader=reader)
                wall = time.perf_counter() - before
                # Includes real indexing and co-access before the next question.
                drained = time.perf_counter()
                status = chat.flush()
                saved = save(prefix.with_suffix('.response.json'), dict(request_sha256=request.sha256,
                    question=q, measurement=result['response']['measurement'], packet=result['packet'],
                    io_total_s=wall, drain_s=time.perf_counter()-drained, status=status, **backend.packet))
                old.emit(phase='answered', ordinal=ordinal, questions=100, io_total_s=round(wall,3),
                         pending=status['pending_events'], packet_refs=len(result['packet']['references']))
        save(root/'io-complete.json', dict(status=chat.flush(), events=len(chat.events()), worker_pid=os.getpid()))
    old.emit(phase='answers_complete', questions=100, elapsed_s=time.perf_counter()-started)


def report(root, enable):
    plan = read(root/'run.json')
    responses = [read(root/'answers'/f'{i:03d}.response.json') for i in range(100)]
    for q, response in zip(plan.payload['questions'], responses):
        if q != response.payload['question']:
            raise ValueError('Question population changed')
    seal = save(root/'answers-complete.json', dict(plan=old.binding(plan), answers=[old.binding(r) for r in responses]))
    refs = read(SOURCE/'questions/references.json')
    reference = {r['question_id']: r for r in refs.payload['references']}
    prompts = [old.current.frozen.build_judge_prompt(r.payload['question']['retrieval_query'],
        reference[r.payload['question']['question_id']]['answer'], r.payload['measurement']['prediction']) for r in responses]
    preflight = save(root/'judge-preflight.json', dict(answers=old.binding(seal), references=old.binding(refs), prompts=prompts))
    def factory(client):
        return old.current.frozen.FastCompletionRuntime(checkpoint_dir=root/'judge-checkpoints',
            prompt_population=prompts, model=old.MODEL, client=client, max_prompt_tokens=4096,
            max_new_tokens=old.current.frozen.JUDGE_MAX_TOKENS, max_concurrency=8, retries=0,
            request_options={'temperature': 0}, benchmark_provenance={'binding_sha256': preflight.sha256, 'phase': 'judge'})
    with closing(factory(None)) as runtime:
        remaining = runtime.population.unique_prompt_count - len(old.current.frozen._authenticated_records(runtime))
    judged, calls, hits, _ = old.current.frozen._run_exactly_authorized(runtime_factory=factory,
        authorized_provider_calls=remaining, enable_provider=enable,
        client_factory=lambda: old.current.frozen.ThreadLocalProvider(lambda: old.current.frozen._completion_client('LITELLM_KEY', old.current.frozen.GATEWAY)))
    baseline = read(BASELINE).payload['rows']
    rows = []
    for i, (r, verdict) in enumerate(zip(responses, judged.logical_completions)):
        p = r.payload
        rows.append(dict(ordinal=i, correct=bool(old.current.frozen.parse_binary_judge_verdict(verdict)),
            baseline_correct=baseline[i]['correct'], prediction=p['measurement']['prediction'],
            reference=reference[p['question']['question_id']]['answer'], question=p['question']['retrieval_query'],
            support_coverage=all(s['quote'] in p['rendered']['text'] for s in reference[p['question']['question_id']]['supports']),
            prompt_tokens=p['measurement']['usage']['prompt_tokens'], io_total_s=p['io_total_s'],
            api_total_s=p['measurement']['api_total_s'], drain_s=p['drain_s']))
    save(root/'report.json', dict(history_count=1, question_count=100, body_tokens=plan.payload['body_tokens'],
        correct=sum(r['correct'] for r in rows), baseline_correct=sum(r['correct'] for r in baseline), rows=rows,
        mean_prompt_tokens=statistics.fmean(r['prompt_tokens'] for r in rows),
        latency={k: old.current.frozen.latency_distribution([r[k] for r in rows]) for k in ('io_total_s','api_total_s','drain_s')},
        live_native_hierarchy_refresh_exercised=False, new_judge_calls=calls, judge_cache_hits=hits,
        plan=old.binding(plan), answers=old.binding(seal),
        lifecycle=read(root/'reopen-audit.json').payload))
    old.emit(phase='report_complete', correct=sum(r['correct'] for r in rows), questions=100)


def audit(root):
    from memory_condense.persistence.db import Database
    from memory_condense.persistence.transcript_store import TranscriptStore
    with closing(sqlite3.connect(root/'chat/chat-events.sqlite')) as journal, Database(root/'application/memory.db', read_only=True) as db:
        turns = {t.turn_id: t for t in TranscriptStore(db).get_all()}
        events = journal.execute('SELECT event_id, role, text FROM events ORDER BY sequence').fetchall()
        for event_id, role, text in events:
            t = turns[event_id]
            if t.text != text or t.role != (role if role in ('user','assistant','system') else 'system'):
                raise ValueError('Reopened I/O transcript changed')
        packets = journal.execute('SELECT packet_id, input_event_id, refs FROM packets').fetchall()
        span_count = 0
        from memory_condense.search.section_summary import RawSectionSpan
        for packet_id, input_id, refs in packets:
            if input_id not in turns:
                raise ValueError('Missing input pointer')
            for r in json.loads(refs):
                span = RawSectionSpan(**r['span'])
                if span != RawSectionSpan.from_turn(turns[span.turn_id], start_char=span.start_char, end_char=span.end_char):
                    raise ValueError('Original memory pointer changed')
                span_count += 1
        feedback = journal.execute('SELECT COUNT(*) FROM feedback WHERE successful=1 AND applied=1').fetchone()[0]
        learned = db.execute("SELECT COUNT(*) FROM consolidation_access_events WHERE event_id LIKE '_chat:feedback:%'").fetchone()[0]
        if len(packets) != 100 or feedback != 100 or learned != 100 or len(events) != len(turns):
            raise ValueError('Incomplete persisted I/O or Hebbian population')
        save(root/'reopen-audit.json', dict(events=len(events), packets=len(packets), source_span_pointers=span_count,
            applied_feedback=feedback, hebbian_events=learned, separate_process_reopen=True, worker_pid=os.getpid()))
        old.emit(phase='audit_complete', events=len(events), packets=len(packets), hebbian_events=learned)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('prepare','run','audit','report'))
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--enable-provider', action='store_true')
    args = parser.parse_args()
    if args.phase in ('run','report') and not args.enable_provider:
        parser.error('Provider phase requires --enable-provider')
    if args.phase == 'report':
        report(args.root, args.enable_provider)
    else:
        globals()[args.phase](args.root)
