"""Source-authenticated date replay; native storage anchors are not mention times.

Reuses frozen r27 evidence and the timestamp evaluator. Source text, roles and
IDs are verified against the original dataset before attaching session dates.
No gold answer, new retrieval, ingestion, summary or embedding enters the reader.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from memory_condense.domain._discourse_identity import identity_sha256, quote_sha256
from memory_condense.domain._tokenizer import count_tokens
from memory_condense.runtime.artifacts import read, read_sealed_json, save, emit
from tools import evaluate_chronos_packet_time as evaluator
from tools.audit_chronos_packet_time import replay as storage_time_replay
from tools.assess_native_longmemeval import stream_records
from tools.evaluate_longmemeval_native_release import sample_without_gold, events_for


ROOT = Path('eval_results/chronos-source-time-accuracy-20261016-r1')
PREVIOUS = Path('eval_results/chronos-time-accuracy-20261016-r2')


def source_time_packet(packet, query, asked_at, events):
    text, _ = storage_time_replay(packet, query, asked_at)
    by_id = {e.event_id:e for e in events}
    cursor, pieces, changes = 0, [], []
    decoder = json.JSONDecoder()
    for reference in packet['references']:
        span = reference['span']
        event = by_id[span['turn_id']]
        if (event.role != span['role'] or event.metadata['source_id'] != span['source_id']
                or quote_sha256(event.text) != span['turn_text_sha256']
                or quote_sha256(event.text[span['start_char']:span['end_char']]) != span['span_text_sha256']):
            raise ValueError('Evidence does not match its original source turn')
        start = text.index('<MEMORY ', cursor)
        metadata, end = decoder.raw_decode(text, start + len('<MEMORY '))
        if metadata['turn_id'] != event.event_id:
            raise ValueError('Envelope order changed')
        if metadata['mentioned_at'] != event.created_at:
            changes.append(dict(turn_id=event.event_id, source_id=span['source_id'],
                                storage_time=metadata['mentioned_at'], source_time=event.created_at))
        metadata['mentioned_at'] = event.created_at
        pieces.extend((text[cursor:start], '<MEMORY '+json.dumps(metadata,ensure_ascii=False)))
        # Skip this entire exact body, including any literal nested delimiters.
        body_end = end + 2 + span['end_char'] - span['start_char']
        pieces.append(text[end:body_end+len('\n</MEMORY>')])
        cursor = body_end + len('\n</MEMORY>')
    pieces.append(text[cursor:])
    return ''.join(pieces), changes


def prepare(root, previous):
    old_plan, bindings = evaluator.validate(previous)
    source = Path(old_plan['source'])
    source_plan = read(source/'plan.json')
    dataset = Path(source_plan['dataset'])
    by_id = {b['question_id']:b for b in bindings}
    rows, changed, unchanged_prompts, before, after = [], 0, 0, 0, 0
    with dataset.open(encoding='utf-8') as handle:
        for record in stream_records(handle):
            binding = by_id[str(record['question_id'])]
            ordinal = binding['ordinal']
            folder = f"{ordinal:03d}-{binding['question_id']}"
            old = read_sealed_json(source/'samples'/folder/'answer.json')
            if old.sha256 != binding['source_sha256']:
                raise ValueError('Frozen source changed')
            a = old.payload
            if (record['question'],record['question_date']) != (a['question'],a['question_date']):
                raise ValueError('Question changed in dataset')
            # The loader sees a copy with the answer field replaced by a sentinel.
            events = events_for(sample_without_gold(record),ordinal)
            text, changes = source_time_packet(a['packet'],a['question'],a['question_date'],events)
            sha = quote_sha256(text)
            save(root/'packets'/f'{ordinal:03d}.json',dict(source_sha256=old.sha256,
                original_packet_sha256=identity_sha256(a['packet']),text=text,updated_text_sha256=sha,
                corrected_source_dates=changes))
            same = sha == binding['updated_text_sha256']
            unchanged_prompts += same
            changed += len(changes)
            before += count_tokens(a['packet']['text'])
            after += count_tokens(text)
            rows.append(dict(binding,updated_text_sha256=sha,corrected_dates=len(changes)))
    rows.sort(key=lambda r:r['ordinal'])
    if [r['ordinal'] for r in rows] != list(range(500)):
        raise ValueError('Not the full frozen population')
    audit = dict(source_plan_sha256=old_plan['source_plan_sha256'],questions=500,samples=rows,
        corrected_span_dates=changed,identical_previous_prompts=unchanged_prompts,
        before_tokens=before,after_tokens=after,exact_source_text_verified=True,
        limitation='Native stored spans retain export anchors; experiment overlays authenticated dataset dates.')
    save(root/'audit.json',audit)
    plan = dict(old_plan,audit=str((root/'audit.json').resolve()),
        audit_sha256=read_sealed_json(root/'audit.json').sha256,
        arm='Original session timestamps authenticated against exact source turns; same raw evidence and routing',
        limitations=['Historical reader predictions reused; same Sol model and binary rubric.',
            'Source-time metadata replay only; no corrected-clock rerouting or event ledger.',
            'The native compiler still stores export anchors; this experiment is not a production migration.'])
    plan['implementation'] = dict(plan['implementation'])
    plan['implementation'][str(Path(__file__))] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    save(root/'plan.json',plan)
    emit(phase='source_time_prepared',questions=500,corrected_span_dates=changed,
         identical_previous_prompts=unchanged_prompts,before_tokens=before,after_tokens=after)


def run(root, start, count):
    _, bindings = evaluator.validate(root)
    packets = {}
    for b in bindings:
        p = read(root/'packets'/f"{b['ordinal']:03d}.json")
        if p['source_sha256'] != b['source_sha256'] or quote_sha256(p['text']) != b['updated_text_sha256']:
            raise ValueError('Corrected packet binding changed')
        packets[p['original_packet_sha256']] = p['text']
    def prepared(packet, query, asked_at):
        return packets[identity_sha256(packet)], 0
    evaluator.replay = prepared
    evaluator.run(root,start,count)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('prepare','run','report'))
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--previous',type=Path,default=PREVIOUS)
    parser.add_argument('--start',type=int,default=0)
    parser.add_argument('--count',type=int,default=500)
    args = parser.parse_args()
    if args.phase == 'prepare': prepare(args.root,args.previous)
    elif args.phase == 'run': run(args.root,args.start,args.count)
    else: evaluator.report(args.root,categories=True)
