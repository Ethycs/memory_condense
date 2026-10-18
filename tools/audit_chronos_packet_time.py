"""Replay saved evidence envelopes only: no re-ingest, routing, readers or gold."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from memory_condense.application.chat_native import exact_memory_passage
from memory_condense.domain._discourse_identity import quote_sha256
from memory_condense.domain._tokenizer import count_tokens
from memory_condense.domain.question_time import dated_recall_question
from memory_condense.runtime.artifacts import publish_sealed_json, read_sealed_json
from memory_condense.search.section_summary import RawSectionSpan


def replay(packet, query, asked_at):
    """Use sealed span lengths, never delimiter splitting inside quoted text."""
    text = packet['text']
    cursor, pieces, previously_dated = 0, [], 0
    decoder = json.JSONDecoder()
    for reference in packet['references']:
        span = RawSectionSpan(**reference['span'])
        start = text.find('<MEMORY ', cursor)
        if start < 0:
            raise ValueError('Saved packet is missing an evidence envelope')
        metadata, end = decoder.raw_decode(text, start + len('<MEMORY '))
        expected = dict(turn_id=span.turn_id, source_id=span.source_id, role=span.role,
                        start_char=span.start_char, end_char=span.end_char)
        if any(metadata.get(key) != value for key, value in expected.items()):
            raise ValueError('Saved envelope disagrees with its source reference')
        if text[end:end + 2] != '>\n':
            raise ValueError('Malformed saved evidence envelope')
        body_start = end + 2
        body_end = body_start + span.end_char - span.start_char
        if text[body_end:body_end + len('\n</MEMORY>')] != '\n</MEMORY>':
            raise ValueError('Saved evidence length disagrees with source span')
        if 'mentioned_at' in metadata:
            if metadata['mentioned_at'] != span.created_at:
                raise ValueError('Saved mention time disagrees with source reference')
            previously_dated += 1
        passage = exact_memory_passage(span, text[body_start:body_end], span.role,
                                       evidence_id=metadata.get('evidence_id'))
        pieces.extend((text[cursor:start], passage))
        cursor = body_end + len('\n</MEMORY>')
    if '<MEMORY ' in text[cursor:]:
        raise ValueError('Saved packet has evidence without a source reference')
    pieces.append(text[cursor:])
    # This audit accepts historical packets without a question-time header.
    if text.startswith('[Question asked at '):
        raise ValueError('Packet already has a request clock; use the historical control')
    header = dated_recall_question(query, asked_at).split(']', 1)[0] + ']'
    return header + '\n\n' + ''.join(pieces), previously_dated


def audit(source: Path, output: Path):
    plan = read_sealed_json(source / 'plan.json')
    rows = []
    for ordinal, question_id in enumerate(plan.payload['question_ids']):
        artifact = read_sealed_json(source / 'samples' / f'{ordinal:03d}-{question_id}' / 'answer.json')
        saved = artifact.payload
        if (saved['question_id'], saved['ordinal']) != (question_id, ordinal):
            raise ValueError('Saved sample identity differs from the frozen plan')
        updated, previously_dated = replay(saved['packet'], saved['question'], saved['question_date'])
        row = dict(ordinal=ordinal, question_id=question_id, source_sha256=artifact.sha256,
                   evidence_spans=len(saved['packet']['references']), previously_dated=previously_dated,
                   before_tokens=count_tokens(saved['packet']['text']), after_tokens=count_tokens(updated),
                   updated_text_sha256=quote_sha256(updated))
        rows.append(row)
        if ordinal < 3:
            publish_sealed_json(output / f'example-{ordinal:03d}.json', dict(
                mode='metadata-only replay; original routing retained', source_path=str(artifact.path),
                source_sha256=artifact.sha256, text=updated, references=saved['packet']['references']))
    before = sum(row['before_tokens'] for row in rows)
    after = sum(row['after_tokens'] for row in rows)
    report = dict(format='chronos-packet-time-audit-v1', source_plan_sha256=plan.sha256,
                  questions=len(rows), verified_exact_spans=sum(row['evidence_spans'] for row in rows),
                  previously_dated_spans=sum(row['previously_dated'] for row in rows),
                  before_tokens=before, after_tokens=after, added_tokens=after-before,
                  mean_added_tokens=(after-before)/len(rows),
                  added_token_fraction=(after-before)/before,
                  new_routing_calls=0, model_calls=0, accuracy_measured=False, samples=rows)
    publish_sealed_json(output / 'report.json', report)
    print(json.dumps({key: value for key, value in report.items() if key != 'samples'}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    audit(args.source, args.output)
