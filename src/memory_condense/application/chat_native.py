"""Bind native exact-span packets to original transcript and Hebbian nodes."""
from __future__ import annotations

import json

from memory_condense.domain._discourse_identity import quote_sha256
from memory_condense.search.section_summary import RawSectionSpan


def validate_reference(app, reference):
    span = RawSectionSpan(**reference['span'])
    turn = app.transcript.get_turn(span.turn_id)
    if turn is None or RawSectionSpan.from_turn(turn, start_char=span.start_char,
            end_char=span.end_char) != span:
        raise ValueError('Recall reference no longer matches original memory')
    return span


def exact_memory_passage(span, text, role):
    """Canonical exact evidence envelope shared by recall and summary reuse."""
    if len(text)!=span.end_char-span.start_char or quote_sha256(text)!=span.span_text_sha256:
        raise ValueError('Recall passage differs from its exact source span')
    label=json.dumps(dict(turn_id=span.turn_id,source_id=span.source_id,role=role,
                          start_char=span.start_char,end_char=span.end_char),ensure_ascii=False)
    return '<MEMORY '+label+'>\n'+text+'\n</MEMORY>'


def native_packet(app, query, dated_question, events, **limits):
    """Only spans which survived packing acquire pointers or learning credit."""
    result = app.retrieve_native_spine(query, dated_question, **limits)
    routing = result.routing.identity_payload()
    if routing['raw_reads_during_routing'] or routing['query_qwen_passes']:
        raise ValueError('Native chat routing violated the summary-only boundary')
    event_by_id = {e.event_id: e for e in events}
    references, passages = [], []
    for section in result.hydration.sections:
        for evidence in section.evidence:
            event = event_by_id[evidence.span.turn_id]
            receipt = event.metadata.get('_chat', {})
            reference = dict(section_id=section.section.section_id,
                             span=evidence.span.identity_payload(), independent=not bool(receipt))
            if receipt.get('kind') == 'recall':
                # Keep the immediate exact source AND ancestry. A retrieved copy
                # does not manufacture independent support for its own graph.
                originals = {}
                for parent in receipt['references']:
                    for origin in parent.get('original_references', [parent]):
                        originals[origin['span']['receipt_sha256']] = origin
                reference['original_references'] = list(originals.values())
                reference['via_packet_id'] = receipt['packet_id']
            validate_reference(app, reference)
            references.append(reference)
            span = evidence.span
            # Keep exact source addresses visible to the reader so engineering
            # artifacts can cite them. Stored routing summaries stay out of it.
            passages.append(exact_memory_passage(span,evidence.text,event.role))
    return dict(text='\n\n'.join(passages), references=references,
                routing=routing, hydration=result.hydration.identity_payload())


def learn_native_packet(app, packet, events, *, access_event_id):
    """Use the existing scalar co-access graph; never send raw text to Qwen.

    The input is linked to original chunks actually delivered in the packet.
    Recall/feedback copies and graph-derived evidence do not train themselves.
    Existing rank discount, decay, node/degree bounds and deduplication apply.
    """
    event_by_id = {e.event_id: e for e in events}
    source_chunks = []
    for reference in packet.references:
        span = validate_reference(app, reference)
        event = event_by_id[span.turn_id]
        if not reference.get('independent', False) or '_chat' in event.metadata:
            continue
        rows = app._db.execute(
            'SELECT chunk_id FROM chunks WHERE turn_id=? AND start_char<? AND end_char>? ORDER BY start_char, chunk_id',
            (span.turn_id, span.end_char, span.start_char)).fetchall()
        source_chunks.extend(r[0] for r in rows)
    source_chunks = list(dict.fromkeys(source_chunks))
    if not source_chunks:
        return None
    input_event = event_by_id[packet.input_event_id]
    input_turn = app.transcript.get_turn(packet.input_event_id)
    if input_turn is None or quote_sha256(input_turn.text) != quote_sha256(input_event.text):
        raise ValueError('Recall input no longer matches its captured event')
    inputs = [r[0] for r in app._db.execute(
        'SELECT chunk_id FROM chunks WHERE turn_id=? ORDER BY start_char, chunk_id',
        (packet.input_event_id,)).fetchall()]
    # Reserve one node for the triggering input within the existing 16-node cap.
    chunks = list(dict.fromkeys(inputs[:1] + source_chunks))[:16]
    return app.observe_context_access([], chunks, access_event_id=access_event_id)
