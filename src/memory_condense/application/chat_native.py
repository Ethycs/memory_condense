"""Bind native exact-span packets to original transcript and Hebbian nodes."""
from __future__ import annotations

import json

from memory_condense.domain._discourse_identity import quote_sha256
from memory_condense.search.section_summary import RawSectionSpan
from memory_condense.search.summary_time_prior import _ASKED
from memory_condense.search.summary_time_prior_v2 import question_day


def validate_reference(app, reference):
    span = RawSectionSpan(**reference['span'])
    turn = app.transcript.get_turn(span.turn_id)
    if turn is None or RawSectionSpan.from_turn(turn, start_char=span.start_char,
            end_char=span.end_char) != span:
        raise ValueError('Recall reference no longer matches original memory')
    return span


def exact_memory_passage(span, text, role, *, evidence_id=None):
    """Canonical exact evidence envelope shared by recall and summary reuse."""
    if len(text)!=span.end_char-span.start_char or quote_sha256(text)!=span.span_text_sha256:
        raise ValueError('Recall passage differs from its exact source span')
    metadata=dict(turn_id=span.turn_id,source_id=span.source_id,role=role,
                  start_char=span.start_char,end_char=span.end_char,mentioned_at=span.created_at)
    if evidence_id is not None:
        metadata['evidence_id'] = evidence_id
    label=json.dumps(metadata,ensure_ascii=False)
    return '<MEMORY '+label+'>\n'+text+'\n</MEMORY>'


def native_packet(app, query, dated_question, events, **limits):
    """Only spans which survived packing acquire pointers or learning credit."""
    question_day(query, dated_question)  # Validate the clock/query binding before retrieval.
    result = app.retrieve_native_spine(query, dated_question, **limits)
    routing = result.routing.identity_payload()
    if routing['raw_reads_during_routing'] or routing['query_qwen_passes']:
        raise ValueError('Native chat routing violated the summary-only boundary')
    event_by_id = {e.event_id: e for e in events}
    references, passages = [], []
    experiments = getattr(result,'experiments',None)
    extra = experiments.hydration.sections if experiments is not None else ()
    extra_ids = {s.section.section_id for s in extra}
    for section in (*result.hydration.sections,*extra):
        for evidence in section.evidence:
            event = event_by_id[evidence.span.turn_id]
            receipt = event.metadata.get('_chat', {})
            reference = dict(section_id=section.section.section_id,
                             span=evidence.span.identity_payload(), independent=not bool(receipt))
            if section.section.section_id in extra_ids:
                reference.update(learning_eligible=False,
                    experimental_route=experiments.receipt['provenance'][section.section.section_id])
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
            # artifacts can cite them. Summary context is labeled separately.
            passages.append(exact_memory_passage(span,evidence.text,event.role))
    packet = dict(text='\n\n'.join(passages), references=references,
                  routing=routing, hydration=result.hydration.identity_payload())
    if 'policy' in routing:
        # The journal needs the deployed policy and actual packing outcome,
        # not a second copy of every candidate summary in the route receipt.
        packet['retrieval'] = dict(mode='dialogue', policy=routing['policy'],
            route_sha256=routing['receipt_sha256'], hierarchy_sha256=routing['hierarchy_sha256'],
            index_sha256=result.hydration.plan.index_sha256,
            selected_episode_ids=routing['selected_episode_ids'], exhaustive=False,
            max_context_tokens=result.hydration.max_context_tokens,
            context_token_count=result.hydration.context_token_count,
            max_raw_spans=result.hydration.max_raw_spans,
            raw_turn_read_count=result.hydration.raw_turn_read_count,
            raw_reads_during_routing=0, query_qwen_passes=0)
    if result.tactic is not None:
        packet.setdefault('retrieval', dict(mode='cap8'))['tactic'] = result.tactic
    if experiments is not None:
        packet.setdefault('retrieval',dict(mode='cap8')).update(experiments=experiments.receipt,
            query_qwen_passes=experiments.receipt['query_qwen_passes'],raw_reads_during_routing=0)
    if result.hierarchy_context is not None:
        context = result.hierarchy_context
        if context.text:
            packet['text'] = context.text+'\n\nEXACT NEIGHBORHOOD EVIDENCE\n'+packet['text']
        packet['retrieval'].update(mode='hierarchy_neighborhoods', hierarchy_context=context.receipt())
    history = result.tactic is not None and result.tactic.get('selected_tactic') == 'decision_history'
    annotations, labels = '', None
    if history:
        from memory_condense.application.decision_history import annotate, snapshot_revisions
        revisions = getattr(app, 'decision_revisions', None)
        if revisions is None:
            revisions = snapshot_revisions(getattr(app, 'memory', None))
        annotations, labels, receipt = annotate(references, events, revisions=revisions)
        packet['retrieval'].update(mode='decision_history', decision_history=receipt)
    if history or result.hierarchy_context is not None:
        from memory_condense.application.continuity import contract, evidence_labels
        if labels is None:
            labels = evidence_labels(references, events)
        conventions, manifest = contract(references, labels,
            hierarchy=packet['retrieval'].get('hierarchy_context',{}),
            source_scope=result.routing.baseline.eligible_source_ids,
            index_sha256=result.hydration.plan.index_sha256,
            as_of_day=question_day(query,dated_question).isoformat())
        exact = []
        for i, reference in enumerate(references):
            span = RawSectionSpan(**reference['span'])
            event = event_by_id[span.turn_id]
            exact.append(exact_memory_passage(span,event.text[span.start_char:span.end_char],event.role,
                                             evidence_id=labels[i]))
        summary = result.hierarchy_context.text if result.hierarchy_context is not None else ''
        title = 'EXACT HISTORY EVIDENCE' if history else 'EXACT NEIGHBORHOOD EVIDENCE'
        packet['text'] = '\n\n'.join(t for t in (conventions,annotations,summary,title,*exact) if t)
        packet['retrieval']['continuity'] = manifest
    # The reader and router must see the same question clock. Mention dates
    # describe the source turn, not necessarily the date of the event it names.
    # Validate the header shape whatever the retrieval outcome, but never let it
    # become the whole body: an empty retrieval must stay empty so the caller
    # still reports that the store held nothing instead of a bare date.
    header = _ASKED.match(dated_question).group(0).strip()
    if packet['text']:
        packet['text'] = header + '\n\n' + packet['text']
    packet.setdefault('retrieval', dict(mode='cap8'))['dated_question'] = dated_question
    from memory_condense.domain._tokenizer import count_tokens
    packet['retrieval']['packet_token_count'] = count_tokens(packet['text'])
    return packet


def learn_native_packet(app, packet, events, *, access_event_id, weight_summaries=None):
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
        if (not reference.get('independent', False) or not reference.get('learning_eligible',True)
                or '_chat' in event.metadata):
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
    activations = weight_summaries(chunks) if weight_summaries is not None else None
    return app.observe_context_access([], chunks, access_event_id=access_event_id,
                                      **({'node_activations':activations} if activations is not None else {}))
