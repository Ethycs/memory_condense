"""Authenticate copied recall evidence before aliasing its existing summaries."""
from __future__ import annotations

import json

from memory_condense.domain._discourse_identity import quote_sha256
from memory_condense.search.section_summary import RawSectionSpan


def recall_summary_alias(row, originals):
    """Return provenance for an exact packet, or None for ordinary compilation.

    `originals` contains authenticated, already indexed atomic sections keyed by
    span receipt. Copies retain their raw journal event but do not paraphrase
    the same evidence again or become independent assertions in the hierarchy.
    Never trust an event's metadata without checking every byte of its body.
    """
    receipt = row.get('metadata', {}).get('_chat', {})
    references = receipt.get('references', ())
    if (row['role'] != 'system' or receipt.get('kind') != 'recall'
            or row['turn_id'] != '_chat:recall:' + str(receipt.get('packet_id'))
            or not references):
        return None
    text, cursor, reused = row['text'], 0, []
    try:
        for i, reference in enumerate(references):
            span = RawSectionSpan(**reference['span'])
            original = originals.get(span.receipt_sha256)
            if original is None or original.spans != (span,):
                return None  # Partial/clipped/unavailable evidence uses the normal path.
            if i:
                if text[cursor:cursor+2] != '\n\n':
                    return None
                cursor += 2
            if not text.startswith('<MEMORY ', cursor):
                return None
            end = text.index('>\n', cursor)
            label = json.loads(text[cursor+8:end])
            role = label.get('role')
            if role not in ({'system', 'tool', 'source'} if span.role == 'system' else {span.role}):
                return None
            if label != dict(turn_id=span.turn_id, source_id=span.source_id, role=role,
                             start_char=span.start_char, end_char=span.end_char):
                return None
            start = end+2
            cursor = start + span.end_char-span.start_char
            if (quote_sha256(text[start:cursor]) != span.span_text_sha256
                    or text[cursor:cursor+10] != '\n</MEMORY>'):
                return None
            cursor += 10
            reused.append(dict(section_id=original.section_id,
                section_receipt_sha256=original.receipt_sha256,
                span=span.identity_payload()))
    except (KeyError, TypeError, ValueError):
        return None
    if cursor != len(text):
        return None
    return dict(mode='recall-summary-alias-v1', packet_id=receipt['packet_id'],
        text_sha256=quote_sha256(text), reused_sections=reused,
        summary=f'Recall observation: {len(reused)} previously indexed evidence sections delivered. '
                'Their existing summaries and exact sources remain authoritative; this receipt adds no new facts.')
