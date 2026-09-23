"""Render the same authenticated raw spans as ordered conversation excerpts."""
from collections import Counter
from dataclasses import dataclass
from types import MappingProxyType

from memory_condense.application.section_retrieval import SectionRetrievalResult
from memory_condense.domain._discourse_identity import identity_sha256, quote_sha256
from memory_condense.domain._tokenizer import count_tokens


class TranscriptOrder:
    """A resident order index over the authenticated raw-turn population."""
    __slots__ = ('positions', 'identities', 'receipt_sha256', '_locked')

    def __setattr__(self, name, value):
        if getattr(self, '_locked', False):
            raise AttributeError('transcript order is immutable')
        object.__setattr__(self, name, value)

    def __init__(self, turns):
        rows = tuple(turns)
        if len({turn.turn_id for turn in rows}) != len(rows):
            raise ValueError('transcript order requires unique turn identities')
        self.positions = MappingProxyType({t.turn_id: i for i, t in enumerate(rows)})
        self.identities = MappingProxyType({t.turn_id: (
            t.source_id, t.role, t.created_at.isoformat(), quote_sha256(t.text)) for t in rows})
        self.receipt_sha256 = identity_sha256([(t.turn_id, self.identities[t.turn_id]) for t in rows])
        self._locked = True


@dataclass(frozen=True)
class ThreadedSectionContext:
    text: str
    placements: tuple
    hydration_sha256: str
    transcript_order_sha256: str
    conversation_count: int
    token_count: int

    def identity_payload(self):
        return {'text': self.text, 'text_sha256': quote_sha256(self.text),
                'placements': [{'span_sha256': sha, 'start_char': start, 'end_char': end}
                               for sha, start, end in self.placements], 'hydration_sha256': self.hydration_sha256,
                'transcript_order_sha256': self.transcript_order_sha256,
                'conversation_count': self.conversation_count, 'token_count': self.token_count}


def render_threaded_sections(hydrated, order):
    """Preserve every selected span, merging only adjacent same-turn slices.

Sources keep the order of their first selected section. Within each source,
raw transcript order replaces relevance order. Only prompt-local C/T aliases,
roles, original timestamps and exact evidence text enter the rendered packet.
"""
    if type(hydrated) is not SectionRetrievalResult:
        raise TypeError('threaded rendering requires authenticated raw hydration')
    if type(order) is not TranscriptOrder:
        raise TypeError('threaded rendering requires an authenticated transcript order')
    sources, expected = {}, {}
    for section in hydrated.sections:
        for row in section.evidence:
            p = row.span
            if order.identities.get(p.turn_id) != (p.source_id, p.role, p.created_at, p.turn_text_sha256):
                raise ValueError('selected evidence differs from the bound transcript')
            if p.receipt_sha256 in expected:
                raise ValueError('selected raw span appears more than once')
            expected[p.receipt_sha256] = row
            sources.setdefault(p.source_id, []).append(row)
    pieces, placements = [], []
    length = 0
    def append(text):
        nonlocal length
        pieces.append(text)
        length += len(text)
    turn_label = 0
    for source_number, rows in enumerate(sources.values(), 1):
        rows.sort(key=lambda row: (order.positions[row.span.turn_id], row.span.start_char))
        groups = []
        for row in rows:
            if groups and groups[-1][-1].span.turn_id == row.span.turn_id:
                end = groups[-1][-1].span.end_char
                if row.span.start_char < end:
                    raise ValueError('selected raw spans overlap')
                if row.span.start_char == end:
                    groups[-1].append(row)
                    continue
            groups.append([row])
        if pieces:
            append('\n\n')
        timestamp = groups[0][0].span.created_at
        append(f'<C{source_number} {timestamp}>\n')
        for i, group in enumerate(groups):
            first = group[0].span
            if i:
                append('\n')
            if first.created_at != timestamp:
                timestamp = first.created_at
                append(f'<AT {timestamp}>\n')
            turn_label += 1
            append(f'<T{turn_label} {first.role}>\n')
            for row in group:
                start = length
                append(row.text)
                placements.append((row.span.receipt_sha256, start, length))
        append(f'\n</C{source_number}>')
    text = ''.join(pieces)
    if Counter(p[0] for p in placements) != Counter(expected.keys()):
        raise ValueError('threaded rendering changed the evidence population')
    if any(text[start:end] != expected[sha].text for sha, start, end in placements):
        raise ValueError('threaded rendering changed raw evidence bytes')
    tokens = count_tokens(text)
    if tokens > hydrated.max_context_tokens:
        raise ValueError('threaded framing exceeds the original context budget')
    return ThreadedSectionContext(text, tuple(placements), hydrated.receipt_sha256,
                                  order.receipt_sha256, len(sources), tokens)
