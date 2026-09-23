"""User-first layout with timestamp state preserved across role blocks."""
from collections import Counter
from dataclasses import dataclass

from memory_condense.application.threaded_section_context import (
    ThreadedSectionContext, TranscriptOrder, render_threaded_sections,
)
from memory_condense.domain._tokenizer import count_tokens


FORMAT = 'conversation-user-spine-then-context-v2'


@dataclass(frozen=True)
class UserSpineSectionContext(ThreadedSectionContext):
    def identity_payload(self):
        return {**super().identity_payload(), 'format': FORMAT}


def render_user_spine_sections(hydrated, order):
    """Change layout only; retain exact spans and their chronological T labels.

    Each conversation keeps its original selected user statements together in
    transcript order. Other selected roles follow in their own transcript order.
    A lower T number still means an earlier turn, even across the two blocks.
    No question, summaries, relevance scoring or provider call enters rendering.
    """
    # Reuse the established identity, overlap, duplicate and framing checks.
    original = render_threaded_sections(hydrated, order)
    sources, expected = {}, {}
    for section in hydrated.sections:
        for row in section.evidence:
            sources.setdefault(row.span.source_id, []).append(row)
            expected[row.span.receipt_sha256] = row.text
    pieces, placements = [], []
    length = 0
    def append(text):
        nonlocal length
        pieces.append(text)
        length += len(text)
    turn_number = 0
    for source_number, rows in enumerate(sources.values(), 1):
        rows.sort(key=lambda r: (order.positions[r.span.turn_id], r.span.start_char))
        groups = []
        for row in rows:
            if (groups and groups[-1][-1].span.turn_id == row.span.turn_id
                    and groups[-1][-1].span.end_char == row.span.start_char):
                groups[-1].append(row)
            else:
                groups.append([row])
        labelled = [(turn_number + i + 1, group) for i, group in enumerate(groups)]
        turn_number += len(groups)
        if pieces:
            append('\n\n')
        source_time = groups[0][0].span.created_at
        append(f'<C{source_number} {source_time}>\n')
        timestamp = source_time
        for user_phase, tag in ((True, 'USER_STATEMENTS'), (False, 'OTHER_TURNS')):
            selected = [(label, group) for label, group in labelled
                        if (group[0].span.role == 'user') == user_phase]
            if not selected:
                continue
            append(f'<{tag}>\n')
            for label, group in selected:
                first = group[0].span
                if first.created_at != timestamp:
                    timestamp = first.created_at
                    append(f'<AT {timestamp}>\n')
                append(f'<T{label} {first.role}>\n')
                for row in group:
                    start = length
                    append(row.text)
                    placements.append((row.span.receipt_sha256, start, length))
                append('\n')
            append(f'</{tag}>\n')
        append(f'</C{source_number}>')
    text = ''.join(pieces)
    if (Counter(sha for sha, _, _ in placements) != Counter(expected.keys())
            or any(text[start:end] != expected[sha] for sha, start, end in placements)):
        raise ValueError('user-spine presentation changed the exact evidence population')
    tokens = count_tokens(text)
    if tokens > hydrated.max_context_tokens:
        raise ValueError('user-spine framing exceeds the original context budget')
    return UserSpineSectionContext(text, tuple(placements), hydrated.receipt_sha256,
                                  order.receipt_sha256, original.conversation_count, tokens)
