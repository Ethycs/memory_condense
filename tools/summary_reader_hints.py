"""The frozen lightweight summary-navigation treatment, without DSPy imports."""
from __future__ import annotations

import re

from memory_condense.domain._discourse_identity import quote_sha256
from memory_condense.domain._tokenizer import count_tokens

STOP = set('a an the and or of to in on at for from with by as i me my we our you your it its did do does was were is are be been have has had what which who when where how why can could would should say said ask asked want wanted question'.split())
GUIDE_HEAD = ('<MEMORY_GUIDE>\nPartial navigation from stored summaries, not evidence. Dates are recording dates, not event dates. '
              'Read relevant follow-ups, including unlisted turns; retain counts and qualifications. Requests, tentative plans, '
              'completed actions and assistant suggestions are distinct. Verify every claim in the excerpts.\n')
GUIDE_TAIL = '</MEMORY_GUIDE>\n\n'


def words(text):
    return set(re.findall(r'[a-z0-9]+', text.lower())) - STOP


def served_summary_labels(packet):
    """Authenticate raw placement; return only summaries, dates, roles and IDs."""
    context = packet['rendered']['text']
    spans = {}
    for section in packet['hydration']['sections']:
        for evidence in section['evidence']:
            sha = evidence['span']['receipt_sha256']
            if sha in spans:
                raise ValueError('duplicate hydration span')
            spans[sha] = (evidence, section['section']['summary'])
    rows, prior_end, prior_turn = [], None, None
    for placement in packet['rendered']['placements']:
        evidence, summary = spans[placement['span_sha256']]
        span = evidence['span']
        start, end = placement['start_char'], placement['end_char']
        if context[start:end] != evidence['text'] or quote_sha256(evidence['text']) != span['span_text_sha256']:
            raise ValueError('rendered evidence changed')
        if prior_end == start:
            if span['turn_id'] != prior_turn:
                raise ValueError('adjacent text changed turn identity')
            if summary not in rows[-1]['about']:
                rows[-1]['about'].append(summary)
        else:
            header = re.search(r'<T(\d+) ([^>\n]+)>\n$', context[:start])
            if not header or header[2] != span['role']:
                raise ValueError('missing authenticated turn marker')
            rows.append(dict(turn='T'+header[1], recorded_at=span['created_at'],
                             role=span['role'], about=[summary]))
        prior_end, prior_turn = end, span['turn_id']
    return rows


def lightweight_guide(question, labels):
    """Same selection, ordering, verbatim prefixes and cap as the Sep 25 trial."""
    query = words(question)
    candidates = []
    for index, row in enumerate(labels):
        for summary in row['about']:
            candidates.append((len(query & words(summary)), index, summary, row))
    candidates.sort(key=lambda x: (-x[0], x[1], x[2]))
    chosen, seen = [], set()
    for score, index, summary, row in candidates:
        if row['turn'] in seen:
            continue
        parts = summary.split()
        label = ' '.join(parts[:16]) + (' ...' if len(parts) > 16 else '')
        line = f"{row['turn']} | {row['recorded_at'][:10]} | {row['role']} | {label}"
        candidate = chosen + [dict(index=index, line=line, summary=summary, score=score, metadata=row)]
        guide = GUIDE_HEAD + '\n'.join(c['line'] for c in sorted(candidate, key=lambda c: c['index'])) + '\n' + GUIDE_TAIL
        if count_tokens(guide) > 320:
            continue
        chosen = candidate
        seen.add(row['turn'])
        if len(chosen) == 6:
            break
    chosen.sort(key=lambda c: c['index'])
    guide = GUIDE_HEAD + '\n'.join(c['line'] for c in chosen) + '\n' + GUIDE_TAIL
    if not chosen or count_tokens(guide) > 320:
        raise ValueError('No valid bounded summary guide')
    return guide, chosen


def add_hints(packet, plain):
    labels = served_summary_labels(packet)
    guide, selected = lightweight_guide(packet['question']['retrieval_query'], labels)
    context = packet['rendered']['text']
    hinted = [dict(m) for m in plain]
    if hinted[-1]['content'].count(context) != 1:
        raise ValueError('Evidence must occur exactly once in the reader prompt')
    hinted[-1]['content'] = hinted[-1]['content'].replace(context, guide+context, 1)
    restored = [dict(m) for m in hinted]
    restored[-1]['content'] = restored[-1]['content'].replace(guide, '', 1)
    if restored != plain:
        raise ValueError('Hints changed existing instructions, evidence or recent context')
    return hinted, guide, selected
