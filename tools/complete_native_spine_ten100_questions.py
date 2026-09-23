"""Seal ten-session question journals with source-exact evidence formatting repairs.

No questions or answers change. A repair must identify one unique source slice
in the original attributed turn. All corrections are recorded before retrieval.
"""
import argparse
import json
from pathlib import Path
import re

from tools import native_spine_ten100 as campaign


def exact_supports(value, source):
    result = {**value, 'supports': [dict(s) for s in value['supports']]}
    corrections = []
    turns = {t['turn_index']: t['text'] for t in source['user_turns']}
    for support in result['supports']:
        text = turns[support['turn_index']]
        quote = support['quote']
        if quote in text:
            continue
        matches = list(re.finditer(re.escape(quote), text, flags=re.IGNORECASE))
        if len(matches) != 1:
            raise ValueError('quote mismatch is not a unique case-only source match')
        match, = matches
        support['quote'] = text[match.start():match.end()]
        corrections.append({'turn_index': support['turn_index'], 'original_quote': quote,
            'exact_quote': support['quote'], 'start_char': match.start(), 'end_char': match.end(),
            'rule': 'unique case-insensitive literal match in the same attributed source turn'})
    campaign.authoring.validate_authored(result, source)
    return result, corrections


def merge_same_turn_supports(value, source):
    """Preserve every quote while coalescing duplicate-turn citations to <=3 spans."""
    if len(value['supports']) <= 3:
        return value, []
    turns = {t['turn_index']: t['text'] for t in source['user_turns']}
    groups = {}
    for support in value['supports']:
        groups.setdefault(support['turn_index'], []).append(support['quote'])
    if len(groups) > 3:
        raise ValueError('cannot merge citations across different source turns')
    result, corrections = {**value, 'supports': []}, []
    for index, quotes in groups.items():
        text = turns[index]
        if any(not q or text.count(q) != 1 for q in quotes):
            raise ValueError('merged citations require unique exact source matches')
        start = min(text.index(q) for q in quotes)
        end = max(text.index(q) + len(q) for q in quotes)
        exact = text[start:end]
        result['supports'].append({'turn_index': index, 'quote': exact})
        if len(quotes) > 1:
            corrections.append({'turn_index': index, 'original_quotes': quotes,
                'exact_quote': exact, 'start_char': start, 'end_char': end,
                'rule': 'minimal contiguous same-turn span containing every original quote'})
    return result, corrections


def complete(root, batch):
    plan, scope, folder = campaign.history(root, batch)
    target = folder / 'questions'
    if (target / 'questions.json').exists():
        existing = campaign.read_sealed_json(target / 'questions.json')
        campaign.current.baseline.validate_population(existing, scope)
        campaign.emit(phase='questions_already_locked', history=batch)
        return
    preflight = campaign.read_sealed_json(target / 'authoring-preflight.json')
    questions, references, repairs = [], [], []
    for i, source in enumerate(scope.payload['author_sources']):
        prefix = target / 'journal' / f'{i:03d}'
        request = campaign.read_sealed_json(prefix.with_suffix('.request.json'))
        response = campaign.read_sealed_json(prefix.with_suffix('.response.json'))
        if (request.payload['preflight_sha256'] != preflight.sha256
                or request.payload['source_sha256'] != source['body_sha256']
                or response.payload['request_sha256'] != request.sha256
                or response.payload['finish_reason'] != 'stop'):
            raise ValueError('authoring request/response binding changed')
        content = response.payload['content'].strip()
        value, _ = json.JSONDecoder().raw_decode(content[content.index('{'):])
        value, merged = merge_same_turn_supports(value, source)
        value, changes = exact_supports(value, source)
        changes = merged + changes
        repairs.extend({'ordinal': i, **change} for change in changes)
        qid = f'ten100-h{batch:02d}-q{i:03d}'
        questions.append({**scope.payload['case'], 'ordinal': i, 'question_id': qid,
            'question': value['question'], 'category': value['category'],
            'reference_sha256': campaign.quote_sha256(value['answer']),
            'question_origin': 'new source-grounded generated evaluation'})
        references.append({'question_id': qid, 'answer': value['answer'], 'source': source,
            'supports': value['supports'], 'author_response': campaign.binding(response)})
    correction = campaign.publish(target / 'quote-corrections.json', {
        'implementation_sha256': campaign.digest(__file__), 'corrections': repairs,
        'questions_unchanged': True, 'reference_answers_unchanged': True,
        'candidate_outputs_loaded': False, 'new_model_calls': 0})
    refs = campaign.publish(target / 'references.json', {'scope_sha256': scope.sha256,
        'evaluation_only': True, 'ingest_use_permitted': False, 'references': references,
        'quote_corrections': campaign.binding(correction)})
    result = campaign.publish(target / 'questions.json', {'scope': campaign.binding(scope),
        'references': campaign.binding(refs), 'authoring_preflight': campaign.binding(preflight),
        'history_count': 1, 'question_count': 100, 'actual_body_tokens': scope.payload['actual_body_tokens'],
        'questions': questions, 'quote_corrections': campaign.binding(correction)})
    campaign.current.baseline.validate_population(result, scope)
    campaign.emit(phase='questions_locked', history=batch, source_quote_corrections=len(repairs), sha256=result.sha256)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--batch', type=int, required=True, choices=range(1, 11))
    args = parser.parse_args()
    complete(args.root, args.batch)
