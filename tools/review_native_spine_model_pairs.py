"""Source-grounded diagnostic of every changed answer pair, without rescoring.

Use the existing source-review schema and exact-quote validator. A/B order is
fixed by a salted hash; model names and original grades are withheld. Unchanged
answers rejected in either run are included too. This is a selected diagnostic,
not a replacement accuracy score or an independent human adjudication.
"""
import argparse
from collections import Counter
from contextlib import closing
import json
from pathlib import Path
import sqlite3

from memory_condense.domain._discourse_identity import identity_sha256, quote_sha256
from memory_condense.domain._tokenizer import count_chat_prompt_token_proxy
from tools import audit_native_spine_source_answers as source_review
from tools import evaluate_native_spine_answer_model_cap8 as comparison


ROOT = comparison.ROOT / 'source-review'
MODEL = 'codex_sdk/gpt-5.6-terra'
SYSTEM = source_review.SYSTEM + '''

This request contains TWO saved answers to the SAME question, labeled A and B.
Apply the rules above independently and equally to both answers using the same
source evidence. Do not prefer length, prose style, agreement with the reference,
or one position. The supplied predictions object maps labels to answer strings.
Return exactly {"A": <review object with the schema above>, "B": <review object
with the schema above>}. Each prediction_quote must come from that label's own
answer. Include exact supporting source quotes for each review. Do not return
an overall winner or a revised score.'''
publish, bound, binding = comparison.publish, comparison.bound, comparison.binding
frozen = comparison.frozen


def implementation():
    return {str(p): comparison.previous.digest(p) for p in
            (__file__, source_review.__file__, comparison.__file__)}


def prepare(root, comparison_root):
    report = comparison.read_sealed_json(comparison_root / 'report.json')
    plan, packets = comparison.inputs(comparison_root)
    if report.payload['preflight'] != binding(plan):
        raise ValueError('source review requires the completed model comparison')
    answers = bound(report.payload['answers'])
    if answers.payload['preflight'] != binding(plan) or len(answers.payload['answers']) != 100:
        raise ValueError('source review requires the complete answer population')
    questions = bound(plan.payload['questions'])
    refs = comparison.previous.rebased(questions.payload['references'])
    references = {r['question_id']: r for r in refs.payload['references']}
    scope = bound(plan.payload['scope'])
    bank = comparison.previous.relocate(scope.payload['body_bank']['path'])
    if comparison.previous.digest(bank) != scope.payload['body_bank']['sha256']:
        raise ValueError('raw source bank changed')
    inventory = bound(plan.payload['model_inventory'])
    if MODEL not in inventory.payload['model_ids']:
        raise ValueError('review model is not in the gateway inventory')
    items = []
    with closing(sqlite3.connect(bank.as_uri() + '?mode=ro', uri=True)) as raw:
        for row, packet, case in zip(report.payload['rows'], packets, questions.payload['questions'], strict=True):
            if (row['prediction'] == row['baseline_prediction']
                    and row['correct'] and row['baseline_correct']):
                continue
            ref = references[case['question_id']]
            if quote_sha256(ref['answer']) != case['reference_sha256']:
                raise ValueError('reference answer changed')
            body = json.loads(raw.execute('SELECT body_json FROM bodies WHERE body_sha256=?',
                (ref['source']['body_sha256'],)).fetchone()[0])
            users = [{'turn_index': i, 'text': t['text']} for i, t in enumerate(body['turns']) if t['role'] == 'user']
            if users != ref['source']['user_turns']:
                raise ValueError('reference source differs from raw transcript')
            original = bound(packet.payload['source'])
            sources = [{'source_id': 'served-context', 'text': original.payload['rendered']['text'],
                        'origin': 'exact rendered context delivered to both answer models; roles labeled inside'}]
            sources.extend({'source_id': f'reference-user-{t["turn_index"]}', 'text': t['text'],
                            'role': 'user', 'origin': 'reference conversation user turn; may be unserved',
                            'created_at': ref['source']['source']['created_at']} for t in users)
            labels = ('baseline', 'candidate')
            if int(identity_sha256(['cap8-model-source-review-v1', row['ordinal']])[:8], 16) % 2:
                labels = labels[::-1]
            assignment = dict(zip(('A', 'B'), labels, strict=True))
            predictions = {label: row['baseline_prediction' if arm == 'baseline' else 'prediction']
                           for label, arm in assignment.items()}
            data = {'question': packet.payload['question']['prompt_question'], 'reference': ref['answer'],
                    'predictions': predictions, 'sources': sources}
            messages = [{'role': 'system', 'content': SYSTEM},
                        {'role': 'user', 'content': json.dumps(data, ensure_ascii=False)}]
            if count_chat_prompt_token_proxy(messages) > 16384:
                raise ValueError('whole paired source review exceeds prompt limit')
            items.append({'ordinal': row['ordinal'], 'assignment': assignment, 'data': data,
                'messages': messages, 'packet': binding(packet),
                'candidate_answer': answers.payload['answers'][row['ordinal']]})
    artifact = publish(root / 'preflight.json', {'comparison': binding(report), 'implementation': implementation(),
        'references': binding(refs), 'source_bank': scope.payload['body_bank'], 'model': MODEL,
        'items': items, 'selected_pairs': len(items), 'answer_population': 100,
        'selection': 'every nonidentical prediction pair plus any unchanged answer failing either original grade',
        'model_and_grades_withheld': True, 'label_order': 'fixed salted hash per question',
        'max_new_tokens': 2048, 'concurrency': 6, 'automatic_retries': 0,
        'diagnostic_only': True, 'original_scores_changed': False})
    comparison.previous.emit(phase='source_review_prepared', pairs=len(items), sha256=artifact.sha256)
    return artifact


def validate_pair(text, item):
    content = text.strip()
    if content.startswith('```json\n') and content.endswith('\n```'):
        content = content[8:-4]
    def unique(pairs):
        obj = {}
        for k, v in pairs:
            if k in obj:
                raise ValueError('duplicate paired review field')
            obj[k] = v
        return obj
    pair = json.loads(content, object_pairs_hook=unique)
    if type(pair) is not dict or set(pair) != {'A', 'B'}:
        raise ValueError('paired review requires exactly both labels')
    data = item['data']
    return {label: source_review.validate_review(json.dumps(pair[label], ensure_ascii=False),
        {'prediction': data['predictions'][label], 'reference': data['reference'], 'sources': data['sources']})
        for label in ('A', 'B')}


def run(root, comparison_root, enable):
    plan = prepare(root, comparison_root)
    items = plan.payload['items']
    def factory(client):
        return frozen.FastCompletionRuntime(checkpoint_dir=root / 'checkpoints',
            prompt_population=[i['messages'] for i in items], model=MODEL, client=client,
            max_prompt_tokens=16384, max_new_tokens=2048, max_concurrency=6, retries=0,
            request_options={'temperature': 0},
            benchmark_provenance={'binding_sha256': plan.sha256, 'phase': 'paired_source_review'})
    with closing(factory(None)) as runtime:
        remaining = runtime.population.unique_prompt_count - len(frozen._authenticated_records(runtime))
    comparison.previous.emit(phase='source_review_starting', pairs=len(items), new_calls=remaining)
    batch, calls, hits, _ = frozen._run_exactly_authorized(runtime_factory=factory,
        authorized_provider_calls=remaining, enable_provider=enable,
        client_factory=lambda: frozen.ThreadLocalProvider(
            lambda: frozen._completion_client('LITELLM_KEY', frozen.GATEWAY)))
    rows = []
    for item, text in zip(items, batch.logical_completions, strict=True):
        try:
            reviews = validate_pair(text, item)
            rows.append({'ordinal': item['ordinal'], 'reviews': {item['assignment'][label]: value
                         for label, value in reviews.items()}, 'validation_error': None})
        except (ValueError, TypeError, KeyError) as error:
            rows.append({'ordinal': item['ordinal'], 'reviews': None,
                         'validation_error': str(error), 'completion': text})
    counts = {arm: dict(Counter(r['reviews'][arm]['verdict'] if r['reviews'] else 'invalid_pair'
                              for r in rows)) for arm in ('baseline', 'candidate')}
    result = publish(root / 'report.json', {'preflight': binding(plan), 'rows': rows, 'counts': counts,
        'selected_pairs': len(items), 'original_scores_changed': False,
        'diagnostic_only': True, 'new_answer_calls': 0, 'human_adjudication_required': True,
        'response_journal_shas': [r.response_journal_sha256 for r in batch.unique_records]})
    comparison.previous.emit(phase='source_review_complete', counts=counts, pairs=len(items),
                             new_review_calls=calls, cache_hits=hits, sha256=result.sha256)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--comparison-root', type=Path, default=comparison.ROOT)
    parser.add_argument('--enable-provider', action='store_true')
    args = parser.parse_args()
    run(args.root, args.comparison_root, args.enable_provider)
