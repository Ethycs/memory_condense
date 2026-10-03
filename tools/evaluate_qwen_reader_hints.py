"""Replay 100 saved Qwen prompts with the established summary hints added."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
from pathlib import Path
import statistics
import time

from memory_condense.domain._tokenizer import count_tokens
from tools.engineering_research_gateway import read, save, emit
from tools.engineering_research_gateway_reader import GatewayReaderRuntime, _completion_client
from tools.summary_reader_hints import add_hints

BASE = Path('eval_results/chat-io-qwen-gateway100-20260930-r2')
PRIOR_HINTS = Path('eval_results/native-spine-lightweight-hints-20260925-r1')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(root):
    if root.exists():
        raise ValueError('Use a fresh experiment directory')
    # Reproduce every guide from the earlier treatment before using the port.
    parity = []
    for path in sorted((PRIOR_HINTS/'packets').glob('*.json')):
        old = read(path)
        source = Path(old['source']['path'])
        if not source.exists():
            source = Path('eval_results')/str(source).replace('\\', '/').split('/eval_results/', 1)[1]
        if digest(source) != old['source']['sha256']:
            raise ValueError('Prior hint source changed')
        original = read(source)
        _, guide, chosen = add_hints(original, original['messages'])
        if guide != old['guide'] or chosen != old['labels']:
            raise ValueError('Lightweight hint treatment differs from the prior trial')
        parity.append(old['key'])
    if len(parity) != 36:
        raise ValueError('Expected all 36 historical hint cases')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('.cache/models/Qwen3-8B', local_files_only=True)
    cases = []
    for path in sorted((BASE/'answers').glob('*.json')):
        row = read(path)
        started = time.perf_counter()
        messages, guide, labels = add_hints(row, row['served_messages'])
        elapsed = time.perf_counter()-started
        tokens = len(tokenizer.apply_chat_template(messages, tokenize=True,
                    add_generation_prompt=True, enable_thinking=False))
        if tokens > 7168:
            raise ValueError('Hints exceed gateway context budget')
        case = dict(ordinal=row['ordinal'], question=row['question'], messages=messages,
                    guide=guide, selected_summaries=labels, hint_tokens=count_tokens(guide),
                    preparation_s=elapsed, native_qwen_prompt_tokens=tokens,
                    source=str(path), source_sha256=digest(path))
        save(root/'packets'/path.name, case)
        cases.append(dict(ordinal=row['ordinal'], path=str(root/'packets'/path.name),
                          sha256=digest(root/'packets'/path.name)))
    if len(cases) != 100:
        raise ValueError('Expected all 100 saved questions')
    files = [Path(__file__), Path('tools/summary_reader_hints.py'),
             Path('tools/engineering_research_gateway_reader.py'), Path('tools/engineering_research_gateway.py')]
    save(root/'run-plan.json', dict(source=str(BASE), source_report_sha256=digest(BASE/'report.json'),
        gateway='https://central-dev.zt:4000/v1', reader_model='qwen3-8b',
        judge_model='codex_sdk/gpt-5.6-sol', question_count=100, cases=cases,
        answer_call_limit=100, judge_call_limit=100, hint_generation_model_calls=0,
        local_attention_calls=0, new_ingestion=False, new_retrieval=False,
        recent_chat_exactly_preserved=True, evidence_and_question_exactly_preserved=True,
        only_prompt_change='Insert the established summary-derived MEMORY_GUIDE before raw evidence',
        historical_guides_reproduced=parity, hint_policy='top six served summary labels, 16 words each, 320-token cap',
        baseline='sealed prior Qwen answers and Sol grades; no fresh unhinted generation',
        reasoning=False, max_answer_tokens=256, max_judge_tokens=128, automatic_retries=0,
        answer_concurrency=1, judge_concurrency=3, production_changes=False,
        implementation={str(p):digest(p) for p in files}))
    emit(phase='prepared', questions=100, historical_guides_reproduced=36)


def run(root):
    plan = read(root/'run-plan.json')
    if any(digest(p) != sha for p, sha in plan['implementation'].items()):
        raise ValueError('Frozen experiment implementation changed')
    if digest(BASE/'report.json') != plan['source_report_sha256']:
        raise ValueError('Baseline report changed')
    (root/'run.reserved').touch(exist_ok=False)
    runtime = GatewayReaderRuntime(root/'runtime', gateway=plan['gateway'],
        reader_model=plan['reader_model'], judge_model=plan['judge_model'])
    runtime.remote = _completion_client('LITELLM_KEY', plan['gateway']).with_options(timeout=120, max_retries=0)
    results = []
    try:
        for binding in plan['cases']:
            path = Path(binding['path'])
            if digest(path) != binding['sha256']:
                raise ValueError('Prepared prompt changed')
            case = read(path)
            if digest(case['source']) != case['source_sha256']:
                raise ValueError('Source packet changed')
            response = runtime.call('actor', case['messages'], scope=f'hinted-{case["ordinal"]:03}', max_tokens=256)
            row = dict(ordinal=case['ordinal'], response=response, prediction=response['content'],
                       question=case['question'], packet_sha256=binding['sha256'])
            save(root/'answers'/f'{case["ordinal"]:03}.json', row)
            results.append(row)
            if len(results)%10 == 0:
                emit(phase='answered', questions=len(results), mean_reader_s=statistics.mean(r['response']['elapsed_s'] for r in results))
        save(root/'answers-complete.json', dict(count=100,
            answers=[digest(root/'answers'/f'{i:03}.json') for i in range(100)]))
        # References are opened only after all answer requests are sealed.
        from tools import evaluate_chat_io_single100 as historical
        references = {r['question_id']:r for r in read(historical.SOURCE/'questions/references.json')['references']}
        def grade(row):
            ref = references[row['question']['question_id']]['answer']
            messages = historical.old.current.frozen.build_judge_prompt(row['question']['retrieval_query'], ref, row['prediction'])
            response = runtime.call('judge', messages, scope=f'grade-{row["ordinal"]:03}', max_tokens=128)
            try:
                correct = bool(historical.old.current.frozen.parse_binary_judge_verdict(response['content']))
            except ValueError:
                correct = None
            label = dict(ordinal=row['ordinal'], correct=correct, reference=ref,
                         question=row['question']['retrieval_query'], prediction=row['prediction'], judge=response)
            save(root/'grades'/f'{row["ordinal"]:03}.json', label)
            return label
        labels = []
        with ThreadPoolExecutor(max_workers=3) as pool:
            for future in as_completed([pool.submit(grade, row) for row in results]):
                labels.append(future.result())
                if len(labels)%10 == 0:
                    emit(phase='graded', questions=len(labels), correct=sum(r['correct'] is True for r in labels))
    finally:
        runtime.remote.close()
    report(root)


def report(root):
    plan = read(root/'run-plan.json')
    seal = read(root/'answers-complete.json')
    baseline = {r['ordinal']:r for r in read(BASE/'report.json')['rows']}
    pairs = []
    for i in range(100):
        path = root/'answers'/f'{i:03}.json'
        if digest(path) != seal['answers'][i]:
            raise ValueError('Sealed answer changed')
        answer = read(path)
        grade = read(root/'grades'/f'{i:03}.json')
        case = read(root/'packets'/f'{i:03}.json')
        original = read(case['source'])
        messages, guide, selected = add_hints(original, original['served_messages'])
        if messages != case['messages'] or guide != case['guide'] or selected != case['selected_summaries']:
            raise ValueError('Saved hint cannot be reproduced')
        pairs.append(dict(ordinal=i, plain_correct=baseline[i]['correct'], hinted_correct=grade['correct'],
            plain_prediction=baseline[i]['prediction'], hinted_prediction=answer['prediction'],
            plain_reader_s=baseline[i]['reader_s'], hinted_reader_s=answer['response']['elapsed_s'],
            plain_input_tokens=original['result']['response']['usage']['prompt_tokens'],
            hinted_input_tokens=answer['response']['usage']['prompt_tokens'],
            finish_reason=answer['response']['finish_reason'], hint_tokens=case['hint_tokens']))
    wins = [r['ordinal'] for r in pairs if r['plain_correct'] is False and r['hinted_correct'] is True]
    losses = [r['ordinal'] for r in pairs if r['plain_correct'] is True and r['hinted_correct'] is False]
    payload = dict(questions=100, plain_correct=sum(r['plain_correct'] is True for r in pairs),
        hinted_correct=sum(r['hinted_correct'] is True for r in pairs),
        invalid_grades=sum(r['hinted_correct'] is None for r in pairs), recoveries=wins, regressions=losses,
        plain_abstentions=sum(r['plain_prediction'].strip().lower().rstrip('.')=="i don't know" for r in pairs),
        hinted_abstentions=sum(r['hinted_prediction'].strip().lower().rstrip('.')=="i don't know" for r in pairs),
        mean_hint_tokens=statistics.mean(r['hint_tokens'] for r in pairs),
        mean_plain_reader_s=statistics.mean(r['plain_reader_s'] for r in pairs),
        mean_hinted_reader_s=statistics.mean(r['hinted_reader_s'] for r in pairs),
        mean_plain_input_tokens=statistics.mean(r['plain_input_tokens'] for r in pairs),
        mean_hinted_input_tokens=statistics.mean(r['hinted_input_tokens'] for r in pairs),
        evidence_recent_context_and_question_unchanged=True, historical_hint_parity_cases=36,
        new_ingestion=False, hint_generation_model_calls=0, production_changes=False,
        latency_caveat='Sequential reader-only replay versus prior live pipeline reader timing; no new full-cycle latency measurement',
        baseline_is_historical_not_fresh=True, rows=pairs)
    save(root/'report.json', payload)
    emit(phase='complete', plain_correct=payload['plain_correct'], hinted_correct=payload['hinted_correct'],
         recoveries=len(wins), regressions=len(losses), hinted_abstentions=payload['hinted_abstentions'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('prepare','run','report'))
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    globals()[args.phase](args.root)
