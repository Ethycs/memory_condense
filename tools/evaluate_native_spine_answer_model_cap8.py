"""Compare one other gateway answer model on the unchanged 100 cap-8 prompts.

Uses the same sealed-packet protocol as the reader comparison. No ingestion,
retrieval, prompt, output budget, reference or grader changes are allowed.
The first question checks transport before the remaining bounded batch starts.
"""
import argparse
from contextlib import closing
from pathlib import Path
import statistics

from memory_condense.domain._discourse_identity import identity_sha256, quote_sha256
from memory_condense.eval import spine_reader_policy_v7 as reader
from tools.run_native_spine_batches import bounded_dispatch
from memory_condense.eval.streaming_latency import measure_streaming_answer
from tools import run_native_spine_user_completion_answers as previous
from tools.matched_eval.artifacts import read_sealed_json


ROOT = Path('eval_results/native-spine-answer-model-cap8-20260924-r2')
MODEL = 'claude_code/claude-opus-4-7'
BASE = Path('eval_results/native-spine-completion-single100-20260923-r1')
frozen = previous.current.frozen
publish, bound, binding = previous.publish, previous.bound, previous.binding


def implementation():
    return {str(p): previous.digest(p) for p in (
        __file__, reader.__file__, frozen.__file__,
        'src/memory_condense/eval/streaming_latency.py')}


def validate_model(model, inventory):
    if (model != MODEL or model == previous.MODEL or model not in inventory['model_ids']
            or inventory['gateway'] != frozen.GATEWAY):
        raise ValueError('requires the selected alternative answer model on the authorized gateway')


def changed_messages(source):
    if identity_sha256(source['messages']) != source['measurement']['messages_sha256']:
        raise ValueError('baseline messages differ from its actual answer request')
    if source['messages'][0] != {'role': 'system', 'content': reader.SPINE_READER_SYSTEM_PROMPT_V7}:
        raise ValueError('model comparison requires the unchanged v7 reader')
    return [dict(message) for message in source['messages']]


def prepare(root):
    if (root / 'preflight.json').exists():
        raise ValueError('prepare requires a fresh preflight')
    inventory = read_sealed_json(root / 'inventory.json')
    validate_model(MODEL, inventory.payload)
    baseline = read_sealed_json(BASE / 'history-01/report.json')
    old_plan = bound(baseline.payload['run'])
    answers = bound(baseline.payload['answers'])
    questions = bound(old_plan.payload['questions'])
    cases = questions.payload['questions']
    scope = bound(old_plan.payload['scope'])
    if (len(cases) != 100 or len(answers.payload['answers']) != 100
            or scope.payload['through_question_day_body_tokens'] < 1_000_000
            or old_plan.payload['model'] != previous.MODEL
            or old_plan.payload['max_tokens'] != 256):
        raise ValueError('requires the complete measured cap-8 million-token history')
    packets = []
    for i, source_binding in enumerate(answers.payload['answers']):
        source = bound(source_binding)
        if source.payload['question'] != frozen.question(cases[i]):
            raise ValueError('baseline question order changed')
        packet = publish(root / 'packets' / f'{i:03d}.json', {
            'ordinal': i, 'source': source_binding, 'question': source.payload['question'],
            'messages': changed_messages(source.payload)})
        packets.append(binding(packet))
    plan = publish(root / 'preflight.json', {
        'implementation': implementation(), 'baseline': binding(baseline),
        'baseline_answers': binding(answers), 'questions': binding(questions),
        'scope': binding(scope), 'packets': packets, 'model': MODEL,
        'model_inventory': binding(inventory), 'judge_model': previous.MODEL,
        'model_only_comparison': True,
        'max_tokens': 256, 'answer_concurrency': 4, 'question_count': 100,
        'history_count': 1, 'body_tokens': scope.payload['actual_body_tokens'],
        'context_policy': old_plan.payload['context_policy'],
        'reader_prompt_sha256': quote_sha256(reader.SPINE_READER_SYSTEM_PROMPT_V7),
        'new_ingestion': False, 'new_qwen_calls': 0, 'fresh_retrieval': False,
        'warm_latency_measured': False, 'automatic_retries': 0,
        'references_opened_by_answer_worker': False,
        'evidence_preserved_from_audited_baseline': True,
        'development_set_only': True, 'replaces_campaign_score': False})
    previous.emit(phase='prepared', questions=100, sha256=plan.sha256)


def inputs(root):
    plan = read_sealed_json(root / 'preflight.json')
    if plan.payload['implementation'] != implementation():
        raise ValueError('model experiment implementation changed')
    validate_model(plan.payload['model'], bound(plan.payload['model_inventory']).payload)
    if plan.payload['judge_model'] != previous.MODEL or plan.payload['model_only_comparison'] is not True:
        raise ValueError('model comparison changed the grader or boundary')
    packets = [bound(b) for b in plan.payload['packets']]
    sources = bound(plan.payload['baseline_answers']).payload['answers']
    if len(packets) != 100 or [p.payload['ordinal'] for p in packets] != list(range(100)):
        raise ValueError('model comparison requires all 100 questions')
    for packet, source_binding in zip(packets, sources, strict=True):
        source = bound(source_binding)
        if (packet.payload['source'] != source_binding
                or packet.payload['question'] != source.payload['question']
                or packet.payload['messages'] != changed_messages(source.payload)):
            raise ValueError('model comparison changed prompt, evidence or question')
    return plan, packets


def response(root, plan, packet):
    result = read_sealed_json(root / 'answers' / f'{packet.payload["ordinal"]:03d}.json')
    m = result.payload['measurement']
    if (result.payload['preflight'] != binding(plan) or result.payload['packet'] != binding(packet)
            or m['messages_sha256'] != identity_sha256(packet.payload['messages'])
            or m['model'] != plan.payload['model'] or m['max_tokens'] != 256
            or m['prediction_sha256'] != quote_sha256(m['prediction'])):
        raise ValueError('answer request or response differs from the sealed plan')
    return result


def run(root, enable):
    if not enable:
        raise ValueError('fresh answers require --enable-provider')
    plan, packets = inputs(root)
    previous.require_idle()
    def answer(packet):
        path = root / 'answers' / f'{packet.payload["ordinal"]:03d}.json'
        if path.exists():
            return response(root, plan, packet)
        with closing(frozen._completion_client('LITELLM_KEY', frozen.GATEWAY)) as client:
            measured = measure_streaming_answer(client=client, model=plan.payload['model'],
                prepare_prompt=lambda: packet.payload['messages'], max_tokens=256)
        publish(path, {'preflight': binding(plan), 'packet': binding(packet), 'measurement': measured})
        return response(root, plan, packet)
    first = answer(packets[0])
    if first.payload['measurement']['finish_reason'] != 'stop':
        raise ValueError('first answer did not fit the unchanged output budget; batch stopped')
    previous.emit(phase='first_answer_complete', model=plan.payload['model'],
        response_model=first.payload['measurement']['response_model'], completed=1)
    def one(i):
        result = answer(packets[i])
        previous.emit(phase='answer_complete', ordinal=i, finish=result.payload['measurement']['finish_reason'])
        return result.sha256
    _, failures = bounded_dispatch(range(1, 100), one, workers=4)
    if failures:
        publish(root / 'batch-failures.json', {'preflight': binding(plan), 'failures': {str(k): v for k,v in failures.items()}})
        raise RuntimeError('answer batch stopped on provider failure; completed answers retained')
    seal_answers(root, plan, packets)


def seal_answers(root, plan, packets):
    responses = [response(root, plan, packet) for packet in packets]
    seal = publish(root / 'answers-complete.json', {
        'preflight': binding(plan), 'answers': [binding(r) for r in responses]})
    return seal, responses


def report(root, enable):
    plan, packets = inputs(root)
    seal, responses = seal_answers(root, plan, packets)
    questions = bound(plan.payload['questions'])
    refs = previous.rebased(questions.payload['references'])
    reference_map = {r['question_id']: r for r in refs.payload['references']}
    scope = bound(plan.payload['scope'])
    if (len(reference_map) != 100 or refs.payload['scope_sha256'] != scope.sha256
            or refs.payload['ingest_use_permitted'] is not False):
        raise ValueError('evaluation reference population changed')
    judge_rows = []
    for packet, answer, case in zip(packets, responses, questions.payload['questions'], strict=True):
        q = packet.payload['question']
        ref = reference_map[q['question_id']]
        if quote_sha256(ref['answer']) != case['reference_sha256']:
            raise ValueError('reference answer changed')
        judge_rows.append({'ordinal': q['ordinal'], 'messages': frozen.build_judge_prompt(
            q['retrieval_query'], ref['answer'], answer.payload['measurement']['prediction'])})
    judge = publish(root / 'judge-preflight.json', {'answers': binding(seal),
        'references': binding(refs), 'rows': judge_rows, 'model': previous.MODEL})
    def factory(client):
        return frozen.FastCompletionRuntime(checkpoint_dir=root / 'judge-checkpoints',
            prompt_population=[r['messages'] for r in judge_rows], model=previous.MODEL, client=client,
            max_prompt_tokens=4096, max_new_tokens=frozen.JUDGE_MAX_TOKENS,
            max_concurrency=8, retries=0, request_options={'temperature': 0},
            benchmark_provenance={'binding_sha256': judge.sha256, 'phase': 'judge'})
    with closing(factory(None)) as runtime:
        remaining = runtime.population.unique_prompt_count - len(frozen._authenticated_records(runtime))
    previous.emit(phase='judge_starting', new_calls=remaining, questions=100)
    batch, _, _, _ = frozen._run_exactly_authorized(runtime_factory=factory,
        authorized_provider_calls=remaining, enable_provider=enable,
        client_factory=lambda: frozen.ThreadLocalProvider(
            lambda: frozen._completion_client('LITELLM_KEY', frozen.GATEWAY)))
    baseline = bound(plan.payload['baseline'])
    rows = []
    for packet, answer, verdict, old in zip(packets, responses, batch.logical_completions,
                                          baseline.payload['rows'], strict=True):
        m = answer.payload['measurement']
        if packet.payload['ordinal'] != old['ordinal']:
            raise ValueError('baseline grades changed order')
        rows.append({'ordinal': old['ordinal'], 'correct': bool(frozen.parse_binary_judge_verdict(verdict)),
            'baseline_correct': old['correct'], 'prediction': m['prediction'],
            'baseline_prediction': old['prediction'], 'question': old['question'],
            'reference': old['reference'], 'finish_reason': m['finish_reason'],
            'prompt_tokens': (m['usage'] or {}).get('prompt_tokens'),
            'baseline_prompt_tokens': old['prompt_tokens'], 'verdict': verdict})
    reported = [r['prompt_tokens'] for r in rows]
    summary = {'correct': sum(r['correct'] for r in rows), 'questions': 100,
        'baseline_correct': sum(r['baseline_correct'] for r in rows),
        'improvements': [r['ordinal'] for r in rows if r['correct'] and not r['baseline_correct']],
        'regressions': [r['ordinal'] for r in rows if not r['correct'] and r['baseline_correct']],
        'identical_prediction_grade_flips': [r['ordinal'] for r in rows
            if r['prediction'] == r['baseline_prediction'] and r['correct'] != r['baseline_correct']],
        'reported_prompt_usage_available': sum(v is not None for v in reported),
        'mean_prompt_tokens': statistics.fmean(reported) if all(v is not None for v in reported) else None,
        'baseline_mean_prompt_tokens': statistics.fmean(r['baseline_prompt_tokens'] for r in rows),
        'all_answers_stopped': all(r['finish_reason'] == 'stop' for r in rows)}
    result = publish(root / 'report.json', {'preflight': binding(plan), 'answers': binding(seal),
        'judge': binding(judge), 'baseline': binding(baseline), 'summary': summary, 'rows': rows,
        'judge_response_journal_shas': [r.response_journal_sha256 for r in batch.unique_records],
        'evidence_and_questions_identical': True, 'new_qwen_calls': 0,
        'warm_latency_measured': False, 'development_set_only': True,
        'model': plan.payload['model'], 'baseline_model': previous.MODEL,
        'limitation': 'One exposed history; historical Sol baseline; unchanged noisy binary grader. Gateway aliases do not independently establish backend checkpoint identity.'})
    previous.emit(phase='report_complete', **summary, sha256=result.sha256)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('prepare', 'run', 'report'))
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--enable-provider', action='store_true')
    args = parser.parse_args()
    if args.phase == 'prepare':
        prepare(args.root)
    else:
        {'run': run, 'report': report}[args.phase](args.root, args.enable_provider)
