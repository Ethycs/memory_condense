"""Answer and grade the two missing cells of the routing/presentation comparison."""
import argparse
from contextlib import closing
from pathlib import Path
import statistics

from memory_condense.domain._discourse_identity import identity_sha256, quote_sha256
from memory_condense.domain._tokenizer import count_chat_prompt_token_proxy
from tools import assess_native_spine_downstream_compensation as assessment
from tools import evaluate_native_spine_heuristic_ablation as prior
from tools.matched_eval.artifacts import read_sealed_json

current = prior.current

# The first evaluator assumed non-streaming provider usage was available.
# Accept that exact archived answer runner for report-only nullable-usage repair;
# prompts, answers, grading and checkpoint identities are unchanged.
NULLABLE_USAGE_PREDECESSOR = '3fc5b1ac60073befd3f9f53c1a604105744ba66a2c002a34a9c562047ecdfcf1'


def token_statistics(rows):
    reported = [r['reported_prompt_tokens'] for r in rows]
    return {'mean_reported_prompt_tokens': statistics.fmean(reported) if all(v is not None for v in reported) else None,
        'reported_prompt_tokens_available': sum(v is not None for v in reported),
        'mean_prompt_token_proxy': statistics.fmean(r['prompt_token_proxy'] for r in rows),
        'mean_projected_prompt_token_proxy': statistics.fmean(r['projected_prompt_token_proxy'] for r in rows)}


def inputs(root):
    plan = read_sealed_json(root / 'preflight.json')
    if plan.payload['runner_sha256'] != prior.digest(assessment.__file__):
        raise ValueError('packet preparation implementation changed')
    packets = read_sealed_json(root / 'packets.json')
    if packets.payload['preflight'] != prior.binding(plan):
        raise ValueError('packets belong to another experiment')
    rows = [prior.bound(p) for p in packets.payload['packets']]
    if len(rows) != 200:
        raise ValueError('requires exactly two arms of the same 100 questions')
    for arm in assessment.ARMS:
        if sorted(r.payload['ordinal'] for r in rows if r.payload['arm'] == arm) != list(range(100)):
            raise ValueError('question population changed')
    return plan, packets, rows


def execute(root, phase, messages, model, binding_sha, *, enable):
    def factory(client):
        return current.frozen.FastCompletionRuntime(checkpoint_dir=root / f'{phase}-checkpoints',
            prompt_population=messages, model=model, client=client,
            max_prompt_tokens=4096,
            max_new_tokens=256 if phase == 'answer' else current.frozen.JUDGE_MAX_TOKENS,
            max_concurrency=4 if phase == 'answer' else 8, retries=0,
            request_options={} if phase == 'answer' else {'temperature': 0},
            benchmark_provenance={'binding_sha256': binding_sha, 'phase': phase})
    with closing(factory(None)) as runtime:
        remaining = runtime.population.unique_prompt_count - len(current.frozen._authenticated_records(runtime))
    prior.emit(phase=phase + '_starting', logical_questions=len(messages), new_calls=remaining)
    batch, calls, hits, _ = current.frozen._run_exactly_authorized(runtime_factory=factory,
        authorized_provider_calls=remaining, enable_provider=enable,
        client_factory=lambda: current.frozen.ThreadLocalProvider(
            lambda: current.frozen._completion_client('LITELLM_KEY', current.frozen.GATEWAY)))
    prior.emit(phase=phase + '_complete', calls=calls, cache_hits=hits)
    return batch


def run(root, enable):
    plan, packets, rows = inputs(root)
    execution = prior.publish(root / 'execution-preflight.json', {'plan': prior.binding(plan),
        'packets': prior.binding(packets), 'runner_sha256': prior.digest(__file__),
        'model': plan.payload['model'], 'references_opened_by_answer_worker': False,
        'answer_concurrency': 4, 'warm_latency_measured': False})
    messages = [r.payload['messages'] for r in rows]
    batch = execute(root, 'answer', messages, plan.payload['model'], execution.sha256, enable=enable)
    records = {r.messages_sha256: r for r in batch.unique_records}
    answers = []
    for packet, prediction in zip(rows, batch.logical_completions, strict=True):
        record = records[identity_sha256(packet.payload['messages'])]
        if record.completion != prediction or record.finish_reason != 'stop':
            raise ValueError('answer differs from checkpoint or was truncated')
        answers.append({'ordinal': packet.payload['ordinal'], 'arm': packet.payload['arm'],
            'packet': prior.binding(packet), 'prediction': prediction,
            'record': {k: v for k, v in record.model_dump().items() if k not in ('checkpoint_hit', 'physical_call')}})
    sealed = prior.publish(root / 'answers-complete.json', {'execution': prior.binding(execution),
        'question_count': 100, 'arm_count': 2, 'rows': answers})
    prior.emit(phase='answers_sealed', rows=len(answers), sha256=sealed.sha256)


def report(root, enable):
    plan, _, packets = inputs(root)
    answers = read_sealed_json(root / 'answers-complete.json')
    execution = prior.bound(answers.payload['execution'])
    answer_runner = execution.payload['runner_sha256']
    compatible = (answer_runner == prior.digest(__file__) or (
        answer_runner == NULLABLE_USAGE_PREDECESSOR
        and prior.digest(root / 'answer-runner.py') == answer_runner))
    if (not compatible
            or execution.payload['plan'] != prior.binding(plan)):
        raise ValueError('answer implementation or plan changed')
    if [r['packet'] for r in answers.payload['rows']] != [prior.binding(p) for p in packets]:
        raise ValueError('answers changed the sealed population')
    old_plan = prior.bound(plan.payload['prior_preflight'])
    questions = prior.bound(old_plan.payload['questions'])
    refs = prior.bound(questions.payload['references'])
    references = {r['question_id']: r for r in refs.payload['references']}
    judge_rows = []
    for answer in answers.payload['rows']:
        question = questions.payload['questions'][answer['ordinal']]
        ref = references[question['question_id']]
        if quote_sha256(ref['answer']) != question['reference_sha256']:
            raise ValueError('reference changed')
        judge_rows.append({'ordinal': answer['ordinal'], 'arm': answer['arm'],
            'messages': current.frozen.build_judge_prompt(question['question'], ref['answer'], answer['prediction'])})
    judge = prior.publish(root / 'judge-preflight.json', {'answers': prior.binding(answers),
        'references': prior.binding(refs), 'rows': judge_rows, 'model': plan.payload['model']})
    batch = execute(root, 'judge', [r['messages'] for r in judge_rows],
                    plan.payload['model'], judge.sha256, enable=enable)
    verdicts = [bool(current.frozen.parse_binary_judge_verdict(v)) for v in batch.logical_completions]
    old = prior.bound(plan.payload['prior_report'])
    rows = []
    for packet, answer, correct in zip(packets, answers.payload['rows'], verdicts, strict=True):
        i, arm = answer['ordinal'], answer['arm']
        old_row = old.payload['rows'][i]
        historical = old_row['baseline_correct'] if arm == 'routing_on' else old_row['correct']
        original = prior.bound(packet.payload['source'])
        rows.append({'ordinal': i, 'arm': arm, 'neutral_correct': correct,
            'projected_correct': historical, 'prediction': answer['prediction'],
            'projected_prediction': old_row['baseline_prediction'] if arm == 'routing_on' else old_row['prediction'],
            'question': old_row['question'], 'reference': old_row['reference'],
            'reported_prompt_tokens': answer['record']['reported_prompt_tokens'],
            'prompt_token_proxy': answer['record']['prompt_token_proxy'],
            'projected_prompt_token_proxy': count_chat_prompt_token_proxy(original.payload['messages']),
            'context_tokens': packet.payload['rendered']['token_count']})
    summaries = {}
    for arm in assessment.ARMS:
        selected = [r for r in rows if r['arm'] == arm]
        summaries[arm] = {'neutral_correct': sum(r['neutral_correct'] for r in selected),
            'projected_correct': sum(r['projected_correct'] for r in selected), 'questions': 100,
            'improvements_vs_projected': [r['ordinal'] for r in selected if r['neutral_correct'] and not r['projected_correct']],
            'regressions_vs_projected': [r['ordinal'] for r in selected if not r['neutral_correct'] and r['projected_correct']],
            **token_statistics(selected),
            'mean_context_tokens': statistics.fmean(r['context_tokens'] for r in selected)}
    routing_effect = {layout: summaries['routing_on'][layout + '_correct'] - summaries['routing_off'][layout + '_correct']
                      for layout in ('neutral', 'projected')}
    report = prior.publish(root / 'report.json', {'preflight': prior.binding(plan),
        'report_runner_sha256': prior.digest(__file__), 'answer_runner_sha256': answer_runner,
        'answers': prior.binding(answers), 'judge': prior.binding(judge),
        'assessment': prior.binding(read_sealed_json(root / 'assessment.json')),
        'history_count': 1, 'question_count': 100, 'summaries': summaries,
        'routing_accuracy_effect_percentage_points': routing_effect,
        'observed_interaction_percentage_points': routing_effect['projected'] - routing_effect['neutral'],
        'warm_latency_measured': False, 'rows': rows, 'limitations': plan.payload['limitations']})
    prior.emit(phase='report_complete', summaries=summaries,
        routing_accuracy_effect_percentage_points=routing_effect, report=str(report.path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('run', 'report'))
    parser.add_argument('--root', type=Path, default=assessment.ROOT)
    parser.add_argument('--enable-provider', action='store_true')
    args = parser.parse_args()
    {'run': run, 'report': report}[args.phase](args.root, args.enable_provider)
