"""One-history design check: two memory answers, three API controls, two judgments."""
import argparse
from contextlib import closing
import json
from pathlib import Path
import sqlite3
import time

from memory_condense.domain._discourse_identity import identity_sha256, quote_sha256
from memory_condense.eval._binary_judge_protocol import JUDGE_MAX_TOKENS, parse_binary_judge_verdict
from memory_condense.eval.benchmark import build_judge_prompt
from memory_condense.eval.streaming_latency import measure_streaming_answer
from memory_condense.ingest.loader import _as_answer_text
from memory_condense.modeling.embedding import EmbeddingService
from memory_condense.search.summary_semantic_index import summary_embedding_identity
from tools import evaluate_native_spine_full100 as serving
from tools import compile_native_spine_design_slice as compilation
from tools.assemble_native_spine_json_recovered import JsonRecoveredSummaryBodies
from tools.assemble_native_spine_summaries import digest
from tools.assess_native_longmemeval import stream_records
from tools.compile_native_spine_vectors import NativeSummaryVectors
from tools.matched_eval.artifacts import publish_sealed_json, read_sealed_json
from tools.parent_budget_native_spine_namespace import materialize_parent_budget_namespace
from tools.prepare_native_spine_design_slice import binding, bound


ARMS = ('flat', 'flat_api', 'hierarchy', 'hierarchy_api', 'short_api')
POLICY = {'history_count': 1, 'question_count': 1, 'fresh_answer_calls': 5, 'judge_calls': 2,
    'minimum_actual_body_tokens': 1_000_000, 'timed_concurrency': 1, 'automatic_retries': 0,
    'query_qwen_passes': 0, 'raw_inputs_to_qwen': False, 'gold_before_all_answers': False,
    'warm_query_latency': True, 'live_retrieval_inside_timer': True,
    'final_95_percent_claim_permitted': False}


def load_namespace(root):
    selected = compilation.scope(root)
    p = selected.payload
    report = read_sealed_json(root/'parents/result.json')
    parent_plan = read_sealed_json(root/'parents/preflight.json')
    needed = {b['body_sha256'] for b in p['bodies']}
    if (report.payload['scope_sha256'] != selected.sha256
            or report.payload['preflight_sha256'] != parent_plan.sha256
            or parent_plan.payload['implementation_sha256'] != digest(compilation.__file__)
            or report.payload['complete_selected_history'] is not True
            or {b['body_sha256'] for b in report.payload['templates']} != needed):
        raise ValueError('design evaluation requires every selected body hierarchy')
    templates = {b['body_sha256']: bound(b['artifact']).payload for b in report.payload['templates']}
    bank = Path(p['body_bank']['path'])
    if digest(bank) != p['body_bank']['sha256']:
        raise ValueError('design source bank changed')
    with closing(JsonRecoveredSummaryBodies(Path(p['store_root']))) as store, closing(
            sqlite3.connect(bank.as_uri()+'?mode=ro', uri=True)) as raw:
        if store.manifest.sha256 != p['store']['sha256']:
            raise ValueError('design summary store changed')
        namespace = materialize_parent_budget_namespace(bound(p['namespace']).payload['sessions'],
            body_ids=needed, templates=templates,
            load_body=lambda sha: json.loads(raw.execute('SELECT body_json FROM bodies WHERE body_sha256=?', (sha,)).fetchone()[0]),
            load_summaries=store.load, compiler_identity=store.manifest.sha256, allow_partial=False)
    namespace.audit.update(namespace_id=p['case']['namespace_id'], namespace_source_sha256=p['case']['namespace_sha256'])
    namespace.require_complete(minimum_body_tokens=1_000_000)
    if namespace.audit['body_tokens'] != p['actual_body_tokens']:
        raise ValueError('materialized design history changed actual token population')
    return selected, namespace


def reference(case, source, dataset):
    if digest(dataset) != source.payload['source_artifacts']['m_dataset_sha256']:
        raise ValueError('design reference dataset changed')
    with Path(dataset).open(encoding='utf-8') as stream:
        for row in stream_records(stream):
            if row['question_id'] != case['question_id']:
                continue
            answer = _as_answer_text(row['answer'])
            if (row['question'] != case['question'] or row['question_date'] != case['question_date']
                    or quote_sha256(answer) != case['reference_sha256']):
                raise ValueError('design reference does not match the selected question')
            return answer
    raise ValueError('selected design reference is missing')


def observations(target, plan):
    found = []
    for index, arm in enumerate(ARMS):
        call = plan.payload['calls'][index]
        prefix = target/'journal'/f'{index:02d}'
        request = read_sealed_json(prefix.with_suffix('.request.json'))
        response = read_sealed_json(prefix.with_suffix('.response.json'))
        m = response.payload['measurement']
        if (call['arm'] != arm or request.payload != {'preflight_sha256': plan.sha256, 'call': call}
                or response.payload['request_sha256'] != request.sha256
                or response.payload['messages'] != call['messages']
                or response.payload['hydration'] != plan.payload['hydration'].get(arm)
                or response.payload['routing'] != plan.payload['routing'].get(arm)
                or m['messages_sha256'] != identity_sha256(call['messages'])
                or m['prediction_sha256'] != quote_sha256(m['prediction'])
                or m['model'] != serving.MODEL or m['max_tokens'] != 256):
            raise ValueError('design answer journal changed its prompt or response')
        found.append((arm, response))
    return found


def report(target, plan):
    if (plan.payload['policy'] != POLICY or plan.payload['implementation_sha256'] != digest(__file__)
            or plan.payload['gold_loaded'] is not False):
        raise ValueError('design evaluation policy or implementation changed')
    rows = observations(target, plan)
    answers = read_sealed_json(target/'answers.json')
    if answers.payload != {'preflight_sha256': plan.sha256, 'responses': [binding(r) for _, r in rows]}:
        raise ValueError('all five design answers must seal before judging')
    judged, accuracy = [], {}
    for arm in ('flat', 'hierarchy'):
        request = read_sealed_json(target/'judgments'/f'{arm}.request.json')
        response = read_sealed_json(target/'judgments'/f'{arm}.response.json')
        answer_response = dict(rows)[arm]
        if (request.payload['answers_sha256'] != answers.sha256
                or request.payload['answer_response_sha256'] != answer_response.sha256
                or request.payload['reference_sha256'] != plan.payload['case']['reference_sha256']
                or response.payload['request_sha256'] != request.sha256
                or response.payload['finish_reason'] != 'stop'):
            raise ValueError('design judgment changed answer, reference or stopped response')
        correct = bool(parse_binary_judge_verdict(response.payload['verdict']))
        accuracy[arm] = {'correct': int(correct), 'questions': 1}
        judged.append(binding(response))
    latency = {arm: {key: response.payload['measurement'][key] for key in
        ('prepare_s', 'e2e_ttft_s', 'e2e_total_s', 'finish_reason')} for arm, response in rows}
    ratios = {arm: {control: {metric: latency[arm][metric]/latency[control][metric]
        for metric in ('e2e_ttft_s', 'e2e_total_s')} for control in (arm+'_api', 'short_api')}
        for arm in ('flat', 'hierarchy')}
    return publish_sealed_json(target/'report.json', {
        'preflight_sha256': plan.sha256, 'answers_sha256': answers.sha256, 'judgments': judged,
        'history_count': 1, 'question_count': 1, 'actual_body_tokens': plan.payload['actual_body_tokens'],
        'accuracy': accuracy, 'latency_seconds': latency, 'single_observation_latency_ratios': ratios,
        'all_answers_stopped': all(v['finish_reason'] == 'stop' for v in latency.values()),
        'accuracy_generalization_established': False, 'full100_target_passed': False,
    })[0]


def run(root, dataset, enable):
    if not enable:
        raise ValueError('design provider execution flag is required')
    root, dataset = Path(root), Path(dataset)
    target = root/'evaluation'
    if (target/'execution.reserved').exists():
        raise ValueError('design evaluation has already started; no implicit retry')
    serving.require_idle()
    setup_started = time.perf_counter()
    selected, namespace = load_namespace(root)
    vectors = NativeSummaryVectors(root/'vectors')
    if vectors.preflight.payload['design_scope_sha256'] != selected.sha256:
        raise ValueError('design vectors belong to another history')
    case = selected.payload['case']
    serving.population.namespace_receipt(namespace, case, vectors)
    q = serving.question(case)
    with closing(EmbeddingService(device='cuda', batch_size=8)) as encoder:
        if summary_embedding_identity(encoder) != vectors.embedding_identity:
            raise ValueError('design query encoder changed')
        encoder.embed_query('Native memory design evaluation warmup.')
        memory = serving.resident(namespace, vectors, encoder)
        built = {arm: serving.build(memory, q, arm) for arm in ('flat', 'hierarchy')}
        baseline, candidate = built['flat'][1]['sections'], built['hierarchy'][1]['sections']
        if candidate[:len(baseline)] != baseline:
            raise ValueError('design candidate dropped baseline raw evidence')
        messages = {arm: built[arm][0] for arm in built}
        messages.update(flat_api=messages['flat'], hierarchy_api=messages['hierarchy'], short_api=serving.protocol.messages(q))
        calls = [{'arm': arm, 'messages': messages[arm]} for arm in ARMS]
        plan, _ = publish_sealed_json(target/'preflight.json', {
            'format': 'native-spine-single-history-design-evaluation-v1', 'policy': POLICY,
            'scope_sha256': selected.sha256, 'case': case, 'calls': calls,
            'hydration': {a: built[a][1] for a in built}, 'routing': {a: built[a][2] for a in built},
            'implementation_sha256': digest(__file__), 'serving_implementation': serving.implementation(),
            'actual_body_tokens': namespace.audit['body_tokens'], 'gold_loaded': False,
            'resident_setup_s_excluded_from_warm_latency': time.perf_counter()-setup_started,
        })
        with (target/'execution.reserved').open('x', encoding='utf-8') as stream:
            stream.write(plan.sha256+'\n')
        with closing(serving._completion_client('LITELLM_KEY', serving.GATEWAY)) as client:
            for index, call in enumerate(calls):
                arm = call['arm']
                prefix = target/'journal'/f'{index:02d}'
                request, _ = publish_sealed_json(prefix.with_suffix('.request.json'), {
                    'preflight_sha256': plan.sha256, 'call': call})
                with prefix.with_suffix('.reserved').open('x', encoding='utf-8') as stream:
                    stream.write(request.sha256+'\n')
                prepared = {}
                def prompt():
                    if arm in built:
                        value, hydration, routing = serving.build(memory, q, arm)
                        if (value, hydration, routing) != built[arm]:
                            raise ValueError('fresh design retrieval differs from its matched API control')
                    else:
                        value, hydration, routing = call['messages'], None, None
                    prepared.update(messages=value, hydration=hydration, routing=routing)
                    return value
                measured = measure_streaming_answer(client=client, model=serving.MODEL, prepare_prompt=prompt, max_tokens=256)
                response, _ = publish_sealed_json(prefix.with_suffix('.response.json'), {
                    'request_sha256': request.sha256, 'measurement': measured, **prepared})
                print({'design_answer': arm, 'ttft_s': measured['e2e_ttft_s'], 'total_s': measured['e2e_total_s'],
                    'finish_reason': measured['finish_reason']}, flush=True)
            rows = observations(target, plan)
            answers, _ = publish_sealed_json(target/'answers.json', {
                'preflight_sha256': plan.sha256, 'responses': [binding(r) for _, r in rows]})
            gold = reference(case, bound(selected.payload['source']), dataset)
            for arm in ('flat', 'hierarchy'):
                answer = dict(rows)[arm]
                request, _ = publish_sealed_json(target/'judgments'/f'{arm}.request.json', {
                    'answers_sha256': answers.sha256, 'answer_response_sha256': answer.sha256,
                    'reference_sha256': quote_sha256(gold),
                    'messages': build_judge_prompt(case['question'], gold, answer.payload['measurement']['prediction'])})
                with (target/'judgments'/f'{arm}.reserved').open('x', encoding='utf-8') as stream:
                    stream.write(request.sha256+'\n')
                response = client.chat.completions.create(model='codex_sdk/gpt-5.6-sol', messages=request.payload['messages'],
                    max_tokens=JUDGE_MAX_TOKENS, temperature=0, timeout=180.0)
                choice, = response.choices
                publish_sealed_json(target/'judgments'/f'{arm}.response.json', {
                    'request_sha256': request.sha256, 'verdict': choice.message.content,
                    'finish_reason': choice.finish_reason, 'response_model': response.model})
        result = report(target, plan)
        if report(target, plan).sha256 != result.sha256:
            raise ValueError('design report replay changed')
        print({'design_report_sha256': result.sha256, 'accuracy': result.payload['accuracy'],
            'latency_seconds': result.payload['latency_seconds'], 'full100_target_passed': False}, flush=True)
        return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--m-dataset', type=Path, required=True)
    parser.add_argument('--enable-provider', action='store_true')
    args = parser.parse_args()
    run(args.root, args.m_dataset, args.enable_provider)
