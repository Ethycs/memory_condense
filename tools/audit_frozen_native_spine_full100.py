"""Independently check a completed frozen evaluation without calling a model."""
import argparse
from contextlib import closing
from datetime import datetime
import json
import math
from pathlib import Path
import sqlite3
import statistics
from types import SimpleNamespace

from memory_condense.application.section_retrieval import HydratedSection, HydratedSectionSpan
from memory_condense.domain._discourse_identity import identity_sha256, quote_sha256
from memory_condense.domain._tokenizer import count_tokens
from memory_condense.eval._binary_judge_protocol import JUDGE_MAX_TOKENS, parse_binary_judge_verdict
from memory_condense.eval.benchmark import build_judge_prompt
from memory_condense.eval.fast_completion_runtime import FastCompletionRuntime
from memory_condense.search.native_spine_summary import body_identity
from memory_condense.search.section_summary import SectionSummary, RawSectionSpan
from memory_condense.search.summary_time_prior_v2 import question_day
from tools import evaluate_frozen_native_spine_full100 as evaluation
from tools.assemble_native_spine_summaries import digest
from tools.matched_eval.artifacts import publish_sealed_json, read_sealed_json
from tools.prepare_native_spine_design_slice import binding
from tools.run_hot_reduced30_answer_judge import _authenticated_records


def distribution(values):
    values = sorted(values)
    if len(values) != 100 or any(not math.isfinite(v) or v < 0 for v in values):
        raise ValueError('audit requires 100 finite nonnegative timing observations')
    return {'median_s': statistics.median(values), 'p95_s': values[94], 'mean_s': statistics.fmean(values)}


def independent_statistics(observations, judgments):
    by_key = {(c['question']['ordinal'], c['arm']): r for c, r in observations}
    judged = {(r['ordinal'], r['arm']): r for r in judgments}
    expected = {(i, a) for i in range(100) for a in evaluation.ARMS}
    expected_judged = {(i, a) for i in range(100) for a in evaluation.MEMORY_ARMS}
    if len(observations) != 300 or set(by_key) != expected or len(judgments) != 200 or set(judged) != expected_judged:
        raise ValueError('audit requires all 300 responses and 200 distinct logical judgments')
    accuracy = {}
    for arm in evaluation.MEMORY_ARMS:
        correct = 0
        for i in range(100):
            row, response = judged[i, arm], by_key[i, arm]
            verdict = parse_binary_judge_verdict(row['verdict'])
            if (type(row['correct']) is not bool or verdict != row['correct']
                    or row['response_sha256'] != response.sha256
                    or row['prediction_sha256'] != response.payload['measurement']['prediction_sha256']):
                raise ValueError('audited score differs from its measured response or verdict')
            correct += verdict
        accuracy[arm] = {'correct': correct, 'questions': 100}
    latency = {arm: {metric: distribution([by_key[i, arm].payload['measurement'][metric] for i in range(100)])
        for metric in ('prepare_s', 'e2e_ttft_s', 'e2e_total_s')} for arm in evaluation.ARMS}
    ratios = {}
    for metric in ('e2e_ttft_s', 'e2e_total_s'):
        ratios[metric] = {}
        for stat in ('median_s', 'p95_s'):
            denominator = latency['parent_context_api'][metric][stat]
            if denominator <= 0:
                raise ValueError('audited API denominator must be positive')
            ratios[metric][stat] = latency['parent_context'][metric][stat]/denominator
    quality = accuracy['parent_context']['correct'] >= 95
    speed = latency['parent_context']['e2e_total_s']['median_s'] < 5.0
    stopped = all(r.payload['measurement']['finish_reason'] == 'stop' for _, r in observations)
    under_five = sum(by_key[i, 'parent_context'].payload['measurement']['e2e_total_s'] < 5.0 for i in range(100))
    return {'accuracy': accuracy, 'latency': latency, 'matched_api_latency_ratios': ratios,
        'candidate_answers_under_five_seconds': under_five, 'accuracy_threshold_passed': quality,
        'median_latency_threshold_passed': speed, 'all_answers_stopped': stopped,
        'target_gate_passed': quality and speed and stopped, 'tail_latency_threshold_applied': False,
        'generalization_established': False, 'historical_question_exposure': True}


def verify_packet(question, hydration, routing, sessions, load_body):
    """Check served bytes against actual bodies and occurrences, not just saved hashes."""
    if (hydration['max_context_tokens'] != 1024 or hydration['max_raw_spans'] != 128
            or hydration['raw_turn_read_count'] > 128 or hydration['context_token_count'] > 1024
            or routing['query_qwen_passes'] != 0 or routing['raw_reads_during_routing'] != 0
            or routing['protected_direct'] != 0 or routing['ancestor_hops'] != 1
            or hydration['plan'] != routing['expanded']):
        raise ValueError('audited packet violates the frozen retrieval policy')
    sources = {'native-source-'+s['occurrence_id']: s for s in sessions}
    raw_turns, rendered = {}, []
    asked = question_day(question['retrieval_query'], question['prompt_question'])
    span_count = 0
    for number, row in enumerate(hydration['sections'], 1):
        section = SectionSummary.from_dict(row['section'])
        evidence = []
        for item in row['evidence']:
            span = RawSectionSpan(**item['span'])
            source = sources.get(span.source_id)
            if source is None or source['created_at'] != span.created_at or datetime.fromisoformat(span.created_at).date() > asked:
                raise ValueError('served evidence belongs to a foreign or future occurrence')
            if span.source_id not in raw_turns:
                body = load_body(source['body_sha256'])
                if body_identity(body) != source['body_sha256']:
                    raise ValueError('raw audit loader returned a different body')
                raw_turns[span.source_id] = {'native-turn-'+identity_sha256({
                    'occurrence_id': source['occurrence_id'], 'body_sha256': source['body_sha256'],
                    'turn_ordinal': ordinal}): turn for ordinal, turn in enumerate(body['turns'])}
            turn = raw_turns[span.source_id].get(span.turn_id)
            if (turn is None or span.role != turn['role'] or quote_sha256(turn['text']) != span.turn_text_sha256
                    or span.end_char > len(turn['text'])
                    or item['text'] != turn['text'][span.start_char:span.end_char]):
                raise ValueError('served text differs from its exact original raw section')
            evidence.append(HydratedSectionSpan(span, item['text']))
            span_count += 1
        hydrated = HydratedSection(section, tuple(evidence))
        if hydrated.identity_payload() != row:
            raise ValueError('hydrated section receipt differs from its raw evidence')
        rendered.append(hydrated.render_raw(f'S{number}'))
    context = '\n\n'.join(rendered)
    if count_tokens(context) != hydration['context_token_count'] or span_count > 128:
        raise ValueError('served context count differs from actual raw packet')
    messages = evaluation.design.reader_messages(evaluation.serving.protocol.messages(
        question, SimpleNamespace(render_context=lambda: context)), 'v5')
    return messages, span_count


def audit(root):
    root = Path(root)
    # Nothing below may open reference answers without a complete saved report.
    report = read_sealed_json(root/'joint-report.json')
    plan = evaluation.load_preflight(root)
    answers = read_sealed_json(root/'answers.json')
    observations = evaluation.recorded(root, plan)
    expected_answers = {'preflight_sha256': plan.sha256, 'rows': [
        {'call_index': c['call_index'], 'response_sha256': r.sha256,
         'prediction_sha256': r.payload['measurement']['prediction_sha256']} for c, r in observations]}
    if len(observations) != 300 or answers.payload != expected_answers:
        raise ValueError('audit requires all 300 bound and sealed answers')
    p, r = plan.payload, report.payload
    judge_inputs = read_sealed_json(root/'judge-preflight.json')
    if (r['preflight_sha256'] != plan.sha256 or r['answers_sha256'] != answers.sha256
            or r['population_admission_sha256'] != p['population_admission_sha256']
            or r['judge_preflight_sha256'] != judge_inputs.sha256
            or judge_inputs.payload['answers_sha256'] != answers.sha256):
        raise ValueError('report or judgments differ from the complete answer experiment')
    references = evaluation.load_references(p['settings'])
    expected_rows = []
    for call, response in observations:
        prefix = root/'journal'/f'{call["call_index"]:03d}'
        if prefix.with_suffix('.reserved').read_text().strip() != plan.sha256:
            raise ValueError('answer lost its one-call reservation')
        measurement = response.payload['measurement']
        for end_to_end, api in (('e2e_total_s', 'api_total_s'), ('e2e_ttft_s', 'api_ttft_s')):
            if not math.isclose(measurement[end_to_end], measurement[api]+measurement['prepare_s'], abs_tol=1e-8):
                raise ValueError('end-to-end timing omits or duplicates preparation')
        if call['arm'] not in evaluation.MEMORY_ARMS:
            continue
        q = call['question']
        expected_rows.append({'ordinal': q['ordinal'], 'question_id': q['question_id'], 'arm': call['arm'],
            'prediction_sha256': measurement['prediction_sha256'], 'response_sha256': response.sha256,
            'reference_sha256': quote_sha256(references[q['question_id']]),
            'messages': build_judge_prompt(q['retrieval_query'], references[q['question_id']], measurement['prediction'])})
    if judge_inputs.payload['rows'] != expected_rows:
        raise ValueError('judge prompts differ from the measured answers and locked references')
    with closing(FastCompletionRuntime(checkpoint_dir=root/'judge-checkpoints',
            prompt_population=[row['messages'] for row in expected_rows], model='codex_sdk/gpt-5.6-sol',
            client=None, max_prompt_tokens=4096, max_new_tokens=JUDGE_MAX_TOKENS, max_concurrency=8,
            retries=0, request_options={'temperature': 0},
            benchmark_provenance={'binding_sha256': judge_inputs.sha256, 'phase': 'judge'})) as runtime:
        records = _authenticated_records(runtime)
        if len(records) != runtime.population.unique_prompt_count:
            raise ValueError('judge response journal is incomplete; audit cannot call a provider')
        records = {record.messages_sha256: record for record in records.values()}
        judged = []
        for row in expected_rows:
            record = records[identity_sha256(row['messages'])]
            if record.finish_reason != 'stop' or record.requested_model != 'codex_sdk/gpt-5.6-sol':
                raise ValueError('judgment did not complete under the pinned judge model')
            judged.append({**{k: v for k, v in row.items() if k != 'messages'}, 'verdict': record.completion,
                'correct': parse_binary_judge_verdict(record.completion)})
        if (judged != r['rows'] or set(r['judge_response_journal_shas']) !=
                {record.response_journal_sha256 for record in records.values()}
                or len(r['judge_response_journal_shas']) != len(records)):
            raise ValueError('reported judgments differ from their authenticated response journals')
    stats = independent_statistics(observations, judged)
    if any(r.get(k) != value for k, value in stats.items()):
        raise ValueError('reported accuracy, timing or target decision differs from independent recomputation')
    source_root = Path(p['settings']['sources'])
    sources = read_sealed_json(source_root/'sources.json')
    namespace_bindings = {s['namespace_id']: s for s in sources.payload['namespaces']}
    bank_path = (source_root/sources.payload['body_bank_path']).resolve()
    bank_path.relative_to(source_root.resolve())
    if digest(bank_path) != sources.payload['body_bank_sha256']:
        raise ValueError('original raw body bank changed before audit')
    packet_count = span_count = 0
    with closing(sqlite3.connect(bank_path.as_uri()+'?mode=ro', uri=True)) as bank:
        def load_body(sha):
            row = bank.execute('SELECT body_json FROM bodies WHERE body_sha256=?', (sha,)).fetchone()
            if row is None:
                raise ValueError('audit source body is missing')
            return json.loads(row[0])
        for ordinal in range(100):
            prepared = read_sealed_json(root/'evidence'/f'{ordinal:03d}.json')
            source_binding = namespace_bindings[prepared.payload['case']['namespace_id']]
            namespace = read_sealed_json(source_root/source_binding['path'])
            if namespace.sha256 != source_binding['sha256']:
                raise ValueError('audit source occurrence namespace changed')
            for arm in evaluation.MEMORY_ARMS:
                messages, spans = verify_packet(prepared.payload['question'], prepared.payload['hydration'][arm],
                    prepared.payload['routing'][arm], namespace.payload['sessions'], load_body)
                if messages != prepared.payload['messages'][arm]:
                    raise ValueError('saved API messages differ from their exact original raw evidence')
                span_count += spans
                packet_count += 1
            print({'audited_question_packets': ordinal+1, 'total': 100}, flush=True)
    result, _ = publish_sealed_json(root/'independent-audit.json', {
        'implementation_sha256': digest(__file__), 'preflight': binding(plan), 'report': binding(report),
        'answers': binding(answers), 'judge_inputs': binding(judge_inputs), 'source_bank_sha256': digest(bank_path),
        'memory_packets_verified_against_original_raw': packet_count, 'exact_raw_spans_verified': span_count,
        'new_model_calls': 0, 'statistics': stats, 'full_population_admission_sha256': p['population_admission_sha256'],
        'generalization_established': False, 'target_gate_passed': stats['target_gate_passed']})
    print({'independent_audit_sha256': result.sha256, 'accuracy': stats['accuracy'],
        'target_gate_passed': stats['target_gate_passed'], 'new_model_calls': 0}, flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    audit(parser.parse_args().root)
