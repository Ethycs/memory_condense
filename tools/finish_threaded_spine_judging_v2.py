"""Complete uniform judging of sealed timed answers after a terminal TLS failure.

The entire logical judge population is rerun in a separate journal. Old partial
verdicts never select replacements. Worker clients use the original verified
Windows trust-store transport, with one independent TLS context per worker.
"""
import argparse
import hashlib
from pathlib import Path

from memory_condense.domain._discourse_identity import identity_sha256
from memory_condense.eval._binary_judge_protocol import JUDGE_MAX_TOKENS, parse_binary_judge_verdict
from memory_condense.eval.fast_completion_runtime import FastCompletionRuntime, _read_journal
from memory_condense.eval.thread_local_provider_v2 import ThreadLocalProvider
from tools import evaluate_threaded_spine_full100_v2 as evaluation
from tools.matched_eval.artifacts import publish_sealed_json, read_sealed_json
from tools.run_hot_reduced30_answer_judge import _authenticated_records, _completion_client, _run_exactly_authorized


FILES = ('tools/finish_threaded_spine_judging.py', 'tools/finish_threaded_spine_judging_v2.py',
         'src/memory_condense/eval/thread_local_provider_v2.py',
         'src/memory_condense/eval/thread_local_provider.py',
         'src/memory_condense/eval/fast_completion_runtime.py',
         'src/memory_condense/eval/fast_1m_hebbian_answer_runtime.py')


def validate_judge_rows(rows, observations):
    expected = [(c, r) for c, r in observations if c['arm'] in evaluation.MEMORY_ARMS]
    if len(rows) != 200 or len(expected) != 200:
        raise ValueError('uniform recovery requires all200 logical judgments')
    for row, (call, response) in zip(rows, expected, strict=True):
        q, m = call['question'], response.payload['measurement']
        if (row['ordinal'] != q['ordinal'] or row['question_id'] != q['question_id'] or
                row['arm'] != call['arm'] or row['prediction'] != m['prediction'] or
                row['prediction_sha256'] != m['prediction_sha256'] or
                row['response_sha256'] != response.sha256):
            raise ValueError('judging recovery changed a timed prediction')


def prepare(parent, root):
    if root.resolve() == parent.resolve():
        raise ValueError('judging recovery requires a separate root')
    preflight = evaluation.load_preflight(parent)
    answers, observations = evaluation.seal_answers(parent, preflight)
    inputs = read_sealed_json(parent / 'judge-preflight.json')
    if inputs.payload['answers_sha256'] != answers.sha256:
        raise ValueError('judge prompts belong to another answer population')
    rows = inputs.payload['rows']
    validate_judge_rows(rows, observations)
    inventory = []
    for path in sorted((parent / 'judge-checkpoints').glob('*.json')):
        _, digest = _read_journal(path)
        inventory.append({'name': path.name, 'journal_sha256': digest})
    requests = {r['name'].removesuffix('.request.json') for r in inventory if r['name'].endswith('.request.json')}
    responses = {r['name'].removesuffix('.response.json') for r in inventory if r['name'].endswith('.response.json')}
    if len(requests - responses) != 1 or responses - requests or (parent / 'joint-report.json').exists():
        raise ValueError('expected the preserved partial TLS-failed judging attempt')
    result, _ = publish_sealed_json(root / 'preflight.json', {
        'parent_root': str(parent.resolve()), 'parent_preflight_sha256': preflight.sha256,
        'answers_sha256': answers.sha256, 'judge_inputs_sha256': inputs.sha256,
        'prior_journal_inventory': inventory, 'prior_execution_session': 27491,
        'prior_execution_exit_code': 1, 'prior_failure': 'TLS certificate verification during judge connection',
        'all_400_streamed_answers_preserved': True, 'all_200_logical_predictions_rejudged': True,
        'partial_prior_verdicts_not_used': True, 'physical_call_cap': len({identity_sha256(r['messages']) for r in rows}),
        'model': 'codex_sdk/gpt-5.6-sol', 'max_prompt_tokens': 4096,
        'max_new_tokens': JUDGE_MAX_TOKENS, 'temperature': 0, 'max_concurrency': 8,
        'automatic_retries': 0, 'gateway': evaluation.GATEWAY,
        'transport': 'existing verified Windows trust store; independent client and context per worker',
        'implementation': {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in FILES}})
    return result, preflight, answers, inputs, observations


def run(parent, root, phase):
    pre, original, answers, inputs, observations = prepare(parent, root)
    if phase == 'prepare':
        print({'preflight_sha256': pre.sha256, 'physical_call_cap': pre.payload['physical_call_cap'], 'new_calls': 0}, flush=True)
        return
    rows = inputs.payload['rows']
    def factory(client):
        return FastCompletionRuntime(checkpoint_dir=root / 'judge-checkpoints',
            prompt_population=[r['messages'] for r in rows], model=pre.payload['model'], client=client,
            max_prompt_tokens=4096, max_new_tokens=JUDGE_MAX_TOKENS, max_concurrency=8, retries=0,
            request_options={'temperature': 0}, benchmark_provenance={'binding_sha256': pre.sha256, 'phase': 'judge'})
    audit = factory(None)
    try:
        remaining = audit.population.unique_prompt_count - len(_authenticated_records(audit))
    finally:
        audit.close()
    batch, calls, hits, _ = _run_exactly_authorized(runtime_factory=factory, authorized_provider_calls=remaining,
        enable_provider=phase == 'run', client_factory=lambda: ThreadLocalProvider(
            lambda: _completion_client('LITELLM_KEY', evaluation.GATEWAY)))
    judged = [{**{k: v for k, v in row.items() if k != 'messages'},
               'verdict': verdict, 'correct': parse_binary_judge_verdict(verdict)}
              for row, verdict in zip(rows, batch.logical_completions, strict=True)]
    stats = evaluation.joint_statistics(observations, judged)
    report, _ = publish_sealed_json(root / 'joint-report.json', {
        'preflight_sha256': original.sha256, 'judge_recovery_preflight_sha256': pre.sha256,
        'answers_sha256': answers.sha256, 'judge_preflight_sha256': inputs.sha256,
        'rows': judged, **stats, 'judge_response_journal_shas': [r.response_journal_sha256 for r in batch.unique_records]})
    publish_sealed_json(root / 'complete.json', {'joint_report_sha256': report.sha256,
                                               'target_gate_passed': stats['target_gate_passed']})
    print({'report_sha256': report.sha256, 'accuracy': stats['accuracy'],
           'target_gate_passed': stats['target_gate_passed'], 'new_judge_calls': calls, 'replay_hits': hits}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('prepare', 'run', 'replay'))
    parser.add_argument('--parent-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    run(args.parent_root, args.output_root, args.phase)
