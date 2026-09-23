"""Grade every saved API control from the completed parent-user 100-question run.

This diagnostic generates no answers and never replaces the original memory
score. Identical-prediction disagreements expose grading variability separately.
"""
import argparse
from contextlib import closing
from pathlib import Path

from memory_condense.domain._discourse_identity import quote_sha256
from tools import evaluate_native_spine_parent_users100 as evaluation
from tools.assemble_native_spine_summaries import digest
from tools.matched_eval.artifacts import publish_sealed_json, read_sealed_json
from tools.prepare_native_spine_design_slice import binding, bound


SOURCE = Path('eval_results/native-spine-app-parent-users100-20260915-r1')
FORMAT = 'native-spine-saved-paired-answers-v1'
frozen = evaluation.frozen


def paired_rows(observations, cases, references, original_inputs, original_grades):
    """Require the entire fixed population and authenticate the original grader prompt."""
    if (len(observations) != 200 or len(cases) != 100 or len(references) != 100
            or len(original_inputs) != 100 or len(original_grades) != 100
            or [c['ordinal'] for c in cases] != list(range(100))
            or [c['call_index'] for c, _ in observations] != list(range(200))):
        raise ValueError('requires all 100 questions and 200 completed answers')
    refs = {r['question_id']: r for r in references}
    if len(refs) != 100 or set(refs) != {c['question_id'] for c in cases}:
        raise ValueError('reference population changed')
    result = []
    for ordinal, case in enumerate(cases):
        group = observations[ordinal * 2:ordinal * 2 + 2]
        pairs = {c['arm']: (c, r) for c, r in group}
        if set(pairs) != set(evaluation.ARMS):
            raise ValueError('each question requires one memory and one API answer')
        memory_call, memory = pairs['parent_context']
        api_call, api = pairs['parent_context_api']
        expected_question = frozen.question(case)
        if (memory_call['question'] != expected_question or api_call['question'] != expected_question
                or memory_call['messages'] != api_call['messages']):
            raise ValueError('paired answers must have identical questions and prompts')
        mm, am = memory.payload['measurement'], api.payload['measurement']
        if any(m['prediction_sha256'] != quote_sha256(m['prediction']) for m in (mm, am)):
            raise ValueError('saved prediction identity changed')
        answer = refs[case['question_id']]['answer']
        if quote_sha256(answer) != case['reference_sha256']:
            raise ValueError('reference answer changed')
        old_input, old_grade = original_inputs[ordinal], original_grades[ordinal]
        identity = {'ordinal': ordinal, 'question_id': case['question_id'],
            'prediction_sha256': mm['prediction_sha256'], 'response_sha256': memory.sha256,
            'reference_sha256': quote_sha256(answer)}
        if (any(old_input.get(k) != v or old_grade.get(k) != v for k, v in identity.items())
                or old_input['messages'] != frozen.build_judge_prompt(
                    expected_question['retrieval_query'], answer, mm['prediction'])
                or type(old_grade['correct']) is not bool
                or frozen.parse_binary_judge_verdict(old_grade['verdict']) != old_grade['correct']):
            raise ValueError('original memory grade or grading protocol changed')
        result.append({'ordinal': ordinal, 'question_id': case['question_id'],
            'memory_response': binding(memory), 'api_response': binding(api),
            'memory_prediction_sha256': mm['prediction_sha256'],
            'api_prediction_sha256': am['prediction_sha256'],
            'identical_prediction': mm['prediction'] == am['prediction'],
            'memory_correct': old_grade['correct'], 'memory_verdict': old_grade['verdict'],
            'reference_sha256': quote_sha256(answer),
            'messages': frozen.build_judge_prompt(expected_question['retrieval_query'], answer, am['prediction'])})
    return result


def prepare(root):
    plan = read_sealed_json(SOURCE / 'preflight.json')
    questions, scope = evaluation.validate_plan(plan)
    answers, observations = evaluation.seal_answers(SOURCE, plan)
    report = read_sealed_json(SOURCE / 'joint-report.json')
    raw = read_sealed_json(SOURCE / 'raw-audit.json')
    complete = read_sealed_json(SOURCE / 'complete.json')
    original_inputs = read_sealed_json(SOURCE / 'judge-preflight.json')
    if (report.payload['answers_sha256'] != answers.sha256
            or report.payload['preflight_sha256'] != plan.sha256
            or report.payload['judge_preflight_sha256'] != original_inputs.sha256
            or original_inputs.payload['answers_sha256'] != answers.sha256
            or raw.payload['joint_report'] != binding(report)
            or raw.payload['verified_memory_packets'] != 100
            or complete.payload['joint_report'] != binding(report)
            or complete.payload['raw_audit'] != binding(raw)):
        raise ValueError('paired diagnostic requires the complete audited source run')
    # No references open before the complete population has been authenticated.
    refs = bound(questions.payload['references'])
    if (refs.sha256 != report.payload['references_sha256']
            or refs.payload['scope_sha256'] != scope.sha256
            or refs.payload['ingest_use_permitted'] is not False):
        raise ValueError('original reference binding changed')
    rows = paired_rows(observations, questions.payload['questions'], refs.payload['references'],
                       original_inputs.payload['rows'], report.payload['rows'])
    return publish_sealed_json(root / 'preflight.json', {
        'format': FORMAT, 'implementation_sha256': digest(__file__),
        'source_plan': binding(plan), 'source_report': binding(report),
        'source_answers': binding(answers), 'source_raw_audit': binding(raw),
        'source_completion': binding(complete), 'original_judge_inputs': binding(original_inputs),
        'references': binding(refs), 'questions': binding(questions),
        'history_count': 1, 'question_count': 100, 'new_answer_calls': 0,
        'original_memory_score_unchanged': True, 'diagnostic_only': True, 'rows': rows})[0]


def summarize(rows, completions):
    if len(rows) != len(completions):
        raise ValueError('every paired answer requires a verdict')
    judged = [{**{k: v for k, v in r.items() if k != 'messages'},
               'api_verdict': text, 'api_correct': frozen.parse_binary_judge_verdict(text)}
              for r, text in zip(rows, completions, strict=True)]
    return {'memory_accuracy': {'correct': sum(r['memory_correct'] for r in judged), 'questions': len(judged)},
        'api_control_accuracy': {'correct': sum(r['api_correct'] for r in judged), 'questions': len(judged)},
        'api_only_correct_ordinals': [r['ordinal'] for r in judged if r['api_correct'] and not r['memory_correct']],
        'memory_only_correct_ordinals': [r['ordinal'] for r in judged if r['memory_correct'] and not r['api_correct']],
        'identical_prediction_count': sum(r['identical_prediction'] for r in judged),
        'identical_prediction_grade_disagreements': [r['ordinal'] for r in judged
            if r['identical_prediction'] and r['memory_correct'] != r['api_correct']],
        'rows': judged}


def run(root, enable=False):
    # Rebuild inputs from original artifacts on replay too; never trust flags alone.
    plan = prepare(root)
    rows = plan.payload['rows']
    def factory(client):
        return frozen.FastCompletionRuntime(checkpoint_dir=root / 'checkpoints',
            prompt_population=[r['messages'] for r in rows], model='codex_sdk/gpt-5.6-sol', client=client,
            max_prompt_tokens=4096, max_new_tokens=frozen.JUDGE_MAX_TOKENS,
            max_concurrency=8, retries=0, request_options={'temperature': 0},
            benchmark_provenance={'binding_sha256': plan.sha256, 'phase': 'saved_api_controls'})
    with closing(factory(None)) as runtime:
        remaining = runtime.population.unique_prompt_count - len(frozen._authenticated_records(runtime))
    print({'preflight_sha256': plan.sha256, 'question_count': 100, 'new_answer_calls': 0,
           'remaining_judge_calls': remaining}, flush=True)
    batch, calls, hits, _ = frozen._run_exactly_authorized(runtime_factory=factory,
        authorized_provider_calls=remaining, enable_provider=enable,
        client_factory=lambda: frozen.ThreadLocalProvider(lambda: frozen._completion_client('LITELLM_KEY', frozen.GATEWAY)))
    result, _ = publish_sealed_json(root / 'report.json', {'preflight': binding(plan),
        **summarize(rows, batch.logical_completions),
        'response_journal_shas': [r.response_journal_sha256 for r in batch.unique_records],
        'new_answer_calls': 0, 'original_memory_score_unchanged': True,
        'diagnostic_only': True, 'target_completion_claim_permitted': False})
    print({'report_sha256': result.sha256,
        **{k: v for k, v in result.payload.items() if k not in ('rows', 'response_journal_shas', 'preflight')},
        'new_judge_calls': calls, 'cache_hits': hits}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--enable-provider', action='store_true')
    args = parser.parse_args()
    run(args.root, args.enable_provider)
