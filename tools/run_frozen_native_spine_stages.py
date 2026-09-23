"""Advance one frozen corpus through serial local stages and the bound full100 run."""
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil

from tools import evaluate_frozen_native_spine_full100 as evaluation
from tools.assemble_native_spine_summaries import digest
from tools.matched_eval.artifacts import publish_sealed_json, read_sealed_json
from tools.prepare_native_spine_design_slice import binding, bound


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def same_process(worker):
    try:
        process = psutil.Process(worker['pid'])
        return process.create_time() == worker['create_time']
    except psutil.NoSuchProcess:
        return False


def stages(corpus_root, evaluation_root, candidate):
    base = Path(corpus_root).resolve()
    answer_root = Path(evaluation_root).resolve()
    python = [sys.executable, '-X', 'utf8']
    parent = base/'remaining-parents'
    settings = {'sources': Path('eval_results/native-spine-complete-sources-20260910-r1').resolve(),
        'store': Path('eval_results/native-spine-source-completion-20260913-r1/complete-body-store').resolve(),
        'hierarchies': parent/'result.json', 'vectors': base/'vectors',
        'm_dataset': Path('.cache/datasets/longmemeval-cleaned/98d7416c24c778c2fee6e6f3006e7a073259d48f/longmemeval_m_cleaned.json').resolve(),
        'candidate': Path(candidate).resolve()}
    source_flags = [part for key, value in settings.items() for part in ('--'+key.replace('_', '-'), str(value))]
    return [
        {'name': 'parent-prepare', 'command': [*python, '-m', 'tools.compile_remaining_native_spine_hierarchies',
            'prepare', '--root', str(parent), '--scope', str(base/'scope.json'),
            '--attention-root', str(base/'remaining-attention')], 'result': str(parent/'preflight.json')},
        {'name': 'parent-run', 'command': [*python, '-m', 'tools.compile_remaining_native_spine_hierarchies',
            'run', '--root', str(parent), '--budget', '4096'], 'result': str(parent/'result.json')},
        {'name': 'vectors', 'command': [*python, '-m', 'tools.run_frozen_native_spine_stages',
            'vectors', '--corpus-root', str(base)], 'result': str(base/'vectors/result.json')},
        {'name': 'evaluation-prepare', 'command': [*python, '-m', 'tools.evaluate_frozen_native_spine_full100',
            'prepare', '--root', str(answer_root), *source_flags], 'result': str(answer_root/'preflight.json')},
        {'name': 'evaluation-run', 'command': [*python, '-m', 'tools.evaluate_frozen_native_spine_full100',
            'run', '--root', str(answer_root), '--enable-provider'], 'result': str(answer_root/'joint-report.json')},
    ]


def prepare(root, corpus_root, evaluation_root, candidate):
    root, base, answer_root = Path(root), Path(corpus_root), Path(evaluation_root)
    if root.exists() or answer_root.exists() or (base/'remaining-parents').exists():
        raise ValueError('serial continuation requires fresh controller, parent and evaluation roots')
    candidate_artifact = evaluation.validate_candidate(candidate)
    worker = read_sealed_json(base/'remaining-attention/worker-started.json')
    attention = bound(worker.payload['preflight'])
    if attention.path != (base/'remaining-attention/preflight.json').resolve():
        raise ValueError('dependency worker differs from this attention stage')
    plan, _ = publish_sealed_json(root/'preflight.json', {
        'format': 'native-spine-frozen-serial-stages-v1', 'implementation_sha256': digest(__file__),
        'evaluation_implementation': evaluation.implementation(), 'candidate': binding(candidate_artifact),
        'corpus_root': str(base.resolve()), 'evaluation_root': str(answer_root.resolve()),
        'attention_worker': binding(worker), 'scope': binding(read_sealed_json(base/'scope.json')),
        'vector_preflight': binding(read_sealed_json(base/'vectors/preflight.json')),
        'stages': stages(base, answer_root, candidate), 'maximum_new_parent_jobs': 4096,
        'maximum_answer_calls': 300, 'logical_judgments': 200, 'automatic_retries': 0,
        'concurrent_gpu_models': 1, 'raw_inputs_to_qwen': False})
    return plan


def validate_stage_result(stage, artifact):
    p = artifact.payload
    if stage['name'] == 'parent-run' and (p.get('complete_native_hierarchies') is not True
            or p.get('complete_available_body_hierarchies') is not True or p.get('body_count') != 31166):
        raise ValueError('parent stage is incomplete; stop before vectors and answers')
    if stage['name'] == 'vectors' and (p.get('complete_prepared_vectors') is not True
            or p.get('complete_source_compilation') is not True):
        raise ValueError('vector stage is incomplete; stop before answers')
    if stage['name'] == 'evaluation-run' and any(
            p.get('accuracy', {}).get(arm, {}).get('questions') != 100 for arm in evaluation.MEMORY_ARMS):
        raise ValueError('evaluation did not produce both full accuracy populations')


def launch_stage(root, plan, ordinal, stage):
    """One child at a time; preserve its output and exit without any retry."""
    prefix = root/f'{ordinal:02d}-{stage["name"]}'
    if Path(stage['result']).exists():
        raise ValueError('stage output already exists; no implicit rerun or substitution')
    with prefix.with_suffix('.log').open('x', encoding='utf-8') as log:
        child = subprocess.Popen(stage['command'], stdout=log, stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        process = psutil.Process(child.pid)
        started, _ = publish_sealed_json(prefix.with_suffix('.started.json'), {
            'preflight_sha256': plan.sha256, 'stage': stage, 'started_at': utc_now(),
            'worker': {'pid': child.pid, 'create_time': process.create_time()}})
        print({'stage_started': stage['name'], 'pid': child.pid, 'log': str(prefix.with_suffix('.log'))}, flush=True)
        while True:
            try:
                returncode = child.wait(timeout=30)
                break
            except subprocess.TimeoutExpired:
                print({'stage_running': stage['name'], 'pid': child.pid, 'at': utc_now()}, flush=True)
    ended, _ = publish_sealed_json(prefix.with_suffix('.exit.json'), {
        'started': binding(started), 'exit_code': returncode, 'ended_at': utc_now(),
        'log_sha256': digest(prefix.with_suffix('.log')), 'retry_performed': False})
    if returncode != 0:
        raise RuntimeError(f'{stage["name"]} exited {returncode}; inspect {prefix.with_suffix(".log")}')
    result = read_sealed_json(stage['result'])
    validate_stage_result(stage, result)
    publish_sealed_json(prefix.with_suffix('.complete.json'), {'exit': binding(ended), 'result': binding(result)})
    print({'stage_complete': stage['name'], 'result_sha256': result.sha256}, flush=True)
    return result


def run(root):
    root = Path(root)
    plan = read_sealed_json(root/'preflight.json')
    p = plan.payload
    if p['implementation_sha256'] != digest(__file__) or p['evaluation_implementation'] != evaluation.implementation():
        raise ValueError('serial continuation implementation changed')
    candidate = bound(p['candidate'])
    evaluation.validate_candidate(candidate.path)
    bound(p['scope'])
    bound(p['vector_preflight'])
    if p['stages'] != stages(p['corpus_root'], p['evaluation_root'], candidate.path):
        raise ValueError('serial stage commands changed')
    with (root/'execution.reserved').open('x', encoding='utf-8') as handle:
        handle.write(plan.sha256+'\n')
    self_process = psutil.Process(os.getpid())
    publish_sealed_json(root/'worker-started.json', {'preflight_sha256': plan.sha256,
        'worker': {'pid': self_process.pid, 'create_time': self_process.create_time()}, 'at': utc_now()})
    worker = bound(p['attention_worker'])
    while same_process(worker.payload['worker']):
        print({'waiting_for_existing_attention': worker.payload['worker']['pid'], 'at': utc_now()}, flush=True)
        time.sleep(30)
    attention = read_sealed_json(Path(p['corpus_root'])/'remaining-attention/result.json')
    if (attention.payload['preflight_sha256'] != bound(worker.payload['preflight']).sha256
            or attention.payload['all_prepared_attention_complete'] is not True):
        raise ValueError('attention dependency ended without a complete matching result')
    publish_sealed_json(root/'attention-dependency-complete.json', {'worker': binding(worker),
        'result': binding(attention), 'same_process_absent': True, 'at': utc_now()})
    for ordinal, stage in enumerate(p['stages']):
        if p['evaluation_implementation'] != evaluation.implementation():
            raise ValueError('frozen implementation changed between stages')
        result = launch_stage(root, plan, ordinal, stage)
    publish_sealed_json(root/'complete.json', {'joint_report': binding(result),
        'target_gate_passed': result.payload['target_gate_passed'], 'at': utc_now()})


def vectors(corpus_root):
    from contextlib import closing
    from memory_condense.modeling.embedding import EmbeddingService
    evaluation.require_idle()
    with closing(EmbeddingService(device='cuda', batch_size=8)) as encoder:
        evaluation.vector_compiler.execute(Path(corpus_root)/'vectors', encoder)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('prepare', 'run', 'vectors'))
    for name in ('root', 'corpus-root', 'evaluation-root', 'candidate'):
        parser.add_argument('--'+name, type=Path)
    args = parser.parse_args()
    if args.phase == 'prepare':
        plan = prepare(args.root, args.corpus_root, args.evaluation_root, args.candidate)
        print({'serial_stages_preflight_sha256': plan.sha256, 'new_model_calls': 0}, flush=True)
    elif args.phase == 'run':
        run(args.root)
    else:
        vectors(args.corpus_root)
