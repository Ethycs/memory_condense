"""Full-context control at the recorded memory arm's exact pre-edit checkpoint.

Generation runs in a separate gateway process; candidate tools remain sandboxed.
No prior generated candidate or future solution enters either arm.
"""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time

from tools import native_spine_engineering_session as s
from tools import native_spine_engineering_live_session as live
from tools import native_spine_engineering_read240 as reader
from tools import native_spine_engineering_transport_resume as transport
from tools import native_spine_engineering_compression_resume as compression
from tools.native_spine_engineering_working_state import readable_observation

SOURCE = Path('eval_results/native-spine-engineering-live-session-20260922-r1')
CAPS = {'memory': 65536, 'full': 1000000}
TEST_NAMES = ('decay', 'db', 'memory_store', 'transcript_store', 'condenser',
              'mcp_server', 'eval_recall', 'ranking', 'architecture')
# Only context delivery changes. Task, tool and engineering instructions match
# all 73 already-recorded memory continuation responses.
SYSTEM = live.SYSTEM.replace(
    'stored in memory. Each request provides retrieved memory and a bounded working conversation for the current user turn. Use recall if something needed is missing.',
    'provided in full chronological context. Each request includes all available earlier conversation and tool observations. Use the supplied history if something needed is missing.')
SYSTEM = SYSTEM.replace(
    'Recent work receipts are recovered from memory and list tools already executed.',
    'Earlier tool observations in the full history list tools already executed.')


def full_messages(rows):
    """Every available original/new conversation record; no truncation/summary."""
    messages = [{'role': 'system', 'content': SYSTEM}]
    for row in rows:
        if row.get('kind') == 'activity':
            continue  # Internal memory receipts duplicate recorded operations.
        if row.get('kind') == 'tool':
            messages.append({'role': 'user', 'content': 'Tool result:\n' + readable_observation(row['text'])})
        else:
            role = row['role'] if row['role'] in ('user', 'assistant') else 'user'
            messages.append({'role': role, 'content': row['text']})
    return messages


class FullHistory:
    active = None

    def __init__(self, root, gateway):
        self.root, self.rows, self.snapshot = root, [], None
        FullHistory.active = self

    def sync(self, rows, folder):
        self.rows = list(rows)
        self.snapshot = {'mode': 'direct_full_history', 'turn_count': len(rows),
                         'history_sha256': s.identity_sha256(rows)}
        s.save(folder / 'ingest.json', dict(self.snapshot, elapsed_s=0., application_memory_used=False))

    def retrieve(self, query, current_turn_id, **kwargs):
        if kwargs.get('include_reservations') is False:
            return {'text': 'All available earlier conversation and tool results are already in this request.',
                    'elapsed_s': 0., 'history_sha256': s.identity_sha256(self.rows)}
        return {'text': '', 'elapsed_s': 0., 'history_sha256': s.identity_sha256(self.rows)}

    def close(self):
        pass


def execute(root, action, folder, memory, current_turn_id, plan):
    return reader.execute(root, action, folder, memory, current_turn_id, plan)


def implementation_hashes():
    modules = (s, live, reader, transport, compression, compression.dedup, compression.dedup.quote)
    return {Path(m.__file__).name: s.evaluation.digest(m.__file__) for m in modules} | {
        Path(__file__).name: s.evaluation.digest(__file__)}


def prepare(root):
    if root.exists():
        raise ValueError('comparison requires a fresh root')
    previous = s.old.load(SOURCE / 'plan.json')
    p = previous.payload
    report = s.payload(SOURCE / 'report.json')
    recorded = report['live_session_actions']
    first = recorded[0]
    first_request = s.payload(SOURCE / 'steps' / f"{first['step']:02d}" / 'actions' /
                              f"{first['action']:03d}" / 'request.json')
    for action in recorded:
        request = s.payload(SOURCE / 'steps' / f"{action['step']:02d}" / 'actions' /
                            f"{action['action']:03d}" / 'request.json')
        if request['messages'][0]['content'] != live.SYSTEM:
            raise ValueError('recorded memory arm changed actor instructions')
    predecessor = Path(p['continuation']['source_root'])
    initial_rows = [dict(row, kind='seed') for row in p['seed']]
    initial_rows += [s.payload(x) for x in sorted((predecessor / 'events').glob('*.json'))]
    if s.identity_sha256(initial_rows) != first_request['history_sha256']:
        raise ValueError('control fork differs from the recorded memory continuation')
    initial_files = {x.relative_to(predecessor / 'workspace').as_posix(): s.evaluation.digest(x)
                     for x in (predecessor / 'workspace').rglob('*') if x.is_file()
                     and not any(part in ('__pycache__', '.pytest_cache') for part in x.parts)}
    if initial_files != p['starting_file_sha256s']:
        raise ValueError('fork checkout contains prior candidate edits')
    root.mkdir(parents=True)
    acceptance = root / 'acceptance'
    shutil.copytree(SOURCE / 'acceptance', acceptance,
                    ignore=shutil.ignore_patterns('__pycache__', '.pytest_cache'))
    original = (acceptance / 'test_independent_behavior.py').read_text(encoding='utf-8')
    begin = original.index('def test_wall_time_does_not_change_retrieval_energy(')
    end = original.index('\ndef test_normal_ingest_', begin)
    clock = '''def test_wall_time_does_not_change_retrieval_energy(mc, monkeypatch):
    item = create(mc)
    instant = datetime.now(timezone.utc)
    monkeypatch.setattr(decay, 'now_utc', lambda: instant)
    first = mc.recall_memories('SQLite', k=1, reheat=False)[0].energy
    monkeypatch.setattr(decay, 'now_utc', lambda: instant + timedelta(days=1000))
    later = mc.recall_memories('SQLite', k=1, reheat=False)[0].energy
    assert first == pytest.approx(item.energy)
    assert later == pytest.approx(first)
    assert mc.memory.get(item.mem_id).energy == item.energy

'''
    (acceptance / 'test_paired_behavior.py').write_bytes((original[:begin] + clock + original[end:]).encode())
    common = dict(p, format='paired-engineering-continuation-v1', arm='full',
                  system=SYSTEM, max_prompt_tokens=CAPS['full'],
                  paired_implementation_sha256s=implementation_hashes(),
                  implementation_sha256=s.evaluation.digest(s.__file__),
                  acceptance_sha256s={x.name: s.evaluation.digest(x) for x in acceptance.glob('*.py')})
    common.update({
        'memory_arm_root': str(SOURCE.resolve()), 'memory_report_sha256': s.evaluation.digest(SOURCE / 'report.json'),
        'fork_history_sha256': first_request['history_sha256'], 'fork_turn_count': len(initial_rows),
        'fork_step': first['step'], 'fork_action': first['action'],
        'inherited_responses': report['live_session_adapter']['retained_completed_actions'],
        'new_memory_actor_calls': 0, 'context_instruction_changes_only': True,
        'memory_system': live.SYSTEM, 'full_context_system': SYSTEM,
        'latency_for_comparison': 'Assume equal earlier seconds-level latency; do not score timing.',
        'primary_checks': ['wall-clock independence', 'turn decay and reopen', 'once-per-turn reinforcement',
                           'stable pins', 'duplicate evidence reinforcement', 'unrelated memory untouched',
                           'current MCP energy display', 'frequent-use retention'],
        'secondary_quality': ['completion of remaining original prompts', 'user correction retained in later work',
                              'actual edits and relevant tests', 'documentation consistent with implementation'],
        'limitations': ['One continuation from the same pre-edit checkpoint; one sample per condition.',
                       'Memory arm reuses its 73 recorded responses; the full-context control is new.',
                       'Three discussion prompts and 28 actions in the first coding prompt are common history.',
                       'Historical pre-episode tool dumps were absent in both conditions.',
                       'Memory ingestion repairs occurred during its recorded run without changing actor context policy or candidate code.',
                       'Normalized behavioral tests are frozen before the full-context run but informed by the earlier memory failures.',
                       'The gateway exposes no verified context limit: fail rather than truncate.',
                       'The fixed nine-module actor test allowlist is identical; benchmark tests can only be validated afterward.']})
    s.save(root / 'comparison-plan.json', common)
    folder = root / 'full'
    workspace = folder / 'workspace'
    workspace.mkdir(parents=True)
    archive = subprocess.check_output(['git', 'archive', p['starting_revision']])
    with tarfile.open(fileobj=io.BytesIO(archive)) as stream:
        for member in stream.getmembers():
            (workspace / member.name).resolve().relative_to(workspace.resolve())
            if not (member.isdir() or member.isfile()):
                raise ValueError('only ordinary checkout entries allowed')
        stream.extractall(workspace, filter='data')
    actual = {x.relative_to(workspace).as_posix(): s.evaluation.digest(x)
              for x in workspace.rglob('*') if x.is_file()}
    if actual != p['starting_file_sha256s']:
        raise ValueError('control code differs from the common pre-edit checkpoint')
    for name in ('events', 'steps'):
        shutil.copytree(predecessor / name, folder / name,
                        ignore=shutil.ignore_patterns('__pycache__', '.pytest_cache'))
    shutil.copytree(acceptance, folder / 'acceptance')
    s.save(folder / 'plan.json', common)
    s.emit(phase='control_prepared', root=str(root), fork_step=first['step'], fork_action=first['action'],
           inherited_responses=33, new_memory_calls=0)


def configure(root):
    plan = s.payload(root / 'plan.json')
    if plan['paired_implementation_sha256s'] != implementation_hashes():
        raise ValueError('frozen implementation changed')
    for name, digest in plan['acceptance_sha256s'].items():
        if s.evaluation.digest(root / 'acceptance' / name) != digest:
            raise ValueError('hidden evaluation suite changed')
    s.SYSTEM, s.PROMPT_CAP, s.ACTION_CAP = SYSTEM, CAPS[plan['arm']], 80
    s.ALLOWED_TESTS = {'test_' + name + '.py' for name in TEST_NAMES}
    return plan


def run(root):
    configure(root)
    if (root / 'STOP').exists():
        raise ValueError('stopped runs require explicit reconciliation')
    s.execute = execute
    s.Gateway = transport.RecoveringGateway
    s.SessionMemory = FullHistory

    def messages_for(prompt, context, latest):
        messages = full_messages(FullHistory.active.rows)
        if s.count_chat_prompt_token_proxy(messages) > CAPS['full']:
            raise ValueError('full context exceeded its declared limit; no compaction allowed')
        return messages

    s.messages_for = messages_for
    try:
        s.run(root)
    except Exception as error:
        s.save(root / 'failure.json', {'error_type': type(error).__name__, 'detail': str(error),
            'complete_prompts': len(list((root / 'steps').glob('*/complete.json')))})
        raise


def validate(root):
    configure(root)
    groups = {'behavior': [str((root / 'acceptance/test_paired_behavior.py').resolve())],
              'legacy-api': [str((root / 'acceptance/test_independent_behavior.py').resolve()) +
                             '::test_wall_time_does_not_change_retrieval_energy'],
              'candidate-regression': ['tests/test_' + name + '.py' for name in TEST_NAMES]}
    for group, paths in groups.items():
        folder = root / 'quality' / group
        if folder.exists():
            raise ValueError('quality result already exists; do not overwrite')
        folder.mkdir(parents=True)
        args = [*paths, '-q', '-m', 'not slow', '--basetemp', (folder / 'temp').as_posix(),
                '--junitxml', (folder / 'results.xml').as_posix()]
        code = 'import sys,pytest;sys.path.insert(0,"src");raise SystemExit(pytest.main(' + repr(args) + '))'
        start = time.perf_counter()
        completed = subprocess.run([sys.executable, '-X', 'utf8', '-c', code], cwd=root / 'workspace',
                                   text=True, encoding='utf-8', capture_output=True, timeout=240)
        (folder / 'pytest.log').write_bytes((completed.stdout + completed.stderr).encode())
        s.save(folder / 'result.json', {'exit_code': completed.returncode, 'elapsed_s': time.perf_counter()-start,
            'log_sha256': s.evaluation.digest(folder / 'pytest.log'), 'actor_received_results': False})
        s.emit(phase='quality_checked', arm=root.name, group=group, exit_code=completed.returncode,
               tail=completed.stdout[-900:])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('prepare', 'run', 'gateway', 'validate'))
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.phase == 'prepare':
        prepare(root)
    elif args.phase == 'gateway':
        configure(root)
        s.gateway_worker(root)
    elif args.phase == 'run':
        run(root)
    else:
        validate(root)
