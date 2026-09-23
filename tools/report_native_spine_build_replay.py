"""Produce a prompt-by-prompt audit of the recorded eight-prompt build replay."""
import argparse
from collections import Counter
import difflib
import hashlib
import json
from pathlib import Path
import re
import subprocess


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def candidate_files(files):
    # PYTEST_ADDOPTS strips unquoted Windows backslashes. Earlier test runs
    # consequently created drive-relative temp folders inside the checkout.
    # Keep those recorded artifacts, but exclude test runtime files from code diffs.
    return {name: value for name, value in files.items()
        if not re.fullmatch(r'KeytoneDocumentsGitHubmemory_condense\.worktreesingest-speedeval_resultsnative-spine-build-replay-20260916-r1pytest-tempstep-\d+-action-\d+',
            name.replace('\\', '/').split('/')[0])}


def report(root):
    root = root.resolve()
    plan = read(root / 'replay-plan.json')
    previous = candidate_files(read(root / 'baseline-file-sha256s.json')['file_sha256s'])
    steps = []
    for index, prompt in enumerate(plan['prompts']):
        folder = root / 'steps' / f'{index:02d}'
        done = read(folder / 'complete.json')
        context = read(folder / 'memory-context.json')
        actions, tests, generation_s, proxies = [], [], 0, []
        for path in sorted((folder / 'actions').glob('*.response.json')):
            response = read(path)
            request = read(path.with_name(path.name.replace('.response.json', '.request.json')))
            tool = read(path.with_name(path.name.replace('.response.json', '.tool.json')))
            actions.append(tool['action'])
            generation_s += response['elapsed_s']
            proxies.append(request.get('prompt_token_proxy'))
            if tool['action'] == 'test':
                tests.append({'action': path.stem.split('.')[0], 'result': tool['result'],
                    'log': path.with_name(path.name.replace('.response.json', '.pytest.txt')).name})
        current = candidate_files(done['file_sha256s'])
        changed = sorted(p for p in set(previous) | set(current) if previous.get(p) != current.get(p))
        steps.append({'step': index + 1, 'prompt': prompt, 'reply': done['message'],
            'actions': dict(Counter(actions)), 'action_count': len(actions),
            'generation_s': generation_s, 'request_token_proxies': proxies,
            'changed_files': changed, 'tests': tests,
            'historical_message_count': context['snapshot']['turn_count'],
            'retrieved_evidence_tokens': context['rendered']['token_count'],
            'raw_reads_during_routing': context['routing']['raw_reads_during_routing'],
            'query_qwen_passes': context['routing']['query_qwen_passes'],
            'retrieved_text': context['text'],
            'protocol_deviation': done.get('protocol_deviation')})
        previous = current
    baseline = read(root / 'baseline-file-sha256s.json')['file_sha256s']
    changed = sorted(p for p in set(baseline) | set(previous) if baseline.get(p) != previous.get(p))
    diffs = []
    for name in changed:
        path = root / 'workspace' / name
        if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() != previous[name]:
            raise ValueError(f'File changed after final reply: {name}')
        old = subprocess.run(['git', 'show', plan['starting_revision'] + ':' + name.replace('\\', '/')],
            capture_output=True, check=False)
        before = old.stdout.decode('utf-8').splitlines(keepends=True) if old.returncode == 0 else []
        after = path.read_text(encoding='utf-8').splitlines(keepends=True) if path.exists() else []
        diffs.extend(difflib.unified_diff(before, after, fromfile='a/' + name, tofile='b/' + name))
    (root / 'candidate.patch').write_text(''.join(diffs), encoding='utf-8')
    payload = {'source_snapshot_sha256': plan['source_snapshot_sha256'],
        'starting_revision': plan['starting_revision'], 'seed_messages': len(plan['seed']),
        'seed_tokens': plan['seed_tokens'], 'steps': steps, 'changed_files': changed,
        'raw_inputs_to_qwen': False, 'original_future_assistant_responses_used': False,
        'limitation': 'Eight-prompt feature episode with live repository tools; not a full-transcript or 1M-token coding benchmark. Harness corrections are recorded in adapter artifacts.'}
    validation = read(root / 'validation-report.json')
    payload['validation'] = validation
    payload['total_generation_s'] = sum(s['generation_s'] for s in steps)
    payload['prompt_selection'] = 'All eight consecutive substantive user prompts from original parsed indices 2052 through 2194.'
    payload['assessment'] = {
        'outcome': 'Actual implementation and documentation delivered; partial behavioral success.',
        'continuity_gap': 'The delivery-plan packet omitted the recent explicit turn-decay correction while the latest generated history-review reply remained in live context.',
        'behavioral_findings': [
            'Turn decay, wall-time invariance for decay, pins and close/reopen persistence pass.',
            'Subsequent rapid turns and duplicate evidence do not reinforce memories under the restored 300-second throttle.',
            'After 16 rapid turns the repeatedly used item has energy 0.5535639528 versus 0.5656854249 for the unused item.',
            'MCP stats prints stored energy 0.80 for an item whose current decayed energy is 0.40.'],
        'historical_oracle_limit': '85/118 historical implementation checks pass; 33 fail, including API naming and clock-layout differences. This is not the behavioral build score.',
        'independent_checks_limit': 'Eight review checks were designed during the replay and hidden from the actor. Four pass and four fail; correlated checks are not a statistical coding-accuracy estimate.',
        'scope': 'The final Go response implements the immediate compatibility milestone, not every future stage listed in the roadmap.'}
    (root / 'build-report.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    lines = ['# Raw-transcript build replay', '',
        '**Partial behavioral success:** the model produced real changes in 16 files. Final regression: 240 passed. Independent review: four checks passed and four failed.', '',
        'Three failures concern the same reinforcement policy: rapid subsequent use and repeated evidence get no reinforcement under the restored 300-second throttle. The fourth is an MCP display bug: it shows energy 0.80 when current decayed energy is 0.40. Wall-time immunity for decay, turn advancement, pins, and persistence across reopening pass.', '',
        'The delivery-plan packet omitted the recent explicit user correction and instead returned 173 tokens of older generic architecture requests. The latest generated reply remained available and its reinterpretation carried into the plan. Active user requirements need stronger protection in this workflow.', '',
        'The actor took 185 model actions and 2,542.7 seconds (42.4 minutes) of API generation, excluding tool execution, memory setup and orchestration. This custom one-action-per-response harness had recorded corrections to tool history, action limits, file search and temporary-directory handling. Its coding latency must not be compared with the short-answer QA median.', '',
        '| Original prompt | Observed outcome |', '| --- | --- |',
        '| Day 14 | Explained the historical evaluation checkpoint. |',
        '| Differential decay per subsequent turn | Stated the intended reinforcement and decay behavior. |',
        '| Goldilocks zone | Preserved the retention/token-saving tradeoff in its reply. |',
        '| Decay per turn, not real time | Implemented a turn-based refactor; 216 selected checks passed at that stage. |',
        '| Review commit history | Identified compatibility and documentation gaps, but proposed reverting reinforcement to a wall-clock gate. |',
        '| Delivery plan | Carried that reinterpretation into a staged plan; the recent explicit correction was missing from retrieval. |',
        '| Put it in docs | Actually edited the roadmap. |',
        '| Go | Implemented the immediate compatibility milestone; final independent review finds the failures above. |', '',
        'The historical later-implementation oracle also ran: 85 passed, 33 failed. Several failures require different API names or a particular persisted clock layout, so this is a compatibility diagnostic rather than a behavioral score. Independent checks were hidden review checks designed during the run, not a pre-registered benchmark.', '',
        payload['limitation'], '',
        f"Seed: {len(plan['seed'])} earlier messages, {plan['seed_tokens']:,} tokens. Model: {plan['model']}.", '',
        'These are all eight consecutive substantive user prompts in the selected source segment. Only newly generated responses carry forward. Qwen receives summaries; the answer model receives selected raw user evidence, the latest exchange, and live file-tool results. All eight routes record zero raw reads during routing and zero query-time Qwen passes.', '',
        'Artifacts: [candidate patch](candidate.patch), [validation](validation-report.json), [independent failures](independent-behavior.log), [regression log](regression.log). The generated checkout remains isolated; it was not merged into the live application.', '']
    for step in steps:
        lines += [f"## Prompt {step['step']}", '', step['prompt']['text'], '',
            f"Actions: {step['action_count']}; generation time: {step['generation_s']:.1f} s; historical messages: {step['historical_message_count']}.", '',
            'Generated reply:', '', step['reply'], '',
            'Changed files: ' + (', '.join('`' + p + '`' for p in step['changed_files']) or 'none') + '.', '']
        for test in step['tests']:
            lines += [f"Test action {test['action']}: `{test['result'].splitlines()[0]}`.", '']
    lines += ['Final behavior and regression checks are recorded separately in `validation-report.json` and the associated test logs.', '']
    (root / 'build-report.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps({'steps': len(steps), 'changed_files': len(changed), 'action_count': sum(s['action_count'] for s in steps)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    report(parser.parse_args().root)
