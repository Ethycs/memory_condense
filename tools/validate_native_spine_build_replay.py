"""Run independent final validation against the isolated generated checkout."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time


def validate(root):
    root = root.resolve()
    if not (root / 'steps/07/complete.json').exists():
        raise ValueError('all eight original prompts must finish before final validation')
    output_path = root / 'validation-report.json'
    if output_path.exists():
        raise ValueError('final validation already recorded')
    groups = {
        'regression': ['tests/test_' + name + '.py' for name in (
            'decay', 'db', 'memory_store', 'transcript_store', 'condenser',
            'mcp_server', 'eval_recall', 'ranking', 'architecture')],
        'independent-behavior': [str(root / 'acceptance/test_independent_behavior.py')],
        'historical-api-compatibility': [str(root / 'acceptance' / ('test_' + name + '.py'))
            for name in ('decay', 'db', 'memory_store')],
    }
    results = {}
    for name, paths in groups.items():
        temp = root / 'validation-temp' / name
        if temp.exists() or not temp.is_relative_to(root):
            raise ValueError('validation temp must be fresh and inside replay root')
        temp.parent.mkdir(parents=True, exist_ok=True)
        argv = [*paths, '-q', '-m', 'not slow', '--basetemp', str(temp),
            '--junitxml', str(root / (name + '.xml'))]
        code = 'import sys,pytest;sys.path.insert(0,"src");raise SystemExit(pytest.main(' + repr(argv) + '))'
        started = time.perf_counter()
        completed = subprocess.run([sys.executable, '-X', 'utf8', '-c', code],
            cwd=root / 'workspace', capture_output=True, text=True, timeout=180)
        text = completed.stdout + completed.stderr
        logfile = root / (name + '.log')
        logfile.write_text(text, encoding='utf-8')
        summaries = [line for line in text.splitlines()
            if re.search(r'\d+ (passed|failed|error|skipped)', line)]
        results[name] = {'exit_code': completed.returncode,
            'elapsed_s': time.perf_counter() - started, 'paths': paths,
            'summary': summaries[-1] if summaries else None,
            'log': str(logfile), 'log_sha256': hashlib.sha256(logfile.read_bytes()).hexdigest()}
        print(json.dumps({'group': name, **results[name]}), flush=True)
    results['note'] = ('Historical tests depend on the original later implementation API; '
        'API collection failures are not a per-test behavioral failure count. Independent '
        'behavioral checks use the candidate public interface and were hidden from its tools.')
    output_path.write_text(json.dumps(results, indent=2), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    validate(parser.parse_args().root)
