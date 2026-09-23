"""Run the fixed ten-session campaign with separate ingest and answer processes."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def save_status(root, **values):
    payload = dict(controller_pid=os.getpid(), updated_at=time.time(), **values)
    temp = root / 'controller-status.tmp'
    temp.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    temp.replace(root / 'controller-status.json')
    print(json.dumps(payload), flush=True)


def start(root, batch, phase):
    folder = root / f'history-{batch:02d}'
    log = (folder / f'{phase}.log').open('ab', buffering=0)
    command = [sys.executable, '-X', 'utf8', '-u', '-m', 'tools.native_spine_ten100',
               phase, '--root', str(root), '--batch', str(batch)]
    if phase in ('author', 'run', 'report'):
        command.append('--enable-provider')
    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                               creationflags=subprocess.CREATE_NO_WINDOW)
    log.close()
    return process


def run(root):
    # Only source-based authoring overlaps ingestion; timed answers are serial
    # and run after both workers exit, without competing local GPU work.
    for batch in range(1, 11):
        if (root / 'STOP').exists():
            save_status(root, state='stopped', history=batch)
            return
        folder = root / f'history-{batch:02d}'
        workers = {}
        for phase, receipt in (('author', 'questions/questions.json'),
                               ('ingest', 'ingest-complete.json')):
            if not (folder / receipt).exists():
                workers[phase] = start(root, batch, phase)
        save_status(root, state='preparing_history', history=batch,
                    workers={k: p.pid for k, p in workers.items()})
        codes = {phase: process.wait() for phase, process in workers.items()}
        if any(codes.values()):
            save_status(root, state='failed', history=batch, exit_codes=codes)
            raise RuntimeError(f'history {batch} preparation failed: {codes}')
        for phase, receipt in (('run', 'answers-complete.json'), ('report', 'report.json')):
            if (folder / receipt).exists():
                continue
            process = start(root, batch, phase)
            save_status(root, state=phase, history=batch, worker_pid=process.pid)
            code = process.wait()
            if code:
                save_status(root, state='failed', history=batch, phase=phase, exit_code=code)
                raise RuntimeError(f'history {batch} {phase} failed: {code}')
        result = json.loads((folder / 'report.json').read_text(encoding='utf-8'))
        save_status(root, state='history_complete', history=batch, accuracy=result['accuracy'])
    subprocess.run([sys.executable, '-X', 'utf8', '-m', 'tools.report_native_spine_ten100',
                    '--root', str(root)], check=True, creationflags=subprocess.CREATE_NO_WINDOW)
    save_status(root, state='complete', histories=10, questions=1000)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    run(parser.parse_args().root)
