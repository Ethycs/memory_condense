"""Controller-side isolated unittest execution with bounded Windows job resources."""
from __future__ import annotations

import ctypes
from ctypes import wintypes
import os
from pathlib import Path
import subprocess
import sys
import time


def attach_job(process):
    if os.name != 'nt':
        raise RuntimeError('This controller currently requires Windows job resource limits')
    class Basic(ctypes.Structure):
        _fields_ = [('process_time', ctypes.c_int64), ('job_time', ctypes.c_int64), ('flags', wintypes.DWORD),
                    ('min_working', ctypes.c_size_t), ('max_working', ctypes.c_size_t), ('active', wintypes.DWORD),
                    ('affinity', ctypes.c_size_t), ('priority', wintypes.DWORD), ('scheduling', wintypes.DWORD)]
    class IO(ctypes.Structure):
        _fields_ = [(name, ctypes.c_uint64) for name in ('read_ops', 'write_ops', 'other_ops', 'read_bytes', 'write_bytes', 'other_bytes')]
    class Extended(ctypes.Structure):
        _fields_ = [('basic', Basic), ('io', IO), ('process_memory', ctypes.c_size_t), ('job_memory', ctypes.c_size_t),
                    ('peak_process', ctypes.c_size_t), ('peak_job', ctypes.c_size_t)]
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.CreateJobObjectW.restype = wintypes.HANDLE
    kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    kernel.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    kernel.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    job = kernel.CreateJobObjectW(None, None)
    if not job:
        raise ctypes.WinError(ctypes.get_last_error())
    limit = Extended()
    limit.basic.flags = 0x2000 | 0x100 | 0x8 | 0x2  # kill-on-close, memory, active-process, CPU time
    limit.basic.active = 1
    limit.basic.process_time = 60 * 10_000_000
    limit.process_memory = 512 * 1024 * 1024
    if (not kernel.SetInformationJobObject(job, 9, ctypes.byref(limit), ctypes.sizeof(limit))
            or not kernel.AssignProcessToJobObject(job, wintypes.HANDLE(int(process._handle)))):
        error = ctypes.get_last_error()
        kernel.CloseHandle(job)
        raise ctypes.WinError(error)
    return lambda: kernel.CloseHandle(job)


def execute_tests(workspace, *, acceptance_case=None):
    workspace = Path(workspace).resolve()
    sandbox = Path(__file__).with_name('engineering_research_sandbox.py').resolve()
    cmd = [sys.executable, '-I', '-B', str(sandbox), str(workspace), 'acceptance' if acceptance_case else 'unit']
    if acceptance_case:
        cmd += [str(Path(__file__).with_name('engineering_research_checks.py').resolve()), acceptance_case]
    env = {k: v for k, v in os.environ.items() if k.upper() in ('SYSTEMROOT', 'WINDIR', 'COMSPEC')}
    env.update(PYTHONUTF8='1', TEMP=str(workspace), TMP=str(workspace), OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
    started = time.perf_counter()
    process = subprocess.Popen(cmd, cwd=workspace, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, encoding='utf-8', creationflags=subprocess.CREATE_NO_WINDOW)
    close_job = None
    try:
        close_job = attach_job(process)
        try:
            output, _ = process.communicate('RUN\n', timeout=75)
            timed_out = False
        except subprocess.TimeoutExpired:
            process.kill()
            output, _ = process.communicate()
            timed_out = True
        return dict(exit_code=process.returncode, output=output[:100_000], output_truncated=len(output)>100_000,
                    elapsed_s=time.perf_counter()-started, timed_out=timed_out,
                    limits=dict(memory_mib=512, cpu_seconds=60, wall_seconds=75, processes=1),
                    environment_credentials_removed=True)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        if close_job:
            close_job()
