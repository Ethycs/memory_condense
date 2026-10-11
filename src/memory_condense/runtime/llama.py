"""Pinned local summarizer invocation; no evaluation-run dependency."""
import json
from pathlib import Path
import subprocess
import time

def gpu(*, strict=False):
    """Driver-reported inventory for the run receipt.

    An absent or failing nvidia-smi is a reporting gap, not a reason to fail an
    otherwise healthy CUDA run. This is called from CompressedAttention.prepare()
    on conversation open, which nothing wraps in a tolerant handler, and the tool
    is routinely missing inside a container that exposes only compute. doctor()
    passes strict=True so the gap is still reported as its own failed check.
    The return stays a str either way: it is embedded in sealed JSON receipts.
    """
    try:
        return subprocess.check_output([
            'nvidia-smi', '--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu',
            '--format=csv',
        ], text=True).strip()
    except (OSError, subprocess.SubprocessError):
        if strict:
            raise
        return 'nvidia-smi unavailable'

def invoke(client, job):
    started = time.perf_counter()
    content, first, finish, usage, timings = '', None, None, None, None
    payload = dict(model='Llama-3.2-3B-Instruct-Q4_K_M', messages=job['messages'],
        temperature=0, seed=42, max_tokens=job['max_tokens'], stream=True,
        stream_options={'include_usage': True}, cache_prompt=False)
    # llama.cpp exposes a small OpenAI-compatible SSE endpoint. Calling it
    # directly avoids importing the SDK's large generated type tree merely to
    # talk to a loopback server; provider-facing code retains its own clients.
    with client.stream('POST', 'chat/completions', json=payload) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line or not line.startswith('data:'):
                continue
            encoded = line[5:].strip()
            if encoded == '[DONE]':
                break
            chunk = json.loads(encoded)
            timings = chunk.get('timings', timings)
            if chunk.get('usage') is not None:
                usage = chunk['usage']
            for choice in chunk.get('choices', ()):
                value = choice.get('delta', {}).get('content')
                if value:
                    if first is None:
                        first = time.perf_counter() - started
                    content += value
                finish = choice.get('finish_reason') or finish
    return dict(content=content, elapsed_s=time.perf_counter()-started,
                ttft_s=first, finish_reason=finish, usage=usage, timings=timings)

MODEL_SHA = '6c1a2b41161032677be168d354123594c0e6e67d2b9227c84f296ad037c728ff'
