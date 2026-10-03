"""Pinned local summarizer invocation; no evaluation-run dependency."""
from pathlib import Path
import subprocess
import time

def gpu():
    return subprocess.check_output([
        'nvidia-smi', '--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu',
        '--format=csv',
    ], text=True).strip()

def invoke(client, job):
    started = time.perf_counter()
    content, first, finish, usage, timings = '', None, None, None, None
    with client.chat.completions.create(
        model='Llama-3.2-3B-Instruct-Q4_K_M', messages=job['messages'],
        temperature=0, seed=42, max_tokens=job['max_tokens'],
        stream=True, stream_options={'include_usage': True},
        extra_body={'cache_prompt': False},
    ) as stream:
        for chunk in stream:
            timings = (chunk.model_extra or {}).get('timings', timings)
            if chunk.usage:
                usage = chunk.usage.model_dump()
            for choice in chunk.choices:
                if choice.delta.content:
                    if first is None:
                        first = time.perf_counter() - started
                    content += choice.delta.content
                finish = choice.finish_reason or finish
    return dict(content=content, elapsed_s=time.perf_counter()-started,
                ttft_s=first, finish_reason=finish, usage=usage, timings=timings)

MODEL_SHA = '6c1a2b41161032677be168d354123594c0e6e67d2b9227c84f296ad037c728ff'
