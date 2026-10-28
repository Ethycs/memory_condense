"""Measure how much of one serving linker pass is CPU tokenization.

Step B0 of ``docs/01 - Design/05 - Local Model Handoff and Concurrent
Linking.md``: before any change makes the encoder safe for concurrent forward
passes, find the share of a pass that a CPU-only tokenizer pool could take off
the GPU gate. If that share is small, step B0 is the whole task.

The pass is the one the proxy runs: ``QwenAttentionHeadSurpriseScorer`` over a
window of eight user summaries, built exactly as ``runtime/attention.py``
builds it. Tokenizer time is the outermost tokenizer call only, so a
``__call__`` that delegates to ``encode_plus`` is counted once.

Run: ``python -m tools.measure_linker_tokenization --model-dir <Qwen3-8B>``.
"""
from __future__ import annotations

import argparse
import functools
import json
import random
import statistics
import threading
import time
from pathlib import Path

_WORDS = (
    "the user asked about deployment budget schedule migration database cache index proxy "
    "session memory summary link evidence receipt model runtime queue window token layer "
    "attention review release branch commit test failure retry latency throughput GPU CPU "
    "conversation decision plan design note package milestone owner policy profile handoff"
).split()


class _TokenizerClock:
    """Times only the outermost tokenizer call on each thread."""

    _METHODS = ("__call__", "encode", "encode_plus", "batch_encode_plus", "tokenize")

    def __init__(self, tokenizer_type):
        self.type, self.seconds, self.calls = tokenizer_type, 0.0, 0
        self._depth = threading.local()
        self._originals = {}

    def __enter__(self):
        for name in self._METHODS:
            original = getattr(self.type, name, None)
            if original is None:
                continue
            self._originals[name] = original
            setattr(self.type, name, self._timed(original))
        return self

    def __exit__(self, *_exc):
        for name, original in self._originals.items():
            setattr(self.type, name, original)

    def _timed(self, original):
        clock = self

        @functools.wraps(original)
        def timed(*args, **kwargs):
            depth = getattr(clock._depth, "value", 0)
            clock._depth.value = depth + 1
            started = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                clock._depth.value = depth
                if depth == 0:
                    clock.seconds += time.perf_counter() - started
                    clock.calls += 1
        return timed


def _window(rng, size=8, words=(60, 100)):
    return [" ".join(rng.choice(_WORDS) for _ in range(rng.randint(*words))) + "."
            for _ in range(size)]


def measure(model_dir, *, passes, warmup, seed):
    import torch
    from memory_condense.associations.qwen_memory_linker import QwenMemoryLinker
    from memory_condense.modeling.qwen_prefix import Qwen3PrefixEncoder
    from memory_condense.search.episodes.qwen_episode_signal import QwenAttentionHeadSurpriseScorer

    # The same construction as runtime/attention.py:LocalAttention.load_scorer.
    encoder = Qwen3PrefixEncoder(model_dir, layers=6, device="cuda", dtype="float16",
                                 verify_checkpoint_files=False)
    scorer = QwenAttentionHeadSurpriseScorer(
        QwenMemoryLinker(encoder, layer=5, max_candidates=8, max_workspace_tokens=4096),
        max_spans=8, span_token_cap=128)
    rng = random.Random(seed)
    for _ in range(warmup):
        scorer.score_sequence(_window(rng))
    torch.cuda.synchronize()
    rows = []
    with _TokenizerClock(type(encoder.tokenizer)) as clock:
        for _ in range(passes):
            texts = _window(rng)
            before_s, before_calls = clock.seconds, clock.calls
            started = time.perf_counter()
            scorer.score_sequence(texts)
            torch.cuda.synchronize()
            total = time.perf_counter() - started
            rows.append(dict(total_s=total, tokenizer_s=clock.seconds - before_s,
                             tokenizer_calls=clock.calls - before_calls))
    shares = [row["tokenizer_s"] / row["total_s"] for row in rows]
    return dict(
        model_dir=str(model_dir), device=torch.cuda.get_device_name(0),
        passes=passes, warmup=warmup, seed=seed,
        pass_s_median=statistics.median(r["total_s"] for r in rows),
        tokenizer_s_median=statistics.median(r["tokenizer_s"] for r in rows),
        tokenizer_calls_per_pass=statistics.median(r["tokenizer_calls"] for r in rows),
        tokenizer_share_median=statistics.median(shares),
        tokenizer_share_max=max(shares),
        rows=rows,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--passes", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20261027)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    result = measure(args.model_dir, passes=args.passes, warmup=args.warmup, seed=args.seed)
    text = json.dumps(result, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    summary = {k: v for k, v in result.items() if k != "rows"}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
