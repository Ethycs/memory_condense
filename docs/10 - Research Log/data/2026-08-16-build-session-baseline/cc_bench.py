"""R6: the build session itself as a benchmark. Free, keyless, in-regime."""
import argparse
import json, random, tempfile, time
from pathlib import Path
from memory_condense._tokenizer import count_tokens
from memory_condense.eval.benchmark import ingest_sample
from memory_condense.eval.recall import _assemble, contains_answer
from memory_condense.eval.schemas import EvalConfig, RetrievalConfig
from memory_condense.loader import BenchmarkQuestion, BenchmarkSample

ROOT = Path(__file__).resolve().parents[4]
BASE = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--turns",
    type=Path,
    default=ROOT / "data" / "build-session-8f7f7561.turns.json",
)
parser.add_argument("--probe", type=Path, default=BASE / "cc_probe.json")
parser.add_argument(
    "--output",
    type=Path,
    default=ROOT / "eval_results" / "build_session_b0_reproduction.json",
)
parser.add_argument(
    "--skip-memory",
    action="store_true",
    help="Reproduce only B0's raw retrieval arms; skip the slower extracted-memory treatment.",
)
args = parser.parse_args()
turns = json.loads(args.turns.read_text(encoding="utf-8"))
probe = json.loads(args.probe.read_text(encoding="utf-8"))

sample = BenchmarkSample(
    sample_id="build-session-8f7f7561",
    turns=[(r if r in ("user", "assistant") else "system", t) for r, _, t in turns],
    questions=[
        BenchmarkQuestion(question_id=f"q{i}", question=q, answer=a, category=c)
        for i, (q, a, c, _) in enumerate(probe)
    ],
)
print(f"{sample.sample_id}: {len(sample.turns)} turns, {len(sample.questions)} questions", flush=True)

diagnostics = {
    "sample_id": sample.sample_id,
    "protocol": "bs-probe-v1 / unchanged B0 raw-retrieval arms",
    "turns_path": str(args.turns),
    "probe_path": str(args.probe),
    "arms": {},
}

t0 = time.time()
mc = ingest_sample(sample, EvalConfig(retrieval=RetrievalConfig()), tempfile.mkdtemp())
print(f"ingest: {time.time()-t0:.0f}s", flush=True)
try:
    rows = mc._db.execute("SELECT text, token_count FROM chunks ORDER BY rowid").fetchall()
    chunk_texts = [r[0] for r in rows]
    sizes = sorted(r[1] for r in rows)
    n_ch = len(sizes)
    print(f"chunks: {n_ch}, median {sizes[n_ch//2]} tok, total {sum(sizes):,} tok", flush=True)

    qs = sample.questions
    n = len(qs)

    def report(name, per_q_texts, per_q_tokens):
        hits = sum(contains_answer(ts, q.answer) for ts, q in zip(per_q_texts, qs))
        ctx = sum(per_q_tokens) / n
        print(f"{name:<18}{hits/n:>7.1%}{ctx:>9.0f}", flush=True)
        diagnostics["arms"][name] = {
            "recall": hits / n,
            "mean_context_tokens": ctx,
            "questions": [
                {
                    "question_id": q.question_id,
                    "category": q.category,
                    "question": q.question,
                    "answer": q.answer,
                    "hit": contains_answer(texts, q.answer),
                    "context_tokens": tokens,
                    "answer_rank": next(
                        (
                            rank
                            for rank, text in enumerate(texts, start=1)
                            if contains_answer([text], q.answer)
                        ),
                        None,
                    ),
                }
                for q, texts, tokens in zip(qs, per_q_texts, per_q_tokens)
            ],
        }
        return hits / n

    print(f"\n{'arm':<18}{'recall':>7}{'ctx tok':>9}", flush=True)
    print("-" * 34, flush=True)

    # ceilings
    turn_texts = [t for _, t in sample.turns]
    ceil_turns = sum(contains_answer(turn_texts, q.answer) for q in qs) / n
    ceil_chunks = sum(contains_answer(chunk_texts, q.answer) for q in qs) / n
    tot = sum(sizes)
    print(f"{'ceiling (turns)':<18}{ceil_turns:>7.1%}{sum(count_tokens(t) for t in turn_texts):>9,}", flush=True)
    print(f"{'ceiling (chunks)':<18}{ceil_chunks:>7.1%}{tot:>9,}", flush=True)

    # naive baselines
    rng = random.Random(0)
    rand = [rng.sample(chunk_texts, min(40, n_ch)) for _ in range(n)]
    report("random k=40", rand, [sum(count_tokens(t) for t in ts) for ts in rand])
    recent = chunk_texts[-40:]
    rtoks = sum(count_tokens(t) for t in recent)
    report("recent k=40", [recent] * n, [rtoks] * n)

    # arms
    ARMS = [
        ("dense k=10",  RetrievalConfig(k=10, mode="dense")),
        ("hybrid k=10", RetrievalConfig(k=10, mode="hybrid")),
        ("hybrid k=50", RetrievalConfig(k=50, mode="hybrid")),
        ("span x2",     RetrievalConfig(mode="span", k_per_level=2)),
        ("span x4",     RetrievalConfig(mode="span", k_per_level=4)),
    ]
    hitsets = {}
    for name, rc in ARMS:
        cfg = EvalConfig(retrieval=rc)
        texts_per_q, toks_per_q, got = [], [], set()
        for i, q in enumerate(qs):
            h, b = _assemble(mc, q.question, cfg)
            ts = h + b
            texts_per_q.append(ts)
            toks_per_q.append(sum(count_tokens(t) for t in ts))
            if contains_answer(ts, q.answer):
                got.add(i)
        hitsets[name] = got
        report(name, texts_per_q, toks_per_q)

    u = hitsets["span x4"] | hitsets["hybrid k=50"]
    print(f"\nunion span x4 + hybrid k=50: {len(u)/n:.1%}", flush=True)
    print("\nby phase (span x4):", flush=True)
    from collections import Counter
    tot_c, hit_c = Counter(), Counter()
    for i, q in enumerate(qs):
        tot_c[q.category] += 1
        if i in hitsets["span x4"]:
            hit_c[q.category] += 1
    for c in sorted(tot_c):
        print(f"  {c}: {hit_c[c]}/{tot_c[c]}", flush=True)
    missed = [qs[i].answer for i in range(n) if i not in (hitsets["span x4"] | hitsets["hybrid k=50"])]
    print("\nmissed by both best arms:", missed, flush=True)

    # Preserve enough evidence to diagnose B0's misses without repeating the
    # 15-minute corpus ingest.  Only questions missed by hybrid k=10 are
    # included; excerpts are bounded because this file is a diagnostic, not a
    # second copy of the private transcript snapshot.
    b0_misses = sorted(set(range(n)) - hitsets["hybrid k=10"])
    diagnostics["b0_misses"] = []
    for i in b0_misses:
        question = qs[i]
        ranked = mc.search_hybrid(question.question, k=50)
        diagnostics["b0_misses"].append(
            {
                "question_id": question.question_id,
                "question": question.question,
                "answer": question.answer,
                "top_50": [
                    {
                        "rank": rank,
                        "score": result.score,
                        "dense_score": result.dense_score,
                        "lexical_score": result.lexical_score,
                        "token_count": result.chunk.token_count,
                        "answer_hit": contains_answer(
                            [result.chunk.text], question.answer
                        ),
                        "text_excerpt": result.chunk.text[:600],
                    }
                    for rank, result in enumerate(ranked, start=1)
                ],
            }
        )
finally:
    mc.close()

args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
print(f"diagnostics: {args.output}", flush=True)

# Memory is a stateful treatment, not a different read over the chunk store
# above.  It must be ingested in memory mode so extraction is enabled and the
# packer is configured for k expansions.  The original B0 script reused the
# dense-ingested store here, so its row had no memory items, used dense rather
# than hybrid expansions, and packed at most three hits.  Keep the extra ingest
# explicit even though it is slow; otherwise the row is not the system.
if not args.skip_memory:
    memory_config = EvalConfig(
        retrieval=RetrievalConfig(k=10, mode="memory", k_memories=8)
    )
    print("\ningesting a separate, extraction-enabled memory treatment...", flush=True)
    t0 = time.time()
    memory_mc = ingest_sample(sample, memory_config, tempfile.mkdtemp())
    print(f"memory ingest: {time.time()-t0:.0f}s", flush=True)
    try:
        texts_per_q, toks_per_q = [], []
        for question in qs:
            header, body = _assemble(memory_mc, question.question, memory_config)
            texts = header + body
            texts_per_q.append(texts)
            toks_per_q.append(sum(count_tokens(text) for text in texts))
        report("memory k=10 (true)", texts_per_q, toks_per_q)
    finally:
        memory_mc.close()
