"""Compile complete-memory hierarchies with dependency waves of summary-only Qwen jobs.

Ingest has no question or gold reader. Completed batches and scalar attention
signals replay locally. Network failures retain reservations; no retries occur.
The prepare phase seals the next concrete wave without making gateway calls.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from memory_condense.domain._discourse_identity import identity_sha256
from memory_condense.eval.fast_completion_runtime import FastCompletionRuntime
from memory_condense.search.episodes.user_spine_hierarchy import (
    UserSpineHierarchy, build_user_spine_hierarchy, compile_user_spine_exchanges,
)
from memory_condense.search.section_summary import SectionSummary
from memory_condense.search.spine_merge_batch import (
    PendingMerge, SummaryMergeCache, merge_batch_messages, pack_merge_batches,
)
from memory_condense.search.spine_summary import SpineSummaryFragment, SpineSummaryRequest
from tools.matched_eval.artifacts import publish_sealed_json, read_sealed_json
from tools.run_hot_reduced30_answer_judge import _authenticated_records, _completion_client, _run_exactly_authorized


GATEWAY = "https://central-dev.zt:4000/v1"
MODEL = "qwen3-8b"
IMPLEMENTATION = ("tools/build_spine_corpus_hierarchy.py", "src/memory_condense/search/spine_merge_batch.py",
    "src/memory_condense/search/spine_summary.py", "src/memory_condense/search/episodes/user_spine_hierarchy.py",
    "src/memory_condense/search/episodes/qwen_episode_signal.py", "src/memory_condense/modeling/qwen_prefix.py",
    "src/memory_condense/associations/qwen_memory_linker.py", "src/memory_condense/search/section_routing.py")


def restore_request(row):
    return SpineSummaryRequest(**dict(row, fragments=tuple(SpineSummaryFragment(**f) for f in row["fragments"])))


class MergeJournal:
    def __init__(self, root, preflight, enable, budget):
        self.root, self.preflight, self.enable, self.budget = root, preflight, enable, budget
        self.cache = SummaryMergeCache()
        self.calls = self.hits = 0

    def runtime(self, request, client=None):
        return FastCompletionRuntime(checkpoint_dir=self.root / "checkpoints" / request.sha256,
            prompt_population=[request.payload["messages"]], model=MODEL, client=client,
            max_prompt_tokens=7000, max_new_tokens=2048, max_concurrency=1, retries=0,
            request_options={"temperature": 0, "extra_body": {"enable_thinking": False}},
            benchmark_provenance={"summary_batch_sha256": request.sha256})

    def remaining(self, request):
        runtime = self.runtime(request)
        try:
            return 1 - len(_authenticated_records(runtime))
        finally:
            runtime.close()

    def replay(self):
        # Recover completed requests even if execution stopped before parsing.
        for path in sorted((self.root / "requests").glob("*.json")):
            request = read_sealed_json(path)
            if request.payload["preflight_sha256"] != self.preflight.sha256:
                raise ValueError("summary batch belongs to another hierarchy")
            jobs = tuple(restore_request(row) for row in request.payload["jobs"])
            if merge_batch_messages(jobs) != request.payload["messages"]:
                raise ValueError("summary batch reconstruction changed")
            if self.remaining(request):
                continue
            runtime = self.runtime(request)
            try:
                batch = runtime.run()
            finally:
                runtime.close()
            self.cache.accept(jobs, batch.logical_completions[0])
            self.hits += 1

    def resolve(self, pending, phase, wave):
        requests = []
        for jobs in pack_merge_batches(tuple(pending.values())):
            payload = {"preflight_sha256": self.preflight.sha256, "jobs": [asdict(r) for r in jobs],
                       "messages": merge_batch_messages(jobs), "model": MODEL, "raw_inputs": False}
            artifact, _ = publish_sealed_json(self.root / "requests" / (identity_sha256(payload) + ".json"), payload)
            requests.append((artifact, jobs))
        needed = sum(self.remaining(request) for request, _ in requests)
        wave_payload = {
            "preflight_sha256": self.preflight.sha256, "phase": phase, "wave": wave,
            "request_shas": [r.sha256 for r, _ in requests], "merge_jobs": len(pending),
            "maximum_calls": len(requests), "max_concurrency": 4, "retries": 0}
        plan, _ = publish_sealed_json(self.root / "waves" / (identity_sha256(wave_payload) + ".json"), wave_payload)
        print({"phase": phase, "wave": wave, "merge_jobs": len(pending), "missing_calls": needed,
               "wave_sha256": plan.sha256, "new_calls_this_run": self.calls}, flush=True)
        if needed and (not self.enable or self.calls + needed > self.budget):
            return False

        def one(pair):
            request, jobs = pair
            batch, calls, hits, _ = _run_exactly_authorized(
                runtime_factory=lambda client: self.runtime(request, client),
                authorized_provider_calls=self.remaining(request), enable_provider=self.enable,
                client_factory=lambda: _completion_client("LITELLM_KEY", GATEWAY))
            return jobs, batch.logical_completions[0], calls, hits

        # Finish and collect every submitted call before parsing; completed
        # responses remain recoverable if a different batch fails validation.
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(one, requests))
        for jobs, response, calls, hits in results:
            self.calls += calls
            self.hits += hits
            self.cache.accept(jobs, response)
        return True


class ScalarAttentionCache:
    max_spans = 8
    span_token_cap = 128

    def __init__(self, root, preflight):
        self.root, self.preflight = root, preflight
        self.scorer = None
        self.values = {}

    def score_sequence(self, texts):
        from memory_condense.search.episodes.surprise_models import AttentionHeadSurpriseReceipt, ScoredSurpriseSequence
        key = identity_sha256({"preflight_sha256": self.preflight.sha256, "texts": list(texts)})
        if key not in self.values:
            path = self.root / "attention" / (key + ".json")
            if path.exists():
                artifact = read_sealed_json(path)
                row = artifact.payload
                if row["preflight_sha256"] != self.preflight.sha256:
                    raise ValueError("attention cache belongs to another compilation")
                signal = ScoredSurpriseSequence(row["scores"], row["similarities"], AttentionHeadSurpriseReceipt(**row["receipt"]))
            else:
                if self.scorer is None:
                    from memory_condense.associations.qwen_memory_linker import QwenMemoryLinker
                    from memory_condense.modeling.qwen_prefix import Qwen3PrefixEncoder
                    from memory_condense.search.episodes.qwen_episode_signal import QwenAttentionHeadSurpriseScorer
                    print("Loading local Qwen for query-independent user-summary attention...", flush=True)
                    encoder = Qwen3PrefixEncoder(Path("../../.cache/models/Qwen3-8B").resolve(),
                        layers=6, device="cuda", dtype="float16")
                    linker = QwenMemoryLinker(encoder, layer=5, max_candidates=8, max_workspace_tokens=4096)
                    self.scorer = QwenAttentionHeadSurpriseScorer(linker, max_spans=8, span_token_cap=128)
                signal = self.scorer.score_sequence(texts)
                publish_sealed_json(path, {"preflight_sha256": self.preflight.sha256,
                    "scores": signal.scores, "similarities": signal.similarities,
                    "receipt": signal.receipt.identity_payload()})
            signal.validate_inputs(texts)
            self.values[key] = signal
        return self.values[key]


def compile_waves(groups, build, journal, phase):
    done = {}
    wave = 0
    while len(done) < len(groups):
        pending = {}
        for source, rows in groups.items():
            if source in done:
                continue
            try:
                done[source] = build(rows)
            except PendingMerge as missing:
                pending.setdefault(missing.request.prompt_sha256, missing.request)
        if not pending:
            break
        if not journal.resolve(pending, phase, wave):
            return None
        wave += 1
    return {source: done[source] for source in groups}


def run(atoms_path, root, enable=False, max_new_calls=0):
    atoms_artifact = read_sealed_json(atoms_path)
    data = atoms_artifact.payload
    if data.get("format") != "memory-condense-spine-source-bound-atoms-v1":
        raise ValueError("expected source-bound routing atoms")
    atoms = tuple(SectionSummary.from_dict(row) for row in data["atoms"])
    if identity_sha256([a.spans[0].receipt_sha256 for a in atoms]) != data["raw_span_population_sha256"]:
        raise ValueError("atomic raw population changed")
    preflight, _ = publish_sealed_json(root / "preflight.json", {
        "format": "memory-condense-batched-spine-hierarchy-preflight-v1", "atoms_sha256": atoms_artifact.sha256,
        "complete_namespace": data["complete_namespace"], "raw_span_population_sha256": data["raw_span_population_sha256"],
        "max_channel_tokens": 128, "leaf_token_cap": 512, "max_leaf_exchanges": 2,
        "attention_window_exchanges": 8, "raw_inputs_to_qwen": False, "query_independent": True,
        "model": MODEL, "gateway": GATEWAY, "max_jobs_per_batch": 8, "max_concurrency": 4,
        "retries": 0, "lossless_fitting_merges_reused": True, "overflow_policy": "fail; no summary omission",
        "implementation": {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in IMPLEMENTATION}})
    groups = {}
    for atom in atoms:
        groups.setdefault(atom.source_id, []).append(atom)
    journal = MergeJournal(root, preflight, enable, max_new_calls)
    journal.replay()
    exchanges = compile_waves(groups, lambda rows: compile_user_spine_exchanges(rows,
        summarize=journal.cache, summarizer_identity=preflight.sha256, max_channel_tokens=128), journal, "exchanges")
    if exchanges is None:
        return
    scorer = ScalarAttentionCache(root, preflight)
    hierarchies = compile_waves(exchanges, lambda rows: build_user_spine_hierarchy(rows, scorer=scorer,
        summarize=journal.cache, summarizer_identity=preflight.sha256, max_channel_tokens=128,
        max_leaf_exchanges=2, window_exchange_cap=8), journal, "hierarchy")
    if hierarchies is None:
        return
    parts = tuple(hierarchies.values())
    hierarchy = UserSpineHierarchy(
        *(tuple(value for part in parts for value in getattr(part, field)) for field in
          ("sections", "root_section_ids", "exchanges", "windows", "splits", "oversized_exchange_ids")),
        leaf_token_cap=512, max_leaf_exchanges=2, max_channel_tokens=128, window_exchange_cap=8)
    spans = [s.receipt_sha256 for exchange in hierarchy.exchanges for s in exchange.section.spans]
    if identity_sha256(spans) != data["raw_span_population_sha256"]:
        raise ValueError("compiled hierarchy lost or reordered source spans")
    result, _ = publish_sealed_json(root / "hierarchy.json", {"preflight_sha256": preflight.sha256,
        "atoms_sha256": atoms_artifact.sha256, "complete_namespace": data["complete_namespace"],
        "hierarchy": hierarchy.identity_payload(), "index_json": hierarchy.summary_index().to_json(),
        "raw_span_population_sha256": identity_sha256(spans), "raw_inputs_to_qwen": False,
        "summary_merge_jobs": len(journal.cache.values), "target_gate_passed": False})
    print({"hierarchy_sha256": result.sha256, "sources": len(parts), "exchanges": len(hierarchy.exchanges),
           "sections": len(hierarchy.sections), "attention_windows": len(hierarchy.windows),
           "new_provider_calls": journal.calls, "replay_hits": journal.hits,
           "complete_namespace": data["complete_namespace"]}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "run"))
    parser.add_argument("--atoms", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--enable-provider", action="store_true")
    parser.add_argument("--max-new-calls", type=int, default=0)
    args = parser.parse_args()
    if args.max_new_calls < 0 or (args.enable_provider and args.phase != "run"):
        parser.error("provider execution requires run and a nonnegative call budget")
    run(args.atoms, args.output_root, args.enable_provider, args.max_new_calls)
