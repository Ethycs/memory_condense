"""Resume complete-memory hierarchy compilation with bounded summary recovery.

Completed batches remain immutable. Valid unambiguous outputs are reused; an
ambiguous batch is recovered one job at a time from its original typed summary
inputs. Over-budget or malformed completions get at most two explicit semantic
recovery calls, never an SDK retry. Every scheduled call consumes the run budget
before outbound I/O, including calls already queued in the regular wave.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from memory_condense.domain._discourse_identity import identity_sha256, quote_sha256
from memory_condense.eval.fast_completion_runtime import FastCompletionRuntime
from memory_condense.search.episodes.user_spine_hierarchy import (
    UserSpineHierarchy, build_user_spine_hierarchy, compile_user_spine_exchanges,
)
from memory_condense.search.episodes.surprise_models import AttentionHeadSurpriseReceipt, ScoredSurpriseSequence
from memory_condense.search.section_summary import SectionSummary
from memory_condense.search.spine_merge_batch import SummaryMergeCache, merge_batch_messages, pack_merge_batches, parse_merge_batch
from memory_condense.search.spine_summary import parse_spine_summary
from tools.build_spine_corpus_hierarchy import (
    GATEWAY, MODEL, IMPLEMENTATION as ORIGINAL_IMPLEMENTATION,
    MergeJournal, ScalarAttentionCache, compile_waves, restore_request,
)
from tools.matched_eval.artifacts import read_sealed_json, publish_sealed_json
from tools.run_hot_reduced30_answer_judge import _authenticated_records, _completion_client, _run_exactly_authorized


IMPLEMENTATION = (*ORIGINAL_IMPLEMENTATION, "tools/build_spine_corpus_hierarchy_resilient.py")


class NeedsProviderWork(Exception):
    pass


def recoverable_slots(response, jobs):
    """Keep slots only when the entire response has unambiguous ordered labels."""
    try:
        return list(parse_merge_batch(response, jobs)), []
    except (ValueError, TypeError, KeyError):
        pass
    try:
        body = json.loads(response)
        rows = body["summaries"]
        if (type(body) is not dict or set(body) != {"summaries"} or type(rows) is not list or len(rows) != len(jobs) or
            any(type(row) is not dict or set(row) != {"label", "summary"} or row["label"] != f"S{i}"
                for i, row in enumerate(rows))):
            raise ValueError("ambiguous batch")
    except (ValueError, TypeError, KeyError):
        return [None] * len(jobs), list(range(len(jobs)))
    values, missing = [], []
    for i, (row, job) in enumerate(zip(rows, jobs, strict=True)):
        try:
            values.append(parse_spine_summary(json.dumps({"summary": row["summary"]}), job))
        except (ValueError, TypeError):
            values.append(None)
            missing.append(i)
    return values, missing


class RecoveringCache(SummaryMergeCache):
    def __init__(self, owner):
        super().__init__()
        self.owner = owner

    def accept(self, jobs, response):
        values, missing = recoverable_slots(response, jobs)
        for i in missing:
            values[i] = self.owner.recover(jobs[i], quote_sha256(response), i)
        # Every original slot must have a bounded attributed result before the
        # parent dependency can advance. No valid output is rewritten here.
        super().accept(jobs, json.dumps({"summaries": [
            {"label": f"S{i}", "summary": value} for i, value in enumerate(values)]}))


class RecoveryJournal(MergeJournal):
    def __init__(self, root, preflight, enable, budget):
        super().__init__(root, preflight, enable, budget)
        self.cache = RecoveringCache(self)
        self.scheduled_calls = 0
        self.recoveries = []

    def reserve(self, needed):
        if type(needed) is not int or needed < 0:
            raise ValueError("call reservation must be a nonnegative integer")
        if needed and (not self.enable or self.scheduled_calls + needed > self.budget):
            raise NeedsProviderWork("next prepared work exceeds this execution's call allowance")
        self.scheduled_calls += needed

    def recover(self, job, response_sha, slot):
        for attempt, words in enumerate((48, 24), 1):
            messages = [dict(row) for row in job.messages]
            messages[0]["content"] += (
                f" This is one recovery job. Merge ALL supplied fragments into ONE summary of at most {words} words. "
                "Return exactly one JSON object with only the key summary. Do not return a list or one output per fragment.")
            body = {"preflight_sha256": self.preflight.sha256, "original_job_sha256": job.prompt_sha256,
                "failed_response_sha256": response_sha, "slot": slot, "attempt": attempt, "job": asdict(job),
                "messages": messages, "model": MODEL, "raw_inputs": False}
            request, _ = publish_sealed_json(self.root / "repair-requests" / (identity_sha256(body) + ".json"), body)
            def factory(client):
                return FastCompletionRuntime(checkpoint_dir=self.root / "repair-checkpoints" / request.sha256,
                    prompt_population=[messages], model=MODEL, client=client,
                    max_prompt_tokens=7000, max_new_tokens=512, max_concurrency=1, retries=0,
                    request_options={"temperature": 0, "extra_body": {"enable_thinking": False}},
                    benchmark_provenance={"summary_repair_request_sha256": request.sha256})
            runtime = factory(None)
            try:
                needed = 1 - len(_authenticated_records(runtime))
            finally:
                runtime.close()
            print({"repair_job_sha256": job.prompt_sha256, "attempt": attempt,
                   "request_sha256": request.sha256, "missing_calls": needed}, flush=True)
            self.reserve(needed)
            batch, calls, hits, _ = _run_exactly_authorized(runtime_factory=factory, authorized_provider_calls=needed,
                enable_provider=self.enable, client_factory=lambda: _completion_client("LITELLM_KEY", GATEWAY))
            self.calls += calls
            self.hits += hits
            response = batch.logical_completions[0]
            try:
                value = parse_spine_summary(response, job)
            except (ValueError, TypeError) as error:
                publish_sealed_json(self.root / "repair-validation" / (request.sha256 + ".json"), {
                    "request_sha256": request.sha256, "response_sha256": quote_sha256(response),
                    "status": "invalid_summary", "error": str(error)})
                continue
            receipt, _ = publish_sealed_json(self.root / "repair-validation" / (request.sha256 + ".json"), {
                "request_sha256": request.sha256, "response_sha256": quote_sha256(response),
                "status": "accepted", "original_job_sha256": job.prompt_sha256,
                "summary": value, "summary_sha256": quote_sha256(value)})
            self.recoveries.append(receipt.sha256)
            return value
        raise ValueError("single-job summary recovery exhausted its two completed attempts")

    def import_parent(self, root, preflight):
        parent = MergeJournal(root, preflight, False, 0)
        for path in sorted((root / "requests").glob("*.json")):
            request = read_sealed_json(path)
            if request.payload["preflight_sha256"] != preflight.sha256:
                raise ValueError("parent batch binding changed")
            jobs = tuple(restore_request(row) for row in request.payload["jobs"])
            if merge_batch_messages(jobs) != request.payload["messages"]:
                raise ValueError("parent summary input reconstruction changed")
            runtime = parent.runtime(request)
            try:
                batch = runtime.run()  # No client: incomplete parents cannot be imported.
            finally:
                runtime.close()
            self.cache.accept(jobs, batch.logical_completions[0])
            self.hits += 1

    def resolve(self, pending, phase, wave):
        requests = []
        for jobs in pack_merge_batches(tuple(pending.values())):
            body = {"preflight_sha256": self.preflight.sha256, "jobs": [asdict(r) for r in jobs],
                    "messages": merge_batch_messages(jobs), "model": MODEL, "raw_inputs": False}
            request, _ = publish_sealed_json(self.root / "requests" / (identity_sha256(body) + ".json"), body)
            requests.append((request, jobs))
        allowances = [self.remaining(r) for r, _ in requests]
        plan = {"preflight_sha256": self.preflight.sha256, "phase": phase, "wave": wave,
                "request_shas": [r.sha256 for r, _ in requests], "merge_jobs": len(pending),
                "maximum_calls": len(requests), "max_concurrency": 4, "retries": 0}
        artifact, _ = publish_sealed_json(self.root / "waves" / (identity_sha256(plan) + ".json"), plan)
        print({"phase": phase, "wave": wave, "merge_jobs": len(pending), "missing_calls": sum(allowances),
               "wave_sha256": artifact.sha256, "new_calls_this_run": self.calls}, flush=True)
        # Charge every regular call before recovery can schedule anything else.
        self.reserve(sum(allowances))
        def one(item):
            (request, jobs), needed = item
            batch, calls, hits, _ = _run_exactly_authorized(
                runtime_factory=lambda client: self.runtime(request, client), authorized_provider_calls=needed,
                enable_provider=self.enable, client_factory=lambda: _completion_client("LITELLM_KEY", GATEWAY))
            return jobs, batch.logical_completions[0], calls, hits
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(one, zip(requests, allowances, strict=True)))
        self.calls += sum(row[2] for row in results)
        self.hits += sum(row[3] for row in results)
        for jobs, response, _, _ in results:
            self.cache.accept(jobs, response)
        return True


class ReusingAttentionCache(ScalarAttentionCache):
    def __init__(self, root, preflight, parents):
        super().__init__(root, preflight)
        self.parents = parents

    def score_sequence(self, texts):
        key = identity_sha256({"preflight_sha256": self.preflight.sha256, "texts": list(texts)})
        path = self.root / "attention" / (key + ".json")
        if key not in self.values and not path.exists():
            for parent_root, parent in self.parents:
                old_key = identity_sha256({"preflight_sha256": parent.sha256, "texts": list(texts)})
                old_path = parent_root / "attention" / (old_key + ".json")
                if not old_path.exists():
                    continue
                artifact = read_sealed_json(old_path)
                row = artifact.payload
                if row["preflight_sha256"] != parent.sha256:
                    raise ValueError("parent attention binding changed")
                signal = ScoredSurpriseSequence(row["scores"], row["similarities"], AttentionHeadSurpriseReceipt(**row["receipt"]))
                signal.validate_inputs(texts)
                publish_sealed_json(path, {"preflight_sha256": self.preflight.sha256, "scores": signal.scores,
                    "similarities": signal.similarities, "receipt": signal.receipt.identity_payload(),
                    "reused_parent_attention_sha256": artifact.sha256})
                self.values[key] = signal
                break
        return super().score_sequence(texts)


def run(atoms_path, root, parent_roots, enable, budget):
    atoms_artifact = read_sealed_json(atoms_path)
    data = atoms_artifact.payload
    if data.get("format") != "memory-condense-spine-source-bound-atoms-v1" or not data["complete_namespace"]:
        raise ValueError("resilient compilation requires a complete bound namespace")
    atoms = tuple(SectionSummary.from_dict(row) for row in data["atoms"])
    if identity_sha256([a.spans[0].receipt_sha256 for a in atoms]) != data["raw_span_population_sha256"]:
        raise ValueError("atomic source partition changed")
    parents = []
    for parent_root in parent_roots:
        parent = read_sealed_json(parent_root / "preflight.json")
        if parent_root.resolve() == root.resolve() or parent.payload["atoms_sha256"] != atoms_artifact.sha256:
            raise ValueError("parent cache must be distinct and bound to the same complete atoms")
        expected = {"model": MODEL, "gateway": GATEWAY, "max_channel_tokens": 128,
            "leaf_token_cap": 512, "max_leaf_exchanges": 2, "attention_window_exchanges": 8,
            "raw_inputs_to_qwen": False, "query_independent": True, "complete_namespace": True,
            "raw_span_population_sha256": data["raw_span_population_sha256"],
            "max_jobs_per_batch": 8, "max_concurrency": 4, "retries": 0}
        if any(parent.payload.get(key) != value for key, value in expected.items()):
            raise ValueError("parent compilation policy or raw partition changed")
        if any(hashlib.sha256(Path(name).read_bytes()).hexdigest() != sha for name, sha in parent.payload["implementation"].items()):
            raise ValueError("parent compilation implementation changed")
        parents.append((parent_root, parent))
    method = {"format": "memory-condense-resilient-spine-hierarchy-v1", "model": MODEL, "gateway": GATEWAY,
        "max_channel_tokens": 128, "leaf_token_cap": 512, "max_leaf_exchanges": 2, "attention_window_exchanges": 8,
        "max_jobs_per_batch": 8, "max_concurrency": 4, "max_recovery_calls_per_failed_job": 2,
        "recovery_words": [48, 24], "recovery_policy": "preserve unambiguous bounded slots; recover ambiguous jobs singly",
        "retries": 0, "raw_inputs_to_qwen": False, "query_independent": True,
        "implementation": {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in IMPLEMENTATION}}
    preflight, _ = publish_sealed_json(root / "preflight.json", {**method, "method_policy_sha256": identity_sha256(method),
        "atoms_sha256": atoms_artifact.sha256, "raw_span_population_sha256": data["raw_span_population_sha256"],
        "complete_namespace": True, "parent_caches": [{"root": str(p.resolve()), "preflight_sha256": a.sha256} for p, a in parents]})
    journal = RecoveryJournal(root, preflight, enable, budget)
    try:
        for parent_root, parent in parents:
            journal.import_parent(parent_root, parent)
        journal.replay()
        groups = {}
        for atom in atoms:
            groups.setdefault(atom.source_id, []).append(atom)
        exchanges = compile_waves(groups, lambda rows: compile_user_spine_exchanges(rows, summarize=journal.cache,
            summarizer_identity=preflight.sha256, max_channel_tokens=128), journal, "exchanges")
        scorer = ReusingAttentionCache(root, preflight, parents)
        hierarchies = compile_waves(exchanges, lambda rows: build_user_spine_hierarchy(rows, scorer=scorer,
            summarize=journal.cache, summarizer_identity=preflight.sha256, max_channel_tokens=128,
            max_leaf_exchanges=2, window_exchange_cap=8), journal, "hierarchy")
    except NeedsProviderWork:
        print({"preflight_sha256": preflight.sha256, "status": "next_work_prepared", "new_calls": journal.calls,
               "scheduled_calls": journal.scheduled_calls, "replay_hits": journal.hits}, flush=True)
        return
    parts = tuple(hierarchies.values())
    hierarchy = UserSpineHierarchy(
        *(tuple(value for part in parts for value in getattr(part, field)) for field in
          ("sections", "root_section_ids", "exchanges", "windows", "splits", "oversized_exchange_ids")),
        leaf_token_cap=512, max_leaf_exchanges=2, max_channel_tokens=128, window_exchange_cap=8)
    spans = [s.receipt_sha256 for exchange in hierarchy.exchanges for s in exchange.section.spans]
    if identity_sha256(spans) != data["raw_span_population_sha256"]:
        raise ValueError("compiled hierarchy changed the complete raw partition")
    artifact, _ = publish_sealed_json(root / "hierarchy.json", {"preflight_sha256": preflight.sha256,
        "atoms_sha256": atoms_artifact.sha256, "complete_namespace": True, "hierarchy": hierarchy.identity_payload(),
        "index_json": hierarchy.summary_index().to_json(), "raw_span_population_sha256": identity_sha256(spans),
        "raw_inputs_to_qwen": False, "summary_merge_jobs": len(journal.cache.values),
        "recovery_receipts": sorted(set(journal.recoveries)), "target_gate_passed": False})
    print({"hierarchy_sha256": artifact.sha256, "exchanges": len(hierarchy.exchanges), "sections": len(hierarchy.sections),
           "attention_windows": len(hierarchy.windows), "new_calls": journal.calls, "replay_hits": journal.hits,
           "complete_namespace": True}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "run"))
    parser.add_argument("--atoms", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--parent-cache-root", type=Path, action="append", default=[])
    parser.add_argument("--enable-provider", action="store_true")
    parser.add_argument("--max-new-calls", type=int, default=0)
    args = parser.parse_args()
    if args.max_new_calls < 0 or (args.enable_provider and args.phase != "run"):
        parser.error("provider execution requires run and a nonnegative call allowance")
    run(args.atoms, args.output_root, args.parent_cache_root, args.enable_provider, args.max_new_calls)
