"""Compile native user-led exchanges with exact reuse and local summary-only Qwen."""
import argparse
from contextlib import closing
from dataclasses import asdict
import json
from pathlib import Path
import sqlite3

from memory_condense.domain._discourse_identity import identity_sha256
from memory_condense.search.episodes.user_spine_hierarchy import compile_user_spine_exchanges
from memory_condense.search.native_spine_memory import materialize_history
from memory_condense.search.native_spine_merges import NeutralMergeCache, neutral_key, neutral_messages
from memory_condense.search.section_summary import SectionSummary
from memory_condense.search.spine_merge_batch import PendingMerge
from memory_condense.search.spine_summary import parse_spine_summary
from memory_condense.search.spine_summary_reuse import ReusingSpineSummarizer
from tools.assemble_native_spine_admitted import AdmittedSummaryBodies, implementation as admission_implementation
from tools.assemble_native_spine_summaries import digest
from tools.build_spine_corpus_hierarchy import restore_request
from tools.matched_eval.artifacts import publish_sealed_json, read_sealed_json
from tools.native_qwen_spine_backend import NativeQwenBackend
from tools.run_hot_reduced30_answer_judge import _phase_lock


FILES = (
    "tools/compile_native_spine_exchanges.py", "tools/native_qwen_spine_backend.py",
    "tools/local_qwen_spine_backend.py", "tools/probe_local_qwen_parent_summaries.py",
    "tools/build_spine_corpus_hierarchy.py", "src/memory_condense/search/native_spine_merges.py",
    "src/memory_condense/search/spine_summary.py", "src/memory_condense/search/spine_merge_batch.py",
    "src/memory_condense/search/spine_summary_reuse.py", "src/memory_condense/search/section_summary.py",
    "src/memory_condense/search/episodes/user_spine_hierarchy.py",
)


def implementation():
    return {**admission_implementation(), **{name: digest(name) for name in FILES}}


def prepare(store_root, source_root, root):
    with _phase_lock(root, "native-exchange-preparation"), closing(AdmittedSummaryBodies(store_root)) as store:
        sources = read_sealed_json(source_root / "sources.json")
        if sources.sha256 != store.manifest.payload["sources_sha256"]:
            raise ValueError("native summaries belong to another source corpus")
        needed = {r[0] for r in store.connection.execute("SELECT body_sha256 FROM bodies")}
        selected = {}
        for binding in sources.payload["namespaces"]:
            namespace = read_sealed_json(source_root / binding["path"])
            if namespace.sha256 != binding["sha256"]:
                raise ValueError("actual source occurrences changed")
            for source in namespace.payload["sessions"]:
                if source["body_sha256"] in needed:
                    selected[source["body_sha256"]] = (source, namespace.sha256)
                    needed.remove(source["body_sha256"])
            if not needed:
                break
        if needed or not selected:
            raise ValueError("each cached body requires an actual source occurrence")
        path = (source_root / sources.payload["body_bank_path"]).resolve()
        path.relative_to(source_root.resolve())
        if digest(path) != sources.payload["body_bank_sha256"]:
            raise ValueError("original native body bank changed")
        bindings, atom_count = [], 0
        with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as raw:
            def load_body(sha):
                return json.loads(raw.execute("SELECT body_json FROM bodies WHERE body_sha256=?", (sha,)).fetchone()[0])

            for sha, (source, namespace_sha) in sorted(selected.items()):
                history = materialize_history([source], load_body=load_body, load_summaries=store.load,
                                              compiler_identity=store.manifest.sha256)
                body, _ = publish_sealed_json(root / "bodies" / f"{sha}.json", {
                    "summary_body_store_sha256": store.manifest.sha256,
                    "namespace_sha256": namespace_sha, "source": source,
                    "atoms": [asdict(a) for a in history.atoms], "raw_text_included": False,
                })
                bindings.append({"body_sha256": sha, "path": str(body.path.relative_to(root)), "sha256": body.sha256})
                atom_count += len(history.atoms)
        result, _ = publish_sealed_json(root / "inputs.json", {
            "format": "native-spine-exchange-inputs-v1", "summary_body_store_sha256": store.manifest.sha256,
            "sources_sha256": sources.sha256, "bodies": bindings, "atom_count": atom_count,
            "body_count": len(bindings), "one_actual_occurrence_per_body": True,
            "complete_source_compilation": store.manifest.payload["complete_source_compilation"],
            "question_or_gold_inputs": False, "raw_text_included": False,
            "implementation": implementation(),
        })
        print({"inputs_sha256": result.sha256, "bodies": len(bindings), "atoms": atom_count}, flush=True)
        return result


class NeutralJournal:
    def __init__(self, root, preflight, backend, budget):
        self.root, self.preflight, self.backend, self.budget = root, preflight, backend, budget
        self.cache = NeutralMergeCache()
        self.attempted = set()
        self.calls = self.jobs = 0

    def accept(self, request, response):
        p, r = request.payload, response.payload
        jobs = tuple(restore_request(row) for row in p["jobs"])
        if (p["preflight_sha256"] != self.preflight.sha256
                or p["backend_sha256"] != self.backend.identity_sha256
                or p["messages"] != [neutral_messages(j, p["attempt"]) for j in jobs]
                or p["raw_inputs_to_qwen"] is not False
                or r["request_sha256"] != request.sha256 or r["backend_sha256"] != self.backend.identity_sha256
                or r["raw_inputs_to_qwen"] is not False or r["remote_provider_calls"] != 0
                or r["timestamp_metadata_in_model_inputs"] is not False
                or len(r["rows"]) != len(jobs)):
            raise ValueError("native summary-only execution binding changed")
        for job, row in zip(jobs, r["rows"], strict=True):
            key = neutral_key(job)
            if row["merge_key"] != key:
                raise ValueError("native generation changed job attribution")
            self.attempted.add((key, p["attempt"]))
            if row["stopped"] is not True:
                continue
            try:
                parse_spine_summary(row["response"], job)
            except (ValueError, TypeError):
                continue
            self.cache.accept(job, row["response"])

    def replay(self):
        for path in sorted((self.root / "requests").glob("*.json")):
            request = read_sealed_json(path)
            response_path = self.root / "responses" / f"{request.sha256}.json"
            reserved = self.root / "executions" / f"{request.sha256}.reserved"
            if response_path.exists():
                self.accept(request, read_sealed_json(response_path))
            elif reserved.exists():
                raise ValueError("local execution lacks a response; refusing an implicit retry")

    def resolve(self, pending):
        unique = {neutral_key(job): job for job in pending.values()}
        for attempt in range(3):
            missing = [job for key, job in sorted(unique.items())
                       if key not in self.cache.values and (key, attempt) not in self.attempted]
            for start in range(0, len(missing), self.backend.max_batch_size):
                jobs = tuple(missing[start:start+self.backend.max_batch_size])
                payload = {"preflight_sha256": self.preflight.sha256,
                           "backend_sha256": self.backend.identity_sha256, "attempt": attempt,
                           "jobs": [asdict(job) for job in jobs],
                           "messages": [neutral_messages(job, attempt) for job in jobs],
                           "raw_inputs_to_qwen": False}
                request, _ = publish_sealed_json(self.root / "requests" / f"{identity_sha256(payload)}.json", payload)
                response_path = self.root / "responses" / f"{request.sha256}.json"
                if response_path.exists():
                    self.accept(request, read_sealed_json(response_path))
                    continue
                if self.jobs + len(jobs) > self.budget:
                    return False
                reserved = self.root / "executions" / f"{request.sha256}.reserved"
                reserved.parent.mkdir(parents=True, exist_ok=True)
                with reserved.open("x", encoding="utf-8") as stream:
                    stream.write(request.sha256 + "\n")
                self.jobs += len(jobs)
                self.calls += 1
                result = self.backend.generate(jobs, attempt)
                response, _ = publish_sealed_json(response_path, {"request_sha256": request.sha256, **result})
                self.accept(request, response)
                print({"local_batches": self.calls, "local_jobs": self.jobs, "attempt": attempt,
                       "accepted_merge_keys": len(self.cache.values), "elapsed_s": result["elapsed_s"]}, flush=True)
        if any(key not in self.cache.values for key in unique):
            raise ValueError("native local summary recovery exhausted its two refinements")
        return True


def execute(root, backend, budget=0):
    if type(budget) is not int or not 0 <= budget <= 128:
        raise ValueError("native exchange compilation allows at most 128 new local jobs per invocation")
    with _phase_lock(root, "native-exchange-compilation"):
        inputs = read_sealed_json(root / "inputs.json")
        p = inputs.payload
        if (p["implementation"] != implementation() or p["raw_text_included"] is not False
                or p["question_or_gold_inputs"] is not False):
            raise ValueError("native exchange source inputs changed")
        preflight, _ = publish_sealed_json(root / "preflight.json", {
            "inputs_sha256": inputs.sha256, "backend": backend.identity,
            "backend_sha256": backend.identity_sha256, "max_channel_tokens": 128,
            "max_prompt_tokens": 2048, "maximum_new_local_jobs_per_invocation": 128,
            "maximum_recovery_attempts": 2, "automatic_retries": 0,
            "raw_inputs_to_qwen": False, "timestamp_metadata_in_model_inputs": False,
            "implementation": implementation(),
        })
        groups, population = {}, []
        for binding in p["bodies"]:
            path = (root / binding["path"]).resolve()
            path.relative_to((root / "bodies").resolve())
            body = read_sealed_json(path)
            b = body.payload
            if (body.sha256 != binding["sha256"] or b["summary_body_store_sha256"] != p["summary_body_store_sha256"]
                    or b["source"]["body_sha256"] != binding["body_sha256"] or b["raw_text_included"] is not False
                    or binding["body_sha256"] in groups):
                raise ValueError("native summary body input changed")
            atoms = tuple(SectionSummary.from_dict(a) for a in b["atoms"])
            groups[binding["body_sha256"]] = (body, atoms)
            population.extend(s.receipt_sha256 for a in atoms for s in a.spans)
        if len(groups) != p["body_count"] or len(population) != p["atom_count"] or len(set(population)) != len(population):
            raise ValueError("native exchange atomic population changed")
        journal = NeutralJournal(root, preflight, backend, budget)
        journal.replay()
        summarize = ReusingSpineSummarizer(journal.cache)
        done, exchange_count = {}, 0
        while len(done) < len(groups):
            pending = {}
            for sha, (body, atoms) in groups.items():
                if sha in done:
                    continue
                try:
                    exchanges = compile_user_spine_exchanges(atoms, summarize=summarize,
                        summarizer_identity=preflight.sha256, max_channel_tokens=128, max_prompt_tokens=2048)
                except PendingMerge as missing:
                    pending.setdefault(neutral_key(missing.request), missing.request)
                    continue
                expected = tuple(s for atom in atoms for s in atom.spans)
                if tuple(s for e in exchanges for s in e.section.spans) != expected:
                    raise ValueError("native exchange compilation changed exact raw coverage")
                artifact, _ = publish_sealed_json(root / "exchanges" / f"{sha}.json", {
                    "preflight_sha256": preflight.sha256, "body_input_sha256": body.sha256,
                    "body_sha256": sha, "exchanges": [asdict(e) for e in exchanges],
                    "raw_span_population_sha256": identity_sha256([s.receipt_sha256 for s in expected]),
                    "raw_inputs_to_qwen": False,
                })
                done[sha] = {"path": str(artifact.path.relative_to(root)), "sha256": artifact.sha256}
                exchange_count += len(exchanges)
            print({"complete_body_exchanges": len(done), "pending_merge_jobs": len(pending)}, flush=True)
            if pending and not journal.resolve(pending):
                break
            if not pending:
                break
        payload = {"preflight_sha256": preflight.sha256, "body_count": len(done),
                   "prepared_body_count": len(groups), "exchange_count": exchange_count,
                   "compiled_bodies": [done[sha] for sha in sorted(done)],
                   "complete_available_body_exchanges": len(done) == len(groups),
                   "complete_source_compilation": p["complete_source_compilation"],
                   "raw_span_population_sha256": identity_sha256(population),
                   "raw_inputs_to_qwen": False, "remote_provider_calls": 0,
                   "hierarchies_compiled": False, "full100_target_passed": False}
        filename = "result.json" if len(done) == len(groups) else f"partial-{identity_sha256(payload)}.json"
        result, _ = publish_sealed_json(root / filename, payload)
        print({"result_sha256": result.sha256, "complete_body_exchanges": len(done),
               "exchanges": exchange_count, "new_local_jobs": journal.jobs, "new_local_batches": journal.calls}, flush=True)
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "run"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--store-root", type=Path)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--budget", type=int, default=0)
    parser.add_argument("--probe-root", type=Path, default=Path("eval_results/local-qwen-parent-summary-probe-20260910-r1"))
    parser.add_argument("--dependency-root", type=Path, default=Path(".cache/local-qwen-runtime/site-packages"))
    parser.add_argument("--model-root", type=Path, default=Path("../../.cache/models/Qwen3-8B"))
    args = parser.parse_args()
    if args.phase == "prepare":
        prepare(args.store_root, args.source_root, args.output_root)
    else:
        backend = NativeQwenBackend(args.probe_root, args.dependency_root, args.model_root)
        execute(args.output_root, backend, args.budget)
