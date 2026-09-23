"""Finish full-corpus compilation and run one fresh matched native full100."""
from contextlib import closing
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil

from tools import complete_native_spine_ingestion as source_completion
from tools import evaluate_native_spine_full100 as evaluation
from tools.matched_eval.artifacts import publish_sealed_json, read_sealed_json
from tools.run_hot_reduced30_answer_judge import _phase_lock


FORMAT = "native-spine-full-corpus-pipeline-v1"
STAGES = ("exchange-inputs", "exchanges", "attention-prepare", "attention",
          "parents", "vectors-prepare", "vectors", "evaluation-prepare", "evaluation", "replay")
PATHS = ("sources", "store", "reuse_exchanges", "reuse_parents", "reuse_vectors", "attention_cache", "m_dataset")


def implementation():
    return {**evaluation.implementation(), **source_completion.implementation(),
            "tools/run_native_spine_full_corpus.py": evaluation.digest(__file__)}


def prepare(config, root):
    root = Path(root).resolve()
    root.relative_to(Path.cwd().resolve())
    if root.exists():
        raise ValueError("full pipeline requires a fresh output root")
    paths = {key: str(Path(config[key]).resolve()) for key in PATHS}
    compiler_flavor = config.get('compiler_flavor', 'expanding')
    if compiler_flavor not in ('expanding', 'bounded'):
        raise ValueError('unsupported native compiler flavor')
    previous_exchanges = config.get('previous_exchanges')
    if previous_exchanges is not None:
        if compiler_flavor != 'bounded':
            raise ValueError('retired exchange journals require bounded recovery')
        previous_exchanges = str(Path(previous_exchanges).resolve())
    if set(config["dependencies"]) != {"ingestion", "source_completion", "exchanges", "attention", "parents", "vectors"}:
        raise ValueError("all active producer dependencies are required")
    dependencies = []
    for name, item in config["dependencies"].items():
        control = Path(item["control"]).resolve()
        started = read_sealed_json(control / "started.json")
        policy = read_sealed_json(control / "policy.json")
        if started.payload["policy_sha256"] != policy.sha256:
            raise ValueError("predecessor start does not bind its policy")
        if name == "source_completion":
            if (policy.payload["format"] != source_completion.FORMAT
                    or policy.payload["implementation"] != source_completion.implementation()
                    or Path(paths["store"]) != control / "complete-body-store"):
                raise ValueError("full pipeline must consume the bound complete store")
        dependencies.append({"name": name, "control": str(control),
            "started": source_completion.binding(started.path),
            "policy": source_completion.binding(policy.path), "required": item.get("required", {})})
    sources = read_sealed_json(Path(paths["sources"]) / "sources.json")
    limits = {"maximum_wait_seconds": 172800, "poll_seconds": 30,
              "maximum_exchange_jobs": 8192, "maximum_parent_jobs": 8192}
    limits.update(config.get("limits", {}))
    if any(type(v) is not int or v <= 0 for v in limits.values()) or limits["poll_seconds"] > 60:
        raise ValueError("invalid pipeline limits")
    if not Path(paths["m_dataset"]).is_file():
        raise ValueError("the bound evaluation dataset must be available")
    policy, _ = publish_sealed_json(root / "policy.json", {
        "format": FORMAT, "paths": paths, "dependencies": dependencies, "limits": limits,
        "sources": source_completion.binding(sources.path),
        "attention_method": source_completion.binding(Path(paths["attention_cache"]) / "method.json"),
        "implementation": implementation(), "evaluation_policy": evaluation.POLICY,
        "stages": list(STAGES), "maximum_answer_calls": 400, "maximum_logical_judgments": 200,
        "timed_concurrency": 1, "automatic_retries": 0, "raw_inputs_to_qwen": False,
        "query_qwen_passes": 0, "complete_corpus_required": True,
        "compilation_and_evaluation_in_separate_child_processes": True,
        "compiler_flavor": compiler_flavor, "previous_exchanges": previous_exchanges,
        "target_gate_passed": False,
    })
    print({"full_corpus_pipeline_policy_sha256": policy.sha256,
           "required_bodies": sources.payload["body_count"], "model_calls_released": 0}, flush=True)
    return policy


def load(root):
    policy = read_sealed_json(Path(root) / "policy.json")
    p = policy.payload
    if (p["format"] != FORMAT or p["implementation"] != implementation()
            or p["evaluation_policy"] != evaluation.POLICY or p["stages"] != list(STAGES)):
        raise ValueError("full pipeline implementation or benchmark policy changed")
    source_completion.bound(p["sources"])
    source_completion.bound(p["attention_method"])
    return policy


def dependency_state(dependencies):
    live, finished = [], {}
    for item in dependencies:
        started = source_completion.bound(item["started"])
        policy = source_completion.bound(item["policy"])
        try:
            process = psutil.Process(started.payload["pid"])
            if process.create_time() == started.payload["create_time"]:
                live.append({"name": item["name"], "pid": process.pid})
                continue
        except psutil.NoSuchProcess:
            pass
        terminal = read_sealed_json(Path(item["control"]) / "finished.json")
        if (terminal.payload["policy_sha256"] != policy.sha256
                or any(type(terminal.payload.get(k)) is not type(v) or terminal.payload[k] != v
                       for k, v in item["required"].items())):
            raise ValueError("pipeline predecessor did not complete its required population")
        finished[item["name"]] = source_completion.binding(terminal.path)
    return live, finished


def wait_dependencies(policy):
    started = time.monotonic()
    while True:
        live, finished = dependency_state(policy.payload["dependencies"])
        if not live:
            return finished
        if time.monotonic() - started > policy.payload["limits"]["maximum_wait_seconds"]:
            raise TimeoutError("predecessors remain live; pipeline has not released a model")
        print({"waiting_for_full_corpus_dependencies": live}, flush=True)
        time.sleep(policy.payload["limits"]["poll_seconds"])


def full_store(policy, release):
    paths = policy.payload["paths"]
    completed = source_completion.bound(release.payload["predecessor_completions"]["source_completion"])
    with closing(source_completion.admission.JsonRecoveredSummaryBodies(Path(paths["store"]))) as store:
        sources = source_completion.bound(policy.payload["sources"])
        p = store.manifest.payload
        if (store.manifest.sha256 != completed.payload["summary_store_sha256"]
                or p["sources_sha256"] != sources.sha256
                or p["body_count"] != sources.payload["body_count"]
                or p["complete_source_compilation"] is not True
                or p["all_prepared_bodies_admitted"] is not True):
            raise ValueError("pipeline requires the actual complete source corpus")
        return store.manifest


def complete_merges(root, backend, execute, *, limit, complete_key):
    """One resident generator per stage, with exact existing 128-job invocations."""
    jobs = batches = 0
    def wrap(generate):
        def counted(group, attempt):
            nonlocal jobs, batches
            group = tuple(group)
            if jobs + len(group) > limit:
                raise ValueError("full-corpus merge allowance exhausted")
            jobs += len(group)
            batches += 1
            return generate(group, attempt)
        return counted
    backend.generate = wrap(backend.generate)
    if hasattr(backend, 'generate_bounded'):
        backend.generate_bounded = wrap(backend.generate_bounded)
    result = execute(0)
    if jobs != 0 or backend.model is not None:
        raise ValueError("initial compilation replay loaded a model")
    invocation = 0
    while not result.payload[complete_key]:
        if jobs >= limit:
            raise ValueError("full-corpus merge allowance exhausted")
        before = jobs
        result = execute(min(128, limit - jobs))
        invocation += 1
        publish_sealed_json(Path(root) / "pipeline-progress" / f"{invocation:04d}.json", {
            "result_sha256": result.sha256, "completed_bodies": result.payload["body_count"],
            "new_local_jobs": jobs, "new_local_batches": batches})
        if jobs == before and not result.payload[complete_key]:
            raise ValueError("incomplete compilation made no further generation progress")
    if result.payload["complete_source_compilation"] is not True:
        raise ValueError("compiled result is still a partial source population")
    return result


def stage(root, name, *, enable_provider=False):
    root = Path(root)
    policy = load(root)
    release = read_sealed_json(root / "released.json")
    request = read_sealed_json(root / "stages" / f"{name}.request.json")
    if (release.payload["policy_sha256"] != policy.sha256
            or request.payload != {"policy_sha256": policy.sha256, "stage": name,
                                   "released_sha256": release.sha256} or name not in STAGES):
        raise ValueError("stage does not belong to this released full pipeline")
    store = full_store(policy, release)
    paths = {k: Path(v) for k, v in policy.payload["paths"].items()}
    bounded = policy.payload.get('compiler_flavor', 'expanding') == 'bounded'
    ex, att, parents, vectors, target = [root / name for name in ("exchanges", "attention", "parents", "vectors", "evaluation")]
    if name == "exchange-inputs":
        if bounded and policy.payload.get('previous_exchanges'):
            from tools import compile_bounded_native_spine_exchanges as compiler
            previous_root = Path(policy.payload['previous_exchanges'])
            previous_inputs = read_sealed_json(previous_root/'inputs.json')
            if previous_inputs.payload['summary_body_store_sha256'] != store.sha256:
                raise ValueError('retired exchange inputs belong to another summary store')
            result = compiler.copy_inputs(previous_root, ex)
        else:
            from tools import compile_expanding_native_spine_exchanges as compiler
            result = compiler.original.prepare(paths["store"], paths["sources"], ex)
        if result.payload["body_count"] != store.payload["body_count"]:
            raise ValueError("exchange preparation omitted complete bodies")
    elif name in {"exchanges", "parents"}:
        from tools.native_qwen_spine_backend import NativeQwenBackend
        backend = NativeQwenBackend(Path("eval_results/local-qwen-parent-summary-probe-20260910-r1"),
            Path(".cache/local-qwen-runtime/site-packages"), Path("../../.cache/models/Qwen3-8B"))
        if bounded:
            from tools.native_spine_bounded_journal import install_backend
            install_backend(backend)
        if name == "exchanges":
            if bounded:
                from tools import compile_bounded_native_spine_exchanges as compiler
                options = {'previous_root': policy.payload.get('previous_exchanges')}
            else:
                from tools import compile_expanding_native_spine_exchanges as compiler
                options = {}
            result = complete_merges(ex, backend,
                lambda budget: compiler.execute(ex, backend, budget, reuse_roots=[paths["reuse_exchanges"]], **options),
                limit=policy.payload["limits"]["maximum_exchange_jobs"], complete_key="complete_available_body_exchanges")
        else:
            if bounded:
                from tools import compile_bounded_native_spine_hierarchy as compiler
            else:
                from tools import compile_expanding_native_spine_hierarchy as compiler
            result = complete_merges(parents, backend,
                lambda budget: compiler.execute(parents, ex, att, backend, budget, reuse_roots=[paths["reuse_parents"]]),
                limit=policy.payload["limits"]["maximum_parent_jobs"], complete_key="complete_available_body_hierarchies")
    elif name in {"attention-prepare", "attention"}:
        if bounded:
            from tools import prepare_bounded_native_spine_attention as compiler
        else:
            from tools import prepare_expanding_native_spine_attention as compiler
        if name == "attention-prepare":
            from tools.native_qwen_spine_backend import NativeQwenBackend
            backend = NativeQwenBackend(Path("eval_results/local-qwen-parent-summary-probe-20260910-r1"),
                Path(".cache/local-qwen-runtime/site-packages"), Path("../../.cache/models/Qwen3-8B"))
            result = compiler.prepare(ex, att, paths["attention_cache"], backend)
        else:
            result = compiler.execute(att)
    elif name in {"vectors-prepare", "vectors"}:
        from memory_condense.modeling.embedding import EmbeddingService
        from tools import compile_native_spine_vectors as compiler
        with closing(EmbeddingService(device="cuda", batch_size=8)) as encoder:
            if name == "vectors-prepare":
                result = compiler.prepare(paths["store"], vectors, encoder, reuse_roots=[paths["reuse_vectors"]])
                if encoder._model is not None:
                    raise ValueError("vector preparation loaded an encoder")
            else:
                result = compiler.execute(vectors, encoder)
    elif name == "evaluation-prepare":
        result = evaluation.prepare(target, {"sources": paths["sources"], "store": paths["store"],
            "hierarchies": parents / "result.json", "vectors": vectors, "m_dataset": paths["m_dataset"]})
    elif name == "evaluation":
        result = evaluation.run(target, enable_provider)
    else:
        original = read_sealed_json(target / "joint-report.json")
        result = evaluation.judge(target, False)
        if result.sha256 != original.sha256:
            raise ValueError("fresh full100 joint report did not replay unchanged")
    receipt, _ = publish_sealed_json(root / "stages" / f"{name}.result.json", {
        "policy_sha256": policy.sha256, "stage": name, "artifact": source_completion.binding(result.path)})
    print({"full_pipeline_stage": name, "receipt_sha256": receipt.sha256}, flush=True)
    return receipt


def run_child(root, policy, release, name, enable_provider):
    stages = Path(root) / "stages"
    with (stages / f"{name}.reserved").open("x", encoding="utf-8") as stream:
        stream.write(policy.sha256 + "\n")
    publish_sealed_json(stages / f"{name}.request.json", {
        "policy_sha256": policy.sha256, "stage": name, "released_sha256": release.sha256})
    command = [sys.executable, "-X", "utf8", "-m", "tools.run_native_spine_full_corpus",
               "stage", "--root", str(root), "--stage", name]
    if name == "evaluation" and enable_provider:
        command.append("--enable-provider")
    with (stages / f"{name}.log").open("xb") as output:
        child = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT,
                                 creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        process = psutil.Process(child.pid)
        publish_sealed_json(stages / f"{name}.started.json", {
            "policy_sha256": policy.sha256, "pid": child.pid, "create_time": process.create_time()})
        print({"started_full_pipeline_stage": name, "child_pid": child.pid}, flush=True)
        while child.poll() is None:
            time.sleep(15)
        publish_sealed_json(stages / f"{name}.exit.json", {
            "policy_sha256": policy.sha256, "stage": name, "returncode": child.returncode})
    if child.returncode != 0:
        raise RuntimeError(f"full-corpus stage {name} failed; no automatic retry")
    result = read_sealed_json(stages / f"{name}.result.json")
    if result.payload["policy_sha256"] != policy.sha256 or result.payload["stage"] != name:
        raise ValueError("child result does not bind the released stage")
    source_completion.bound(result.payload["artifact"])
    return result


def run(root, *, enable_provider=False):
    if not enable_provider:
        raise ValueError("full pipeline requires the authorized evaluation provider flag")
    root = Path(root)
    policy = load(root)
    with _phase_lock(root, "full-corpus-pipeline"):
        with (root / "execution.reserved").open("x", encoding="utf-8") as stream:
            stream.write(policy.sha256 + "\n")
        process = psutil.Process(os.getpid())
        publish_sealed_json(root / "started.json", {
            "policy_sha256": policy.sha256, "pid": process.pid, "create_time": process.create_time()})
        print({"full_pipeline_pid": process.pid, "policy_sha256": policy.sha256}, flush=True)
        current = "waiting"
        try:
            completed = wait_dependencies(policy)
            load(root)
            release, _ = publish_sealed_json(root / "released.json", {
                "policy_sha256": policy.sha256, "predecessor_completions": completed,
                "all_owned_predecessor_processes_gone": True})
            full_store(policy, release)
            (root / "stages").mkdir()
            for current in STAGES:
                run_child(root, policy, release, current, enable_provider)
            report = read_sealed_json(root / "evaluation" / "joint-report.json")
            done, _ = publish_sealed_json(root / "finished.json", {
                "policy_sha256": policy.sha256, "joint_report_sha256": report.sha256,
                "accuracy": report.payload["accuracy"], "target_gate_passed": report.payload["target_gate_passed"],
                "joint_report_replayed": True})
            print({"full_pipeline_finished_sha256": done.sha256, **done.payload}, flush=True)
            return done
        except Exception as error:
            publish_sealed_json(root / "failure.json", {"policy_sha256": policy.sha256,
                "stage": current, "error_type": type(error).__name__, "automatic_retry": False})
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "run", "stage"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--stage", choices=STAGES)
    parser.add_argument("--enable-provider", action="store_true")
    args = parser.parse_args()
    if args.phase == "prepare":
        prepare(json.loads(args.config.read_text(encoding="utf-8")), args.root)
    elif args.phase == "stage":
        stage(args.root, args.stage, enable_provider=args.enable_provider)
    else:
        run(args.root, enable_provider=args.enable_provider)
