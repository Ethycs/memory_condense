"""Resumable LongMemEval-S evaluation through the production native-memory route.

The answer phase intentionally never reads ``answer``.  Gold answers are opened
only by the separate judge phase after a response artifact exists.  Local Qwen
attention and BGE route summaries; exact raw spans are hydrated for Sol.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import time

from memory_condense.runtime.artifacts import read, save
from memory_condense.runtime.config import RuntimeAssets


DATASET = Path(r"C:\Users\Keytone\Downloads\memory-condense-rig\datasets\longmemeval_s_cleaned.json")
ROOT = Path("eval_results/longmemeval-s-native-release-sol-20261010-r6")
GATEWAY = "https://central-dev.zt:4000/v1"
MODEL = "codex_sdk/gpt-5.6-sol"


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def records_and_digest(dataset: Path) -> tuple[list[dict], str]:
    raw = dataset.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, list) or len(value) != 500 or any(type(row) is not dict for row in value):
        raise ValueError("Expected the official 500-record LongMemEval-S JSON array")
    return value, hashlib.sha256(raw).hexdigest()


def records(dataset: Path) -> list[dict]:
    return records_and_digest(dataset)[0]


def sample_without_gold(record: dict):
    from memory_condense.ingest.loader import parse_longmemeval
    # Keep the public loader's exact ordering/date/source behavior while making
    # the answer unavailable to the ingest, retrieval and responder code below.
    blinded = {key: value for key, value in record.items() if key != "answer"}
    blinded["answer"] = "[withheld until judge phase]"
    parsed = parse_longmemeval([blinded])
    if len(parsed) != 1 or len(parsed[0].questions) != 1:
        raise ValueError("LongMemEval record did not produce exactly one question")
    return parsed[0]


def events_for(sample, ordinal: int) -> list:
    from memory_condense.application.chat_session import ChatEvent
    if not (len(sample.turns) == len(sample.turn_source_ids) == len(sample.turn_created_at)):
        raise ValueError("LongMemEval turn metadata is not parallel")
    result = []
    for i, ((role, text), source_id, stamp) in enumerate(zip(
            sample.turns, sample.turn_source_ids, sample.turn_created_at, strict=True)):
        if stamp is None:
            raise ValueError("LongMemEval-S turn is missing its session timestamp")
        result.append(ChatEvent(f"lmes-{ordinal:03d}-t{i:04d}", role, text,
            stamp.astimezone(timezone.utc).isoformat(), {"source_id": source_id}))
    return result


def plan(root: Path, dataset: Path) -> dict:
    return read(root / "plan.json")


def runtime_attempt(root: Path, phase: str) -> Path:
    """Reserve a fresh process-receipt directory without replacing prior runs."""
    parent = root / "runtime-runs"
    parent.mkdir(parents=True, exist_ok=True)
    for ordinal in range(10000):
        candidate = parent / f"{phase}-{ordinal:04d}"
        try:
            candidate.mkdir()
        except FileExistsError:
            continue
        return candidate
    raise RuntimeError(f"No free runtime attempt directory for {phase}")


def prepare(root: Path, dataset: Path) -> None:
    if root.exists():
        raise ValueError("Use a fresh evaluation root")
    rows, dataset_sha256 = records_and_digest(dataset)
    question_ids = [str(row.get("question_id")) for row in rows]
    if len(set(question_ids)) != 500:
        raise ValueError("LongMemEval-S question IDs are not unique")
    source_files = [Path(__file__), Path("src/memory_condense/runtime/resident.py"),
        Path("src/memory_condense/runtime/local.py"),
        Path("src/memory_condense/runtime/config.py"),
        Path("src/memory_condense/runtime/embedding.py"),
        Path("src/memory_condense/runtime/attention.py"),
        Path("src/memory_condense/modeling/qwen_prefix.py"),
        Path("src/memory_condense/runtime/summary_fallback.py"),
        Path("tools/engineering_research_gateway_reader.py"),
        Path("src/memory_condense/application/chat_native.py"),
        Path("src/memory_condense/application/native_spine_retrieval.py"),
        Path("src/memory_condense/application/dialogue_retrieval.py"),
        Path("src/memory_condense/application/tactic_router.py"),
        Path("src/memory_condense/eval/benchmark.py")]
    assets = RuntimeAssets.resolve()
    save(root / "plan.json", {
        "format": "longmemeval-s-native-release-sol-v2",
        "dataset": str(dataset.resolve()), "dataset_sha256": dataset_sha256,
        "question_count": 500, "question_ids": question_ids,
        "answer_model": MODEL, "judge_model": MODEL, "gateway": GATEWAY,
        "retrieval_policy": "auto", "summary_api_fallback": True,
        "gold_boundary": "answer phase removes the answer field before parsing; judge opens gold only after saved response",
        "raw_to_qwen": False, "exact_raw_hydration": True,
        "runtime_assets": {"root": str(assets.root), "bge_model": str(assets.bge)},
        "implementation": {str(path): digest(path) for path in source_files},
    })
    print({"phase": "prepared", "questions": 500, "dataset_sha256": dataset_sha256}, flush=True)


def validate_plan(root: Path, dataset: Path, *, dataset_sha256: str | None = None) -> dict:
    value = plan(root, dataset)
    if (value["format"] != "longmemeval-s-native-release-sol-v2"
            or value["dataset_sha256"] != (dataset_sha256 or digest(dataset))
            or value["question_count"] != 500
            or value["answer_model"] != MODEL or value["judge_model"] != MODEL):
        raise ValueError("Benchmark plan or dataset changed")
    assets = RuntimeAssets.resolve()
    if value["runtime_assets"] != {"root": str(assets.root), "bge_model": str(assets.bge)}:
        raise ValueError("Benchmark runtime asset layout changed")
    for path, sha in value["implementation"].items():
        if digest(Path(path)) != sha:
            raise ValueError(f"Frozen implementation changed: {path}")
    return value


def answer(root: Path, dataset: Path, start: int, count: int) -> None:
    from memory_condense.application.dialogue_retrieval import DialogueRetrievalPolicy
    from memory_condense.eval.benchmark import build_qa_prompt
    from memory_condense.runtime.resident import ResidentNativeBackend
    from tools.engineering_research_gateway_reader import GatewayReaderRuntime

    class SharedModelBackend(ResidentNativeBackend):
        """A benchmark history owns its DB; the campaign owns resident model weights."""
        def close(self):
            if self.app is not None:
                self.app.close()
                self.app = None

    rows, dataset_sha256 = records_and_digest(dataset)
    p = validate_plan(root, dataset, dataset_sha256=dataset_sha256)
    stop = min(start + count, len(rows))
    if not 0 <= start < stop <= 500:
        raise ValueError("Requested answer slice is empty or outside the 500 records")
    runtime = GatewayReaderRuntime(runtime_attempt(root, "answer"), gateway=p["gateway"],
        reader_model=p["answer_model"], judge_model=p["judge_model"],
        empty_response_retries=2, summary_api_fallback=True)
    try:
        for ordinal in range(start, stop):
            target = root / "samples" / f"{ordinal:03d}-{p['question_ids'][ordinal]}"
            response_path = target / "answer.json"
            if response_path.exists():
                print({"phase": "answer_replay", "ordinal": ordinal}, flush=True)
                continue
            target.mkdir(parents=True, exist_ok=True)
            sample = sample_without_gold(rows[ordinal])
            question = sample.questions[0]
            if question.answer != "[withheld until judge phase]":
                raise ValueError("Gold crossed the answer-phase boundary")
            events = events_for(sample, ordinal)
            actor = {"case_id": f"longmemeval-s-{ordinal:03d}", "source": {
                "family": f"longmemeval-s-{ordinal:03d}",
                "export_timestamp": max(e.created_at for e in events)}}
            save(target / "input.json", {"ordinal": ordinal, "question_id": question.question_id,
                "question": question.question, "question_date": question.question_date,
                "turn_count": len(events), "answer_field_loaded": False})
            backend = SharedModelBackend(root / "compiler", target, actor, runtime=runtime,
                dialogue_policy=DialogueRetrievalPolicy(), retrieval_policy="auto")
            started = time.perf_counter()
            try:
                backend.sync(events)
                ingested_s = time.perf_counter() - started
                recall_started = time.perf_counter()
                packet = backend.recall(question.question)
                recall_s = time.perf_counter() - recall_started
                messages = build_qa_prompt(question.dated_question, [packet["text"]])
                result = runtime.call("actor", messages, scope=f"longmemeval-s-{ordinal:03d}", max_tokens=256)
                if result["finish_reason"] != "stop" or not result["content"].strip():
                    raise ValueError("Sol answer did not complete")
                payload = {"ordinal": ordinal, "question_id": question.question_id,
                    "question": question.question, "question_date": question.question_date,
                    "prediction": result["content"].strip(), "packet": packet,
                    "answer_response": result, "ingest_s": ingested_s, "recall_s": recall_s,
                    "total_s": time.perf_counter() - started, "turn_count": len(events),
                    "answer_field_loaded": False, "backend_timings": backend.timings}
                save(response_path, payload)
                print({"phase": "answered", "ordinal": ordinal,
                    "ingest_s": round(ingested_s, 3), "recall_s": round(recall_s, 3),
                    "answer_s": round(result["elapsed_s"], 3), "prediction": payload["prediction"]}, flush=True)
            finally:
                backend.close()
    finally:
        runtime.close()


def judge(root: Path, dataset: Path, start: int, count: int) -> None:
    from memory_condense.eval._binary_judge_protocol import parse_binary_judge_verdict
    from memory_condense.eval.benchmark import build_judge_prompt
    from memory_condense.ingest.loader import _as_answer_text
    from tools.engineering_research_gateway_reader import GatewayReaderRuntime

    rows, dataset_sha256 = records_and_digest(dataset)
    p = validate_plan(root, dataset, dataset_sha256=dataset_sha256)
    stop = min(start + count, len(rows))
    runtime = GatewayReaderRuntime(runtime_attempt(root, "judge"), gateway=p["gateway"],
        reader_model=p["answer_model"], judge_model=p["judge_model"], empty_response_retries=2)
    try:
        runtime.start_gateway()
        for ordinal in range(start, stop):
            target = root / "samples" / f"{ordinal:03d}-{p['question_ids'][ordinal]}"
            if (target / "judgment.json").exists():
                continue
            response = read(target / "answer.json")
            gold = _as_answer_text(rows[ordinal].get("answer"))
            if not gold or response["question_id"] != str(rows[ordinal]["question_id"]):
                raise ValueError("Gold record does not match the saved answer")
            messages = build_judge_prompt(response["question"], gold, response["prediction"])
            result = runtime.call("judge", messages, scope=f"longmemeval-s-{ordinal:03d}-judge", max_tokens=1024)
            correct = parse_binary_judge_verdict(result["content"])
            save(target / "judgment.json", {"ordinal": ordinal, "question_id": response["question_id"],
                "gold": gold, "prediction": response["prediction"], "correct": correct,
                "verdict": result["content"], "judge_response": result})
            print({"phase": "judged", "ordinal": ordinal, "correct": correct}, flush=True)
    finally:
        runtime.close()


def report(root: Path, dataset: Path) -> None:
    p = validate_plan(root, dataset)
    answers, judgments = [], []
    for ordinal, qid in enumerate(p["question_ids"]):
        target = root / "samples" / f"{ordinal:03d}-{qid}"
        if (target / "answer.json").exists():
            answers.append(read(target / "answer.json"))
        if (target / "judgment.json").exists():
            judgments.append(read(target / "judgment.json"))
    latencies = [row["answer_response"]["elapsed_s"] for row in answers]
    result = {"format": "longmemeval-s-native-release-sol-report-v1",
        "answered": len(answers), "judged": len(judgments),
        "correct": sum(row["correct"] for row in judgments),
        "judge_accuracy": (sum(row["correct"] for row in judgments) / len(judgments) if judgments else None),
        "mean_answer_s": statistics.fmean(latencies) if latencies else None,
        "mean_recall_s": statistics.fmean(row["recall_s"] for row in answers) if answers else None,
        "mean_ingest_s": statistics.fmean(row["ingest_s"] for row in answers) if answers else None,
        "complete": len(judgments) == 500}
    save(root / "report.json", result)
    print(result, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "answer", "judge", "report"))
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=500)
    args = parser.parse_args()
    globals()[args.phase](args.root.resolve(), args.dataset.resolve(),
        *(() if args.phase in ("prepare", "report") else (args.start, args.count)))
