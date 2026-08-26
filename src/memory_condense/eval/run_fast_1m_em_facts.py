"""Fast S1 episodic-memory facts experiment: preflight, run, score."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import ssl
import statistics
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence, cast

from dotenv import load_dotenv

from memory_condense.domain.discourse import identity_sha256, quote_sha256
from memory_condense.eval._artifact_json import canonical_json_bytes
from memory_condense.eval.benchmark import exact_match, f1_score
from memory_condense.eval.fast_completion_runtime import (
    FastCompletionBatch,
    FastCompletionRuntime,
    preflight_fast_completion_prompts,
)
from memory_condense.eval.fast_em_fact_memory import (
    DEFAULT_EM_FACT_POLICY,
    DEFAULT_EM_STAGE_ID,
    EM_FACT_ARMS,
    EM_FACT_POLICIES,
    EMFactArm,
    EMFactAnswerPrompt,
    EMFactCompression,
    EMFactPolicy,
    build_em_fact_answer_prompt,
    build_fact_compression_messages,
    parse_fact_compression,
)
from memory_condense.eval.recall_guarded_cumulative_fast_artifact import (
    ORIGINAL_1M_RETRIEVAL_SHA256,
    FastRetrievalArtifact,
    load_fast_retrieval_artifact,
)

PREFLIGHT_FORMAT = "memory-condense-fast-1m-em-facts-preflight-v1"
RUN_FORMAT = "memory-condense-fast-1m-em-facts-run-v1"
SCORE_FORMAT = "memory-condense-fast-1m-em-facts-score-v1"
DEFAULT_RETRIEVAL = Path(
    "eval_results/longmemeval-1m-recall-guarded-cumulative-"
    "development-20260821/retrieval.json"
)
DEFAULT_SPLIT = Path(
    "docs/10 - Research Log/data/longmemeval-95-target-split-v2.json"
)
DEFAULT_OUTPUT_ROOT = Path(
    "eval_results/longmemeval-1m-em-facts-development-20260825"
)
DEFAULT_GATEWAY_URL = "https://central-dev.zt:4000/v1"
DEFAULT_MODEL = "codex_sdk/gpt-5.6-terra"
DEFAULT_EXPECTED_QUESTION_COUNT = 10
MAX_PROMPT_TOKENS = 8_000
MAX_COMPRESSION_OUTPUT_TOKENS = 1_024
MAX_ANSWER_OUTPUT_TOKENS = 256
MAX_FACTS = 24


def _memory_policy(args: argparse.Namespace) -> EMFactPolicy:
    value = str(getattr(args, "memory_policy", DEFAULT_EM_FACT_POLICY))
    if value not in EM_FACT_POLICIES:
        raise ValueError(f"unknown EM fact-memory policy: {value!r}")
    return cast(EMFactPolicy, value)


def _answer_arms(args: argparse.Namespace) -> tuple[EMFactArm, ...]:
    requested = getattr(args, "answer_arms", None)
    if requested is None:
        if _memory_policy(args) == "v2":
            return ("facts",)
        return EM_FACT_ARMS
    values = tuple(str(value) for value in requested)
    if not values or len(values) != len(set(values)):
        raise ValueError("--answer-arms must name one or more unique arms")
    unknown = tuple(value for value in values if value not in EM_FACT_ARMS)
    if unknown:
        raise ValueError(f"unknown EM fact-memory arm: {unknown[0]!r}")
    return cast(tuple[EMFactArm, ...], values)


def _effective_output_root(args: argparse.Namespace) -> Path:
    root = Path(args.output_root)
    policy = _memory_policy(args)
    arms = _answer_arms(args)
    if root != DEFAULT_OUTPUT_ROOT or (
        policy == DEFAULT_EM_FACT_POLICY and arms == EM_FACT_ARMS
    ):
        return root
    arm_slug = "-".join(arms)
    return root.with_name(f"{root.name}-{policy}-{arm_slug}")


def _publish(path: Path, value: object) -> str:
    raw = canonical_json_bytes(value)
    digest = hashlib.sha256(raw).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    outputs = (
        (path, raw),
        (path.with_name(path.name + ".sha256"), f"{digest}  {path.name}\n".encode()),
    )
    for target, payload in outputs:
        if target.exists():
            if target.is_symlink() or not target.is_file() or target.read_bytes() != payload:
                raise FileExistsError(f"refusing to replace artifact: {target}")
            continue
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
        finally:
            if temporary.exists():
                temporary.unlink()
    return digest


def _read(path: Path) -> tuple[dict[str, Any], str]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"artifact must be a regular file: {path}")
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"artifact is not JSON: {path}") from exc
    if type(value) is not dict or raw != canonical_json_bytes(value):
        raise ValueError(f"artifact is not canonical JSON: {path}")
    digest = hashlib.sha256(raw).hexdigest()
    sidecar = path.with_name(path.name + ".sha256")
    expected = f"{digest}  {path.name}\n".encode()
    if sidecar.is_symlink() or not sidecar.is_file() or sidecar.read_bytes() != expected:
        raise ValueError(f"artifact sidecar is invalid: {path}")
    return value, digest


def _run_path(args: argparse.Namespace) -> Path:
    return Path(args.run_artifact or _effective_output_root(args) / "run.json")


def _calls(args: argparse.Namespace, kind: str) -> Path:
    return _run_path(args).parent / f"{kind}-calls"


def _load(args: argparse.Namespace) -> FastRetrievalArtifact:
    digest = str(args.expected_retrieval_sha256)
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("--expected-retrieval-sha256 must be an exact digest")
    if type(args.expected_question_count) is not int or args.expected_question_count < 1:
        raise ValueError("--expected-question-count must be positive")
    artifact = load_fast_retrieval_artifact(Path(args.retrieval), expected_sha256=digest)
    if artifact.question_count != args.expected_question_count:
        raise ValueError("sealed retrieval question count changed")
    for question in artifact.questions:
        question.stage(str(args.source_stage_id))
    return artifact


def _binding(artifact: FastRetrievalArtifact, stage_id: str) -> dict[str, Any]:
    body: dict[str, Any] = {
        "retrieval_sha256": artifact.raw_sha256,
        "population_identity_sha256": artifact.population_identity_sha256,
        "source_stage_id": stage_id,
        "questions": [
            {
                "question_id": question.question_id,
                "retrieval_receipt_sha256": question.retrieval_receipt_sha256,
                "stage_receipt_sha256": question.stage(stage_id).stage_receipt_sha256,
            }
            for question in artifact.questions
        ],
    }
    body["binding_sha256"] = identity_sha256(body)
    return body


def _guard_existing_run_identity(
    args: argparse.Namespace,
    artifact: FastRetrievalArtifact,
    settings: Mapping[str, Any],
    arms: Sequence[EMFactArm],
    exact_calls: int,
) -> None:
    path = _run_path(args)
    if not path.exists():
        return
    existing, _digest = _read(path)
    expected_binding = _binding(artifact, str(settings["stage_id"]))
    existing_answers = existing.get("answers")
    if (
        existing.get("format") != RUN_FORMAT
        or existing.get("retrieval_binding") != expected_binding
        or existing.get("settings") != dict(settings)
        or existing.get("question_count") != artifact.question_count
        or existing.get("authorized_physical_calls") != exact_calls
        or existing.get("journaled_completion_calls") != exact_calls
        or not isinstance(existing_answers, Mapping)
        or existing_answers.get("arms") != list(arms)
    ):
        raise FileExistsError(
            "refusing provider access because the run path contains a different "
            f"experiment: {path}"
        )


def _compression_prompts(
    artifact: FastRetrievalArtifact,
    stage_id: str,
    *,
    policy: EMFactPolicy = DEFAULT_EM_FACT_POLICY,
) -> tuple[tuple[dict[str, str], ...], ...]:
    return tuple(
        build_fact_compression_messages(
            question,
            stage_id=stage_id,
            policy=policy,
        )
        for question in artifact.questions
    )


def _answer_prompts(
    artifact: FastRetrievalArtifact,
    compressions: Sequence[EMFactCompression],
    *,
    policy: EMFactPolicy = DEFAULT_EM_FACT_POLICY,
    arms: Sequence[EMFactArm] = EM_FACT_ARMS,
) -> tuple[EMFactAnswerPrompt, ...]:
    return tuple(
        build_em_fact_answer_prompt(
            question,
            compression,
            arm=arm,
            max_prompt_tokens=MAX_PROMPT_TOKENS,
            responder_output_token_reserve=MAX_ANSWER_OUTPUT_TOKENS,
            policy=policy,
        )
        for question, compression in zip(artifact.questions, compressions, strict=True)
        for arm in arms
    )


def _settings(args: argparse.Namespace) -> dict[str, Any]:
    if type(args.max_concurrency) is not int or args.max_concurrency < 1:
        raise ValueError("--max-concurrency must be positive")
    values = {
        "stage_id": str(args.source_stage_id),
        "gateway_url": str(args.gateway_url),
        "model": str(args.model),
        "max_concurrency": args.max_concurrency,
    }
    if any(not str(values[name]).strip() for name in ("stage_id", "gateway_url", "model")):
        raise ValueError("stage, gateway, and model must be non-empty")
    policy = _memory_policy(args)
    arms = _answer_arms(args)
    # Preserve the exact v1 settings object used by the sealed 40-call run.
    # Candidate policies are explicit additions rather than silent reinterpretations.
    if policy != DEFAULT_EM_FACT_POLICY or arms != EM_FACT_ARMS:
        values["memory_policy"] = policy
        values["answer_arms"] = list(arms)
    return values


def _runtime(
    artifact: FastRetrievalArtifact,
    args: argparse.Namespace,
    settings: Mapping[str, Any],
    *,
    kind: str,
    prompts: Sequence[Sequence[Mapping[str, str]]],
    client: Any | None,
) -> FastCompletionRuntime:
    max_new_tokens = (
        MAX_COMPRESSION_OUTPUT_TOKENS if kind == "compression" else MAX_ANSWER_OUTPUT_TOKENS
    )
    benchmark_provenance: dict[str, Any] = {
        "experiment_format": RUN_FORMAT,
        "kind": kind,
        "retrieval_sha256": artifact.raw_sha256,
        "source_stage_id": settings["stage_id"],
        "gateway_url": settings["gateway_url"],
        "gold_loaded": False,
    }
    if "memory_policy" in settings:
        benchmark_provenance["memory_policy"] = settings["memory_policy"]
        benchmark_provenance["answer_arms"] = settings["answer_arms"]
    return FastCompletionRuntime(
        checkpoint_dir=_calls(args, kind),
        prompt_population=prompts,
        model=str(settings["model"]),
        client=client,
        max_prompt_tokens=MAX_PROMPT_TOKENS,
        max_new_tokens=max_new_tokens,
        max_concurrency=int(settings["max_concurrency"]),
        retries=0,
        benchmark_provenance=benchmark_provenance,
    )


def _runtime_receipt(batch: FastCompletionBatch) -> dict[str, Any]:
    return {
        "runtime_identity_sha256": batch.runtime_identity_sha256,
        "prompt_population_sha256": batch.prompt_population.prompt_population_sha256,
        "response_journal_sha256s": [
            row.response_journal_sha256 for row in batch.unique_records
        ],
    }


def _exact_calls(
    question_count: int,
    arms: Sequence[EMFactArm] = EM_FACT_ARMS,
) -> int:
    return question_count * (1 + len(arms))


def run_preflight(args: argparse.Namespace) -> dict[str, Any]:
    if args.enable_provider or args.authorized_provider_calls:
        raise ValueError("preflight forbids provider access and authorization")
    artifact = _load(args)
    stage_id = str(args.source_stage_id)
    policy = _memory_policy(args)
    arms = _answer_arms(args)
    population = preflight_fast_completion_prompts(
        _compression_prompts(artifact, stage_id, policy=policy),
        max_prompt_tokens=MAX_PROMPT_TOKENS,
    )
    if population.unique_prompt_count != artifact.question_count:
        raise ValueError("compression prompts are not one-per-question")
    return {
        "format": PREFLIGHT_FORMAT,
        "retrieval_binding": _binding(artifact, stage_id),
        "question_count": artifact.question_count,
        "compression_prompt_population": population.model_dump(),
        "memory_policy": policy,
        "dependent_answer_arms": list(arms),
        "exact_authorized_physical_calls": _exact_calls(artifact.question_count, arms),
        "max_prompt_tokens": MAX_PROMPT_TOKENS,
        "compression_output_tokens": MAX_COMPRESSION_OUTPUT_TOKENS,
        "answer_output_tokens": MAX_ANSWER_OUTPUT_TOKENS,
        "provider_calls": 0,
        "writes": 0,
        "gold_loaded": False,
    }


def _make_provider_client(api_key: str, gateway_url: str) -> Any:
    import httpx
    import truststore
    from openai import OpenAI

    return OpenAI(
        api_key=api_key,
        base_url=gateway_url,
        http_client=httpx.Client(
            verify=truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        ),
        max_retries=0,
    )


def run_experiment(args: argparse.Namespace) -> tuple[dict[str, Any], str]:
    artifact, settings = _load(args), _settings(args)
    policy = _memory_policy(args)
    arms = _answer_arms(args)
    exact_calls = _exact_calls(artifact.question_count, arms)
    if not args.enable_provider:
        raise ValueError("run requires --enable-provider")
    if args.authorized_provider_calls != exact_calls:
        raise ValueError(
            f"--authorized-provider-calls must exactly equal {exact_calls} "
            f"({args.authorized_provider_calls} given)"
        )
    _guard_existing_run_identity(args, artifact, settings, arms, exact_calls)
    api_key = os.environ.get(str(args.api_key_env), "").strip()
    if not api_key:
        raise RuntimeError(f"provider API key is empty: {args.api_key_env}")

    stage_id = str(settings["stage_id"])
    compression_prompts = _compression_prompts(
        artifact,
        stage_id,
        policy=policy,
    )
    client = _make_provider_client(api_key, str(settings["gateway_url"]))
    try:
        compression_batch = _runtime(
            artifact, args, settings, kind="compression",
            prompts=compression_prompts, client=client,
        ).run()
        compressions = tuple(
            parse_fact_compression(
                question, response, stage_id=stage_id, max_facts=MAX_FACTS
            )
            for question, response in zip(
                artifact.questions, compression_batch.logical_completions, strict=True
            )
        )
        answer_prompts = _answer_prompts(
            artifact,
            compressions,
            policy=policy,
            arms=arms,
        )
        answer_messages = tuple(prompt.as_mappings() for prompt in answer_prompts)
        answer_population = preflight_fast_completion_prompts(
            answer_messages, max_prompt_tokens=MAX_PROMPT_TOKENS
        )
        if answer_population.unique_prompt_count != (
            artifact.question_count * len(arms)
        ):
            raise ValueError("answer prompts are not one-per-question-and-arm")
        answer_batch = _runtime(
            artifact, args, settings, kind="answer",
            prompts=answer_messages, client=client,
        ).run()
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            close()

    physical_calls = (
        compression_batch.usage.physical_calls + answer_batch.usage.physical_calls
    )
    checkpoint_hits = (
        compression_batch.usage.checkpoint_hits + answer_batch.usage.checkpoint_hits
    )
    if physical_calls + checkpoint_hits != exact_calls:
        raise RuntimeError("completion journal population changed")
    result = {
        "format": RUN_FORMAT,
        "retrieval_binding": _binding(artifact, stage_id),
        "settings": settings,
        "question_count": artifact.question_count,
        "authorized_physical_calls": exact_calls,
        "journaled_completion_calls": exact_calls,
        "physical_calls_this_invocation": physical_calls,
        "checkpoint_hits_this_invocation": checkpoint_hits,
        "compression": _runtime_receipt(compression_batch),
        "answers": {
            "arms": list(arms),
            "runtime": _runtime_receipt(answer_batch),
        },
        "gold_loaded": False,
        "retained_request_token_state_bytes": 0,
    }
    return result, _publish(_run_path(args), result)


def _stored_policy_and_arms(
    settings: Mapping[str, Any],
) -> tuple[EMFactPolicy, tuple[EMFactArm, ...]]:
    has_policy = "memory_policy" in settings
    has_arms = "answer_arms" in settings
    if has_policy != has_arms:
        raise ValueError("run settings contain a partial EM policy identity")
    if not has_policy:
        return DEFAULT_EM_FACT_POLICY, EM_FACT_ARMS
    policy = str(settings["memory_policy"])
    raw_arms = settings["answer_arms"]
    if policy not in EM_FACT_POLICIES or type(raw_arms) is not list:
        raise ValueError("run settings contain an invalid EM policy identity")
    arms = tuple(str(value) for value in raw_arms)
    if (
        not arms
        or len(arms) != len(set(arms))
        or any(value not in EM_FACT_ARMS for value in arms)
    ):
        raise ValueError("run settings contain invalid EM answer arms")
    return cast(EMFactPolicy, policy), cast(tuple[EMFactArm, ...], arms)


def _verified_predictions(
    args: argparse.Namespace,
    artifact: FastRetrievalArtifact,
    run: Mapping[str, Any],
) -> list[dict[str, Any]]:
    settings, stage_id = run.get("settings"), str(args.source_stage_id)
    if (
        run.get("format") != RUN_FORMAT
        or run.get("retrieval_binding") != _binding(artifact, stage_id)
        or run.get("question_count") != artifact.question_count
        or run.get("gold_loaded") is not False
        or run.get("retained_request_token_state_bytes") != 0
        or not isinstance(settings, Mapping)
        or settings.get("stage_id") != stage_id
    ):
        raise ValueError("run artifact changed experiment provenance")
    policy, arms = _stored_policy_and_arms(settings)
    physical_calls = run.get("physical_calls_this_invocation")
    checkpoint_hits = run.get("checkpoint_hits_this_invocation")
    if (
        policy != _memory_policy(args)
        or arms != _answer_arms(args)
        or run.get("authorized_physical_calls")
        != _exact_calls(artifact.question_count, arms)
        or run.get("journaled_completion_calls")
        != run.get("authorized_physical_calls")
        or type(physical_calls) is not int
        or physical_calls < 0
        or type(checkpoint_hits) is not int
        or checkpoint_hits < 0
        or physical_calls + checkpoint_hits
        != run.get("journaled_completion_calls")
    ):
        raise ValueError("run artifact changed experiment provenance")

    compression_replay = _runtime(
        artifact, args, settings, kind="compression",
        prompts=_compression_prompts(artifact, stage_id, policy=policy), client=None,
    ).run()
    if run.get("compression") != _runtime_receipt(compression_replay):
        raise ValueError("compression checkpoints changed")
    compressions = tuple(
        parse_fact_compression(
            question, response, stage_id=stage_id, max_facts=MAX_FACTS
        )
        for question, response in zip(
            artifact.questions,
            compression_replay.logical_completions,
            strict=True,
        )
    )

    answer_prompts = _answer_prompts(
        artifact,
        compressions,
        policy=policy,
        arms=arms,
    )
    answer_replay = _runtime(
        artifact, args, settings, kind="answer",
        prompts=tuple(prompt.as_mappings() for prompt in answer_prompts), client=None,
    ).run()
    if (
        run.get("answers", {}).get("arms") != list(arms)
        or run.get("answers", {}).get("runtime")
        != _runtime_receipt(answer_replay)
    ):
        raise ValueError("answer checkpoints changed")
    return [
        {
            "question_id": prompt.question_id,
            "arm": prompt.arm,
            "completion": completion,
        }
        for prompt, completion in zip(
            answer_prompts,
            answer_replay.logical_completions,
            strict=True,
        )
    ]


def _load_gold_population(dataset: Path, split: Path) -> Any:
    from memory_condense.eval.recall_guarded_cumulative_1m import (
        load_original_population,
    )

    return load_original_population(dataset, split)


def run_score(args: argparse.Namespace) -> tuple[dict[str, Any], str]:
    if args.enable_provider or args.authorized_provider_calls:
        raise ValueError("score forbids provider access and authorization")
    if args.dataset is None:
        raise ValueError("score requires --dataset")
    artifact = _load(args)
    arms = _answer_arms(args)
    run, run_sha256 = _read(_run_path(args))
    predictions = _verified_predictions(args, artifact, run)
    # Gold is unreachable until both completion checkpoint populations replay.
    gold = _load_gold_population(Path(args.dataset), Path(args.split))
    gold_by_id = {row.question_id: row for row in gold.questions}
    if tuple(gold_by_id) != tuple(row.question_id for row in artifact.questions):
        raise RuntimeError("post-hoc gold population changed question order")
    rows = []
    for ordinal, source in enumerate(predictions):
        gold_row = gold_by_id[source["question_id"]]
        prediction = source["completion"].strip()
        rows.append(
            {
                "logical_ordinal": ordinal,
                "question_id": source["question_id"],
                "arm": source["arm"],
                "category": gold_row.category,
                "prediction_sha256": quote_sha256(prediction),
                "gold_answer_sha256": quote_sha256(gold_row.answer),
                "exact_match": exact_match(prediction, gold_row.answer),
                "f1": f1_score(prediction, gold_row.answer),
            }
        )
    aggregates = []
    for arm in arms:
        selected = [row for row in rows if row["arm"] == arm]
        aggregates.append(
            {
                "arm": arm,
                "questions": len(selected),
                "exact_matches": sum(bool(row["exact_match"]) for row in selected),
                "exact_match_rate": statistics.fmean(
                    float(row["exact_match"]) for row in selected
                ),
                "mean_f1": statistics.fmean(float(row["f1"]) for row in selected),
            }
        )
    result = {
        "format": SCORE_FORMAT,
        "run_artifact_sha256": run_sha256,
        "retrieval_binding": run["retrieval_binding"],
        "question_count": artifact.question_count,
        "logical_score_count": len(rows),
        "gold_loaded_posthoc": True,
        "aggregates": aggregates,
        "rows": rows,
    }
    return result, _publish(_run_path(args).parent / "scores.json", result)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", choices=("preflight", "run", "score"), default="preflight"
    )
    parser.add_argument("--retrieval", type=Path, default=DEFAULT_RETRIEVAL)
    parser.add_argument(
        "--expected-retrieval-sha256", default=ORIGINAL_1M_RETRIEVAL_SHA256
    )
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-artifact", type=Path)
    parser.add_argument("--source-stage-id", default=DEFAULT_EM_STAGE_ID)
    parser.add_argument(
        "--memory-policy",
        choices=EM_FACT_POLICIES,
        default=DEFAULT_EM_FACT_POLICY,
        help="v1 replays the sealed experiment; v2 enables the streamlined candidate",
    )
    parser.add_argument(
        "--answer-arms",
        nargs="+",
        choices=EM_FACT_ARMS,
        help=(
            "answer only these unique arms; defaults to all arms for v1 and "
            "facts-only for v2"
        ),
    )
    parser.add_argument(
        "--expected-question-count", type=int, default=DEFAULT_EXPECTED_QUESTION_COUNT
    )
    parser.add_argument("--gateway-url", default=DEFAULT_GATEWAY_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--api-key-env", default="LITELLM_KEY")
    parser.add_argument("--max-concurrency", type=int, default=4)
    parser.add_argument("--enable-provider", action="store_true")
    parser.add_argument("--authorized-provider-calls", type=int, default=0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    load_dotenv()
    args = build_parser().parse_args(argv)
    if args.phase == "preflight":
        result = run_preflight(args)
        print(
            f"EM-fact preflight: questions={result['question_count']}; "
            f"authorized_calls={result['exact_authorized_physical_calls']}; "
            "provider_calls=0; writes=0"
        )
        return 0
    if args.phase == "run":
        result, digest = run_experiment(args)
        print(
            f"EM-fact run {digest}: "
            f"physical={result['physical_calls_this_invocation']}; "
            f"checkpoint_hits={result['checkpoint_hits_this_invocation']}"
        )
        return 0
    result, digest = run_score(args)
    print(f"EM-fact scores {digest}")
    for row in result["aggregates"]:
        print(
            f"  {row['arm']}: EM={row['exact_matches']}/{row['questions']} "
            f"F1={row['mean_f1']:.6f}"
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
