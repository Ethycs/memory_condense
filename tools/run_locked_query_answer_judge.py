#!/usr/bin/env python3
"""Judge locked query answer arms with a split changed-only Sol lifecycle."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    repository = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(repository / "src"), str(repository)]

from dotenv import load_dotenv  # noqa: E402

from tools import run_locked_query_fact_answers as fact_cli  # noqa: E402
from tools import (  # noqa: E402
    run_locked_partition_payload_answers as partition_payload_cli,
)
from tools import (  # noqa: E402
    run_locked_query_guided_payload_answers as guided_payload_cli,
)
from tools import (  # noqa: E402
    run_locked_query_operator_refinement_answers as operator_cli,
)
from tools import run_locked_query_payload_answers as payload_cli  # noqa: E402
from tools import run_locked_adaptive_evidence_solver_v3 as adaptive_cli  # noqa: E402
from tools import (  # noqa: E402
    run_locked_query_evidence_map_solver_v2 as evidence_v2_cli,
)
from tools.matched_eval import judging, live  # noqa: E402
from tools.matched_eval.adaptive_evidence_solver_judge_adapter import (  # noqa: E402
    adapt_verified_adaptive_evidence_solver,
)
from tools.matched_eval.contracts import (  # noqa: E402
    MatchedEvalContractError,
    require_sha256,
)
from tools.matched_eval.query_answer_judging import (  # noqa: E402
    JUDGE_CHECKPOINT_DIR_NAME,
    load_query_answer_judge_journals,
    load_query_answer_judge_provider_population,
    materialize_query_answer_changed_only_judge,
    preflight_query_answer_changed_only_judge,
    replay_query_answer_changed_only_judge,
    run_sealed_query_answer_judge_provider,
)
from tools.matched_eval.query_fact_answer_live import (  # noqa: E402
    replay_query_fact_answers,
)
from tools.matched_eval.query_evidence_map_solver_v2_live import (  # noqa: E402
    replay_evidence_solver,
)
from tools.matched_eval.query_operator_refinement_live import (  # noqa: E402
    replay_query_operator_refinement_answers,
)
from tools.matched_eval.query_payload_live import (  # noqa: E402
    replay_query_payload_answers,
)
from tools.matched_eval.payload_arm_identity import (  # noqa: E402
    QUERY_PAYLOAD_PROFILE,
    load_verified_payload_semantic_arm_binding,
    profile_for_cli_arm,
)
from tools.matched_eval.source_history_fact_union import FactLane  # noqa: E402
from tools.run_matched_eval_spine import DEFAULT_SPLIT  # noqa: E402


DEFAULT_DATASET = Path(
    "C:/Users/Keytone/Downloads/memory-condense-rig/datasets/"
    "longmemeval_s_cleaned.json"
)
DEFAULT_PARENT_JUDGE_ROOT = payload_cli.DEFAULT_PARENT_ROOT
DEFAULT_PARENT_JUDGE_SHA256 = (
    "05fec9a7f284bb4e95d286f44e7378a8bbc1737a03e7c2ed60aefd50e6ddc689"
)
DEFAULT_PARENT_SCORE_LEDGER_SHA256 = (
    "3422ce2825bdcdc347c8307bd3fed5a46de3dff6d33510c8bc3a3ba1c31c56e1"
)
DEFAULT_OPERATOR_PARENT_JUDGE_ROOT = payload_cli.DEFAULT_OUTPUT
DEFAULT_OPERATOR_PARENT_JUDGE_SHA256 = (
    "f0460baa796220f9975ab2f4e8250e231ed67da128182f4880f7ac9ef5a4c097"
)
DEFAULT_OPERATOR_PARENT_SCORE_LEDGER_SHA256 = (
    "41ef567a1d27d4c840489def844372892fb029f7f57ea9f215780e19886d21bb"
)


@dataclass(frozen=True, slots=True)
class _AdaptiveCliProfile:
    source_root: Path
    answer_root: Path
    judge_root: Path
    lanes: tuple[FactLane, ...]
    direct_base_cap: int
    partition_base_cap: int
    guided_base_cap: int
    source_preflight_sha256: str
    source_materialization_sha256: str
    solver_preflight_sha256: str
    solver_run_sha256: str


_ADAPTIVE_CAMPAIGN_ROOT = Path("eval_results/matched_eval_100")
_ADAPTIVE_SOURCE_CAMPAIGN_ROOT = (
    _ADAPTIVE_CAMPAIGN_ROOT
    / "adaptive-source-pareto-consolidated-authority-v1"
)
_DG_SOURCE_PREFLIGHT_SHA256 = (
    "216be985c901e47b2bc8ae21917f7417e1443704051f150aba7a4b40dec1a3e6"
)
_DG_SOURCE_MATERIALIZATION_SHA256 = (
    "21f4c79c1c0d4d663bca8fffbfb3f38933ae5ab72492434b2af860babfdd03e6"
)


def _adaptive_cli_profile(
    *,
    answer_name: str,
    source_name: str,
    lanes: tuple[FactLane, ...],
    caps: tuple[int, int, int],
    source_preflight_sha256: str,
    source_materialization_sha256: str,
    solver_preflight_sha256: str,
    solver_run_sha256: str,
) -> _AdaptiveCliProfile:
    answer_root = _ADAPTIVE_CAMPAIGN_ROOT / answer_name
    return _AdaptiveCliProfile(
        source_root=_ADAPTIVE_SOURCE_CAMPAIGN_ROOT / source_name,
        answer_root=answer_root,
        judge_root=_ADAPTIVE_CAMPAIGN_ROOT / f"{answer_name}-judge",
        lanes=lanes,
        direct_base_cap=caps[0],
        partition_base_cap=caps[1],
        guided_base_cap=caps[2],
        source_preflight_sha256=source_preflight_sha256,
        source_materialization_sha256=source_materialization_sha256,
        solver_preflight_sha256=solver_preflight_sha256,
        solver_run_sha256=solver_run_sha256,
    )


_ADAPTIVE_CLI_PROFILES = {
    "adaptive-solver-v3-d": _adaptive_cli_profile(
        answer_name="adaptive-solver-v3-d-only",
        source_name="d1-p0-g1",
        lanes=(FactLane.DIRECT,),
        caps=(1, 0, 1),
        source_preflight_sha256=_DG_SOURCE_PREFLIGHT_SHA256,
        source_materialization_sha256=_DG_SOURCE_MATERIALIZATION_SHA256,
        solver_preflight_sha256=(
            "8324ad0e8b10180f84a957fadf9837905ca7b71ac957fa756cfef62caed41980"
        ),
        solver_run_sha256=(
            "a7f2ba44aaf1867bfdf98b571d0f53e8da08f062476c67011553e4c728ad5bf2"
        ),
    ),
    "adaptive-solver-v3-p": _adaptive_cli_profile(
        answer_name="adaptive-solver-v3-p-only",
        source_name="d0-p1-g0",
        lanes=(FactLane.PARTITION,),
        caps=(0, 1, 0),
        source_preflight_sha256=(
            "e1a4e95302b99a6327f9946b55620617cbd2113a6d5f8163807b07a570cffd36"
        ),
        source_materialization_sha256=(
            "9c06b5adb7ab83049b8bf8210372bff54978787f62739a92099b3c4111594a8e"
        ),
        solver_preflight_sha256=(
            "923768b1e9273eee27399cd70b1694f5a00d9b6275b253ffed650c1397c5717b"
        ),
        solver_run_sha256=(
            "ee693a9e7548fbe990a027ddc2e29a9f95712464be3d452396ac9bd71331458c"
        ),
    ),
    "adaptive-solver-v3-g": _adaptive_cli_profile(
        answer_name="adaptive-solver-v3-g-only",
        source_name="d1-p0-g1",
        lanes=(FactLane.GUIDED,),
        caps=(1, 0, 1),
        source_preflight_sha256=_DG_SOURCE_PREFLIGHT_SHA256,
        source_materialization_sha256=_DG_SOURCE_MATERIALIZATION_SHA256,
        solver_preflight_sha256=(
            "7e8fe981d57e7c1f8246e45b96c0bac7dd36f67af8b2ecad77a4a3e479f859b5"
        ),
        solver_run_sha256=(
            "1e412517a8166b7f0045469d3b65b228cac315618c1bd9ff100e43078f6b872a"
        ),
    ),
    "adaptive-solver-v3-dg": _adaptive_cli_profile(
        answer_name="adaptive-solver-v3-dg",
        source_name="d1-p0-g1",
        lanes=(FactLane.DIRECT, FactLane.GUIDED),
        caps=(1, 0, 1),
        source_preflight_sha256=_DG_SOURCE_PREFLIGHT_SHA256,
        source_materialization_sha256=_DG_SOURCE_MATERIALIZATION_SHA256,
        solver_preflight_sha256=(
            "ba5419cb94c1431ed61b3b519fd8eea0b8aeeb716b433d99c55692920613222e"
        ),
        solver_run_sha256=(
            "bf1f5238feb67c1ffc2044192f946dcf755d0e14f235cc7252a61c5236c552ca"
        ),
    ),
}


def _answer_root(args: argparse.Namespace) -> Path:
    if args.answer_root is not None:
        return Path(args.answer_root)
    if args.arm == "query-payload":
        return payload_cli.DEFAULT_OUTPUT
    if args.arm == "partition-payload":
        return partition_payload_cli.DEFAULT_OUTPUT
    if args.arm == "query-guided-payload":
        return guided_payload_cli.DEFAULT_OUTPUT
    if args.arm == "query-operator-refinement":
        return operator_cli.DEFAULT_OUTPUT
    if args.arm == "query-evidence-map-solver-v2":
        return evidence_v2_cli.DEFAULT_OUTPUT
    if args.arm in _ADAPTIVE_CLI_PROFILES:
        return _ADAPTIVE_CLI_PROFILES[args.arm].answer_root
    return fact_cli.DEFAULT_OUTPUT


def _judge_root(args: argparse.Namespace) -> Path:
    if args.judge_output_root is not None:
        return Path(args.judge_output_root)
    if args.arm in _ADAPTIVE_CLI_PROFILES:
        return _ADAPTIVE_CLI_PROFILES[args.arm].judge_root
    return _answer_root(args)


def _parent_judge_binding(args: argparse.Namespace) -> tuple[Path, str, str]:
    if args.arm in {
        "query-operator-refinement",
        "query-evidence-map-solver-v2",
        *_ADAPTIVE_CLI_PROFILES,
    }:
        root = (
            DEFAULT_OPERATOR_PARENT_JUDGE_ROOT
            if args.parent_judge_root is None
            else Path(args.parent_judge_root)
        )
        judge_sha = (
            DEFAULT_OPERATOR_PARENT_JUDGE_SHA256
            if args.expected_parent_judge_sha256 is None
            else args.expected_parent_judge_sha256
        )
        score_sha = (
            DEFAULT_OPERATOR_PARENT_SCORE_LEDGER_SHA256
            if args.expected_parent_score_ledger_sha256 is None
            else args.expected_parent_score_ledger_sha256
        )
        return root, judge_sha, score_sha
    return (
        DEFAULT_PARENT_JUDGE_ROOT
        if args.parent_judge_root is None
        else Path(args.parent_judge_root),
        DEFAULT_PARENT_JUDGE_SHA256
        if args.expected_parent_judge_sha256 is None
        else args.expected_parent_judge_sha256,
        DEFAULT_PARENT_SCORE_LEDGER_SHA256
        if args.expected_parent_score_ledger_sha256 is None
        else args.expected_parent_score_ledger_sha256,
    )


def _add_full_inputs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--arm",
        choices=(
            "query-payload",
            "partition-payload",
            "query-guided-payload",
            "query-operator-refinement",
            "query-evidence-map-solver-v2",
            "adaptive-solver-v3-d",
            "adaptive-solver-v3-p",
            "adaptive-solver-v3-g",
            "adaptive-solver-v3-dg",
            "query-fact",
        ),
        default="query-payload",
    )
    parser.add_argument("--answer-root", type=Path)
    parser.add_argument("--judge-output-root", type=Path)
    parser.add_argument("--retrieval", type=Path, default=payload_cli.DEFAULT_RETRIEVAL)
    parser.add_argument(
        "--query-preflight",
        type=Path,
        default=payload_cli.DEFAULT_QUERY_PREFLIGHT,
    )
    parser.add_argument("--query-run", type=Path, default=payload_cli.DEFAULT_QUERY_RUN)
    parser.add_argument(
        "--query-parent-root",
        type=Path,
        default=guided_payload_cli.DEFAULT_QUERY_PARENT_ROOT,
    )
    parser.add_argument(
        "--guided-root",
        type=Path,
        default=guided_payload_cli.DEFAULT_GUIDED_ROOT,
    )
    parser.add_argument(
        "--partition-generation",
        type=Path,
        default=partition_payload_cli.DEFAULT_GENERATION,
    )
    parser.add_argument(
        "--compression-root",
        type=Path,
        default=fact_cli.DEFAULT_COMPRESSION_ROOT,
    )
    parser.add_argument(
        "--direct-answer-root",
        type=Path,
        default=operator_cli.DEFAULT_DIRECT_ANSWER_ROOT,
    )
    parser.add_argument("--parent-root", type=Path, default=payload_cli.DEFAULT_PARENT_ROOT)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument(
        "--parent-judge-root",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--expected-parent-judge-sha256",
        default=None,
    )
    parser.add_argument(
        "--expected-parent-score-ledger-sha256",
        default=None,
    )
    parser.add_argument(
        "--expected-retrieval-sha256",
        default=payload_cli.EXPECTED_RETRIEVAL_SHA256,
    )
    parser.add_argument(
        "--expected-source-population-id",
        default=payload_cli.EXPECTED_SOURCE_POPULATION_ID,
    )
    parser.add_argument(
        "--expected-query-preflight-sha256",
        default=payload_cli.EXPECTED_QUERY_PREFLIGHT_SHA256,
    )
    parser.add_argument(
        "--expected-query-run-sha256",
        default=payload_cli.EXPECTED_QUERY_RUN_SHA256,
    )
    parser.add_argument(
        "--expected-query-runtime-ledger-sha256",
        default=guided_payload_cli.EXPECTED_QUERY_RUNTIME_LEDGER_SHA256,
    )
    parser.add_argument(
        "--expected-query-population-id",
        default=payload_cli.EXPECTED_QUERY_POPULATION_ID,
    )
    parser.add_argument(
        "--expected-query-prompt-population-sha256",
        default=payload_cli.EXPECTED_QUERY_PROMPT_POPULATION_SHA256,
    )
    parser.add_argument(
        "--expected-guided-run-sha256",
        default=guided_payload_cli.EXPECTED_GUIDED_RUN_SHA256,
    )
    parser.add_argument(
        "--expected-guided-runtime-ledger-sha256",
        default=guided_payload_cli.EXPECTED_GUIDED_RUNTIME_LEDGER_SHA256,
    )
    parser.add_argument(
        "--expected-partition-generation-sha256",
        default=partition_payload_cli.EXPECTED_GENERATION_SHA256,
    )
    parser.add_argument(
        "--expected-eligibility-sha256",
        default=partition_payload_cli.EXPECTED_ELIGIBILITY_SHA256,
    )
    parser.add_argument(
        "--expected-compression-sha256",
        default=fact_cli.EXPECTED_COMPRESSION_SHA256,
    )
    parser.add_argument(
        "--expected-compression-runtime-ledger-sha256",
        default=fact_cli.EXPECTED_COMPRESSION_RUNTIME_LEDGER_SHA256,
    )
    parser.add_argument(
        "--expected-direct-answer-preflight-sha256",
        default=operator_cli.EXPECTED_DIRECT_ANSWER_PREFLIGHT_SHA256,
    )
    parser.add_argument(
        "--expected-direct-answer-run-sha256",
        default=operator_cli.EXPECTED_DIRECT_ANSWER_RUN_SHA256,
    )
    parser.add_argument(
        "--expected-direct-semantic-binding-sha256",
        default=evidence_v2_cli.EXPECTED_DIRECT_SEMANTIC_BINDING_SHA256,
    )
    parser.add_argument("--expected-map-preflight-sha256")
    parser.add_argument("--expected-map-run-sha256")
    parser.add_argument(
        "--expected-parent-answer-run-sha256",
        default=payload_cli.EXPECTED_PARENT_ANSWER_RUN_SHA256,
    )
    parser.add_argument("--expected-answer-preflight-sha256", required=True)
    parser.add_argument("--expected-answer-run-sha256", required=True)
    parser.add_argument("--max-concurrency", type=int, default=4)
    parser.add_argument("--gateway-url", default=live.DEFAULT_GATEWAY_URL)


def _add_provider_inputs(parser: argparse.ArgumentParser) -> None:
    # Network execution accepts no dataset, gold, answer, retrieval, or parent
    # path.  Its only input is the explicitly pinned judge preflight.
    parser.add_argument(
        "--judge-output-root",
        type=Path,
        default=payload_cli.DEFAULT_OUTPUT,
    )
    parser.add_argument("--expected-judge-preflight-sha256", required=True)
    parser.add_argument("--enable-provider", action="store_true")
    parser.add_argument("--authorized-provider-calls", type=int, default=0)
    parser.add_argument("--api-key-env", default=live.DEFAULT_API_KEY_ENV)
    parser.add_argument("--max-concurrency", type=int, default=4)
    parser.add_argument("--gateway-url", default=live.DEFAULT_GATEWAY_URL)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight = subparsers.add_parser("preflight")
    _add_full_inputs(preflight)
    provider = subparsers.add_parser("provider-run")
    _add_provider_inputs(provider)
    materialize = subparsers.add_parser("materialize")
    _add_full_inputs(materialize)
    materialize.add_argument("--expected-judge-preflight-sha256", required=True)
    replay = subparsers.add_parser("replay")
    _add_full_inputs(replay)
    replay.add_argument("--expected-judge-preflight-sha256", required=True)
    replay.add_argument("--expected-judge-sha256", required=True)
    replay.add_argument("--expected-score-ledger-sha256", required=True)
    return parser


def _plan_namespace(args: argparse.Namespace) -> argparse.Namespace:
    values = vars(args).copy()
    values["output_root"] = _answer_root(args)
    if args.arm == "partition-payload":
        values["generation"] = args.partition_generation
        values["expected_generation_sha256"] = (
            args.expected_partition_generation_sha256
        )
    return argparse.Namespace(**values)


def _load_answer_plane(args: argparse.Namespace) -> object:
    """Replay the complete answer/runtime plane before any judge loads gold."""

    plan_args = _plan_namespace(args)
    answer_root = _answer_root(args)
    if args.arm in _ADAPTIVE_CLI_PROFILES:
        profile = _ADAPTIVE_CLI_PROFILES[args.arm]
        if (
            args.expected_answer_preflight_sha256
            != profile.solver_preflight_sha256
            or args.expected_answer_run_sha256 != profile.solver_run_sha256
        ):
            raise MatchedEvalContractError(
                "adaptive solver arm requires its exact locked preflight/run pins"
            )
        adaptive_args = argparse.Namespace(
            source_root=profile.source_root,
            expected_source_preflight_sha256=(
                profile.source_preflight_sha256
            ),
            expected_source_materialization_sha256=(
                profile.source_materialization_sha256
            ),
            output_root=answer_root,
            lanes=profile.lanes,
            direct_base_cap=profile.direct_base_cap,
            partition_base_cap=profile.partition_base_cap,
            guided_base_cap=profile.guided_base_cap,
            model=live.DEFAULT_TERRA_GATEWAY_MODEL,
            gateway_url=args.gateway_url,
            max_concurrency=args.max_concurrency,
            expected_preflight_sha256=profile.solver_preflight_sha256,
            expected_run_sha256=profile.solver_run_sha256,
        )
        loaded_run = adaptive_cli.load_verified_adaptive_solver_run(
            adaptive_args
        )
        loaded = loaded_run.loaded
        return adapt_verified_adaptive_evidence_solver(
            lanes=loaded.lanes,
            plan=loaded.plan,
            preflight=loaded.preflight,
            completion_plane=loaded_run.completion_plane,
            run=loaded_run.run,
            verified_plane=loaded_run.verified_plane,
            terminal_run_sha256=loaded_run.terminal.sha256,
            solver_preflight_artifact_sha256=(
                loaded_run.provider_preflight.sha256
            ),
            source_preflight_sha256=loaded.source_preflight.sha256,
            source_work_manifest_sha256=loaded.source_work_manifest.sha256,
            source_materialization_sha256=(
                loaded.source_materialization.sha256
            ),
            lane_filter_receipt_sha256=loaded.lane_filter_receipt_sha256,
        )
    if args.arm == "query-evidence-map-solver-v2":
        if args.expected_map_preflight_sha256 is None:
            raise MatchedEvalContractError(
                "query-evidence-map-solver-v2 requires --expected-map-preflight-sha256"
            )
        if args.expected_map_run_sha256 is None:
            raise MatchedEvalContractError(
                "query-evidence-map-solver-v2 requires --expected-map-run-sha256"
            )
        plan = evidence_v2_cli._load_solver_plan(plan_args)
        return replay_evidence_solver(
            plan,
            output_root=answer_root,
            expected_preflight_sha256=args.expected_answer_preflight_sha256,
            expected_run_sha256=args.expected_answer_run_sha256,
            max_concurrency=args.max_concurrency,
            gateway_url=args.gateway_url,
        )
    if args.arm == "query-operator-refinement":
        plan = operator_cli._load_plan(plan_args)
        load_verified_payload_semantic_arm_binding(
            Path(args.direct_answer_root),
            expected_profile=QUERY_PAYLOAD_PROFILE,
        )
        return replay_query_operator_refinement_answers(
            plan,
            output_root=answer_root,
            expected_preflight_sha256=args.expected_answer_preflight_sha256,
            expected_run_sha256=args.expected_answer_run_sha256,
            max_concurrency=args.max_concurrency,
            gateway_url=args.gateway_url,
        )
    if args.arm in {
        "query-payload",
        "partition-payload",
        "query-guided-payload",
    }:
        if args.arm == "query-payload":
            plan = payload_cli._load_plan(plan_args)
        elif args.arm == "partition-payload":
            plan = partition_payload_cli._load_plan(plan_args)
        else:
            plan = guided_payload_cli._load_plan(plan_args)
        plane = replay_query_payload_answers(
            plan,
            output_root=answer_root,
            expected_preflight_sha256=args.expected_answer_preflight_sha256,
            expected_run_sha256=args.expected_answer_run_sha256,
            max_concurrency=args.max_concurrency,
            gateway_url=args.gateway_url,
        )
        # This sidecar remains outside the immutable responder/judge protocol,
        # but its semantic profile must match the selected CLI arm before the
        # judging core is entered and therefore before gold can be opened.
        load_verified_payload_semantic_arm_binding(
            answer_root,
            expected_profile=profile_for_cli_arm(args.arm),
        )
        return plane
    plan = fact_cli._load_plan(plan_args)
    return replay_query_fact_answers(
        plan,
        output_root=answer_root,
        expected_preflight_sha256=args.expected_answer_preflight_sha256,
        expected_run_sha256=args.expected_answer_run_sha256,
        max_concurrency=args.max_concurrency,
        gateway_url=args.gateway_url,
    )


def _judge_request(args: argparse.Namespace) -> dict[str, Any]:
    parent_root, parent_judge_sha, parent_score_sha = _parent_judge_binding(args)
    return {
        "answer_plane": _load_answer_plane(args),
        "dataset_path": Path(args.dataset),
        "split_path": Path(args.split),
        "parent_judge_root": parent_root,
        "expected_parent_judge_sha256": parent_judge_sha,
        "expected_parent_score_ledger_sha256": parent_score_sha,
        "output_root": _judge_root(args),
    }


def _preflight(args: argparse.Namespace) -> dict[str, Any]:
    artifact = preflight_query_answer_changed_only_judge(**_judge_request(args))
    return {
        "arm_label": artifact.payload["arm_label"],
        "changed_prediction_count": artifact.payload[
            "changed_prediction_count"
        ],
        "judge_output_root": str(_judge_root(args)),
        "judge_preflight_sha256": artifact.sha256,
        "physical_provider_calls": 0,
        "required_authorized_provider_calls": artifact.payload[
            "required_authorized_provider_calls"
        ],
    }


def _provider(args: argparse.Namespace) -> dict[str, Any]:
    expected = require_sha256(
        args.expected_judge_preflight_sha256,
        "expected query-answer judge preflight",
    )
    population = load_query_answer_judge_provider_population(
        output_root=args.judge_output_root,
        expected_preflight_sha256=expected,
    )
    if (
        type(args.authorized_provider_calls) is not int
        or args.authorized_provider_calls != population.required_calls
        or args.enable_provider != bool(population.required_calls)
    ):
        raise MatchedEvalContractError(
            "provider-run requires exact authorization for "
            f"{population.required_calls} calls"
        )
    # Exact authorization precedes environment loading, client construction,
    # and every request/response journal write.
    client = None
    if population.required_calls:
        load_dotenv()
        api_key = os.environ.get(args.api_key_env, "").strip()
        if not api_key:
            raise MatchedEvalContractError(
                f"provider API key is empty: {args.api_key_env}"
            )
        client = judging._make_provider_client(api_key, args.gateway_url)
    result = run_sealed_query_answer_judge_provider(
        population,
        enable_provider=bool(args.enable_provider),
        authorized_provider_calls=args.authorized_provider_calls,
        client=client,
        max_concurrency=args.max_concurrency,
        gateway_url=args.gateway_url,
    )
    checkpoint = Path(args.judge_output_root) / JUDGE_CHECKPOINT_DIR_NAME
    return {
        "arm_label": population.adapter.arm_label,
        "checkpoint_hits": result.checkpoint_hits,
        "judge_preflight_sha256": result.preflight_artifact.sha256,
        "physical_provider_calls": result.physical_provider_calls,
        "request_journal_count": len(tuple(checkpoint.glob("*.request.json"))),
        "response_journal_count": len(tuple(checkpoint.glob("*.response.json"))),
    }


def _materialize(args: argparse.Namespace) -> dict[str, Any]:
    request = _judge_request(args)
    journals = load_query_answer_judge_journals(
        output_root=request["output_root"],
        expected_preflight_sha256=args.expected_judge_preflight_sha256,
        max_concurrency=args.max_concurrency,
        gateway_url=args.gateway_url,
    )
    result = materialize_query_answer_changed_only_judge(
        **request,
        expected_preflight_sha256=args.expected_judge_preflight_sha256,
        completion_batch=journals.batch,
    )
    return {
        "checkpoint_hits": result.checkpoint_hits,
        "correct": result.correct,
        "judge_sha256": result.judge_artifact.sha256,
        "physical_provider_calls": 0,
        "score_ledger_sha256": result.score_ledger_artifact.sha256,
    }


def _replay(args: argparse.Namespace) -> dict[str, Any]:
    result = replay_query_answer_changed_only_judge(
        **_judge_request(args),
        expected_preflight_sha256=args.expected_judge_preflight_sha256,
        expected_judge_sha256=args.expected_judge_sha256,
        expected_score_ledger_sha256=args.expected_score_ledger_sha256,
        max_concurrency=args.max_concurrency,
        gateway_url=args.gateway_url,
    )
    return {
        "checkpoint_hits": result.checkpoint_hits,
        "correct": result.correct,
        "judge_replay_sha256": result.judge_artifact.sha256,
        "physical_provider_calls": 0,
        "score_ledger_replay_sha256": result.score_ledger_artifact.sha256,
    }


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "preflight":
        result = _preflight(args)
    elif args.command == "provider-run":
        result = _provider(args)
    elif args.command == "materialize":
        result = _materialize(args)
    elif args.command == "replay":
        result = _replay(args)
    else:  # pragma: no cover
        raise AssertionError("unreachable command")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]
