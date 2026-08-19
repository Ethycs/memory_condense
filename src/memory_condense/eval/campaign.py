"""Compatibility and CLI facade for locked campaign report assembly."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from memory_condense.eval.campaign_merge import (
    _category_metrics,
    _distribution,
    _mean,
    _nearest_rank,
    _sum_usage,
    merge_benchmark_reports,
)
from memory_condense.eval.campaign_models import (
    CampaignMergeError,
    ExpectedStressShard,
    LockedValidationPlan,
)
from memory_condense.eval.campaign_plan import (
    _assert_locked_plan_unchanged,
    _load_json_object,
    _revalidate_locked_claim_profile,
    _safe_repository_file,
    build_locked_validation_plan,
)
from memory_condense.eval.campaign_validation import (
    _BINARY_JUDGE_VERDICT,
    _HASH_FIELDS,
    _QUESTION_ERROR_FIELDS,
    _assert_policy_retrieval_identity,
    _canonical_json,
    _ensure_same_identity,
    _file_sha256,
    _has_error,
    _identity,
    _json_constant,
    _load_report,
    _locked_judge_verdict,
    _require_bool,
    _require_float,
    _require_int,
    _require_list,
    _require_mapping,
    _require_nonempty_string,
    _require_sha256,
    _validate_question,
    _validate_usage,
)
from memory_condense.eval.context_stress import transcript_tokens
from memory_condense.eval.reproducibility import file_sha256


def save_campaign_report(report: dict[str, Any], output: str | Path) -> Path:
    """Write a deterministic campaign JSON document and checksum sidecar."""

    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    (path.parent / f"{path.name}.sha256").write_text(
        f"{file_sha256(path)}  {path.name}\n", encoding="ascii"
    )
    return path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Merge locked benchmark validation shards"
    )
    parser.add_argument("--reports", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-questions", type=int, default=100)
    parser.add_argument("--accuracy-target", type=float, default=0.95)
    parser.add_argument("--benchmark-file", type=Path)
    parser.add_argument("--benchmark-format", default="auto")
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--policy-manifest", type=Path)
    parser.add_argument("--repository-root", type=Path)
    parser.add_argument(
        "--allow-unverified-summary",
        action="store_true",
        help="Merge metrics without certifying the population",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if any(path.resolve() == output for path in args.reports):
        parser.error("--output must not overwrite an input report")
    try:
        verification_paths = (
            args.benchmark_file,
            args.split_manifest,
            args.policy_manifest,
        )
        if any(verification_paths) and not all(verification_paths):
            raise CampaignMergeError(
                "--benchmark-file, --split-manifest, and --policy-manifest "
                "must be supplied together"
            )
        if not all(verification_paths) and not args.allow_unverified_summary:
            raise CampaignMergeError(
                "locked certification requires --benchmark-file, "
                "--split-manifest, and --policy-manifest; use "
                "--allow-unverified-summary only for non-claim diagnostics"
            )
        locked_plan = (
            build_locked_validation_plan(
                benchmark_file=args.benchmark_file,
                benchmark_format=args.benchmark_format,
                split_manifest=args.split_manifest,
                policy_manifest=args.policy_manifest,
                repository_root=args.repository_root,
            )
            if all(verification_paths)
            else None
        )
        report = merge_benchmark_reports(
            args.reports,
            min_questions=args.min_questions,
            accuracy_target=args.accuracy_target,
            locked_plan=locked_plan,
        )
        path = save_campaign_report(report, output)
    except CampaignMergeError as exc:
        parser.error(str(exc))
    print(
        f"Merged {report['input_count']} shards / {report['num_questions']} "
        f"questions: judge={report['judge_accuracy']:.1%}, "
        f"target={report['target_status']}; saved {path}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
