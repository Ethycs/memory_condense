"""Publish or replay provider-free append-only Hebbian H2 receipts.

Every phase reconstructs H2 from the exact sealed S3 retrieval, historical
access artifact, and derived association store.  ``preflight`` performs no
writes.  ``publish`` writes only the canonical text-free H2 identity payload
and its digest sidecar.  ``replay`` requires those bytes to match a fresh
reconstruction exactly.  No phase can load gold, call a provider, or compute
CAV features or links.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from memory_condense.eval._artifact_json import (
    canonical_json_bytes as _canonical_json_bytes,
)
from memory_condense.eval._fast_hebbian_h2_io import (
    read_canonical_json,
    verify_digest_anchor,
)
from memory_condense.eval.fast_hebbian_h2 import (
    FAST_HEBBIAN_H2_MAX_PROMPT_TOKENS,
    FastHebbianH2Population,
    build_fast_hebbian_h2_population,
    load_fast_hebbian_h2_history,
    load_fast_hebbian_h2_retrieval_source,
)
from memory_condense.eval.recall_guarded_cumulative_fast_artifact import (
    ORIGINAL_1M_RETRIEVAL_SHA256,
    load_fast_retrieval_artifact,
)


PREFLIGHT_FORMAT = (
    "memory-condense-fast-hebbian-h2-static-local-closure-runner-preflight-v3"
)
H2_RECEIPTS_NAME = "h2-receipts.json"
DEFAULT_RETRIEVAL = Path(
    "eval_results/longmemeval-1m-recall-guarded-cumulative-"
    "development-20260821/retrieval.json"
)
DEFAULT_HISTORY = Path(
    "eval_results/longmemeval-1m-fast-hebbian-history-"
    "development-20260822/history.json"
)
DEFAULT_HISTORY_SHA256 = (
    "b610d482ddfd0c662d80755b0a1f93eb8921eb5a61254c1ee00c97073a692ba2"
)
DEFAULT_DERIVED_STORE = Path(
    "eval_results/longmemeval-1m-fast-hebbian-history-"
    "development-20260822/derived-store"
)
DEFAULT_OUTPUT_ROOT = Path(
    "eval_results/longmemeval-1m-fast-hebbian-h2-static-local-closure-"
    "development-20260823"
)
DEFAULT_EXPECTED_QUESTION_COUNT = 10

_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
_FORBIDDEN_RECEIPT_KEYS = frozenset(
    {
        "answer",
        "dated_question",
        "final_evidence",
        "gold_answer",
        "question",
        "source_path",
        "text",
    }
)


def _digest(value: object, label: str) -> str:
    if type(value) is not str or _DIGEST_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _validate_args(args: argparse.Namespace) -> None:
    _digest(args.expected_retrieval_sha256, "--expected-retrieval-sha256")
    _digest(args.expected_history_sha256, "--expected-history-sha256")
    if args.expected_receipts_sha256 is not None:
        _digest(args.expected_receipts_sha256, "--expected-receipts-sha256")
    if (
        type(args.expected_question_count) is not int
        or args.expected_question_count < 1
    ):
        raise ValueError("--expected-question-count must be positive")


def _load_population(args: argparse.Namespace) -> FastHebbianH2Population:
    _validate_args(args)
    retrieval_path = Path(args.retrieval)
    artifact = load_fast_retrieval_artifact(
        retrieval_path,
        expected_sha256=args.expected_retrieval_sha256,
    )
    retrieval_source = load_fast_hebbian_h2_retrieval_source(
        retrieval_path,
        artifact=artifact,
    )
    history_source = load_fast_hebbian_h2_history(
        Path(args.history),
        expected_sha256=args.expected_history_sha256,
    )
    population = build_fast_hebbian_h2_population(
        artifact,
        retrieval_source,
        history_source,
        Path(args.derived_store),
    )
    if population.question_count != args.expected_question_count:
        raise ValueError(
            "H2 population changed expected question count "
            f"({population.question_count} != {args.expected_question_count})"
        )
    return population


def _reject_raw_fields(value: object) -> None:
    if type(value) is dict:
        for key, child in value.items():
            if key in _FORBIDDEN_RECEIPT_KEYS:
                raise ValueError(f"H2 receipt exposed forbidden raw field {key!r}")
            _reject_raw_fields(child)
    elif type(value) is list:
        for child in value:
            _reject_raw_fields(child)


def _receipt_payload(population: FastHebbianH2Population) -> dict[str, Any]:
    if type(population) is not FastHebbianH2Population:
        raise TypeError("population must be an exact FastHebbianH2Population")
    payload = population.identity_payload()
    if (
        payload.get("gold_fields_consumed") is not False
        or payload.get("provider_calls") != 0
        or payload.get("cav_links_computed") is not False
        or payload.get("retained_request_token_state_bytes") != 0
    ):
        raise ValueError("H2 receipt crossed a forbidden runtime boundary")
    _reject_raw_fields(payload)
    return payload


def _metrics(
    population: FastHebbianH2Population,
    *,
    writes: int,
) -> dict[str, Any]:
    candidates = tuple(
        candidate
        for receipt in population.question_receipts
        for candidate in receipt.ranked_candidates
    )
    return {
        "format": PREFLIGHT_FORMAT,
        "population_sha256": population.population_sha256,
        "policy_sha256": population.policy.policy_sha256,
        "question_count": population.question_count,
        "appended_question_count": population.appended_question_count,
        "appended_evidence_count": population.appended_evidence_count,
        "no_robust_question_count": sum(
            receipt.outcome == "no_robust_candidate"
            for receipt in population.question_receipts
        ),
        "budget_blocked_question_count": sum(
            receipt.outcome == "no_budget_admissible_candidate"
            for receipt in population.question_receipts
        ),
        "budget_rejected_candidate_count": sum(
            candidate.admission_status == "budget_rejected"
            for candidate in candidates
        ),
        "addition_cap_rejected_candidate_count": sum(
            candidate.admission_status == "addition_cap_rejected"
            for candidate in candidates
        ),
        "max_prompt_token_proxy": max(
            receipt.final_prompt_token_proxy
            for receipt in population.question_receipts
        ),
        "hard_prompt_token_cap": FAST_HEBBIAN_H2_MAX_PROMPT_TOKENS,
        "gold_fields_consumed": False,
        "provider_calls": 0,
        "cav_links_computed": False,
        "writes": writes,
    }


def _receipts_path(args: argparse.Namespace) -> Path:
    return Path(args.output_root) / H2_RECEIPTS_NAME


def _publish_bytes(path: Path, payload: bytes) -> bool:
    root = path.parent
    if root.is_symlink():
        raise FileExistsError(f"refusing symbolic-link output root: {root}")
    root.mkdir(parents=True, exist_ok=True)
    if not root.is_dir() or root.is_symlink():
        raise FileExistsError(f"output root must be a regular directory: {root}")
    if path.is_symlink():
        raise FileExistsError(f"refusing to replace a symbolic link: {path}")
    if path.exists():
        if not path.is_file() or path.read_bytes() != payload:
            raise FileExistsError(f"refusing to replace another artifact: {path}")
        return False
    descriptor, raw_temporary = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=root,
    )
    temporary = Path(raw_temporary)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return True


def _atomic_write_receipts(
    path: Path,
    payload: Mapping[str, Any],
    *,
    expected_sha256: str | None,
) -> tuple[str, int]:
    raw = _canonical_json_bytes(payload)
    digest = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError("reconstructed H2 receipt changed expected digest")
    writes = int(_publish_bytes(path, raw))
    writes += int(
        _publish_bytes(
            path.with_name(path.name + ".sha256"),
            f"{digest}  {path.name}\n".encode("ascii"),
        )
    )
    return digest, writes


def _read_receipts(
    path: Path,
    *,
    expected_sha256: str | None,
) -> tuple[dict[str, Any], str, bytes]:
    payload, digest, resolved = read_canonical_json(path)
    verify_digest_anchor(
        resolved,
        digest,
        expected_sha256=expected_sha256,
        verify_sidecar=True,
    )
    return payload, digest, resolved.read_bytes()


def run_preflight(args: argparse.Namespace) -> dict[str, Any]:
    """Reconstruct and count H2 without creating an output path."""

    return _metrics(_load_population(args), writes=0)


def run_publish(args: argparse.Namespace) -> tuple[dict[str, Any], str]:
    """Publish the immutable text-free H2 receipt and verify it immediately."""

    population = _load_population(args)
    payload = _receipt_payload(population)
    path = _receipts_path(args)
    digest, writes = _atomic_write_receipts(
        path,
        payload,
        expected_sha256=args.expected_receipts_sha256,
    )
    observed, observed_digest, observed_raw = _read_receipts(
        path,
        expected_sha256=args.expected_receipts_sha256,
    )
    expected_raw = _canonical_json_bytes(payload)
    if observed != payload or observed_raw != expected_raw or observed_digest != digest:
        raise ValueError("published H2 receipt is not byte-identical")
    return _metrics(population, writes=writes), digest


def run_replay(args: argparse.Namespace) -> tuple[dict[str, Any], str]:
    """Rebuild H2 and require the published identity payload byte-for-byte."""

    population = _load_population(args)
    expected_raw = _canonical_json_bytes(_receipt_payload(population))
    _observed, digest, observed_raw = _read_receipts(
        _receipts_path(args),
        expected_sha256=args.expected_receipts_sha256,
    )
    if observed_raw != expected_raw:
        raise ValueError(
            "published H2 receipt differs from byte-identical reconstruction"
        )
    return _metrics(population, writes=0), digest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=("preflight", "publish", "replay"),
        default="preflight",
    )
    parser.add_argument("--retrieval", type=Path, default=DEFAULT_RETRIEVAL)
    parser.add_argument(
        "--expected-retrieval-sha256",
        default=ORIGINAL_1M_RETRIEVAL_SHA256,
    )
    parser.add_argument("--history", type=Path, default=DEFAULT_HISTORY)
    parser.add_argument(
        "--expected-history-sha256",
        default=DEFAULT_HISTORY_SHA256,
    )
    parser.add_argument(
        "--derived-store",
        type=Path,
        default=DEFAULT_DERIVED_STORE,
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--expected-receipts-sha256")
    parser.add_argument(
        "--expected-question-count",
        type=int,
        default=DEFAULT_EXPECTED_QUESTION_COUNT,
    )
    return parser


def _summary(metrics: Mapping[str, Any]) -> str:
    return (
        f"questions={metrics['question_count']}; "
        f"appended_questions={metrics['appended_question_count']}; "
        f"appended_evidence={metrics['appended_evidence_count']}; "
        "budget_rejected_candidates="
        f"{metrics['budget_rejected_candidate_count']}; "
        f"budget_blocked_questions={metrics['budget_blocked_question_count']}; "
        f"max_prompt={metrics['max_prompt_token_proxy']}/"
        f"{metrics['hard_prompt_token_cap']}; provider_calls=0; "
        f"writes={metrics['writes']}"
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.phase == "preflight":
        metrics = run_preflight(args)
        print(f"Fast 1M Hebbian H2 preflight passed: {_summary(metrics)}", flush=True)
        return 0
    if args.phase == "publish":
        metrics, digest = run_publish(args)
        print(
            f"Fast 1M Hebbian H2 published ({digest}): {_summary(metrics)}",
            flush=True,
        )
        return 0
    metrics, digest = run_replay(args)
    print(
        "Fast 1M Hebbian H2 replay verified byte-identical "
        f"({digest}): {_summary(metrics)}",
        flush=True,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "DEFAULT_DERIVED_STORE",
    "DEFAULT_EXPECTED_QUESTION_COUNT",
    "DEFAULT_HISTORY",
    "DEFAULT_HISTORY_SHA256",
    "DEFAULT_OUTPUT_ROOT",
    "DEFAULT_RETRIEVAL",
    "H2_RECEIPTS_NAME",
    "PREFLIGHT_FORMAT",
    "build_parser",
    "main",
    "run_preflight",
    "run_publish",
    "run_replay",
]
