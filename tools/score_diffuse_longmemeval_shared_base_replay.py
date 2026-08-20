"""Score the pinned shared-base replay without loading models or providers.

This CLI verifies and reconstructs all three frozen packets before its exact
one-record firebreak loader is invoked.  Gold and source IDs are never written
to the report or stdout; callers externally pin the label artifact identities
printed by the separate analysis-label exporter.
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

import memory_condense.eval.diffuse_longmemeval_replay as replay_module
from memory_condense.domain.discourse import identity_sha256, quote_sha256
from memory_condense.eval._diffuse_replay_provider_history import (
    certify_historical_provider_identity,
)
from memory_condense.eval.diffuse_longmemeval_base import (
    owned_build_runtime_identity,
    verify_diffuse_longmemeval_base,
)
from memory_condense.eval.diffuse_longmemeval_replay import ReplayExecutionIdentity
from memory_condense.eval.diffuse_longmemeval_replay_scoring import (
    publish_diffuse_longmemeval_posthoc_score,
    score_diffuse_longmemeval_replay_package,
)
from memory_condense.eval.diffuse_longmemeval_runtime import (
    gold_blind_from_treatment_sample,
)
from tools import run_diffuse_longmemeval_shared_base_replay as replay_launcher
from tools.run_diffuse_longmemeval_shared_base_replay import (
    CAMPAIGN_RECEIPT_NAME,
    PINNED_SAMPLE_ORDINAL,
    PinnedReplayCampaignReceipt,
    _canonical_json_bytes,
    _canonical_cuda_device,
    _load_nested_replay_manifest,
    _load_pinned_treatment,
    _new_owned_binding,
    _population_receipt,
    _require_campaign_binding,
    _treatment_identity,
    verify_pinned_replay_campaign_receipt,
)
from tools.v4_population_firebreak import load_analysis_scoring_label


def _historical_campaign_launcher_identity(
    campaign_root: Path,
) -> ReplayExecutionIdentity:
    """Verify the recorded launcher against its ancestor commit and live bytes."""

    receipt_path = campaign_root / CAMPAIGN_RECEIPT_NAME
    if receipt_path.is_symlink() or not receipt_path.is_file():
        raise RuntimeError("campaign receipt must be a regular file")
    raw = receipt_path.read_bytes()
    try:
        receipt = PinnedReplayCampaignReceipt.model_validate_json(raw)
    except Exception as exc:
        raise RuntimeError("invalid campaign receipt") from exc
    if raw != _canonical_json_bytes(receipt.model_dump(mode="json")):
        raise RuntimeError("campaign receipt is not canonical JSON")
    return _certify_historical_launcher(
        receipt.launcher,
        Path(replay_launcher.__file__),
    )


def _certify_historical_launcher(
    expected: ReplayExecutionIdentity,
    launcher_path: Path,
) -> ReplayExecutionIdentity:
    """Reprove a frozen launcher while allowing unrelated descendant commits."""

    launcher = launcher_path.resolve()
    if launcher.is_symlink() or not launcher.is_file():
        raise ValueError("historical launcher must be a regular file")
    root_result = subprocess.run(
        ("git", "rev-parse", "--show-toplevel"),
        cwd=launcher.parent,
        check=False,
        capture_output=True,
        text=True,
    )
    if root_result.returncode != 0:
        raise RuntimeError("historical launcher git certification failed")
    root = Path(root_result.stdout.strip()).resolve()
    try:
        relative = launcher.relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError("historical launcher is outside its git worktree") from exc

    def git(*arguments: str, binary: bool = False):
        result = subprocess.run(
            ("git", *arguments),
            cwd=root,
            check=False,
            capture_output=True,
            text=not binary,
        )
        if result.returncode != 0:
            raise RuntimeError("historical launcher git certification failed")
        return result.stdout

    commit = str(expected.source_commit).casefold()
    resolved = str(git("rev-parse", "--verify", f"{commit}^{{commit}}"))
    if resolved.strip().casefold() != commit:
        raise RuntimeError("campaign launcher commit did not resolve exactly")
    git("merge-base", "--is-ancestor", commit, "HEAD")
    historical = git("show", f"{commit}:{relative}", binary=True)
    active = launcher.read_bytes()
    if historical != active:
        raise RuntimeError("current launcher bytes differ from the frozen launcher")
    launcher_sha256 = hashlib.sha256(historical).hexdigest()
    if launcher_sha256 != expected.launcher_sha256:
        raise RuntimeError("historical launcher hash differs from the campaign")
    return ReplayExecutionIdentity(
        launcher_sha256=launcher_sha256,
        source_commit=commit,
        tracked_worktree_clean=expected.tracked_worktree_clean,
    )


def _score_campaign(
    *,
    campaign_root: Path,
    treatment_input: Path,
    scoring_label: Path,
    label_file_sha256: str,
    label_record_sha256: str,
    raw_record_sha256: str,
    raw_record_span_sha256: str,
    output: Path,
    device: str,
):
    root = campaign_root.resolve()
    target = output.resolve()
    if target == root or target.is_relative_to(root):
        raise ValueError("score output must stay outside the frozen campaign root")
    if target in {treatment_input.resolve(), scoring_label.resolve()}:
        raise ValueError("score output must not replace an input artifact")
    if target.exists():
        raise FileExistsError(f"refusing to overwrite score output: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.parent.is_symlink() or not target.parent.is_dir():
        raise ValueError("score output parent must be a regular directory")

    normalized_device = _canonical_cuda_device(device)
    treatment = _load_pinned_treatment(treatment_input.resolve())
    treatment_identity = _treatment_identity(treatment)
    sample = treatment.samples[PINNED_SAMPLE_ORDINAL]
    population = _population_receipt(treatment_identity, sample)

    # Construction derives code/config identities only.  Neither resident
    # model is loaded, and the Qwen filesystem locator is not identity-bearing.
    binding = _new_owned_binding(root / ".identity-only-qwen", normalized_device)
    _require_campaign_binding(binding)
    runtime_sha256 = binding.binding_sha256
    launcher = _historical_campaign_launcher_identity(root)
    campaign = verify_pinned_replay_campaign_receipt(
        root,
        expected_population=population,
        expected_launcher=launcher,
        expected_runtime_binding_sha256=runtime_sha256,
    )
    nested = _load_nested_replay_manifest(root / "replay" / "replay-manifest.json")
    provider_identity_proof = certify_historical_provider_identity(
        execution_identity=campaign.launcher,
        recorded_identity=nested.verified_base_provider_identity,
        current_source_path=Path(replay_module.__file__),
    )
    blind = gold_blind_from_treatment_sample(sample)
    base = verify_diffuse_longmemeval_base(
        root / "cache",
        treatment_identity=treatment_identity,
        sample=blind,
        config=binding.config,
        embedding_identity=binding.embedding_identity,
        build_runtime_identity=owned_build_runtime_identity(binding.new_condenser),
        # This is a post-hoc read of a frozen package.  Its content-addressed
        # build recipe predates this scorer module; replay reconstruction below
        # still re-executes current deterministic closure and packing code.
        implementation_digest=nested.base_manifest.implementation_sha256,
        environment_digest=nested.base_manifest.environment_lock_sha256,
    )
    question = blind.questions[0]

    def load_bound_label():
        return load_analysis_scoring_label(
            scoring_label.resolve(),
            expected_file_sha256=label_file_sha256,
            expected_label_record_sha256=label_record_sha256,
            expected_dataset_sha256=treatment_identity.dataset_sha256,
            expected_split_manifest_sha256=treatment_identity.split_manifest_sha256,
            expected_analysis_ordered_question_ids_sha256=(
                treatment_identity.ordered_question_ids_sha256
            ),
            expected_analysis_sample_count=treatment_identity.sample_count,
            expected_sample_ordinal=treatment_identity.sample_ordinal,
            expected_sample_id_sha256=identity_sha256(
                {"sample_id": blind.sample_id}
            ),
            expected_question_id_sha256=identity_sha256(
                {"question_id": question.question_id}
            ),
            expected_question_text_sha256=quote_sha256(question.retrieval_query),
            expected_question_probe_sha256=question.probe_sha256,
            expected_raw_record_sha256=raw_record_sha256,
            expected_raw_record_span_sha256=raw_record_span_sha256,
        )

    report = score_diffuse_longmemeval_replay_package(
        root / "replay",
        base=base,
        expected_runtime_binding_sha256=runtime_sha256,
        label_loader=load_bound_label,
        historical_provider_identity_proof=provider_identity_proof,
    )
    if (
        report.replay_receipt_sha256 != campaign.artifacts.replay_receipt_sha256
        or report.replay_manifest_file_sha256
        != campaign.artifacts.replay_manifest_file_sha256
    ):
        raise RuntimeError("score report belongs to another frozen campaign")
    publish_diffuse_longmemeval_posthoc_score(target, report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="provider-free post-hoc scoring of the pinned shared-base replay"
    )
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--treatment-input", type=Path, required=True)
    parser.add_argument("--scoring-label", type=Path, required=True)
    parser.add_argument("--label-file-sha256", required=True)
    parser.add_argument("--label-record-sha256", required=True)
    parser.add_argument("--raw-record-sha256", required=True)
    parser.add_argument("--raw-record-span-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = _score_campaign(
            campaign_root=args.campaign_root,
            treatment_input=args.treatment_input,
            scoring_label=args.scoring_label,
            label_file_sha256=args.label_file_sha256,
            label_record_sha256=args.label_record_sha256,
            raw_record_sha256=args.raw_record_sha256,
            raw_record_span_sha256=args.raw_record_span_sha256,
            output=args.output,
            device=args.device,
        )
    except Exception as exc:
        print(f"shared-base post-hoc scoring failed: {exc}", file=sys.stderr)
        return 2
    print(
        "SHARED_BASE_POSTHOC_SCORE_PASS "
        f"receipt={report.receipt_sha256} "
        f"replay={report.replay_receipt_sha256}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
