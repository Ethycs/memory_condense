from __future__ import annotations

from types import SimpleNamespace

import pytest

from tools import run_locked_semantic_residual_construction_v4 as r7_cli
from tools import run_reduced_semantic_global_completion_assay as v7_cli
from tools.matched_eval import semantic_residual_search as residual


def _args(policy: residual.SemanticResidualPolicy) -> SimpleNamespace:
    return SimpleNamespace(
        max_cell_tokens=policy.max_cell_tokens,
        cosine_upper_bound_floor=policy.cosine_upper_bound_floor,
        specificity_upper_bound_ratio=policy.specificity_upper_bound_ratio,
        dual_gate_enabled=policy.dual_gate_enabled,
    )


@pytest.mark.parametrize(
    "mode",
    [
        residual.LEGACY_RESIDUAL_CLASSIFIER_MODE,
        residual.EVIDENCE_CONSERVING_RESIDUAL_CLASSIFIER_MODE,
    ],
)
def test_v7_derives_classifier_mode_from_sealed_r7_policy(mode: str) -> None:
    sealed = residual.SemanticResidualPolicy(classifier_mode=mode)
    rebuilt = v7_cli._sealed_r7_semantic_policy(  # noqa: SLF001
        {"residual_search_policy": sealed.projection()},
        _args(sealed),
        payload_token_cap=sealed.payload_token_cap,
    )

    assert rebuilt.classifier_mode == mode
    assert rebuilt.projection() == sealed.projection()


def test_r7_cli_defaults_successor_and_accepts_explicit_legacy_mode() -> None:
    required = ["construct", "--expected-vector-sha256", "a" * 64]
    default_args = r7_cli.build_parser().parse_args(required)
    legacy_args = r7_cli.build_parser().parse_args(
        [
            *required,
            "--residual-classifier-mode",
            residual.LEGACY_RESIDUAL_CLASSIFIER_MODE,
        ]
    )

    assert default_args.residual_classifier_mode == (
        residual.EVIDENCE_CONSERVING_RESIDUAL_CLASSIFIER_MODE
    )
    assert r7_cli.construction_output_root_for_args(default_args) == (
        r7_cli.DEFAULT_SUCCESSOR_OUTPUT_ROOT
    )
    assert legacy_args.residual_classifier_mode == (
        residual.LEGACY_RESIDUAL_CLASSIFIER_MODE
    )
    assert r7_cli.construction_output_root_for_args(legacy_args) == (
        r7_cli.DEFAULT_OUTPUT_ROOT
    )


def test_v7_r7_replay_component_failure_names_the_first_boundary() -> None:
    with pytest.raises(
        v7_cli.ReducedSemanticGlobalCompletionAssayError,
        match=r"R7 question 14 semantic search replay changed",
    ):
        v7_cli._require_r7_replay_component(  # noqa: SLF001
            ordinal=14,
            component="semantic search",
            actual={"receipt": "new"},
            expected={"receipt": "sealed"},
        )
