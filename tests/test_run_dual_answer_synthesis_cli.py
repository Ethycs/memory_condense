from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tools.matched_eval.artifacts import publish_sealed_json, read_sealed_json
from tools.matched_eval.contracts import identity_sha256
import tools.run_dual_answer_synthesis as launcher


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64


@pytest.mark.parametrize(
    ("command", "extra"),
    [
        ("preflight", []),
        ("answer", []),
        ("answer-replay", ["--expected-answer-run-sha256", SHA_A]),
        (
            "judge-preflight",
            [
                "--expected-answer-run-sha256",
                SHA_A,
                "--dataset",
                "locked.json",
            ],
        ),
        (
            "judge",
            [
                "--expected-answer-run-sha256",
                SHA_A,
                "--dataset",
                "locked.json",
            ],
        ),
        (
            "judge-replay",
            [
                "--expected-answer-run-sha256",
                SHA_A,
                "--expected-judge-sha256",
                SHA_B,
                "--dataset",
                "locked.json",
            ],
        ),
    ],
)
def test_parser_exposes_exact_six_commands(
    command: str, extra: list[str]
) -> None:
    args = launcher._parser().parse_args([command, *extra])

    assert args.command == command
    assert args.flip10 is False
    assert args.output_root is None
    assert args.expected_v2_answer_run_sha256 == (
        launcher.DEFAULT_V2_ANSWER_RUN_SHA256
    )
    assert args.expected_v4_preflight_sha256 == (
        launcher.DEFAULT_V4_PREFLIGHT_SHA256
    )
    assert launcher._scope(args) == (launcher.DEFAULT_FULL_ROOT, None)


def test_flip10_selects_only_the_fixed_root_and_ordinals(tmp_path: Path) -> None:
    args = launcher._parser().parse_args(["preflight", "--flip10"])
    assert launcher._scope(args) == (
        launcher.DEFAULT_FLIP10_ROOT,
        launcher.FLIP10_ORDINALS,
    )

    custom = launcher._parser().parse_args(
        ["preflight", "--flip10", "--output-root", str(tmp_path)]
    )
    with pytest.raises(ValueError, match="fixed output root"):
        launcher._scope(custom)


def test_provider_authorization_is_explicit_and_non_ambient() -> None:
    parser = launcher._parser()
    missing = parser.parse_args(["answer", "--enable-provider"])
    ambient = parser.parse_args(
        ["answer", "--authorized-provider-calls", "10"]
    )
    exact = parser.parse_args(
        [
            "answer",
            "--enable-provider",
            "--authorized-provider-calls",
            "10",
        ]
    )

    with pytest.raises(ValueError, match="positive exact"):
        launcher._validate_provider_authorization(missing)
    with pytest.raises(ValueError, match="must be zero"):
        launcher._validate_provider_authorization(ambient)
    launcher._validate_provider_authorization(exact)


def _source_bindings() -> dict[str, Any]:
    return {
        "legacy": {
            "run_sha256": SHA_A,
            "replay_sha256": SHA_A,
            "prompt_population_sha256": SHA_B,
        },
        "v2": {
            "answer_run_sha256": SHA_B,
            "answer_replay_sha256": SHA_B,
            "prompt_population_sha256": SHA_C,
        },
        "v4": {
            "preflight_sha256": SHA_C,
            "prompt_population_sha256": SHA_D,
        },
    }


def _row(ordinal: int) -> dict[str, Any]:
    digit = format(ordinal % 16, "x")
    return {
        "source_ordinal": ordinal,
        "question_sha256": digit * 64,
        "legacy_prompt_messages_sha256": SHA_A,
        "v2_prompt_messages_sha256": SHA_B,
        "v4_prompt_messages_sha256": SHA_C,
        "synthesis_prompt_messages_sha256": SHA_D,
    }


def _preflight(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "format": "test-dual-answer-synthesis-preflight",
        "ordered_rows": rows,
        "source_bindings": _source_bindings(),
    }


def test_flip_preflight_seals_exact_gate_and_all_source_bindings(
    tmp_path: Path,
) -> None:
    full_root = tmp_path / "full100"
    flip_root = tmp_path / "flip10"
    full_rows = [_row(ordinal) for ordinal in range(100)]
    full, _ = publish_sealed_json(
        full_root / launcher.SYNTHESIS_PREFLIGHT_NAME,
        _preflight(full_rows),
    )
    diagnostic, _ = publish_sealed_json(
        flip_root / launcher.SYNTHESIS_PREFLIGHT_NAME,
        _preflight([full_rows[ordinal] for ordinal in launcher.FLIP10_ORDINALS]),
    )

    gate = launcher._seal_flip10_gate(
        output_root=flip_root,
        diagnostic_preflight=diagnostic,
        full_root=full_root,
    )
    replayed = launcher._seal_flip10_gate(
        output_root=flip_root,
        diagnostic_preflight=diagnostic,
        full_root=full_root,
    )
    payload = read_sealed_json(gate.path).payload

    assert replayed.sha256 == gate.sha256
    assert payload["full_preflight_sha256"] == full.sha256
    assert payload["diagnostic_preflight_sha256"] == diagnostic.sha256
    assert payload["selected_ordinals"] == list(launcher.FLIP10_ORDINALS)
    assert payload["rescue_ordinals"] == [29, 34, 50]
    assert payload["regression_ordinals"] == [5, 16, 52, 65, 79, 83, 97]
    assert payload["minimum_proceed_gate"] == {
        "minimum_correct": 8,
        "minimum_regressions_recovered": 5,
        "minimum_rescues_retained": 3,
        "question_count": 10,
        "retain_all_rescues": True,
    }
    assert payload["strong_recovery_goal"] == {
        "minimum_correct": 10,
        "recover_all_regressions": True,
        "retain_all_rescues": True,
    }
    assert payload["full_100_promotion_gate"] == {
        "baseline_arm": "S0_CONTROL_V2",
        "baseline_correct": 53,
        "minimum_correct": 60,
        "minimum_paired_net_improvement_vs_v2": 7,
        "question_count": 100,
    }
    assert payload["source_bindings"] == _source_bindings()
    assert payload["source_bindings_sha256"] == identity_sha256(
        _source_bindings()
    )
    prompt_bindings = payload["source_prompt_hash_bindings"]
    assert prompt_bindings["diagnostic_rows_are_exact_full_view"] is True
    assert prompt_bindings["full_ordered_rows_sha256"] == identity_sha256(
        launcher._row_hash_projection(full_rows)
    )
    assert prompt_bindings["diagnostic_ordered_rows_sha256"] == identity_sha256(
        launcher._row_hash_projection(
            [full_rows[ordinal] for ordinal in launcher.FLIP10_ORDINALS]
        )
    )


def test_flip_gate_rejects_non_view_or_changed_sources(tmp_path: Path) -> None:
    full_root = tmp_path / "full100"
    full_rows = [_row(ordinal) for ordinal in range(100)]
    publish_sealed_json(
        full_root / launcher.SYNTHESIS_PREFLIGHT_NAME,
        _preflight(full_rows),
    )

    changed_row_root = tmp_path / "changed-row"
    changed_rows = [
        dict(full_rows[ordinal]) for ordinal in launcher.FLIP10_ORDINALS
    ]
    changed_rows[0]["synthesis_prompt_messages_sha256"] = "e" * 64
    changed_row, _ = publish_sealed_json(
        changed_row_root / launcher.SYNTHESIS_PREFLIGHT_NAME,
        _preflight(changed_rows),
    )
    with pytest.raises(RuntimeError, match="not an exact view"):
        launcher._seal_flip10_gate(
            output_root=changed_row_root,
            diagnostic_preflight=changed_row,
            full_root=full_root,
        )

    changed_source_root = tmp_path / "changed-source"
    changed_payload = _preflight(
        [full_rows[ordinal] for ordinal in launcher.FLIP10_ORDINALS]
    )
    changed_payload["source_bindings"]["legacy"]["run_sha256"] = "e" * 64
    changed_source, _ = publish_sealed_json(
        changed_source_root / launcher.SYNTHESIS_PREFLIGHT_NAME,
        changed_payload,
    )
    with pytest.raises(RuntimeError, match="do not share exact"):
        launcher._seal_flip10_gate(
            output_root=changed_source_root,
            diagnostic_preflight=changed_source,
            full_root=full_root,
        )


def test_judge_preflight_verifies_synthesis_plane_before_gold_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[object] = []
    plane = object()
    artifact = SimpleNamespace(
        sha256=SHA_A,
        payload={"required_authorized_provider_calls": 10},
    )

    def load_plane(*_args: Any, **_kwargs: Any) -> object:
        events.append("verified-plane")
        return plane

    def judge_preflight(**kwargs: Any) -> object:
        events.append(("gold-judge", kwargs))
        return artifact

    monkeypatch.setattr(launcher, "_load_synthesis_plane", load_plane)
    monkeypatch.setattr(
        "tools.matched_eval.judging.preflight_verified_answer_plane_judge",
        judge_preflight,
    )

    assert launcher.main(
        [
            "judge-preflight",
            "--output-root",
            str(tmp_path / "full"),
            "--expected-answer-run-sha256",
            SHA_A,
            "--dataset",
            str(tmp_path / "locked.json"),
        ]
    ) == 0
    assert events[0] == "verified-plane"
    label, kwargs = events[1]
    assert label == "gold-judge"
    assert kwargs["answer_plane"] is plane
    assert kwargs["dataset_path"] == tmp_path / "locked.json"
