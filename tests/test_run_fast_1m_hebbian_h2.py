from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from memory_condense.eval import run_fast_1m_hebbian_h2 as runner
from tests.test_fast_hebbian_h2 import _fixture


def _values(
    *,
    phase: str,
    retrieval_path: Path,
    retrieval_sha256: str,
    history_path: Path,
    history_sha256: str,
    derived_store: Path,
    output_root: Path,
    receipts_sha256: str | None = None,
) -> list[str]:
    values = [
        "--phase",
        phase,
        "--retrieval",
        str(retrieval_path),
        "--expected-retrieval-sha256",
        retrieval_sha256,
        "--history",
        str(history_path),
        "--expected-history-sha256",
        history_sha256,
        "--derived-store",
        str(derived_store),
        "--output-root",
        str(output_root),
        "--expected-question-count",
        "1",
    ]
    if receipts_sha256 is not None:
        values.extend(["--expected-receipts-sha256", receipts_sha256])
    return values


def _rewrite_canonical(path: Path, payload: object) -> None:
    raw = runner._canonical_json_bytes(payload)
    digest = hashlib.sha256(raw).hexdigest()
    path.write_bytes(raw)
    path.with_name(path.name + ".sha256").write_bytes(
        f"{digest}  {path.name}\n".encode("ascii")
    )


def test_provider_free_preflight_publish_and_byte_exact_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    artifact, retrieval, history, derived_store, _derived = _fixture(tmp_path)
    retrieval_path = Path(retrieval.source_path)
    history_path = Path(history.source_path)
    output_root = tmp_path / "h2-output"

    def load_retrieval(path: Path, *, expected_sha256: str):
        assert path == retrieval_path
        assert expected_sha256 == artifact.raw_sha256
        return artifact

    monkeypatch.setattr(runner, "load_fast_retrieval_artifact", load_retrieval)
    common = {
        "retrieval_path": retrieval_path,
        "retrieval_sha256": artifact.raw_sha256,
        "history_path": history_path,
        "history_sha256": history.raw_sha256,
        "derived_store": derived_store,
        "output_root": output_root,
    }

    assert runner.main(_values(phase="preflight", **common)) == 0
    preflight_output = capsys.readouterr().out
    assert "questions=1; appended_questions=1; appended_evidence=1" in (
        preflight_output
    )
    assert "budget_rejected_candidates=0" in preflight_output
    assert "budget_blocked_questions=0" in preflight_output
    assert "provider_calls=0; writes=0" in preflight_output
    assert not output_root.exists()

    publish_args = runner.build_parser().parse_args(
        _values(phase="publish", **common)
    )
    publish_metrics, receipt_sha256 = runner.run_publish(publish_args)
    receipt_path = output_root / runner.H2_RECEIPTS_NAME
    raw = receipt_path.read_bytes()
    payload = json.loads(raw)

    assert publish_metrics["writes"] == 2
    assert publish_metrics["provider_calls"] == 0
    assert publish_metrics["gold_fields_consumed"] is False
    assert publish_metrics["cav_links_computed"] is False
    assert receipt_sha256 == hashlib.sha256(raw).hexdigest()
    assert receipt_path.with_name(receipt_path.name + ".sha256").read_bytes() == (
        f"{receipt_sha256}  {receipt_path.name}\n".encode("ascii")
    )
    assert raw == runner._canonical_json_bytes(payload)
    assert "final_evidence" not in payload
    assert "h2_consumer_source_manifest" in payload
    assert "h2_consumer_source_sha256" in payload
    assert "h2_consumer_environment_lock_sha256" in payload
    assert "h2_consumer_implementation_sha256" not in payload
    assert b"SECRET source" not in raw
    assert artifact.questions[0].question.encode("utf-8") not in raw

    replay_args = runner.build_parser().parse_args(
        _values(
            phase="replay",
            receipts_sha256=receipt_sha256,
            **common,
        )
    )
    monkeypatch.setattr(
        runner,
        "_publish_bytes",
        lambda *_args, **_kwargs: pytest.fail("replay attempted a write"),
    )
    replay_metrics, replay_sha256 = runner.run_replay(replay_args)

    assert replay_sha256 == receipt_sha256
    assert replay_metrics["population_sha256"] == payload["population_sha256"]
    assert replay_metrics["writes"] == 0
    assert receipt_path.read_bytes() == raw

    payload["provider_calls"] = 1
    _rewrite_canonical(receipt_path, payload)
    unpinned_replay = runner.build_parser().parse_args(
        _values(phase="replay", **common)
    )
    with pytest.raises(ValueError, match="byte-identical reconstruction"):
        runner.run_replay(unpinned_replay)


def test_defaults_pin_sealed_development_inputs() -> None:
    args = runner.build_parser().parse_args([])

    assert args.phase == "preflight"
    assert "20260821" in str(args.retrieval)
    assert "20260822" in str(args.history)
    assert "20260822" in str(args.derived_store)
    assert "hebbian-h2-static-local-closure" in str(args.output_root)
    assert runner.PREFLIGHT_FORMAT.endswith(
        "static-local-closure-runner-preflight-v3"
    )
    assert args.expected_history_sha256 == runner.DEFAULT_HISTORY_SHA256
    assert args.expected_question_count == 10
