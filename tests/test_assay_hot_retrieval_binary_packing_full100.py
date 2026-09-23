from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import pytest

from tools import assay_hot_retrieval_binary_packing_full100 as assay


def _selected_question(
    prompt: str,
    rendered_texts: tuple[str, ...] = ("alpha fact", "beta fact"),
) -> dict[str, object]:
    evidence = [
        {
            "chunk_id": f"chunk-{index}",
            "rendered_text": text,
        }
        for index, text in enumerate(rendered_texts)
    ]
    envelope = assay.hot._pack_provider_prompt(  # noqa: SLF001
        rendered_texts,
        prompt_question=prompt,
        max_context_tokens=assay.MAX_CONTEXT_TOKENS,
        max_prompt_tokens=assay.MAX_PROMPT_TOKENS,
    )
    arm = assay.hot._raw_packet_semantic(evidence, envelope)  # noqa: SLF001
    return {
        "ordinal": 0,
        "shard_offset": 0,
        "local_ordinal": 0,
        "question_id": "question-0",
        "probe_sha256": "a" * 64,
        "prompt_question_sha256": assay.quote_sha256(prompt),
        "arms": {"a3_protected_union": arm},
    }


def _runtime(question_id: str = "question-0") -> dict[str, object]:
    return {
        "samples": [
            {
                "ordinal": 0,
                "question_id": question_id,
                "shard_offset": 0,
                "timings_ns": {"pack_and_prompt_render_count_ns": 50_000},
            }
        ]
    }


def test_repack_is_exactly_equivalent_to_sealed_provider_payload() -> None:
    prompt = "What are the facts?"
    selected = _selected_question(prompt)

    row, timing = assay._repack_question(  # noqa: SLF001
        selected,
        prompt_question=prompt,
    )

    arm = selected["arms"]["a3_protected_union"]  # type: ignore[index]
    assert row["all_equivalent"] is True
    assert all(row["equivalence"].values())
    assert row["packed_count"] == len(arm["packed_evidence"])
    assert row["context_token_proxy"] == arm["context_token_proxy"]
    assert row["prompt_token_proxy"] == arm["prompt_token_proxy"]
    assert row["prompt_workspace_token_proxy"] == (
        arm["prompt_workspace_token_proxy"]
    )
    assert row["provider_payload_sha256"] == arm["provider_payload_sha256"]
    assert row["packer_audit"]["prompt_render_call_count"] == 1
    assert row["packer_audit"]["complete_prefix_fast_path"] is True
    assert timing["binary_pack_only_ns"] >= 0


def test_repack_rejects_changed_sealed_payload_digest() -> None:
    prompt = "What are the facts?"
    selected = copy.deepcopy(_selected_question(prompt))
    arm = selected["arms"]["a3_protected_union"]  # type: ignore[index]
    arm["provider_payload_sha256"] = (
        "0" * 64
    )

    with pytest.raises(ValueError, match="provider payload bytes changed"):
        assay._repack_question(selected, prompt_question=prompt)  # noqa: SLF001


def test_v7_loader_rejects_any_selection_other_than_exact_pin(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        assay.v7,
        "_load_selection",
        lambda _root: ({}, "0" * 64),
    )

    with pytest.raises(ValueError, match="sealed v7 selection changed"):
        assay._load_v7_bundle(tmp_path)  # noqa: SLF001


def test_run_and_replay_publish_sidecar_sealed_byte_identical_receipts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prompt = "What are the facts?"
    selected = _selected_question(prompt)
    selection = {
        "bindings": {},
        "questions": [selected],
    }
    v7_runtime = _runtime()
    v7_runtime_sha = "b" * 64
    probes_sha = "c" * 64
    implementation = {
        "format": "test-implementation-v1",
        "files": {"test": "d" * 64},
        "sha256": "e" * 64,
    }
    monkeypatch.setattr(assay, "EXPECTED_QUESTION_COUNT", 1)
    monkeypatch.setattr(
        assay,
        "_load_v7_bundle",
        lambda _root: (
            selection,
            assay.EXPECTED_V7_SELECTION_SHA256,
            v7_runtime,
            v7_runtime_sha,
        ),
    )
    monkeypatch.setattr(
        assay,
        "_load_prompt_map",
        lambda _selection: ({0: prompt}, probes_sha),
    )
    monkeypatch.setattr(
        assay,
        "_implementation_identity",
        lambda: copy.deepcopy(implementation),
    )
    monkeypatch.setattr(
        assay,
        "_relative_to_repository",
        lambda _path, *, label: f"test/{label.replace(' ', '-')}",
    )
    v7_root = tmp_path / "sealed-v7"
    output_root = tmp_path / "binary-v8"

    run_sha = assay.run(v7_root=v7_root, output_root=output_root)

    run_body, loaded_run_sha = assay.hot._read_json_artifact(  # noqa: SLF001
        output_root / assay.RUN_NAME
    )
    runtime_body, _runtime_sha = assay.hot._read_json_artifact(  # noqa: SLF001
        output_root / assay.RUNTIME_NAME
    )
    assert loaded_run_sha == run_sha
    assert run_body["aggregate"]["byte_equivalent_question_count"] == 1
    assert run_body["aggregate"]["payload_change_count"] == 0
    assert run_body["provider_calls"] == 0
    assert runtime_body["timing_scope"]["binary"] == (
        "pack_ranked_prefix_prompt_call_only"
    )
    assert runtime_body["sealed_v7_linear_pack_only_ns"]["total"] == 50_000
    assert runtime_body["speedup"]["total_ratio"] > 0
    for name in (assay.RUN_NAME, assay.RUNTIME_NAME):
        artifact = output_root / name
        sidecar = output_root / f"{name}.sha256"
        assert artifact.is_file()
        assert sidecar.is_file()
        assert hashlib.sha256(artifact.read_bytes()).hexdigest() in (
            sidecar.read_text("ascii")
        )

    replay_sha = assay.replay(v7_root=v7_root, output_root=output_root)
    replay_body, loaded_replay_sha = assay.hot._read_json_artifact(  # noqa: SLF001
        output_root / assay.REPLAY_NAME
    )
    assert loaded_replay_sha == replay_sha
    assert replay_body["byte_identical"] is True
    assert replay_body["run_payload_sha256"] == run_sha
    assert replay_body["run_payload_sha256"] == (
        replay_body["replayed_payload_sha256"]
    )
    assert (output_root / f"{assay.REPLAY_NAME}.sha256").is_file()

    # Existing receipts are authenticated and reused rather than overwritten.
    assert assay.run(v7_root=v7_root, output_root=output_root) == run_sha
