from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools import run_locked_semantic_residual_construction_v4 as r7_cli
from tools import run_reduced_semantic_global_completion_assay as v7_cli
from tools import run_reduced_source_group_reinjection_assay as v6_cli
from tools.matched_eval.artifacts import read_sealed_json
from tools.matched_eval.contracts import identity_sha256
from tools.matched_eval.semantic_global_terminal_adapter import (
    TerminalSealedSources,
    compile_semantic_global_terminal,
    load_selected_protected_owner_evidence,
    replay_semantic_global_terminal,
)


GARMIN_FACT = (
    "I'm also planning to track my ride with my new Garmin bike computer. "
    "Can you tell me how to set it up to get the most accurate distance and "
    "speed readings?"
)
CHAIN_FACT = (
    "I replaced the old bike's chain and cassette on February 1st, which has "
    "contributed to an improvement in the bike's performance. Can you "
    "recommend some routes or apps to help me find the most bike-friendly "
    "roads for my 20-mile ride?"
)
Q82_DIRECT_EPISODE = "episode-f43e2711dea9ce00e405e21f"


@pytest.mark.slow
def test_exact_q82_terminal_preserves_frozen_v61_l_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Assay the real q82 namespace without persisting any live index object."""

    captured: dict[str, object] = {}
    original_residual = v7_cli.residual.search_semantic_residual
    original_local = v7_cli.search_source_group_reinjection
    original_global = v7_cli.search_semantic_global_completion
    original_terminal = v7_cli.r7_cli.build_separate_terminal_prompt

    def capture_residual(*args, **kwargs):
        result = original_residual(*args, **kwargs)
        captured["residual_index"] = args[0]
        captured["query"] = args[1]
        captured["protected"] = tuple(kwargs.get("protected_evidence", ()))
        captured["residual_result"] = result
        return result

    def capture_local(*args, **kwargs):
        result = original_local(*args, **kwargs)
        captured["local_result"] = result
        return result

    def capture_global(*args, **kwargs):
        result = original_global(*args, **kwargs)
        captured["global_result"] = result
        return result

    def capture_terminal(*args, **kwargs):
        result = original_terminal(*args, **kwargs)
        captured["parent_prediction"] = kwargs["current_prediction"]
        captured["r7_terminal"] = result[0]
        return result

    monkeypatch.setattr(
        v7_cli.residual, "search_semantic_residual", capture_residual
    )
    monkeypatch.setattr(v7_cli, "search_source_group_reinjection", capture_local)
    monkeypatch.setattr(v7_cli, "search_semantic_global_completion", capture_global)
    monkeypatch.setattr(
        v7_cli.r7_cli, "build_separate_terminal_prompt", capture_terminal
    )

    args = v7_cli.build_parser().parse_args(
        [
            "construct",
            "--ordinals",
            "82",
            "--auto-resolve-episode-artifact",
            "--output-root",
            str(tmp_path / "unused-assay-root"),
        ]
    )
    assay = v7_cli.build_assay(args)
    assert assay["question_count"] == 1
    assert assay["questions"][0]["ordinal"] == 82

    residual_index = captured["residual_index"]
    query = captured["query"]
    protected = captured["protected"]
    residual_result = captured["residual_result"]
    local_result = captured["local_result"]
    global_result = captured["global_result"]
    r7_terminal = captured["r7_terminal"]
    assert r7_terminal is not None
    selected_owners = load_selected_protected_owner_evidence(
        r7_terminal["provider_input"]["protected_owner_evidence"]
    )

    r7_artifact = v6_cli._verified_r7_construction(args)  # noqa: SLF001
    parent_artifact = read_sealed_json(Path(r7_cli.DEFAULT_CONSTRUCTION))
    sealed_sources = TerminalSealedSources(
        protected_owner_artifact_sha256=r7_artifact.sha256,
        residual_artifact_sha256=r7_artifact.sha256,
        parent_artifact_sha256=parent_artifact.sha256,
    )
    compiled = compile_semantic_global_terminal(
        dated_question=query.dated_question,
        parent_prediction=captured["parent_prediction"],
        residual_index=residual_index,
        query=query,
        protected_owner_universe_bindings=protected,
        selected_protected_owner_evidence=selected_owners,
        residual_result=residual_result,
        local_result=local_result,
        global_result=global_result,
        sealed_sources=sealed_sources,
    )

    assert len(local_result.attempted_selection) == 64
    assert sum(
        row.disposition == "budget_unpacked"
        for row in local_result.attempted_selection
    ) == 41
    l_rows = tuple(
        row for row in compiled.local_rows if row["candidate"]["plane"] == "L"
    )
    assert len(l_rows) == 64
    assert tuple(
        row["candidate"]["selection_receipt_sha256"] for row in l_rows
    ) == tuple(row.receipt_sha256 for row in local_result.attempted_selection)
    assert sum(
        row["candidate"]["upstream_disposition"] == "budget_unpacked"
        for row in l_rows
    ) == 41
    l_selection = next(
        row for row in compiled.plane_selections if row.plane == "L"
    )
    assert l_selection.candidate_receipt_sha256s == tuple(
        row["candidate"]["receipt_sha256"] for row in l_rows
    )
    assert set(l_selection.selected_candidate_receipt_sha256s).isdisjoint(
        l_selection.skipped_candidate_receipt_sha256s
    )
    assert set(l_selection.selected_candidate_receipt_sha256s) | set(
        l_selection.skipped_candidate_receipt_sha256s
    ) == set(l_selection.candidate_receipt_sha256s)

    evidence_by_quote = {row.quote: row for row in local_result.evidence}
    assert {GARMIN_FACT, CHAIN_FACT} <= set(evidence_by_quote)
    assert compiled.fitted.prompt_token_proxy + 768 <= 8_000
    assert local_result.packed_local_evidence_tokens <= 1_200

    garmin_segment = evidence_by_quote[GARMIN_FACT].segment_receipt_sha256
    garmin_attempt = next(
        row
        for row in local_result.attempted_selection
        if row.segment_receipt_sha256 == garmin_segment
    )
    anchor_attempt = next(
        row
        for row in local_result.attempted_selection
        if row.selection_rank < garmin_attempt.selection_rank
        and row.disposition == "protected_exact_duplicate"
        and any(Q82_DIRECT_EPISODE in route for route in row.selection_routes)
    )
    assert garmin_attempt.selection_rank == 11
    assert anchor_attempt.selection_rank == 8
    terminal_l_by_selection = {
        row["candidate"]["selection_receipt_sha256"]: row for row in l_rows
    }
    anchor_candidate_receipt = terminal_l_by_selection[
        anchor_attempt.receipt_sha256
    ]["candidate"]["receipt_sha256"]
    garmin_candidate_receipt = terminal_l_by_selection[
        garmin_attempt.receipt_sha256
    ]["candidate"]["receipt_sha256"]
    assert anchor_candidate_receipt in l_selection.selected_candidate_receipt_sha256s
    assert garmin_candidate_receipt in l_selection.selected_candidate_receipt_sha256s
    assert any(
        row["candidate_receipt_sha256"] == anchor_candidate_receipt
        and row["containment_proven"] is True
        for row in compiled.post_selection_dedup.substitutions
    )

    replayed = replay_semantic_global_terminal(
        dated_question=query.dated_question,
        parent_prediction=captured["parent_prediction"],
        residual_index=residual_index,
        query=query,
        protected_owner_universe_bindings=protected,
        selected_protected_owner_evidence=selected_owners,
        residual_result=residual_result,
        local_result=local_result,
        global_result=global_result,
        sealed_sources=sealed_sources,
        sealed_compilation=compiled,
    )
    assert replayed.projection(include_local=True) == compiled.projection(
        include_local=True
    )
    assert compiled.projection()["new_provider_calls"] == 0
    assert compiled.projection()["retained_transformer_token_state_bytes"] == 0

    provider_projection = compiled.provider_projection()
    provider_bytes = json.dumps(
        provider_projection,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    provider_text = provider_bytes.decode("utf-8")
    metrics = {
        "budget_unpacked": 41,
        "chain_fact_present": CHAIN_FACT in provider_text,
        "compilation_receipt_sha256": compiled.receipt_sha256,
        "garmin_fact_present": GARMIN_FACT in provider_text,
        "l_attempt_count": len(l_rows),
        "l_selected_count": len(l_selection.selected_candidate_receipt_sha256s),
        "local_result_receipt_sha256": local_result.receipt_sha256,
        "prompt_token_proxy": compiled.fitted.prompt_token_proxy,
        "provider_identity_sha256": identity_sha256(provider_projection),
        "provider_json_byte_count": len(provider_bytes),
        "replay_identical": True,
    }
    assert GARMIN_FACT in provider_text and CHAIN_FACT in provider_text, json.dumps(
        metrics, sort_keys=True
    )
