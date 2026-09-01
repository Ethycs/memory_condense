from __future__ import annotations

import copy
import json
from typing import Any

import pytest

from memory_condense.domain.discourse import EvidenceSpan, quote_sha256
from tools import run_r7_a1_terminal_answer as sealed_terminal
from tools.matched_eval.after_union_fact_closure import SelectedHLeaf
from tools.matched_eval.contracts import identity_sha256
from tools.matched_eval.full_store_slot_closure import LocalCitationBinding
from tools.matched_eval.r7_linked_terminal_repair import (
    R7LinkedTerminalRepairError,
    compile_r7_linked_terminal_repair,
)
from tools.matched_eval.selected_evidence_discourse_links import (
    SelectedEvidenceDiscourseLinks,
)


def _sha(label: str) -> str:
    return identity_sha256({"label": label})


def _signed(body: dict[str, Any]) -> dict[str, Any]:
    return {**body, "receipt_sha256": identity_sha256(body)}


def _fixture(monkeypatch: pytest.MonkeyPatch) -> tuple[dict[str, Any], dict[str, Any]]:
    dated = "[Question asked at 2025-01-03] What changed?"
    summaries = (
        "The original plan was approved.",
        "Then the plan was revised to use cobalt.",
    )
    leaves = tuple(
        SelectedHLeaf(
            handle_id=f"H{index + 1}00",
            group_handle=f"G{index + 1:04d}",
            text=summary,
            source_receipt_sha256=_sha(f"leaf-source-{index}"),
            topic_labels=("kind:event", "status:completed"),
            boundary_labels=(
                f"group:G{index + 1:04d}",
                f"date:2025-01-0{index + 1}",
                "relation:authored-by-user",
            ),
        )
        for index, summary in enumerate(summaries)
    )
    selection = _signed(
        {
            "cross_boundary_edges": [],
            "format": "test-selection-v1",
            "leaves": [leaf.projection() for leaf in leaves],
            "semantic_result": {
                "retained_leaf_cell_ids": [leaf.handle_id for leaf in leaves]
            },
        }
    )
    question = {
        "dated_question": dated,
        "question_id": "q-linked",
        "question_receipt_sha256": _sha("a1-question"),
        "question_sha256": quote_sha256(dated),
        "semantic_selection": selection,
    }
    base_provider = {
        "dated_question": dated,
        "format": "sealed-hybrid-provider-v1",
        "frontier": {"exact_retained_cover": True},
        "graph_links": [],
        "memory": {
            "raw_summaries": [
                {
                    "group_handle": leaves[0].group_handle,
                    "handle_id": leaves[0].handle_id,
                    "summary": leaves[0].text,
                }
            ],
            "typed_facts": [
                {
                    "citations": [
                        {
                            "group_handle": leaves[1].group_handle,
                            "handle_id": leaves[1].handle_id,
                            "quote": leaves[1].text,
                        }
                    ],
                    "fact_id": "T001",
                    "handle_ids": [leaves[1].handle_id],
                    "text": leaves[1].text,
                }
            ],
        },
        "memory_representation": "deduplicated_typed_facts_plus_unresolved_raw",
        "operator_projection": {"operation": "retrieve"},
        "response_contract": {
            "response_text": "nonempty concise text",
            "used_handle_ids": ["H100"],
        },
    }
    base_prompt = {
        "allowed_handle_ids": [leaf.handle_id for leaf in leaves],
        "messages": [
            {"role": "system", "content": "Answer only from memory."},
            {
                "role": "user",
                "content": json.dumps(
                    base_provider,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            },
        ],
        "presented_handle_ids": [leaf.handle_id for leaf in leaves],
        "prompt_row_receipt_sha256": _sha("base-hybrid-prompt"),
    }
    monkeypatch.setattr(
        sealed_terminal,
        "_question_prompt_rows",
        lambda _question: ({}, ({}, {}, base_prompt)),
    )

    local_rows: list[dict[str, Any]] = []
    handle_rows: list[dict[str, Any]] = []
    item_rows: list[dict[str, Any]] = []
    for index, leaf in enumerate(leaves):
        span = EvidenceSpan(
            chunk_id=f"chunk-{index}",
            start_char=0,
            end_char=len(leaf.text),
            quote_sha256=quote_sha256(leaf.text),
            ordinal=index,
            source_id="source-one",
            turn_start_char=0,
            turn_id=f"turn-{index}",
            role="user",
            created_at=f"2025-01-0{index + 1}T00:00:00Z",
        )
        binding = LocalCitationBinding(
            candidate_id=_sha(f"candidate-{index}"),
            source_group_handle=leaf.group_handle,
            namespace_id=_sha("namespace"),
            cache_receipt_sha256=_sha("cache"),
            source_database_sha256=_sha("database"),
            source_store_receipt_sha256=_sha("store"),
            source_id="source-one",
            partition_id="partition-0",
            span=span,
            quote_sha256=quote_sha256(leaf.text),
        ).projection()
        candidate = _signed(
            {
                "binding_receipt_sha256": binding["receipt_sha256"],
                "format": "test-r7-candidate-v1",
                "quote_sha256": quote_sha256(leaf.text),
            }
        )
        typed = {
            "binding": binding,
            "candidate": candidate,
            "final_handle_id": leaf.handle_id,
            "retained_in_final_prompt": True,
        }
        local_rows.append(
            {"binding": binding, "candidate": candidate, "typed_terminal": typed}
        )
        handle_rows.append(
            {"group_handle": leaf.group_handle, "handle_id": leaf.handle_id}
        )
        item_rows.append(
            {
                "handle_ids": [leaf.handle_id],
                "included": True,
                "summary": leaf.text,
            }
        )
    source_provider = {
        "dated_question": dated,
        "story_coherence": {
            "group_links": [],
            "link_overlays": [
                {
                    "group_handles": [leaves[0].group_handle, leaves[1].group_handle],
                    "relation": "exact_local_candidate_comembership",
                }
            ],
        },
        "typed_evidence": {"handles": handle_rows, "items": item_rows},
    }
    compilation_format = "test-r7-terminal-compilation-v1"
    local_audit = {
        "exact_span_support_population": {"format": "test-support-v1"},
        "local_rows": local_rows,
        "mechanism_by_handle": {leaf.handle_id: "test" for leaf in leaves},
        "terminal_prompt": {"story_link_local_bindings": []},
    }
    local_receipt = identity_sha256(
        {
            "format": f"{compilation_format}-local-audit-v1",
            "exact_span_support_population": local_audit[
                "exact_span_support_population"
            ],
            "local_rows": local_audit["local_rows"],
            "mechanism_by_handle": local_audit["mechanism_by_handle"],
        }
    )
    compilation_body = {
        "format": compilation_format,
        "local_audit_receipt_sha256": local_receipt,
        "new_provider_calls": 0,
        "retained_transformer_token_state_bytes": 0,
    }
    compilation = {
        **compilation_body,
        "local_audit": local_audit,
        "receipt_sha256": identity_sha256(compilation_body),
    }
    plan_body = {
        "dated_question": dated,
        "provider_input": source_provider,
        "provider_input_sha256": identity_sha256(source_provider),
        "question_id": "q-linked",
        "terminal_compilation": compilation,
        "terminal_compilation_receipt_sha256": compilation["receipt_sha256"],
    }
    plan = {
        **plan_body,
        "answer_plan_receipt_sha256": identity_sha256(plan_body),
    }
    return question, {"question_id": "q-linked", "terminal_answer_plan": plan}


def _provider_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        return set(value) | {
            key for child in value.values() for key in _provider_keys(child)
        }
    if isinstance(value, list):
        return {key for child in value for key in _provider_keys(child)}
    return set()


def test_repair_authenticates_enriches_links_and_keeps_locators_local(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    question, source = _fixture(monkeypatch)
    repair = compile_r7_linked_terminal_repair(question, source)

    assert repair["allowed_handle_ids"] == ["H100", "H200"]
    assert repair["presented_handle_ids"] == ["H100", "H200"]
    assert repair["prompt_token_proxy"] + repair["output_token_reserve"] <= 8_000
    provider = repair["provider_input"]
    assert provider["memory_representation"].endswith("linked_metadata")
    assert provider["memory"]["raw_summaries"][0]["handle_id"] == "H100"
    assert provider["memory"]["raw_summaries"][0]["terminal_leaf_metadata"][
        "event_date"
    ] == "2025-01-01"
    citation = provider["memory"]["typed_facts"][0]["citations"][0]
    assert citation["terminal_leaf_metadata"]["kind"] == "event"
    assert {row["relation"] for row in provider["graph_links"]} == {
        "exact_local_candidate_comembership",
    }
    typed_by_relation = {row["relation"]: row for row in provider["typed_links"]}
    assert set(typed_by_relation) == {"revises", "sequence"}
    sequence = typed_by_relation["sequence"]
    assert [row["role"] for row in sequence["members"]] == ["previous", "next"]
    assert [row["evidence_role"] for row in sequence["members"]] == ["user", "user"]
    assert [row["ordinal"] for row in sequence["members"]] == [0, 1]
    revision = typed_by_relation["revises"]
    assert [row["role"] for row in revision["members"]] == [
        "predecessor",
        "successor",
    ]
    assert not (
        _provider_keys(provider)
        & {"chunk_id", "source_id", "partition_id", "turn_id", "namespace_id"}
    )
    audit_binding = repair["local_audit"]["local_handle_bindings"][0]
    assert audit_binding["binding"]["source_id"] == "source-one"
    assert audit_binding["summary_sha256"] == audit_binding["binding"]["span"][
        "quote_sha256"
    ]
    assert repair["local_audit"]["discourse_compilation"][
        "retained_transformer_token_state_bytes"
    ] == 0


def test_repair_fails_closed_on_summary_span_digest_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    question, source = _fixture(monkeypatch)
    changed = copy.deepcopy(source)
    provider = changed["terminal_answer_plan"]["provider_input"]
    provider["typed_evidence"]["items"][0]["summary"] = "A forged summary."
    plan = changed["terminal_answer_plan"]
    plan["provider_input_sha256"] = identity_sha256(provider)
    plan["answer_plan_receipt_sha256"] = identity_sha256(
        {key: value for key, value in plan.items() if key != "answer_plan_receipt_sha256"}
    )
    with pytest.raises(R7LinkedTerminalRepairError, match="exact local span"):
        compile_r7_linked_terminal_repair(question, changed)


def test_repair_fails_closed_on_local_audit_receipt_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    question, source = _fixture(monkeypatch)
    changed = copy.deepcopy(source)
    plan = changed["terminal_answer_plan"]
    local_audit = plan["terminal_compilation"]["local_audit"]
    local_audit["mechanism_by_handle"]["H100"] = "forged-mechanism"
    plan["answer_plan_receipt_sha256"] = identity_sha256(
        {key: value for key, value in plan.items() if key != "answer_plan_receipt_sha256"}
    )
    with pytest.raises(R7LinkedTerminalRepairError, match="local audit receipt"):
        compile_r7_linked_terminal_repair(question, changed)


def test_repair_trims_only_extras_and_preserves_evidence_population(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    question, source = _fixture(monkeypatch)

    def no_links(_inputs: object) -> SelectedEvidenceDiscourseLinks:
        return SelectedEvidenceDiscourseLinks((), (), ())

    repair = compile_r7_linked_terminal_repair(
        question,
        source,
        discourse_linker=no_links,
        hard_total_token_cap=600,
        output_token_reserve=256,
    )
    assert repair["allowed_handle_ids"] == ["H100", "H200"]
    assert repair["provider_input"]["memory"]["raw_summaries"][0]["summary"].startswith(
        "The original"
    )
    assert repair["provider_input"]["memory"]["typed_facts"][0]["text"].startswith(
        "Then the plan"
    )
    assert repair["prompt_token_proxy"] + 256 <= 600
    assert (
        repair["local_audit"]["trimmed_extra_edge_ids"]
        or repair["local_audit"]["trimmed_metadata_fields"]
    )
