import copy

import pytest

from memory_condense.domain._tokenizer import count_tokens
from tools import assay_hot_raw_packet_reduced30 as assay
from tools.matched_eval.hot_temporal_reference_chain import split_packet, _global_rows
from tests.test_hot_temporal_reference_chain import _arm, _rebind


def _with_tail():
    arm = _arm()
    arm["provider_messages"][1]["content"] = arm["provider_messages"][1]["content"].replace(
        "\n\nQuestion:", "\n\n<FACTS exact_quotes raw_authoritative>\n<F1 backs=G1> I still ride my old bicycle.\n\nQuestion:")
    arm["fact_ledger"] = {"selected_facts": ["audit only"]}
    arm["rendered_fact_ids"] = ["fact-1"]
    arm["context_token_proxy"] = count_tokens(split_packet(arm)[1])
    return _rebind(arm)


def test_removing_hints_keeps_every_original_raw_byte_and_policy():
    arm = _with_tail()
    original = copy.deepcopy(arm)
    prefix, context, question = split_packet(arm)
    rows = _global_rows(arm, context)
    candidate = assay.compose(arm, prefix, context, question, rows)
    assert arm == original
    assert candidate["provider_messages"][0] == arm["provider_messages"][0]
    assert split_packet(candidate)[1] == context.split("\n\n<FACTS ")[0]
    assert split_packet(candidate)[2] == question
    assert candidate["rendered_parent_evidence_ids"] == arm["rendered_parent_evidence_ids"]
    assert candidate["rendered_fact_ids"] == []
    assert "fact_ledger" not in candidate
    assert candidate["raw_packet_ablation"]["removed_hints_parent_audit"]["fact_ledger"] == arm["fact_ledger"]


def test_hint_with_missing_raw_numeric_backing_is_rejected():
    arm = _with_tail()
    arm["numeric_slot_completion"] = {"provider_bindings": [{"backing_evidence_id": "missing", "backing_provider_label": "G99"}]}
    prefix, context, question = split_packet(arm)
    with pytest.raises(ValueError, match="retained raw backing"):
        assay.compose(arm, prefix, context, question, _global_rows(arm, context))
