"""Dependency-light common-parent request accounting authority.

This module deliberately imports no source package or campaign runtime.  The
provider-free Mem0 launcher must be able to authenticate its request budgets
against the historical frozen validation source, which does not contain the
newer answer/judge implementation modules.
"""

from __future__ import annotations


COMPARISON_SEMANTICS = "common_parent"

EXACT_ACCOUNTING = {
    "answer_complete_request_token_cap": 8_000,
    "answer_max_prompt_tokens": 7_232,
    "answer_output_token_reserve": 768,
    "judge_complete_envelope_token_cap": 9_024,
    "judge_max_prompt_tokens": 8_000,
    "judge_model": "codex_sdk/gpt-5.6-sol",
    "judge_output_token_reserve": 1_024,
    "responder_model": "codex_sdk/gpt-5.6-terra",
    "retained_transformer_token_state_bytes": 0,
    "sdk_retries": 0,
}


__all__ = ["COMPARISON_SEMANTICS", "EXACT_ACCOUNTING"]
