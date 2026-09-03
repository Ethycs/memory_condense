"""Provider-free prompt packing shared by retrieval-only execution paths.

This module intentionally owns no benchmark sample loader, answer, reference,
judge, or provider surface.  It is the narrow prompt-construction dependency
needed by prediction-stage cumulative retrieval.
"""

from __future__ import annotations

from memory_condense.domain._tokenizer import (
    count_chat_prompt_token_proxy,
    count_tokens,
    truncate_to_tokens,
)


QA_SYSTEM_PROMPT = (
    "You are answering questions about a long conversation history. "
    "You are given excerpts retrieved from that history as your only source "
    "of information.\n\n"
    "Answer the question using ONLY the retrieved excerpts. Be as short as "
    "possible: reply with just the fact, name, number, or date asked for — "
    "no preamble, no explanation, no full sentences unless the question "
    "requires one. Provenance labels may include an excerpt timestamp and "
    "speaker role. Treat user statements as facts about the user; do not "
    "mistake assistant suggestions for things the user did. For 'now', "
    "'current', or 'latest' questions, use the newest relevant user update. "
    "If that newest update states an approximate current value (for example, "
    "'close to 1300 now' or 'about 20'), return the stated number; do not "
    "abstain merely because the value is approximate. "
    "For ordering questions, compare the relevant timestamps. If the "
    "question asks for a difference, duration, or amount remaining, identify "
    "the relevant operands and calculate the result. Treat statements such "
    "as 'started today' or 'got it today' as events at their excerpt "
    "timestamps; if an approximate recap conflicts with an explicit start "
    "or end boundary, use the explicit boundary. If the "
    "excerpts do not contain the answer, reply exactly: "
    "I don't know."
)
QA_USER_TEMPLATE = (
    "Retrieved excerpts from the conversation history:\n"
    "{context}\n\n"
    "Question: {question}\n"
    "Short answer:"
)
QA_NO_CONTEXT = "(no excerpts retrieved)"
RESPONDER_OUTPUT_TOKEN_RESERVE = 256


def build_qa_prompt(question: str, chunk_texts: list[str]) -> list[dict[str, str]]:
    """Build the frozen retrieval-answer prompt from selected excerpts."""

    context = (
        "\n".join(f"[{index + 1}] {text}" for index, text in enumerate(chunk_texts))
        if chunk_texts
        else QA_NO_CONTEXT
    )
    return [
        {"role": "system", "content": QA_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": QA_USER_TEMPLATE.format(context=context, question=question),
        },
    ]


def cap_context_to_prompt_budget(
    question: str,
    chunk_texts: list[str],
    max_prompt_tokens: int | None,
) -> list[str]:
    """Keep ranked excerpts within the exact prompt-token-proxy ceiling."""

    if max_prompt_tokens is None:
        return list(chunk_texts)
    if max_prompt_tokens < 1:
        raise ValueError("max_prompt_tokens must be positive")
    if count_chat_prompt_token_proxy(build_qa_prompt(question, [])) > max_prompt_tokens:
        raise ValueError(
            "max_prompt_tokens is smaller than the QA prompt without context"
        )

    selected: list[str] = []
    for excerpt in chunk_texts:
        proposal = [*selected, excerpt]
        if (
            count_chat_prompt_token_proxy(build_qa_prompt(question, proposal))
            <= max_prompt_tokens
        ):
            selected.append(excerpt)
            continue
        low = 0
        high = count_tokens(excerpt)
        while low < high:
            midpoint = (low + high + 1) // 2
            prefix = truncate_to_tokens(excerpt, midpoint)
            tokens = count_chat_prompt_token_proxy(
                build_qa_prompt(question, [*selected, prefix])
            )
            if tokens <= max_prompt_tokens:
                low = midpoint
            else:
                high = midpoint - 1
        if low:
            selected.append(truncate_to_tokens(excerpt, low))
        break
    return selected


__all__ = [
    "QA_NO_CONTEXT",
    "QA_SYSTEM_PROMPT",
    "RESPONDER_OUTPUT_TOKEN_RESERVE",
    "build_qa_prompt",
    "cap_context_to_prompt_budget",
]
