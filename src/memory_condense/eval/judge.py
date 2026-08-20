"""Judge LLM scores a generated response against the actual response.

NOTE ON SAMPLING PARAMETERS: this module deliberately does NOT pass
``temperature`` (or ``top_p`` / ``top_k``) to litellm. The default judge is
Claude Sonnet 5, which rejects non-default sampling parameters with a 400.
Do not "helpfully" re-add temperature here — steer the judge with the prompt
instead.

``max_tokens`` is 1024 rather than a tight 256 because Sonnet 5 runs adaptive
thinking by default and ``max_tokens`` caps thinking + visible text together;
256 truncates the JSON verdict.
"""

from __future__ import annotations

import json
import time

import litellm

from memory_condense.eval._completion import _content, build_completion_request
from memory_condense.eval.schemas import DEFAULT_JUDGE_MODEL, UsageStats

JUDGE_SYSTEM = """You are a strict but fair judge evaluating the quality of an AI-generated response.

You will see:
1. The user's message
2. The ACTUAL response (ground truth from the original conversation)
3. The GENERATED response (produced by the system under test)

Score the generated response on a 1-5 scale based on how well it captures the substance and intent of the actual response:

5 - EXCELLENT: Covers the same key information and approach. May differ in wording but is substantively equivalent.
4 - GOOD: Same general direction, captures most key points but misses some details or nuance.
3 - PARTIAL: Gets some things right but misses important content from the actual response.
2 - POOR: Mostly different from the actual response, only tangentially related.
1 - FAIL: Completely off-topic or contradicts the actual response.

IMPORTANT: Judge based on substance, not style. Different wording is fine as long as the key information matches. The generated response does not need to be identical — it needs to convey the same essential information.

Respond with valid JSON only:
{"score": <1-5>, "reasoning": "<1-2 sentences>"}"""

JUDGE_MAX_TOKENS = 1024


def build_judge_prompt(
    user_text: str,
    actual_response: str,
    generated_response: str,
) -> list[dict[str, str]]:
    """Build the messages list for the judge completion call."""
    user_prompt = (
        f"User message:\n{user_text}\n\n"
        f"ACTUAL response:\n{actual_response}\n\n"
        f"GENERATED response:\n{generated_response}\n\n"
        f"Judge the generated response:"
    )
    return [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": user_prompt},
    ]


def judge_response_with_usage(
    user_text: str,
    actual_response: str,
    generated_response: str,
    model: str = DEFAULT_JUDGE_MODEL,
) -> tuple[int, str, UsageStats]:
    """Score a generated response against the actual response.

    Returns (score, reasoning, usage).

    The ``temperature`` argument was removed: Claude Sonnet 5 (the default
    judge) rejects non-default sampling parameters with a 400.
    """
    messages = build_judge_prompt(user_text, actual_response, generated_response)

    start = time.perf_counter()
    response = litellm.completion(
        **build_completion_request(
            model,
            messages,
            max_tokens=JUDGE_MAX_TOKENS,
            num_retries=5,
        )
    )
    elapsed = time.perf_counter() - start

    usage = UsageStats.from_litellm(response, elapsed)
    # ``None`` content (refusal / thinking exhausted max_tokens) becomes "",
    # which fails JSON parsing below and is scored as an explicit 1 with a
    # "Failed to parse" reasoning — not an AttributeError mid-run.
    content = _content(response)

    try:
        result = json.loads(content)
        score = int(result.get("score", 1))
        score = max(1, min(5, score))
        reasoning = result.get("reasoning", "")
    except (json.JSONDecodeError, ValueError):
        score = 1
        reasoning = f"Failed to parse judge response: {content[:200]}"

    return score, reasoning, usage
