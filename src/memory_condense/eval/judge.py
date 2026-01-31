"""Judge LLM scores a generated response against the actual response."""

from __future__ import annotations

import json

import litellm

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


def judge_response(
    user_text: str,
    actual_response: str,
    generated_response: str,
    model: str = "anthropic/claude-3-5-haiku-20241022",
    temperature: float = 0.0,
) -> tuple[int, str]:
    """Score a generated response against the actual response.

    Returns (score, reasoning).
    """
    user_prompt = (
        f"User message:\n{user_text}\n\n"
        f"ACTUAL response:\n{actual_response}\n\n"
        f"GENERATED response:\n{generated_response}\n\n"
        f"Judge the generated response:"
    )

    response = litellm.completion(
        model=model,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=256,
    )

    content = response.choices[0].message.content.strip()

    try:
        result = json.loads(content)
        score = int(result.get("score", 1))
        score = max(1, min(5, score))
        reasoning = result.get("reasoning", "")
    except (json.JSONDecodeError, ValueError):
        score = 1
        reasoning = f"Failed to parse judge response: {content[:200]}"

    return score, reasoning
