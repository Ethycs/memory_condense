"""Generate a response from retrieved memory chunks + recent conversation context."""

from __future__ import annotations

import time

import litellm

from memory_condense.eval.schemas import DEFAULT_RESPONDER_MODEL, UsageStats
from memory_condense.domain.schemas import RetrievalResult

SYSTEM_PROMPT = (
    "You are a helpful assistant. You have access to a memory system that "
    "retrieves relevant context from earlier in the conversation. Use the "
    "memory context and recent conversation to respond to the user.\n\n"
    "If the memory context contains relevant information, incorporate it "
    "naturally into your response. Do not mention the memory system itself."
)


def build_prompt(
    user_text: str,
    retrieved: list[RetrievalResult],
    recent_turns: list[tuple[str, str]],
) -> list[dict[str, str]]:
    """Build the messages list for the litellm completion call.

    Args:
        user_text: The current user message.
        retrieved: Retrieved chunks from memory.
        recent_turns: Recent (role, text) pairs for conversational context.
    """
    messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Memory context
    if retrieved:
        chunk_texts = [
            f"[Memory {i + 1}]: {r.chunk.text}" for i, r in enumerate(retrieved)
        ]
        memory_block = "Relevant memory context:\n" + "\n".join(chunk_texts)
        messages.append({"role": "system", "content": memory_block})

    # Recent conversation turns
    for role, text in recent_turns:
        messages.append({"role": role, "content": text})

    # Current user message
    messages.append({"role": "user", "content": user_text})

    return messages


def generate_from_messages(
    messages: list[dict[str, str]],
    model: str = DEFAULT_RESPONDER_MODEL,
    temperature: float = 0.3,
    max_tokens: int = 1024,
) -> tuple[str, UsageStats]:
    """Generate from an already-assembled messages list.

    Exists so the memory arm can send what ``ContextPacker`` produced rather
    than re-deriving a prompt: ``build_context`` returns a budgeted
    ``PackedContext.messages``, and the whole point of measuring that arm is
    to send exactly those tokens.

    Returns ``(generated_text, usage)``.
    """
    start = time.perf_counter()
    response = litellm.completion(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        num_retries=5,
    )
    elapsed = time.perf_counter() - start

    usage = UsageStats.from_litellm(response, elapsed)
    try:
        text = (response.choices[0].message.content or "").strip()
    except (AttributeError, IndexError, TypeError):
        text = ""
    return text, usage


def generate_response_with_usage(
    user_text: str,
    retrieved: list[RetrievalResult],
    recent_turns: list[tuple[str, str]],
    model: str = DEFAULT_RESPONDER_MODEL,
    temperature: float = 0.3,
    max_tokens: int = 1024,
) -> tuple[str, UsageStats]:
    """Generate a response given memory context and recent conversation.

    Returns (generated_text, usage).

    The default responder is Claude Haiku 4.5, which accepts sampling
    parameters, so ``temperature`` is still passed here (unlike the judge).
    """
    return generate_from_messages(
        build_prompt(user_text, retrieved, recent_turns),
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def generate_response(
    user_text: str,
    retrieved: list[RetrievalResult],
    recent_turns: list[tuple[str, str]],
    model: str = DEFAULT_RESPONDER_MODEL,
    temperature: float = 0.3,
    max_tokens: int = 1024,
) -> str:
    """Generate a response given memory context and recent conversation.

    Returns the generated response text. Use
    :func:`generate_response_with_usage` when you also need token/latency
    accounting.
    """
    text, _ = generate_response_with_usage(
        user_text=user_text,
        retrieved=retrieved,
        recent_turns=recent_turns,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return text
