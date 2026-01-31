"""Generate a response from retrieved memory chunks + recent conversation context."""

from __future__ import annotations

import litellm

from memory_condense.schemas import RetrievalResult

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


def generate_response(
    user_text: str,
    retrieved: list[RetrievalResult],
    recent_turns: list[tuple[str, str]],
    model: str = "anthropic/claude-3-5-haiku-20241022",
    temperature: float = 0.3,
    max_tokens: int = 1024,
) -> str:
    """Generate a response given memory context and recent conversation.

    Returns the generated response text.
    """
    messages = build_prompt(user_text, retrieved, recent_turns)

    response = litellm.completion(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    return response.choices[0].message.content.strip()
