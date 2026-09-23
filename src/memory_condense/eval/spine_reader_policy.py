"""General answer-completeness rules for exact user-spine evidence packets."""
from memory_condense.eval._retrieval_qa_prompt import QA_SYSTEM_PROMPT


SPINE_READER_SYSTEM_PROMPT = QA_SYSTEM_PROMPT + (
    "\n\nWhen the question asks you to identify an entity and the excerpts give "
    "only an identifying description, return that description. Do not invent a "
    "proper name or abstain solely because a proper name is missing. "
    "For recommendation requests, briefly state how the suggestions meet the "
    "user's relevant setup, compatibility requirements and preferences supported "
    "by the excerpts. Keep those constraints explicit rather than giving only "
    "a generic list."
)
