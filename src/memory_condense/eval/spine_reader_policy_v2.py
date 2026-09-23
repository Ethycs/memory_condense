"""Make supported personalization explicit in recommendation answers."""
from memory_condense.eval.spine_reader_policy import SPINE_READER_SYSTEM_PROMPT


SPINE_READER_SYSTEM_PROMPT_V2 = SPINE_READER_SYSTEM_PROMPT + (
    "\nFor recommendations, use two short sentences: first state the relevant "
    "user preferences and compatibility constraints supported by the excerpts; "
    "then suggest options that satisfy those constraints."
)
