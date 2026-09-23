"""Distinguish an explicitly stated attribute from proof of a completed action."""
from memory_condense.eval.spine_reader_policy_v2 import SPINE_READER_SYSTEM_PROMPT_V2


SPINE_READER_SYSTEM_PROMPT_V5 = SPINE_READER_SYSTEM_PROMPT_V2 + (
    "\nWhen the question requests an attribute of a uniquely identified item, "
    "return that attribute if the user states it explicitly. The excerpts need "
    "not repeat every background verb in the question. This does not establish "
    "that an action happened: when the requested fact is whether an action "
    "occurred, require evidence for that action. Respect explicit denials and "
    "corrections even when the question assumes otherwise. Do not turn ownership, "
    "bidding, intentions or assistant recommendations into completed actions. "
    "If the requested attribute is absent or the item match is ambiguous, abstain."
)
