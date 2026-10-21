"""Provider-free QA prompt constants shared by prediction renderers.

Re-exported from the owning module rather than copied. This file previously
held a byte-identical duplicate of the text, which stayed correct only by
coincidence: 9cc4a89 split it off so the renderers would not import the
benchmark surface, and from then until 684c3dd the two texts happened to
match. 684c3dd rewrote benchmark's copy and left this one, which is how a
single eval pipeline came to answer under two different reader policies.

Importing ``eval._retrieval_qa_prompt`` keeps the decoupling this file was
created for -- that module deliberately owns no sample loader, answer,
judge, or provider surface -- while leaving exactly one definition of the
text.
"""

from memory_condense.eval._retrieval_qa_prompt import QA_SYSTEM_PROMPT

__all__ = ["QA_SYSTEM_PROMPT"]
