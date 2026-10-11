"""Serving defaults validated on the persisted native summary-memory path."""
from types import MappingProxyType


CAP8_LIMITS = MappingProxyType({
    'ancestor_hops': 2,
    'context_seed_limit': 4,
    'lexical_reserve': 0,
    'max_additions': 16,
    'max_context_tokens': 2048,
    'max_direct': 8,
    'max_raw_spans': 128,
    'protected_direct': 0,
    'user_completion_atoms': 8,
})


def retrieval_limits(memory, overrides):
    """Resolve the same serving budgets for direct and published-snapshot IO."""
    from memory_condense.search.native_spine_user_completion import NativeSpineUserCompletionRouter
    limits = dict(CAP8_LIMITS) if isinstance(memory.router, NativeSpineUserCompletionRouter) else {}
    if memory.tactic_classifier is not None and overrides.get('augment', True):
        # Choose the hydration budget only after the classifier picks a tactic.
        limits.pop('max_context_tokens', None)
        limits.pop('max_raw_spans', None)
    elif memory.dialogue is not None and overrides.get('augment', True):
        policy = memory.dialogue.policy
        limits.update(max_context_tokens=policy.max_context_tokens, max_raw_spans=policy.max_raw_spans)
    return {**limits, **overrides}
