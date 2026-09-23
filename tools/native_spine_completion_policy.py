"""Bounded user-completion policy on top of the sealed dense-parent limits.

The numeric context policy gains one question-independent key,
``user_completion_atoms``: the maximum number of remaining user atoms appended
from already routed conversations. Every other limit, the exact hydrator, the
user-evidence projection and the reader are unchanged.
"""
from types import SimpleNamespace

from memory_condense.application import user_evidence_projection as renderer
from memory_condense.search.native_spine_user_completion import (
    FORMAT, NativeSpineUserCompletionRoute, route_from_payload,
)
from tools import native_spine_context_policy as context_policy
from tools.native_spine_threaded_presentation import hydration_from_payload, source_order


KEY = 'user_completion_atoms'
BOUNDS = (0, 64)
ROUTING_STRATEGY = FORMAT


def validate_policy(policy):
    if (type(policy) is not dict or KEY not in policy or type(policy[KEY]) is not int
            or not BOUNDS[0] <= policy[KEY] <= BOUNDS[1]):
        raise ValueError('user completion policy needs one bounded integer atom limit')
    base = context_policy.validate_policy({k: v for k, v in policy.items() if k != KEY})
    return {**base, KEY: policy[KEY]}


def base_policy(policy):
    return {k: v for k, v in validate_policy(policy).items() if k != KEY}


def build(memory, question, policy, order):
    result = memory.retrieve(question['retrieval_query'], question['prompt_question'],
                             **validate_policy(policy))
    rendered = renderer.render_user_spine_sections(result.hydration, order)
    messages = context_policy.messages(question, SimpleNamespace(render_context=lambda: rendered.text))
    return (messages, result.hydration.identity_payload(), result.routing.identity_payload(),
            rendered.identity_payload())


def verify_packet(question, hydration, routing, rendered, sessions, load_body, policy, order):
    """Reconstruct exact source excerpts and the projection under the sealed policy."""
    p = validate_policy(policy)
    route = route_from_payload(routing)
    if (type(route) is not NativeSpineUserCompletionRoute or route.user_completion_atoms != p[KEY]
            or len(route.completion_added_atomic_ids) > p[KEY]):
        raise ValueError('audited packet violates its sealed user completion limit')
    _, count = context_policy.verify_packet(question, hydration, routing, sessions, load_body, base_policy(p))
    rebuilt = renderer.render_user_spine_sections(hydration_from_payload(hydration), order)
    if rebuilt.identity_payload() != rendered:
        raise ValueError('served projection differs from exact whole-section source reconstruction')
    return context_policy.messages(question, SimpleNamespace(render_context=lambda: rebuilt.text)), count


__all__ = ['KEY', 'BOUNDS', 'ROUTING_STRATEGY', 'validate_policy', 'base_policy', 'build',
           'verify_packet', 'source_order']
