"""User statements first within each conversation; unchanged exact hydration."""
from types import SimpleNamespace

from memory_condense.application import user_spine_section_context_v2 as renderer
from memory_condense.search.native_spine_parent_user_routing import route_from_payload
from tools import native_spine_context_policy as context_policy
from tools.native_spine_threaded_presentation import hydration_from_payload, source_order


def build(memory, question, policy, order):
    result = memory.retrieve(question['retrieval_query'], question['prompt_question'],
                             **context_policy.validate_policy(policy))
    rendered = renderer.render_user_spine_sections(result.hydration, order)
    messages = context_policy.messages(question, SimpleNamespace(render_context=lambda: rendered.text))
    return (messages, result.hydration.identity_payload(), result.routing.identity_payload(),
            rendered.identity_payload())


def verify_packet(question, hydration, routing, rendered, sessions, load_body, policy, order):
    route_from_payload(routing)
    _, count = context_policy.verify_packet(question, hydration, routing, sessions, load_body, policy)
    rebuilt = renderer.render_user_spine_sections(hydration_from_payload(hydration), order)
    if rebuilt.identity_payload() != rendered:
        raise ValueError('served user-spine layout differs from exact source reconstruction')
    return context_policy.messages(question, SimpleNamespace(render_context=lambda: rebuilt.text)), count
