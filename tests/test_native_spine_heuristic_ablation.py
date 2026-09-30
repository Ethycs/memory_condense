from copy import deepcopy

import pytest

from memory_condense.application.native_spine_context_retrieval import ResidentNativeSpineContextMemory
from tests.test_native_spine_context_routing import parent_fixture
from tests.test_native_spine_routing import DATED, QUERY
from tools.evaluate_native_spine_heuristic_ablation import ablated_policy, validate_ablation
from tools.native_spine_context_policy import DENSE_PARENT_2048


def test_dense_ablation_preserves_seed_order_and_hydration_budget():
    history, semantic, hierarchy, encoder = parent_fixture()
    memory = ResidentNativeSpineContextMemory(semantic, hierarchy, encoder=encoder,
        load_turn=history.get_turn)
    original = {**DENSE_PARENT_2048, 'max_direct': 8}
    baseline = memory.retrieve(QUERY, DATED, **original)
    policy = ablated_policy(original)
    result = memory.retrieve(QUERY, DATED, **policy)
    validate_ablation(result.routing.identity_payload(), baseline.routing.identity_payload())
    assert result.routing.expanded.routes == result.routing.baseline.routes
    assert not result.routing.consulted_chunk_ids
    assert result.hydration.max_context_tokens == baseline.hydration.max_context_tokens == 2048
    assert result.hydration.max_raw_spans == baseline.hydration.max_raw_spans == 128


def test_ablation_rejects_changed_seed_order_and_remaining_additions():
    history, semantic, hierarchy, encoder = parent_fixture()
    memory = ResidentNativeSpineContextMemory(semantic, hierarchy, encoder=encoder,
        load_turn=history.get_turn)
    result = memory.retrieve(QUERY, DATED, **ablated_policy({**DENSE_PARENT_2048, 'max_direct': 8}))
    payload = result.routing.identity_payload()
    changed = deepcopy(payload)
    changed['baseline']['routes'].reverse()
    with pytest.raises(ValueError, match='seed selection'):
        validate_ablation(payload, changed)
    changed = deepcopy(payload)
    changed['consulted_chunk_ids'] = ['unexpected-context']
    with pytest.raises(ValueError, match='retained an addition'):
        validate_ablation(changed, payload)
