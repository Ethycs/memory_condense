import numpy as np
import pytest

from memory_condense.search.section_routing import SectionSummaryIndex
from memory_condense.search.user_spine_addresses import UserSpineAddressIndex, user_spine_text
from tests.test_user_spine_hierarchy import make_exchanges


def fixture():
    _, _, _, exchanges = make_exchanges()
    hierarchy = SectionSummaryIndex(tuple(e.section for e in exchanges[:2]))
    combined = np.array([[1, 0], [0, 1]], dtype=np.float32)
    spine = np.array([[0, 1], [1, 0]], dtype=np.float32)
    return hierarchy, combined, spine


def test_weighting_changes_only_addresses_and_returns_original_hydration_descriptors():
    hierarchy, combined, spine = fixture()
    index = UserSpineAddressIndex(hierarchy, combined, spine, embedding_identity="fixture")
    query = np.array([1, 0], dtype=np.float32)
    old = index.route_vector("orchard", query, embedding_identity="fixture", user_weight=0, max_sections=1)
    new = index.route_vector("orchard", query, embedding_identity="fixture", user_weight=1, max_sections=1)
    assert old.routes[0].section is hierarchy.sections[0]
    assert new.routes[0].section is hierarchy.sections[1]
    assert old.index_sha256 == new.index_sha256 == hierarchy.receipt_sha256
    spine[:] = 0  # Address matrix must own its immutable copy.
    assert index.route_vector("orchard", query, embedding_identity="fixture", user_weight=1, max_sections=1) == new
    with pytest.raises(AttributeError, match="immutable"):
        index.embedding_identity = "changed"


def test_user_projection_excludes_assistant_claims_and_raw_text():
    hierarchy, combined, spine = fixture()
    views = [user_spine_text(s) for s in hierarchy.sections]
    assert "Unowned prelude." in views and "orchard harvest" in views
    assert "machine" not in repr(views) and "RAW_CANARY" not in repr(views)
    index = UserSpineAddressIndex(hierarchy, combined, spine, embedding_identity="fixture")
    route = index.route_vector("orchard", np.array([1, 0]), embedding_identity="fixture", lexical_reserve=1, max_sections=1)
    assert "orchard harvest" in user_spine_text(route.routes[0].section)
    assert "machine suggestion" in route.routes[0].section.summary
    assert not index.route_vector("orchard", np.array([1, 0]), embedding_identity="fixture",
                                   eligible_source_ids=("absent",)).routes


@pytest.mark.parametrize("weight", [-1, 2, float("nan"), True])
def test_invalid_weights_are_rejected(weight):
    hierarchy, combined, spine = fixture()
    index = UserSpineAddressIndex(hierarchy, combined, spine, embedding_identity="fixture")
    with pytest.raises(ValueError, match="weight"):
        index.route_vector("query", np.array([1, 0]), embedding_identity="fixture", user_weight=weight)
