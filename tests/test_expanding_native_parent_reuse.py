import pytest

from tests import test_expanded_native_parent_reuse as fixture
from tests.test_native_spine_exchanges import Backend
from tools import compile_expanding_native_spine_hierarchy as expanding
from tools import compile_recovered_native_spine_hierarchy as recovered


def test_parent_reuse_traverses_original_recovered_and_expanding_ancestry(tmp_path, monkeypatch):
    original = fixture.source_cache(tmp_path / "original", Backend())
    backend = Backend(fail=True)
    expected, ancestors = expanding.reusable_parents([original], backend, "source-corpus", "fixed-method")
    monkeypatch.setattr(fixture, "expanded", recovered)
    recovered_root = fixture.source_cache(tmp_path / "recovered", backend, ancestors=ancestors)
    values, recovered_links = expanding.reusable_parents([recovered_root], backend, "source-corpus", "fixed-method")
    assert values == expected
    monkeypatch.setattr(fixture, "expanded", expanding)
    expanded_root = fixture.source_cache(tmp_path / "expanded", backend, ancestors=recovered_links)
    frozen = {path: path.read_bytes() for path in tmp_path.rglob("*.json")}
    final, links = expanding.reusable_parents([expanded_root], backend, "source-corpus", "fixed-method")
    assert final == expected and backend.calls == 0
    assert links[0]["accepted_merge_count"] == 1
    assert not list((expanded_root / "responses").glob("*.json"))
    assert all(path.read_bytes() == data for path, data in frozen.items())


@pytest.mark.parametrize("producer", [recovered, expanding])
def test_changed_ancestor_receipt_is_rejected_across_each_supported_lineage(tmp_path, monkeypatch, producer):
    original = fixture.source_cache(tmp_path / "original", Backend())
    backend = Backend(fail=True)
    _, ancestors = expanding.reusable_parents([original], backend, "source-corpus", "fixed-method")
    monkeypatch.setattr(fixture, "expanded", producer)
    changed = [dict(ancestors[0], merge_cache_sha256="changed")]
    source = fixture.source_cache(tmp_path / "changed", backend, ancestors=changed)
    with pytest.raises(ValueError, match="ancestry changed"):
        expanding.reusable_parents([source], backend, "source-corpus", "fixed-method")
    assert backend.calls == 0


@pytest.mark.parametrize("defect", ["method", "corpus", "backend", "duplicate", "cycle", "response"])
def test_incompatible_or_corrupt_parent_cache_cannot_trigger_generation(tmp_path, defect):
    source = fixture.source_cache(tmp_path / "source", Backend())
    backend = Backend(fail=True)
    if defect == "backend":
        backend.identity_sha256 = "changed"
    if defect == "response":
        response = next((source / "responses").glob("*.json"))
        response.write_bytes(response.read_bytes() + b" ")
    roots = [source, source] if defect == "duplicate" else [source]
    chain = (source.resolve(),) if defect == "cycle" else ()
    with pytest.raises(ValueError):
        expanding.reusable_parents(roots, backend, "changed" if defect == "corpus" else "source-corpus",
                                   "changed" if defect == "method" else "fixed-method", chain=chain)
    assert backend.calls == 0
