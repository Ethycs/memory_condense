"""Package-boundary invariants for the context-packing decomposition."""

from __future__ import annotations

import memory_condense.search.packing.context_budget as budget_module
import memory_condense.search.packing.context_packer as facade
import memory_condense.search.packing.packing_contracts as contracts_module
import memory_condense.search.packing.source_provenance as provenance_module


def test_context_packer_facade_reexports_canonical_contract_objects() -> None:
    assert facade.ContextBudget is budget_module.ContextBudget
    assert facade.ExpansionSelector is contracts_module.ExpansionSelector
    assert (
        facade.is_source_metadata_text
        is provenance_module.is_source_metadata_text
    )


def test_context_budget_default_remains_patchable_at_the_facade(monkeypatch) -> None:
    replacement = budget_module.ContextBudget(expansion_tokens=123)
    monkeypatch.setattr(facade, "ContextBudget", lambda: replacement)

    assert facade.ContextPacker().budget is replacement


def test_metadata_predicate_remains_patchable_at_the_facade(monkeypatch) -> None:
    captured = {}

    def sentinel(_text):
        return True

    def fake_bind(selected, **kwargs):
        captured.update(kwargs)
        return {}, list(selected)

    monkeypatch.setattr(facade, "is_source_metadata_text", sentinel)
    monkeypatch.setattr(facade, "bind_source_metadata", fake_bind)

    facade.ContextPacker()._bind_source_metadata([])

    assert captured["metadata_predicate"] is sentinel
