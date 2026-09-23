"""Reuse the persisted application with a separately persisted parent index."""
from memory_condense.application.condenser import MemoryCondenser
from memory_condense.persistence import native_spine_parent_store as store
from memory_condense.search.native_spine_parent_user_routing import NativeSpineParentUserRouter


class ParentUserMemoryCondenser(MemoryCondenser):
    def _load_native_spine(self):
        loaded = super()._load_native_spine()
        snapshot, memory = loaded[1:]
        if not isinstance(memory.router, NativeSpineParentUserRouter):
            parents, receipt = store.load(self.database_path.parent / store.FILENAME,
                hierarchy=snapshot.hierarchy, native_receipt=snapshot.receipt)
            memory.router = NativeSpineParentUserRouter(snapshot.semantic, snapshot.hierarchy, parents)
            self._native_parent_user_receipt = receipt
        return loaded

    def native_parent_user_receipt(self):
        self._load_native_spine()
        return dict(self._native_parent_user_receipt)
