"""Opt-in parent-context packet using the existing exact hydration boundary."""
from memory_condense.application.native_spine_retrieval import ResidentNativeSpineMemory
from memory_condense.search.native_spine_context_routing import NativeSpineContextRouter


class ResidentNativeSpineContextMemory(ResidentNativeSpineMemory):
    def __init__(self, atomic_semantic, hierarchy, *, encoder, load_turn):
        super().__init__(atomic_semantic, hierarchy, encoder=encoder, load_turn=load_turn)
        self.router = NativeSpineContextRouter(atomic_semantic, hierarchy)
