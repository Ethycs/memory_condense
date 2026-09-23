"""Opt-in additive summary routing over the existing durable memory lifecycle."""
from memory_condense.application.condenser import MemoryCondenser
from memory_condense.search.native_spine_additive_lexical import NativeSpineAdditiveLexicalRouter


class AdditiveLexicalMemoryCondenser(MemoryCondenser):
    def _load_native_spine(self):
        loaded = super()._load_native_spine()
        memory = loaded[2]
        if not isinstance(memory.router, NativeSpineAdditiveLexicalRouter):
            memory.router = NativeSpineAdditiveLexicalRouter(memory.router.semantic, memory.router.hierarchy)
        return loaded
