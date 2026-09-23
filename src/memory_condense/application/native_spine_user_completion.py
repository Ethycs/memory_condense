"""Reuse the persisted application and parent index with user completion routing."""
from memory_condense.application.native_spine_parent_users import ParentUserMemoryCondenser
from memory_condense.search.native_spine_user_completion import NativeSpineUserCompletionRouter


class UserCompletionMemoryCondenser(ParentUserMemoryCondenser):
    def _load_native_spine(self):
        loaded = super()._load_native_spine()
        snapshot, memory = loaded[1:]
        if not isinstance(memory.router, NativeSpineUserCompletionRouter):
            memory.router = NativeSpineUserCompletionRouter(snapshot.semantic, snapshot.hierarchy,
                                                            memory.router.parent_semantic)
        return loaded
