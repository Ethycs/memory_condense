"""Private single-attempt close binding for externally owned condensers."""

from __future__ import annotations

from threading import RLock
from typing import Callable
import weakref


def install_owned_close(expected_type: type, legacy_close: Callable[[object], None]):
    """Return a binder and exact close wrapper over closure-owned state."""

    reference = weakref.ref
    finalize = weakref.finalize
    detach = weakref.finalize.detach
    lock_factory = RLock
    registry: dict[
        int,
        tuple[
            weakref.ReferenceType[object],
            Callable[[], None],
            weakref.finalize,
            RLock,
        ],
    ] = {}
    closed: dict[int, weakref.ReferenceType[object]] = {}
    lock = lock_factory()

    def bind(
        condenser: object,
        close_op: Callable[[], None],
        abandon_op: Callable[[], None],
    ) -> None:
        if (
            type(condenser) is not expected_type
            or not callable(close_op)
            or not callable(abandon_op)
        ):
            raise TypeError("owned close requires an exact condenser and callback")
        key = id(condenser)

        def abandoned() -> None:
            with lock:
                entry = registry.pop(key, None)
            if entry is not None:
                try:
                    abandon_op()
                except BaseException:
                    pass

        finalizer = finalize(condenser, abandoned)
        created = (reference(condenser), close_op, finalizer, lock_factory())
        with lock:
            if key in registry or (key in closed and closed[key]() is condenser):
                detach(finalizer)
                raise TypeError("condenser already has an owned close binding")
            registry[key] = created

    def dispatch(condenser: object) -> bool:
        with lock:
            entry = registry.get(id(condenser))
            prior = closed.get(id(condenser))
        if entry is None:
            if prior is not None and prior() is condenser:
                return True
            return False
        if entry[0]() is not condenser:
            raise TypeError("owned condenser close binding was forged")
        key = id(condenser)
        with entry[3]:
            with lock:
                current = registry.get(key)
                prior = closed.get(key)
            if current is None:
                if prior is not None and prior() is condenser:
                    return True
                raise TypeError("owned condenser close changed during dispatch")
            if current is not entry or current[0]() is not condenser:
                raise TypeError("owned condenser close changed during dispatch")
            try:
                entry[1]()
            finally:
                with lock:
                    if registry.get(key) is entry:
                        del registry[key]

                        def release_closed(_reference) -> None:
                            with lock:
                                if closed.get(key) is _reference:
                                    del closed[key]

                        closed[key] = reference(condenser, release_closed)
                detach(entry[2])
        return True

    def close(self) -> None:
        """Close a normal facade or its exact registry-owned lifecycle."""

        if not dispatch(self):
            legacy_close(self)

    close.__qualname__ = "MemoryCondenser.close"
    close.__module__ = expected_type.__module__
    return bind, close


__all__ = ["install_owned_close"]
