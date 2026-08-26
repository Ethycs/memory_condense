"""Associative memory: consolidation, expansion arms, and Qwen head features.

This package holds everything that relates one piece of retrieved evidence to
another.  It is read by ``application`` (context building) and ``search``
(episode signals, reranking); it never imports from them.

Storage
    ``association_store`` is the import seam for durable state — a facade over
    ``association_repository`` (the ``AssociationStore`` class), which composes
    three mixins that share its connection and caches: ``association_artifacts``
    (CAV coordinate systems and signatures), ``association_edges`` (sparse
    head-attention edges), and ``hebbian_store`` (co-retrieval counts).  Import
    from the facade, not the modules behind it.  Value objects live in
    ``association_models``; the decay/scoring math shared by the co-access
    stores lives in ``coaccess_graph``.

Expansion arms
    Three arms grow a retrieved set, all sharing the admission machinery in
    ``expansion_guards`` (budget guard, rollback, protected anchors) and the
    slot composition in ``associative_composition``:

    - ``consolidation`` — the production arm.  Causal co-access edges learned
      from what was retrieved together, expanded by
      ``expand_context_associations``.  Reached from ``MemoryCondenser``.
    - ``heat_diffusion`` — source-level heat spreading.  Exercised by
      ``tooling/experiment_rig`` sweeps, not by the production path.
    - ``hebbian_retrieval`` / ``hebbian_store`` — Hebbian co-retrieval.  Active
      research surface for the fast-Hebbian work in ``eval``.

    ``transition_policy`` scores destination transitions for offline replay
    (``eval/transition_replay``).

Qwen head features
    ``qwen_memory_linker`` is production-live, but as a *feature extractor*
    rather than a memory: ``search`` calls it for bounded scalars and CAV
    coordinates that closure and coverage retrieval consume.  ``cav_memory``
    holds the CAV bank, ``head_memory_models`` the shared value objects.

    ``qwen_live_memory``, ``head_kv_store``, ``head_association_graph``, and
    ``head_memory_cli`` implement the live head-memory experiment, reachable
    only through the ``qwen-head-*-smoke`` pixi tasks (they need a local
    Qwen3-8B checkpoint).  ``head_memory`` is their ``python -m`` entry point
    and a re-export facade.

Reading order
    Numeric middles are kept as pure functions over plain inputs so they can be
    read and tested without a store; SQLite connections and loaded torch models
    stay at the edges.  Ranking primitives shared with the rest of the system
    (weighted-fair ordering, round-robin, softmax) live in ``domain/ranking``,
    not here.
"""
