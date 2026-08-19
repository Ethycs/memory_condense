# Code package layout

**Status**: CURRENT
**Date**: 2026-08-18
**Applies to**: `src/memory_condense` after the flat-module reorganization

The implementation is grouped by responsibility instead of keeping every
module at the package root. The move changes ownership and import paths, not
the intended runtime behavior. A second, separately tested pass decomposes the
largest modules by the shape of their work: durable objects and contracts,
pure or stateless transformations, and stateful workflows or adapters.

## Package map

| Package | Responsibility |
| --- | --- |
| `domain` | Shared schemas, token counting, decay, ranking, and immutable discourse/closure contracts |
| `persistence` | SQLite access plus transcript, memory, and source-grounded discourse stores |
| `ingest` | Loading, corpus composition, exact-slice chunking, extraction, validation, and conservative discourse linking |
| `modeling` | Embedding and local Qwen prefix runtimes |
| `associations` | Persistent association artifacts, Hebbian and heat reads, consolidation, and transition policy |
| `search.indexes` | Lexical, vector/hybrid, source-local, and hierarchy retrieval |
| `search.packing` | Context budgets, scalar handling, and event-aware packet construction |
| `search.selectors` | Coverage, forced-choice, cross-encoder, and Qwen reranking policies |
| `search.episodes` | Pluggable event boundaries, bounded refinement, representatives, and source-local temporal expansion |
| `search.closure` | Query obligations, evidenced relation traversal, revision/conflict semantics, and atomic closure plans |
| `application` | `MemoryCondenser` composition root and the hosted-provider binding seam |
| `interfaces` | Long-lived external interfaces, currently the MCP server |
| `tooling` | Probes, experiment rigs, and Qwen smoke/consolidation commands |
| `eval` | The existing validation harness, cache protocol, campaign logic, metrics, and adapters |

The corresponding source tree is:

```text
memory_condense/
├── domain/
├── persistence/
├── ingest/
├── modeling/
├── associations/
├── search/
│   ├── closure/
│   ├── episodes/
│   ├── indexes/
│   ├── packing/
│   └── selectors/
├── application/
├── interfaces/
├── tooling/
└── eval/
```

These groups communicate responsibility. They are not a claim that every
dependency already follows a perfectly strict layer DAG. In particular,
validation-sensitive `eval` code remains in its existing cohesive namespace;
nesting it further would be a separate refactor with its own evidence gate.

## Objects, transformations, and workflows

Responsibility packages answer *where a capability belongs*. Inside a
capability, the preferred dependency direction is:

```text
immutable objects / protocols
  -> pure or stateless transformations
  -> stateful workflows / adapters
  -> thin public facade or composition root
```

- **Objects and contracts** describe facts crossing a boundary: domain
  records, query programs, reports, receipts, budgets, protocols, and frozen
  configuration. Prefer immutable dataclasses or existing validated schemas.
- **Transformations** compile, normalize, score, reduce, validate, order, or
  pack those values. They should receive their dependencies explicitly and
  avoid hidden writes.
- **Workflows and adapters** own real state or effects: SQLite transactions,
  ANN indexes, model runtimes, provider transports, caches, and command-line
  orchestration.
- **Facades** preserve the supported import and monkeypatch surfaces while
  delegating to the focused implementation modules. A facade is not a second
  implementation.

This is not a mandate to turn every function into a class. A class is useful
when it owns an invariant, resource lifetime, or coherent operation state.
Otherwise, a typed value plus a pure function is clearer. Likewise, mixins are
used only to divide one stateful composition root such as `MemoryCondenser` or
`SimilarityRetriever`; extracted workflow modules must not import the concrete
facade back and create a cycle.

Examples in the current tree include:

- association models -> artifact/edge/Hebbian transformations -> repository;
- set program and coverage reports -> evidence/admission/scoring/reservation
  transformations -> selector facade;
- context budget and closure contracts -> ordering/assembly -> context packer;
- query routing and active-partition receipts -> ingest/retrieval/graph/source
  workflows -> `MemoryCondenser`; and
- CLI configuration and policy contracts -> runtime controls and run modes ->
  the executable evaluation facade.

The compatibility surfaces are now intentionally small:

| Capability | Facade lines | Largest focused implementation |
| --- | ---: | ---: |
| Association store | 30 | 503 |
| Head memory | 69 | 676 |
| Context packing | 228 | 736 |
| Coverage selection | 64 | 914 |
| `MemoryCondenser` | 680 | 1,176 |
| Evaluation CLI | 245 | 849 |
| Mem0 adapter | 123 | 818 |
| Campaign merge | 146 | 793 |
| Recall evaluation | 34 | 690 |
| Similarity retrieval | 59 | 594 |

The sizes are regression tripwires rather than quality scores. The material
change is that each file now owns one kind of invariant and the former
1,500--4,400-line orchestration units no longer mix records, normalization,
storage, scoring, packing, and command dispatch in one namespace.

## Import contract

The package root is the supported object-level facade:

```python
from memory_condense import ContextBudget, MemoryCondenser, MemoryItem
```

Its 56 exported objects are loaded lazily and remain identical to their
canonical definitions. Internal code, tests, and advanced integrations use
canonical leaf paths:

```python
from memory_condense.application.condenser import MemoryCondenser
from memory_condense.persistence.db import Database
from memory_condense.search.indexes.retrieval import SimilarityRetriever
```

The old flat implementation paths are not compatibility shims. For example,
`memory_condense.condenser` has become
`memory_condense.application.condenser`. This keeps the package root visibly
organized instead of replacing each moved implementation with another flat
proxy file. Package `__init__.py` files stay minimal except for the lazy public
facade at the root.

Canonical executable entrypoints include:

```powershell
python -m memory_condense.eval
python -m memory_condense.interfaces.mcp_server
python -m memory_condense.modeling.qwen_prefix --help
```

Prefer the named Pixi tasks (`mcp`, `qwen-smoke`, and the probe tasks) when
working in this repository.

## Evaluation identity boundary

The reorganization begins a new **implementation epoch v4**. This label is an
evaluation/provenance epoch and is unrelated to SQLite schema version 11.

`implementation_sha256()` hashes every Python file's package-relative path as
well as its bytes. Moving a byte-identical implementation therefore changes
the digest by design. The frozen validation-v3 treatment remains bound to:

- Git commit `bfa5b6daf6a5e61881ac10f0555e5d9972f9e1c2`
- implementation SHA-256
  `452be3bfa7524bb81676c7abcb032529a32a480311d24d1e17f8513c783ecd83`

The reorganized and decomposed tree's implementation SHA-256 is
`1208e2619194aaf1832e15d59a36e05b4cc08838c95ed47f1872aa37a76dea8b`.
Old v3 policies and prepared cache receipts certify only the frozen v3 bytes;
they must never be relabeled as v4 evidence. A scored v4 campaign requires a
new policy identity and newly prepared/re-attested caches.

Dated analyses, research logs, and frozen experiment scripts retain their
historical module paths. They document the code that produced their artifacts
and are not active import examples.

## Verification

The reorganization gate checks:

- every first-party import resolves;
- the root facade exports exactly the same 56 canonical objects;
- importing the root does not import hosted-model SDKs or executable modules;
- all configured module entrypoints parse;
- architecture, application, evaluation, and provider-free Mem0 tests pass;
- compatibility facades remain small and resolve to canonical objects;
- no source module regrows into an unreviewable multi-thousand-line unit;
- the complete repository suite passes; and
- `git diff --check` is clean.

The implementation hash above must be recomputed if any Python source changes
after this document is updated.

Current verification evidence: 138 package modules import successfully; the
complete non-model suite passes with 1,475 tests, 13 explicitly slow/model
tests deselected, and one pre-existing `pydantic-settings` warning. The source
tree also enforces a 1,300-line review ceiling and tighter per-facade limits in
`tests/test_architecture.py`.
