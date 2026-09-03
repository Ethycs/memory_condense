# Confirmation production runtime

`tools/confirmation_production_runtime.py` closes the production-construction
gap between the tested confirmation adapters and the local BGE/Qwen models.
It reconstructs the frozen S0--S3 retrieval controls directly from attested
code. It does not read the validation policy or any validation artifact at
confirmation runtime.

The factory returns one owned object graph:

1. a direct dense source-ingest config and `ProductionBaseStoreBackend`;
2. the causal-graph cumulative config and fixed-interval compiler;
3. a staged preparation backend sharing the same BGE instance;
4. a Qwen factory that cannot load before the sealed BGE release barrier; and
5. a staged retrieval factory with the frozen episode, representative,
   closure, prompt, and source-router budgets.

The full resolved config, retrieval projection, derived source config, and
derived source retrieval projection each have a fixed SHA-256. Any schema
default drift therefore fails before a model or store is opened. The low-level
runtime factory constructs its shared BGE binding eagerly, so the production
phase environment owns that factory lazily: merely opening or resuming the
prediction pipeline loads no model. The first load occurs only when namespace
ingest or staged cumulative retrieval asks for the initial runtime.

The fixed residency order is:

```text
source stores under BGE
  -> combined stores + frozen query vectors under the same BGE
  -> sealed BGE release barrier
  -> Qwen coverage/representative retrieval
  -> Qwen close
```

After staged retrieval seals its release barrier, the initial runtime cannot
be resurrected. Query expansion later opens one fresh confirmation retriever
session, loads one shared BGE embedder, opens at most one namespace index at a
time, seals its ownership audit, and closes both indexes and BGE before the
terminal/provider phases. Status checks and artifact-only checkpoint replay do
not construct either lifecycle.

The content-addressed confirmation source coordinates deliberately preserve
transcript, timestamp, chunking, embedding, and retrieval semantics without
claiming byte identity with validation databases whose legacy IDs were
position-derived.
