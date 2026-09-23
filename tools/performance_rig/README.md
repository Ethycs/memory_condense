# Memory Condense performance rig

This rig keeps heavy stores and run output outside the repository. Its runtime
workspace is `C:\Users\Keytone\Downloads\memory-condense-rig`.

## Ingest throughput

`ingest_throughput.py` times the new two-step write path without conflating its
boundaries. `capture_many()` establishes durable turn/manifests without model
work; bounded `drain_pending_ingests()` calls make those captures searchable.
The default deterministic fake embedder is suitable for a quick CI smoke:

```powershell
pixi run --frozen -e dev python tools/performance_rig/ingest_throughput.py `
  --embedder fake --turns 64 --tokens-per-turn 500 `
  --capture-batch-size 2 --drain-max-manifests 16 `
  --offered-generation-tokens-per-second 100
```

The opt-in local BGE-M3 assay uses the same report schema:

```powershell
pixi run --frozen -e dev python tools/performance_rig/ingest_throughput.py `
  --embedder real --device cuda --turns 64 --tokens-per-turn 500 `
  --output-json C:\Users\Keytone\Downloads\memory-condense-rig\ingest-speed.json
```

Use `--enforce-sla` only on a controlled performance host. Ordinary CI should
verify the metric boundaries and deterministic fake behavior, not fail on
shared-runner wall-clock noise.

This is deliberately a core capture-first finite-burst assay, not an
asynchronous proxy load generator or a proxy for sustained load. It reports
each capture and drain batch's p50/p95/maximum latency. The SLA gates the ratio
of end-to-end T0-to-T1 chunk-token throughput to the configured offered
generation rate; the faster drain-only ratio is diagnostic. The
generation-headroom gate requires at least 2x service capacity. Proxy admission
loss is explicitly unmeasured.

The public dense verification executes exactly one query embedding after T1
has completed; the lexical verification is model-free. The report separates
that post-T1 query call from ingest embedding calls, so T0's zero-model-call
invariant remains directly auditable.

The two stages are deliberately asymmetric:

1. `start-untouched-compile.ps1` starts one hidden Qwen/embedding pipeline,
   writes logs to disk, persists the unpruned compact association store, and
   freezes a hashed top-10 hybrid anchor pack. A VS Code restart does not own
   its stdout pipe and therefore does not strand the process.
2. `start-sweep.ps1` loads no Qwen and no embedding model. Independent arms
   open separate SQLite readers and evaluate in parallel over the same frozen
   anchors with `touch=False`. Arms with `prune_max_neighbors` use independent
   SQLite backup copies, so they measure physical pruning without modifying the
   prepared source store.

Start the untouched compile:

```powershell
& 'C:\Users\Keytone\Downloads\memory-condense-rig\start-untouched-compile.ps1'
```

Check it using the run directory printed by the launcher:

```powershell
& 'C:\Users\Keytone\Downloads\memory-condense-rig\get-status.ps1' -Directory '<run directory>'
```

After `compile_report.json` and `anchor_pack.json` exist, start the default
eight-arm/five-repeat sweep:

```powershell
& 'C:\Users\Keytone\Downloads\memory-condense-rig\start-sweep.ps1' -RunDir '<run directory>' -Workers 4
```

Only one Qwen compiler should run per 8 GB GPU. Multiple CPU-only sweeps may
run concurrently, but increasing workers beyond available cores or storage
bandwidth can make wall time worse; the report records worker count and per-arm
timing samples.

## Locked confirmation protocol

`start-confirmation.ps1` excludes every source family consumed by development
or the v1 sweep and locks a new six-family selection seed before compilation:

```powershell
& 'C:\Users\Keytone\Downloads\memory-condense-rig\start-confirmation.ps1'
```

After it completes, run `start-sweep.ps1` with
`configs\confirmation-arms.json`. The six arms are fixed: hybrid, CAV-only,
the selected two-hop QK arm, a QK+CAV ablation, and physical degree-2 and
degree-1 pruning copies. Do not add or tune an arm after inspecting the
confirmation questions.

After v2 exposed unconditional slot replacement as harmful, the safe policy
added two admission invariants: protect a reserved tail anchor at normalized
BM25 `>= 0.90`, and roll back the whole composed result if it increases prompt
tokens. `start-confirmation-v3.ps1` excludes every family consumed through v2;
`configs\confirmation-v3-arms.json` was locked before that selection and is
the confirmatory test of the safe policy.
