# Confirmation Executor Readiness Attestation

`tools/attest_confirmation_executor_v2.py` is the last provider-free boundary
before the sanitized confirmation treatment may be opened. It upgrades the
historical v1 boundary attestation only after the complete executor exists and
its fixed offline suite passes from a clean committed tree.

The v2 artifact binds:

- the exact policy-v5-r3 freeze and the executor Git commit/tree;
- the recursive local-Python import closure of the production apparatus;
- every apparatus-file digest and their ordered file-set digest;
- the dependency-lock inventory and digest;
- the production entrypoint bytes;
- a sealed offline-test receipt for the same commit/tree and fixed test list;
- zero provider calls and zero Terra/Sol authorization.

Its release is intentionally asymmetric. It permits the prediction executor
to open the sanitized treatment, but it does not permit opening reference
answers or calling a model. Later stage-local preflight/release artifacts own
those authorizations and preserve exact remaining-call accounting with zero
retries.

The attester separately computes reachability from the single production
prediction entrypoint. This is intentionally narrower than the apparatus
inventory: the standalone treatment exporter and the post-prediction judge
are hashed as apparatus, but neither is a prediction root.

Readiness binds the production code and import closure. After the sanitized
runtime policy and model paths are opened, the immutable run manifest makes
that binding concrete by recording the exact ordered production-adapter
identity SHA-256 for every phase. Checkpoint loading, resume/status, handoff
publication, and evaluator handoff verification all require those identities.

The prediction firebreak follows ordinary and literal dynamic local imports,
including every parent-package `__init__.py` that Python executes. It fails
closed on unresolved dynamic imports (apart from a proved inert, closed-map
package `__getattr__` facade), judge/gold modules, benchmark and locked-split
loaders, the raw-population verifier, and imported dataset/gold/reference/judge
loader callables. Consequently, a sanitized projection is insufficient if its
prediction import closure still retains a callable route to evaluator data.
The judge remains a separate post-prediction program.

After committing the complete apparatus, run the fixed suite and publish the
readiness artifact into an ignored result directory:

```powershell
.pixi\envs\dev\python.exe -m tools.attest_confirmation_executor_v2 test-receipt `
  --output eval_results/confirmation-policy-v5-r3/offline-tests.json

.pixi\envs\dev\python.exe -m tools.attest_confirmation_executor_v2 attest `
  --offline-test-receipt eval_results/confirmation-policy-v5-r3/offline-tests.json `
  --output eval_results/confirmation-policy-v5-r3/executor-readiness-v2.json
```

Both outputs are canonical, no-clobber JSON with filename-bearing SHA-256
sidecars. Any source, lock, test-list, commit, or tree change invalidates the
readiness artifact and requires a fresh test receipt and attestation.
