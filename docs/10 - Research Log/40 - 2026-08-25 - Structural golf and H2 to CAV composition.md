# Structural golf and H2-to-CAV composition

**Status:** the missing provider-free H2-to-CAV seam now works without a new
evaluation framework. Exact append-only H2 evidence is bound into a transient,
digest-addressed S3 compatibility view; the existing genuine CAV router then
creates its rectangular extraction/reinjection links, and the existing matched
synthesis builder performs the authoritative final 8,000-token-proxy preflight.
This is an integration result, not a new 1M or locked-100 accuracy result.

## Compact execution path

```text
sealed S3 retrieval
  -> Hebbian H2 append receipt + hash-bound transient evidence
  -> transient H2 overlay (source stage remains S3; layer is H2)
  -> genuine CAV links (membership preserving)
  -> linked/unlinked synthesis prompts
  -> actual final 8k prompt preflight
```

The implementation is isolated in
`src/memory_condense/eval/fast_hebbian_h2_cav.py`. It does not modify the
existing H2 builder, CAV feature session, CAV link receipts, or synthesis
engine. The bridge:

1. requires the H2 population to name the exact source retrieval digest;
2. rechecks question order, question hashes, source S3 receipt/projection, and
   the exact ordered S3 prefix;
3. replaces only the transient final S3 evidence view with H2's append-only
   result and binds the H2 question receipt as its parent;
4. rebuilds the per-question feature table by first-seen text, preserving
   distinct evidence IDs that share text;
5. derives a domain-separated overlay digest from the source retrieval, H2
   population, H2 question receipts, evidence projections, and prompt seals;
6. runs the existing CAV feature/link and matched synthesis implementations.

The overlay is deliberately in-memory and uses a `transient:` source label. It
must not be published or represented as a replacement retrieval JSON. Persisted
H2 receipts omit evidence text by design, so a real combined run must rebuild
the typed H2 population from the sealed retrieval/history/derived-store inputs.

## What the integration test proves

The focused test uses the real miniature H2 store/history path and the existing
deterministic CAV encoder/router doubles. It proves that:

- H2 preserves exact S3 evidence as a prefix and appends one robust neighbor;
- the H2 receipt becomes the exact parent of the CAV feature and synthesis
  receipts;
- ordered `(evidence_id, source_id, text_sha256)` coordinates agree across H2,
  CAV feature output, genuine CAV links, synthesis aliases, and both prompts;
- the appended evidence occurs exactly once in each final prompt;
- the encoder is called once;
- provider calls remain zero;
- feature, link, synthesis, and composed receipts retain zero token IDs,
  tensors, or persisted transformer token state;
- both actual linked/unlinked prompts remain within the 8,000-token-proxy cap.

This last check is important. H2 admission uses the immutable guide-slot
sentinel. The rendered CAV guide is larger. The composed path therefore treats
the existing synthesis preflight, not the earlier sentinel estimate, as the
final budget authority and fails closed on overflow.

## Literal code golf

Seven equivalent strict canonical-JSON serializers were also replaced with one
leaf helper, `src/memory_condense/eval/_artifact_json.py`. Existing local
function names remain import aliases and the encoded bytes remain sorted,
compact, finite-only UTF-8 with one trailing newline. This removes roughly 48
net source lines and one class of serializer drift. Four superficially similar
legacy encoders were left alone because they currently differ on non-finite
number handling.

## Verification

The affected artifact, runner, H2, CAV, provider-runtime, architecture, and
cumulative-invariant suites completed with:

```text
373 passed, 7 skipped in 83.91s
```

The skips are existing conditional tests. No locked evaluation artifact,
provider journal, corpus store, or published H2/CAV artifact was rewritten.

## Remaining boundary

The bridge currently consumes the development `FastRetrievalArtifact` shape.
The locked 100-question campaign still uses question-local shard/store binding,
and only 70 retrieval questions are sealed. No answer provider, semantic judge,
or official Mem0 arm was run here. The next structural step is a small locked
question adapter into this same bridge, followed by lazy final-stage-only CAV
routing; it should not become another campaign framework.
