# Lossless structural V compression probe

**Date:** 2026-10-15  
**Status:** Artifact-only result; recommends record/page Merkle storage, rejects
fine-grained de Bruijn or Merkle compression for current V bytes

## Clarified question

Can Qwen V landmark data be compressed losslessly by sharing exact structure,
using a de Bruijn-style graph or a content-addressed Merkle tree/DAG?

Lossless here means exact reconstruction of every persisted INT8 V byte and
its scale. The existing INT8 publication is itself an approximation of the
transient FP32 state; this probe introduces no additional approximation.

This question is distinct from the predictive residual experiment in research
log 275. That experiment remains a lossy research avenue but does not answer
the structural-lossless question.

## Population

The probe reused the existing 79-summary development artifact:

- 79 summaries;
- eight KV heads;
- eight V landmarks per summary;
- 128 INT8 dimensions per landmark; and
- 647,168 V payload bytes before scales and framing.

No Qwen, provider or raw-section call was made.

## Ordinary lossless baseline

The raw byte entropy was 6.807 bits per byte. General-purpose lossless codecs
were tested over multiple exact layouts.

| Layout/transform | Best exact size | Ratio to raw |
| --- | ---: | ---: |
| Original summary/head/landmark/dimension order | 548,292 bytes | 84.72% |
| Head/dimension/summary/landmark order | **533,680 bytes** | **82.46%** |
| Landmark modular delta | 560,132 bytes | 86.55% |
| Landmark XOR delta | 566,832 bytes | 87.59% |
| K-lexicographic V order plus modular delta | 562,560 bytes | 86.93% |

The best measured exact saving was 17.54%. Delta and XOR coding made the
stream less compressible.

Conditioning symbol distributions on head and dimension reduced the empirical
zero-order entropy to 6.331 bits/byte. Conditioning on head, landmark and
dimension reduced it to 5.417 bits/byte, although the latter estimate is
optimistic because each distribution saw only 79 samples. This indicates that
a trained per-coordinate ANS/FSE dictionary deserves a larger-corpus test.

## Merkle/block deduplication

One pair of the 79 summaries had identical complete K and V code arrays. A
whole-summary content address can therefore share one complete record, a 1.27%
duplicate-record rate in this small population.

After removing that duplicate summary, exact V block reuse was:

| Fixed block size | Occurrences | Unique blocks | Duplicate occurrences | Raw payload saved before references |
| ---: | ---: | ---: | ---: | ---: |
| 4 bytes | 159,744 | 159,249 | 0.310% | 1,980 bytes |
| 8 bytes | 79,872 | 79,848 | 0.030% | 192 bytes |
| 16 bytes | 39,936 | 39,933 | 0.0075% | 48 bytes |
| 32 bytes | 19,968 | 19,968 | 0% | 0 bytes |
| 64 bytes | 9,984 | 9,984 | 0% | 0 bytes |
| 128 bytes | 4,992 | 4,992 | 0% | 0 bytes |

Every fine-grained Merkle representation expanded the artifact once local
integer references were charged. With real 128-bit content references the
expansion was substantially larger. A bit-plane transpose increased exact
four-byte reuse to 7.69%, but even an optimistic compact-reference DAG was
2.23 times the raw size; at sixteen-byte leaves it was still 1.20 times raw.

The tree branching factor changes height and reference overhead. It cannot
create sharing when leaf content is unique.

## De Bruijn reuse

After removing the duplicate complete summary, exact substring reuse was:

| k-mer width | Total occurrences | Unique k-mers | Repeated occurrence fraction |
| ---: | ---: | ---: | ---: |
| 4 bytes | 624,000 | 618,916 | 0.815% |
| 8 bytes | 604,032 | 603,835 | 0.0326% |
| 16 bytes | 564,096 | 564,062 | 0.0060% |

Two-byte k-mers repeat heavily because the alphabet contains only 256 byte
values, but the graph is almost maximally branching: all 255 observed prefix
bytes branched, with a mean of 135 unique outgoing values and a maximum of
231. Encoding which edge each sequence follows therefore approaches ordinary
symbol entropy and still requires the graph and path metadata.

The V records do not have the long repeated motifs that make de Bruijn graphs
compact for genomic strings. At useful k-mer widths the graph is nearly a list
of unique edges.

## Recommended lossless design

1. Canonically serialize one complete landmark record, including checkpoint,
   layer, format and scale identity, and content-address that record.
2. Deduplicate identical whole records across conversations and snapshots.
3. Pack records into independently decompressible pages and place a Merkle DAG
   over pages for integrity, resumability, replication and copy-on-write
   versioning. Do not Merkleize individual V dimensions or small blocks.
4. Store page payloads in head/dimension-major order and apply a fast lossless
   codec such as Zstandard. The format is specified in
   [RFC 8878](https://www.rfc-editor.org/rfc/rfc8878.html).
5. On the larger corpus, train and validate per-head/per-coordinate ANS or FSE
   tables. Charge dictionary bytes and measure hot/cold random-access latency.
6. Keep raw-section hydration identities outside shared numeric blobs so that
   sharing bytes never merges provenance between memories.

A Merkle DAG is still valuable even when it does not compress the numeric
payload: content-addressed nodes provide exact identity, shared immutable
objects and verified version structure. The mechanism is described in the
[IPFS Merkle-DAG documentation](https://docs.ipfs.tech/concepts/merkle-dag/).
Succinct de Bruijn structures compact graphs when their k-mer edge set is
shared; the measured V edge set is not. See [Bowe et al., Succinct de Bruijn
Graphs](https://doi.org/10.1007/978-3-642-33122-0_18).

## Decision

Adopt whole-record/page content addressing as a storage and versioning feature
if its engineering cost is small. Do not pursue raw-V de Bruijn coding or
fine-grained Merkle block compression. Test coordinate-conditioned entropy
coding on the full landmark publication before choosing a production codec.

