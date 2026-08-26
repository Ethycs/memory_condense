# V2 semantic confirmation and locked 100Q answers

**Status at publication:** independent Sol confirmed the development v2
facts-only result at 10/10 semantic accuracy, and the complete locked
100-question fixed-S1 Terra answer population was sealed and replayed. The
validation answers scored 33/100 normalized exact match and 0.447494 mean token
F1 locally. The later exact 100-call judge scored them 56/100 semantically and
failed the >=95% gate; see
[Research Log 45](45%20-%202026-08-26%20-%20Locked%20100Q%20semantic%20gate%20result.md).

## Exact authorized execution

The two newly authorized provider populations completed without retries:

| Population | Physical calls | Checkpoint hits | Retries |
| --- | ---: | ---: | ---: |
| development v2 Sol judge | 10 | 0 | 0 |
| locked validation fixed-S1 Terra answers | 100 | 0 | 0 |

Provider-free replay then used all 10 v2 judge checkpoints and all 100 Terra
answer checkpoints without making a call. The stable artifacts reproduced
their original hashes.

## V2 semantic result

The local v2 score from
[Research Log 43](43%20-%202026-08-26%20-%20EM%20v2%20result%20and%20locked%20100Q%20retrieval%20merge.md)
was 7/10 exact match and 0.914065 F1. Sol accepted all three form-only misses:

| Measure | Result |
| --- | ---: |
| Normalized exact match | 7/10 |
| Mean token F1 | 0.914065 |
| Independent Sol semantic accuracy | **10/10** |

The judge run contains ten logical and ten unique prompts. Its zero-call replay
hit all ten checkpoints and reproduced 10/10. This confirms that the compact
comma-separated event orders and bare scalar were semantically equivalent to
the references.

The causal limitation from Log 43 remains: protected S0 supplied substantial
answer evidence, including both questions whose v2 compressor emitted no EM
facts. V2 proves that aggressive cited compression can preserve this dev10
semantic result; it does not isolate an EM-only recall gain.

## Locked 100-question Terra answers

The responder consumed the preregistered `direct_episode_additions` stage from
the merged retrieval, under the frozen 8,000-token prompt proxy and 256-token
output reserve. It completed exactly one unique answer per question.

| Runtime measure | Value |
| --- | ---: |
| Questions / unique calls | 100 / 100 |
| Maximum prompt proxy | 7,353 |
| Total input-token proxy | 715,037 |
| Mean input-token proxy | 7,150.37 |
| Total output-token proxy | 378 |
| Mean output-token proxy | 3.78 |
| Wall time | 405.73 seconds |
| Retries | 0 |

The gateway did not expose provider-reported token counts, so the table uses
the sealed local token proxy. No prompt exceeded the cap, and the replay
reproduced the same canonical answer artifact.

## Provider-free lexical score

Only after the answer artifact and all journals passed validation was the
locked gold population loaded for a local diagnostic:

| Category | Questions | Exact | Mean F1 |
| --- | ---: | ---: | ---: |
| knowledge-update | 16 | 11 | 0.776042 |
| multi-session | 27 | 7 | 0.359259 |
| single-session-assistant | 11 | 3 | 0.408081 |
| single-session-preference | 6 | 0 | 0.049733 |
| single-session-user | 14 | 9 | 0.744048 |
| temporal-reasoning | 26 | 3 | 0.285722 |
| **overall** | **100** | **33** | **0.447494** |

Twenty-seven responses were the exact short abstention `I don't know` with
optional punctuation. Some non-exact answers are clearly semantically valid,
such as `7 days` against a reference that explicitly accepts seven days, so
33% exact match is not the formal accuracy. Conversely, the high abstention
count, wrong scalars, and weak preference/temporal categories made a hidden
95% semantic result implausible. The preregistered independent judge later
turned that diagnosis into the 56/100 failed gate recorded in Research Log 45.

## Independent validation judge preflight

The sealed answers produced this provider-free Sol population:

```text
questions=100
unique Sol prompts=100
campaign binding=84c871adac4b73bf4a40103c49b624227e4a12acb104d9335c3f9492171068da
provider calls=0
```

Each call would send one locked validation question, its reference answer, and
the sealed Terra prediction to Sol. This was a new export population not
covered by the prior ten-call v2 authorization. Its later separately authorized
execution and replay are recorded in Research Log 45.

## Artifact hashes

| Artifact | SHA-256 |
| --- | --- |
| v2 Sol semantic judge | `021069f8eeac586b6888f5287f45e10bf6a9a80fa25a2c7cc3d31992227ca4cd` |
| v2 Sol zero-call replay | `4cfdecf20ff2a9fae2806916469f52ce11d8d8e1ab77584bc464d85d650abde4` |
| locked 100Q Terra answers | `d7fc47b8d1f372f002230c6ffe489dac8cd11bd71b35b8d3008b1255da2a38cd` |
| locked 100Q retrieval | `e36b54ec6171aa7b40f75682ad85e5822a64d45bc411ffe03bcd9cad0222007f` |

## Interpretation and next step

The development result answers the representation question: facts-only can
retain semantic accuracy while removing raw EM clutter, whereas full raw
reinjection caused a real precision loss. The locked result exposes the scale
problem: fixed S1 supplies a very large prompt but still produces many
abstentions and weak temporal/preference answers.

The formal judge in item 1 below is now complete; Research Log 45 supersedes
that step. The remaining improvement sequence is:

1. classify the sealed failures;
2. convert the existing 100 S1 neighborhoods to the v2 fact representation;
3. apply cited raw fallback only to misses whose required evidence was
   selected but lost or made ambiguous during compression;
4. change retrieval only for misses where the needed evidence never reached
   S1.

This preserves the intended linear improvement ladder and avoids adding raw
context or complex retrieval layers to cases that already succeed.
