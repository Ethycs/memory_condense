# Native model assets

`install-proxy.ps1` and `install-proxy.sh` run `memory-condense setup`
automatically, as does `docker run ... setup` for the container image. Setup fetches
exact revisions, checks file SHA-256 hashes, and records the verified asset
directory. Existing valid files are reused. `memory-condense setup --reuse PATH`
verifies an existing cache without downloading. Runtime startup never downloads
models implicitly; `memory-condense doctor --verify-models` checks them again.

| Component | Provisioning | Upstream |
| --- | --- | --- |
| Qwen attention | Qwen3-8B revision `b968826d9c46dd6066d109eabc6255188de91218`; first shard plus config/tokenizer/index files | [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B), [Apache-2.0 license](https://huggingface.co/Qwen/Qwen3-8B/blob/main/LICENSE) |
| CPU embeddings | Pinned FP32 BGE-M3 ONNX export and tokenizer, authenticated by the packaged admission manifest | [BGE-M3 ONNX](https://huggingface.co/onnx-community/bge-m3-ONNX) |
| Tactic classifier | Project-trained three-class linear head, about 64 KB, bundled with a sealed identity; reuses the existing CPU BGE encoder | `memory_condense/runtime/data/tactic-router-v1.json` |
| Decision/fix intent | Project-trained binary linear head, about 22 KB; shares the same BGE query vector | `memory_condense/runtime/data/history-intent-v1.json` |
| Summary fallback | Llama-3.2-3B-Instruct Q4_K_M, revision `5ab33fa94d1d04e903623ae72c95d1696f09f9e8` | [GGUF model and upstream license information](https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF) |
| Local generation runtime (Windows) | llama.cpp b11272 Windows CUDA 12.4 and its CUDA runtime archive, with archive and extracted-file hashes | [Upstream release](https://github.com/ggml-org/llama.cpp/releases/tag/b11272) |
| Local generation runtime (Linux) | llama.cpp b11272 Ubuntu CUDA 12.8 and its CUDA runtime archive, with archive and extracted-file hashes. 12.8 is the lowest Linux CUDA build upstream publishes at this tag; its `libggml-base.so` requires glibc 2.38 | [Upstream release](https://github.com/ggml-org/llama.cpp/releases/tag/b11272) |

Qwen loads layers 0–5, using the existing lossless runtime path. The required
first upstream shard is 3,996,250,744 bytes (about 3.72 GiB); it also contains
unused tensors. Setup does not fetch the remaining model shards. This release
ships the six-layer loader and downloads original upstream assets. A separately
trimmed checkpoint is not shipped: that would need its own tensor-identity
manifest, compatible loader and equivalence validation before replacing the
tested asset path. Runtime FP32 attention and upstream on-disk weight precision
are separate properties.

Model licenses remain those of their respective publishers. Neither the source
archive nor the application release redistributes upstream transformer weights.
The small project-trained classifier heads are included in the application package.
