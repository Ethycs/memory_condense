# Native model assets

`install-proxy.ps1` runs `memory-condense setup` automatically. Setup fetches
exact revisions, checks file SHA-256 hashes, and records the verified asset
directory. Existing valid files are reused. `memory-condense setup --reuse PATH`
verifies an existing cache without downloading. Runtime startup never downloads
models implicitly; `memory-condense doctor --verify-models` checks them again.

| Component | Provisioning | Upstream |
| --- | --- | --- |
| Qwen attention | Qwen3-8B revision `b968826d9c46dd6066d109eabc6255188de91218`; first shard plus config/tokenizer/index files | [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B), [Apache-2.0 license](https://huggingface.co/Qwen/Qwen3-8B/blob/main/LICENSE) |
| CPU embeddings | Pinned FP32 BGE-M3 ONNX export and tokenizer, authenticated by the packaged admission manifest | [BGE-M3 ONNX](https://huggingface.co/onnx-community/bge-m3-ONNX) |
| Summary fallback | Llama-3.2-3B-Instruct Q4_K_M, revision `5ab33fa94d1d04e903623ae72c95d1696f09f9e8` | [GGUF model and upstream license information](https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF) |
| Local generation runtime | llama.cpp b11272 Windows CUDA 12.4 and its CUDA runtime archive, with archive and extracted-file hashes | [Upstream release](https://github.com/ggml-org/llama.cpp/releases/tag/b11272) |

Qwen loads layers 0–5, using the existing lossless runtime path. The required
first upstream shard is 3,996,250,744 bytes (about 3.72 GiB); it also contains
unused tensors. Setup does not fetch the remaining model shards. This release
ships the six-layer loader and downloads original upstream assets. A separately
trimmed checkpoint is not shipped: that would need its own tensor-identity
manifest, compatible loader and equivalence validation before replacing the
tested asset path. Runtime FP32 attention and upstream on-disk weight precision
are separate properties.

Model licenses remain those of their respective publishers. Neither the source
archive nor the application release redistributes the model weight files.
