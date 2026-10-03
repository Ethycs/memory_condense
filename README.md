# memory_condense

A local memory proxy for OpenAI Chat Completions and Anthropic Messages.
It captures chat I/O, recalls source-linked evidence, keeps recent exchanges,
and updates its summary hierarchy and learning in the background.

The native runtime currently targets **Windows x64, Python 3.11–3.12, and an
NVIDIA CUDA GPU**. The evaluated machine has 8 GB VRAM. Model files are downloaded
separately or reused from an existing cache; they are not bundled in the package.

## Install using Pixi

Install [Pixi](https://pixi.sh/), download the Windows ZIP from
[GitHub Releases](https://github.com/Ethycs/memory_condense/releases), extract it,
and run:

```powershell
.\install-proxy.ps1
```

This installs the application and automatically downloads the pinned models.
The release ZIP includes checksums and [model asset details](packaging/MODEL_ASSETS.md).
Qwen uses six layers and downloads only its first upstream shard plus metadata.

To build from the repository instead:

```powershell
pixi build --output-dir dist
```

The build produces `dist/memory_condense-0.2.0-pyh4616a5c_0.conda`. Install it
into a separate application environment:

```powershell
.\packaging\install-proxy.ps1 `
  -PackagePath .\dist\memory_condense-0.2.0-pyh4616a5c_0.conda
```

The installer resolves CUDA-enabled PyTorch and the other native dependencies
with Pixi, installs pinned FastEmbed/ONNX Runtime/nvCOMP additions, downloads
the pinned model assets, and runs diagnostics. Downloads are several GB on a
fresh machine. To reuse this project's existing assets instead:

```powershell
.\packaging\install-proxy.ps1 `
  -PackagePath .\dist\memory_condense-0.2.0-pyh4616a5c_0.conda `
  -AssetsDir 'F:\Keytone\Documents\GitHub\memory_condense\.cache'
```

In a downloaded release bundle, `install-proxy.ps1` sits beside the package;
run `.\install-proxy.ps1` there. It copies the package into the application
directory, so the download folder is not needed after installation.

Start the installed proxy from any directory:

```powershell
& "$env:LOCALAPPDATA\memory_condense\proxy\memory-condense.cmd" proxy `
  --openai-base-url https://central-dev.zt:4000/v1
```

Keep that terminal running; stop it with Ctrl+C to drain ingestion and release
the local models. This installs a foreground application, not a Windows service.
`-InstallDir` changes the application location. Memory lives separately under
`%LOCALAPPDATA%\memory_condense\data`; changing the app directory does not
delete stored conversations.

The build follows Pixi's [Python package backend](https://pixi.prefix.dev/latest/build/backends/pixi-build-python/).
Pixi 0.68.1 accepts `pixi build` and prints its replacement local-build command,
`pixi publish --target-dir dist`; neither command above uploads the package.

CI runs proxy/lifecycle tests on Linux and Windows, builds the Windows Pixi
artifact, and installs it in a separate environment without downloading models.
Version tags publish a beta release only after both test jobs and package
installation pass. GPU and provider-backed stress tests remain separate from CI.

## Python package alternative

For a Python environment with CUDA-enabled PyTorch already installed, install
from the checkout once (subsequent execution does not need the checkout):

```powershell
python -m pip install ".[proxy]"
memory-condense setup
memory-condense doctor
memory-condense proxy --openai-base-url https://your-gateway.example/v1
```

Run `setup --reuse C:\path\to\existing\.cache` to use the evaluated assets
without copying or downloading them. The installed program runs from any
working directory and saves its configuration and memory under
`%LOCALAPPDATA%\memory_condense` by default.

## Connect a client

Point an OpenAI-compatible client at `http://127.0.0.1:8787/v1`, retain its
provider API key, and send a stable `x-memory-conversation-id` header. Use a
unique `x-memory-request-id` per generation and reuse it for an identical retry.
For example, after setting `LITELLM_KEY` to your gateway credential:

```python
import os
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8787/v1",
    api_key=os.environ["LITELLM_KEY"],
    default_headers={"x-memory-conversation-id": "my-engineering-session"},
)
response = client.chat.completions.create(
    model="codex_sdk/gpt-5.6-sol",
    messages=[{"role": "user", "content": "Remember: GET /ready must return 204."}],
    extra_headers={"x-memory-request-id": "turn-1"},
)
print(response.choices[0].message.content)
```

Use the same conversation ID for subsequent turns and a fresh ID for a new
conversation or branch. The client may send its full history or only new input.
The proxy preserves system instructions and native tool messages. A different
provider credential selects a separate memory namespace.

For Anthropic clients, configure `--anthropic-base-url` and use the local base
URL `http://127.0.0.1:8787` with the same conversation header.

Check service health with `http://127.0.0.1:8787/_memory/health`. Run
`memory-condense doctor --verify-models` through the installed launcher to check
CUDA, dependencies, and file hashes. `POST /_memory/flush` with the same credential
and conversation headers drains pending work for that conversation; add
`x-memory-provider: anthropic` when applicable.

The installed `proxy` command enables memory by default. `--mode observe`
selects transparent capture. Memory mode buffers streamed replies before
delivery; Responses API and multimodal requests are not supported yet.

The current Sol gateway route returned 404 for native tool requests in the live
probe, including a direct control without this proxy. Select a gateway/model
route that supports native tools when using an engineering agent.

`MEMORY_CONDENSE_CONFIG` overrides the saved configuration-file path;
`MEMORY_CONDENSE_ASSETS` or `--assets-dir` overrides the model directory.
The upstream address can also come from `OPENAI_BASE_URL` or
`ANTHROPIC_BASE_URL`; these must name the real provider when starting the proxy,
not its own local address. Provider credentials remain client-supplied.

Model provenance: [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B),
[FP32 BGE-M3 export](https://huggingface.co/onnx-community/bge-m3-ONNX),
[Llama 3.2 3B GGUF](https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF),
and [llama.cpp b11272](https://github.com/ggml-org/llama.cpp/releases/tag/b11272).
Setup pins revisions and verifies hashes; model/runtime licenses remain with
their upstream distributions. See the [proxy guide](docs/02%20-%20Implementation/06%20-%20Provider%20Proxy%20and%20Transcript%20Import.md)
for the detailed I/O contract and validation evidence.
