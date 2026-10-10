# memory_condense

A local memory proxy for OpenAI Chat Completions and Anthropic Messages.
It captures chat I/O, recalls source-linked evidence, keeps recent exchanges,
and updates its summary hierarchy and learning in the background.

The native runtime targets **Windows x64 and Linux x86_64, Python 3.11–3.12,
and an NVIDIA CUDA GPU**. The evaluated machine has 8 GB VRAM. Model files are
downloaded separately or reused from an existing cache; they are not bundled in
the package.

Linux needs **glibc 2.38 or newer** (Ubuntu 24.04, Debian 13 or RHEL 10 and
later). That floor is set by the pinned llama.cpp CUDA build, whose
`libggml-base.so` imports `GLIBC_2.38`; an older distribution resolves every
other dependency and then fails to load the summarizer. Linux pins llama.cpp
CUDA 12.8 because upstream `b11272` publishes no lower Linux CUDA build. The
archive carries its own `libcudart`, so CUDA minor-version compatibility keeps
the driver requirement at 12.x.

## Install using Pixi

Install [Pixi](https://pixi.sh/), download your platform's archive from
[GitHub Releases](https://github.com/Ethycs/memory_condense/releases), extract it,
and run the installer beside the package.

On Windows:

```powershell
.\install-proxy.ps1
```

On Linux the bundle is a flat tarball, so extract it into a directory of its own:

```bash
mkdir memory-condense && tar -xzf memory-condense-0.2.2-linux64.tar.gz -C memory-condense
cd memory-condense && chmod +x install-proxy.sh && ./install-proxy.sh
```

This installs the application and automatically downloads the pinned models.
Each archive ships its own checksum manifest, `SHA256SUMS-win64.json` or
`SHA256SUMS-linux64.json`, alongside [model asset details](packaging/MODEL_ASSETS.md).
Qwen uses six layers and downloads only its first upstream shard plus metadata.

To build from the repository instead:

```powershell
pixi build --output-dir dist
```

The build produces `dist/memory_condense-0.2.2-pyh4616a5c_0.conda`. Install it
into a separate application environment:

```powershell
.\packaging\install-proxy.ps1 `
  -PackagePath .\dist\memory_condense-0.2.2-pyh4616a5c_0.conda
```

The installer resolves CUDA-enabled PyTorch and the other native dependencies
with Pixi, installs pinned FastEmbed/ONNX Runtime/nvCOMP additions, downloads
the pinned model assets, and runs diagnostics. Downloads are several GB on a
fresh machine. To reuse this project's existing assets instead:

```powershell
.\packaging\install-proxy.ps1 `
  -PackagePath .\dist\memory_condense-0.2.2-pyh4616a5c_0.conda `
  -AssetsDir 'F:\Keytone\Documents\GitHub\memory_condense\.cache'
```

The Linux installer takes the same options as long flags:

```bash
./packaging/install-proxy.sh \
  --package-path ./dist/memory_condense-0.2.2-pyh4616a5c_0.conda \
  --assets-dir /path/to/existing/.cache
```

In a downloaded release bundle the matching installer sits beside the package;
run `.\install-proxy.ps1` or `./install-proxy.sh` there. It copies the package
into the application directory, so the download folder is not needed after
installation.

For an upgrade, stop the existing proxy and install into a new application
directory; the installer preserves existing environments instead of overwriting
their manifests. Reuse your verified model cache:

```powershell
.\install-proxy.ps1 `
  -InstallDir "$env:LOCALAPPDATA\memory_condense\proxy-0.2.2" `
  -AssetsDir 'C:\path\to\existing\models'
```

On Linux the equivalent is `--install-dir` and `--assets-dir`, and the default
application directory is `${XDG_DATA_HOME:-~/.local/share}/memory_condense/proxy`.

Start the new directory's launcher, `memory-condense.cmd` on Windows or
`memory-condense` on Linux, with the same provider options
and the absolute `--data-dir` used by the previous installation. Existing native
snapshots are validated and migrated during startup. Back up the data directory
before upgrading.

Start the installed proxy from any directory:

```powershell
& "$env:LOCALAPPDATA\memory_condense\proxy\memory-condense.cmd" proxy `
  --openai-base-url https://central-dev.zt:4000/v1
```

Or on Linux:

```bash
"${XDG_DATA_HOME:-$HOME/.local/share}/memory_condense/proxy/memory-condense" proxy \
  --openai-base-url https://central-dev.zt:4000/v1
```

Keep that terminal running; stop it with Ctrl+C to drain ingestion and release
the local models. This installs a foreground application, not a service.
`-InstallDir` or `--install-dir` changes the application location. Memory lives
separately under `%LOCALAPPDATA%\memory_condense\data`, or
`${XDG_DATA_HOME:-~/.local/share}/memory_condense/data` on Linux; changing the
app directory does not delete stored conversations.

The build follows Pixi's [Python package backend](https://pixi.prefix.dev/latest/build/backends/pixi-build-python/).
Pixi 0.68.1 accepts `pixi build` and prints its replacement local-build command,
`pixi publish --target-dir dist`; neither command above uploads the package.

CI runs proxy/lifecycle tests on Linux and Windows, builds the Pixi package
once, and installs it in a separate environment on each platform without
downloading models. The package is noarch, so one build serves both. Version
tags publish a beta release only after the test jobs and both platforms'
package installation pass. GPU and provider-backed stress tests remain separate
from CI, so native execution on either platform is only as verified as its
manual GPU assay.

## Run in Docker

The image carries the code and its runtime dependencies but **no models**, so it
stays small and redistributes no upstream weights. Models and memory live on
mounted volumes. It needs the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/)
and `--gpus all`, because the Qwen attention path requires a real CUDA device.

```bash
docker volume create memory-condense-models
docker volume create memory-condense-data

# Provision the pinned assets once. Downloading needs no GPU.
docker run --rm \
  -v memory-condense-models:/models -v memory-condense-data:/data \
  ghcr.io/ethycs/memory_condense:0.2.2 setup

docker run --rm --gpus all \
  -v memory-condense-models:/models -v memory-condense-data:/data \
  ghcr.io/ethycs/memory_condense:0.2.2 doctor

docker run -d --name memory-condense --gpus all \
  -p 127.0.0.1:8787:8787 \
  -v memory-condense-models:/models -v memory-condense-data:/data \
  ghcr.io/ethycs/memory_condense:0.2.2 proxy \
  --openai-base-url https://your-gateway.example/v1
```

To reuse an existing verified cache, mount it at the canonical point so the
saved configuration never records a throwaway container path:

```bash
docker run --rm -v /host/cache:/models:ro -v memory-condense-data:/data \
  ghcr.io/ethycs/memory_condense:0.2.2 setup --reuse /models
```

The container runs as UID 10001. Named volumes inherit that ownership, but a
**bind** mount from the host needs `--user "$(id -u):$(id -g)"` or a
pre-owned directory, otherwise the first `setup` fails with a permission error.
The entrypoint appends `--host 0.0.0.0` to `proxy` unless you pass `--host`
yourself, since the proxy's own default of `127.0.0.1` is unreachable from
outside the container. Build locally with
`docker build -f packaging/Dockerfile -t memory-condense .` after
`pixi build --output-dir dist`.

## Python package alternative

For a Python environment with CUDA-enabled PyTorch already installed, install
from the checkout once (subsequent execution does not need the checkout):

```powershell
python -m pip install ".[proxy]"
memory-condense setup
memory-condense doctor
memory-condense proxy --openai-base-url https://your-gateway.example/v1
```

Run `setup --reuse C:\path\to\existing\.cache`, or
`setup --reuse /path/to/existing/.cache` on Linux, to use the evaluated
assets without copying or downloading them. The installed program runs from
any working directory and saves its configuration and memory under
`%LOCALAPPDATA%\memory_condense`, or
`${XDG_DATA_HOME:-~/.local/share}/memory_condense` on Linux, by default.

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

Version 0.2.2 keeps local Qwen attention, CPU BGE embeddings, and valid
local Llama summaries. Failed local summary jobs escalate to the conversation's
API/model (Sol in our latest evaluation); accepted repairs are cached. This can
add paid background requests. `--no-memory-summary-api-fallback` disables
escalation, and health reports its counts under `summary_routing`. This policy
is included in 0.2.2; see the [proxy guide](docs/02%20-%20Implementation/06%20-%20Provider%20Proxy%20and%20Transcript%20Import.md)
for validation and recovery details.

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

## Automatic dialogue-wide recall

Version 0.2.2 enables BGE tactic selection by default. Specific questions
use focused cap-8 recall; broad questions receive hierarchy summaries plus exact
neighborhood evidence. Decision/fix questions such as “What have we tried?” also
receive chronological evidence labels, speaker attribution and supported stored
revision links. They use the same BGE query vector and existing reader call:

```powershell
memory-condense proxy --memory-retrieval-policy auto `
  --openai-base-url https://your-gateway.example/v1
```

The selector shares the existing CPU BGE model. Ambiguous requests get one
contextual check against recent chat, then default to focused recall if still
uncertain. Force `cap8`, `dialogue` (raw neighborhoods), or
`hierarchy_neighborhoods` to bypass classification.
Optional `--memory-dialogue-config C:\path\dialogue.json` loads a
versioned tuning profile at startup; restart to change it. Health reports the
active profile under `retrieval`, and each recalled packet journals its policy
hash and source pointers. See the [configuration and rollback guide](docs/02%20-%20Implementation/06%20-%20Provider%20Proxy%20and%20Transcript%20Import.md#dialogue-wide-retrieval-module).

Routing uses stored summaries and attention neighborhoods, then hydrates exact
raw evidence for the reader. It reuses the existing query embedding and caches
leaf-summary vectors during publication. Summary context is separately labeled
and limited to 24,576 tokens; omitted sections are reported. Exact raw spans
retain their original pointers and learning behavior.

Broad recall also carries the ADR and fresh-chat handoff conventions from
[claude-chat-decompose](https://github.com/Ethycs/claude-chat-decompose): decisions,
rationale, alternatives, current progress and open work stay linked to their original
evidence. Ask for a handoff in chat; the ordinary reader call produces it with a
source map. The journal retains citation bindings and the snapshot identity.
This formats recall from the current conversation; automatic project sharing across
new conversation namespaces is not yet implemented. See the
[handoff contract](docs/02%20-%20Implementation/06%20-%20Provider%20Proxy%20and%20Transcript%20Import.md#adr-evidence-and-fresh-chat-handoff-conventions).
