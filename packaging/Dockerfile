# Memory Condense proxy, Linux x86_64 with NVIDIA CUDA.
#
# Base choice is measured, not guessed: the pinned llama.cpp runtime's
# libggml-base.so.0.25.3 imports GLIBC_2.38, so Ubuntu 22.04 (glibc 2.35) cannot
# run it. 24.04 ships glibc 2.39. CUDA 12.8 matches the pinned llama.cpp build;
# the conda environment brings its own torch CUDA 12.6, and the two stay apart
# because llama-server runs as a separate process.
#
# Models are deliberately NOT baked in. They live on a mounted volume, so the
# image stays small and no upstream weights are redistributed.
#
# Build from the repository root, after pixi build --output-dir dist:
#   docker build -f packaging/Dockerfile -t memory-condense .
ARG CUDA_IMAGE=nvidia/cuda:12.8.1-runtime-ubuntu24.04
# Pinned release tarball rather than a container image: the GHCR pixi tags are
# not v-prefixed and 0.68.1 is not resolvable there, while this release asset is
# exact and contains a single static binary.
ARG PIXI_VERSION=v0.68.1
ARG PIXI_URL=https://github.com/prefix-dev/pixi/releases/download/${PIXI_VERSION}/pixi-x86_64-unknown-linux-musl.tar.gz

FROM ${CUDA_IMAGE} AS builder
ARG MEMORY_CONDENSE_VERSION=0.2.2
ARG PIXI_URL
RUN apt-get update && apt-get install --no-install-recommends -y ca-certificates curl \
    && rm -rf /var/lib/apt/lists/* \
    && curl -fsSL "${PIXI_URL}" | tar -xz -C /usr/local/bin pixi \
    && pixi --version
# Version-scoped so a stale package in dist/ cannot be picked up instead.
COPY dist/memory_condense-${MEMORY_CONDENSE_VERSION}-*.conda /opt/pkg/
COPY packaging/install-proxy.sh /opt/pkg/install-proxy.sh
# CONDA_OVERRIDE_CUDA only on this layer: there is no GPU during a build, and
# system-requirements cuda = "12.0" would otherwise fail to solve. Persisting it
# as ENV would mask a genuinely missing driver at run time.
RUN set -eux; \
    set -- /opt/pkg/*.conda; \
    chmod +x /opt/pkg/install-proxy.sh; \
    CONDA_OVERRIDE_CUDA=12.8 PIXI_CACHE_DIR=/opt/pixi-cache \
      /opt/pkg/install-proxy.sh --package-path "$1" \
        --install-dir /opt/memory-condense --skip-setup

FROM ${CUDA_IMAGE}
LABEL org.opencontainers.image.title="memory-condense" \
      org.opencontainers.image.source="https://github.com/Ethycs/memory_condense" \
      org.opencontainers.image.description="Memory-enabled OpenAI and Anthropic provider proxy"

# libgomp1 and libstdc++6 are NEEDED by the pinned llama.cpp libraries; curl is
# the healthcheck; tini reaps the llama-server child when the proxy is PID 1.
RUN apt-get update && apt-get install --no-install-recommends -y \
        ca-certificates curl libgomp1 libstdc++6 tini \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/bin/pixi /usr/local/bin/pixi
# COPY rather than building here: pixi reflinks packages out of its cache into
# the environment, so copying materializes real files and drops the cache.
COPY --from=builder /opt/memory-condense /opt/memory-condense
COPY packaging/docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh

# XDG_DATA_HOME is the whole trick for the data volume: cli.py injects
# --data-dir user_directory()/'data', and user_directory() derives from
# XDG_DATA_HOME off Windows, so memory lands under /data with no argument
# juggling. MEMORY_CONDENSE_ASSETS is read before any config file, so the
# mount point wins unconditionally.
ENV MEMORY_CONDENSE_ASSETS=/models \
    MEMORY_CONDENSE_CONFIG=/data/config.json \
    XDG_DATA_HOME=/data \
    HF_HOME=/data/huggingface \
    MEMORY_CONDENSE_HOST=0.0.0.0 \
    KMP_DUPLICATE_LIB_OK=TRUE \
    PYTHONUNBUFFERED=1 \
    PYTHONUTF8=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

# 'utility' above is what makes nvidia-smi resolvable, which runtime/llama.py
# calls for the run receipt. It degrades gracefully now, but the driver tools
# are still wanted.
RUN groupadd -g 10001 memory \
    && useradd -u 10001 -g 10001 -m -d /home/memory -s /usr/sbin/nologin memory \
    && chmod 0755 /usr/local/bin/docker-entrypoint.sh \
    && chown -R 10001:10001 /opt/memory-condense \
    && mkdir -p /models /data \
    && chown 10001:10001 /models /data
VOLUME ["/models", "/data"]

USER memory
WORKDIR /home/memory
EXPOSE 8787

# Liveness only: /_memory/health answers 200 even when it reports degraded.
# start-period covers loading Qwen's first shard and BGE before serving.
HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=3 \
    CMD curl -fsS -o /dev/null http://127.0.0.1:8787/_memory/health || exit 1

ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/docker-entrypoint.sh"]
CMD ["proxy"]
