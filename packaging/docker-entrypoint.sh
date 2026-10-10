#!/usr/bin/env bash
# Container entrypoint. Keeps the image usable with the same subcommands as the
# installed launcher, and fixes the two things a container changes: the bind
# address, and the fact that models arrive on a volume rather than from setup.
set -euo pipefail

launcher=/opt/memory-condense/memory-condense

# Anything that is not one of our subcommands is run verbatim, so `docker run
# IMAGE bash` still works for debugging.
case "${1:-}" in
    proxy|setup|doctor|route) ;;
    *) exec "$@" ;;
esac

command=$1
shift
arguments=("$@")

if [ "$command" = 'proxy' ]; then
    if [ ! -e /models/models ] && [ ! -e "${MEMORY_CONDENSE_CONFIG:-/data/config.json}" ]; then
        cat >&2 <<'MISSING'
No model assets found under /models.

Provision them once into the volume, which needs no GPU:
  docker run --rm -v memory-condense-models:/models -v memory-condense-data:/data IMAGE setup

Or reuse an existing verified cache, mounted at the canonical point:
  docker run --rm -v /host/cache:/models:ro -v memory-condense-data:/data IMAGE setup --reuse /models
MISSING
        exit 1
    fi
    # 127.0.0.1 is the proxy default and is unreachable from outside the
    # container. Appended, not prepended, because argparse takes the last
    # occurrence, so an explicit --host from the caller still wins.
    host_given=0
    for argument in "${arguments[@]+"${arguments[@]}"}"; do
        case "$argument" in
            --host|--host=*) host_given=1 ;;
        esac
    done
    if [ "$host_given" -eq 0 ]; then
        arguments+=(--host "${MEMORY_CONDENSE_HOST:-0.0.0.0}")
    fi
fi

exec "$launcher" "$command" "${arguments[@]+"${arguments[@]}"}"
