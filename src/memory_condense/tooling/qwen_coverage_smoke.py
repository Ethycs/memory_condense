"""Timed local smoke for the generation-free Qwen prefix coverage reader."""

from __future__ import annotations

import argparse
import gc
import time

from memory_condense.associations.head_memory import (
    AssociativeMemoryCandidate,
    QwenMemoryLinker,
)
from memory_condense.modeling.qwen_prefix import (
    add_prefix_encoder_arguments,
    prefix_encoder_from_args,
)


_CANDIDATES = (
    "I visited the Metropolitan Museum of Art and saw the Egyptian wing.",
    "At the Met I spent most of the afternoon looking at Egyptian art.",
    "I toured the Museum of Modern Art during my New York trip.",
    "Later that week I went to the Guggenheim Museum.",
    "I bought paint and replacement brushes for the kitchen.",
    "The Museum of Modern Art visit included a photography exhibition.",
    "My train was delayed for forty minutes outside Newark.",
    "The Guggenheim's spiral gallery was the highlight of that visit.",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_prefix_encoder_arguments(
        parser,
        layers_flags=("--layers", "--prefix-layers"),
        layers_default=6,
        dtype_default="float16",
    )
    parser.add_argument("--attention-layer", type=int, default=5)
    parser.add_argument("--candidates", type=int, default=2)
    parser.add_argument("--workspace-tokens", type=int, default=1024)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if not 1 <= args.candidates <= len(_CANDIDATES):
        raise ValueError(f"candidates must be in [1, {len(_CANDIDATES)}]")

    started = time.perf_counter()
    encoder = prefix_encoder_from_args(args)
    loaded_s = time.perf_counter() - started
    print(
        f"load_s={loaded_s:.3f} layers=0..{args.prefix_layers - 1} "
        f"readout_layer={args.attention_layer} lm_head=absent",
        flush=True,
    )

    linker = QwenMemoryLinker(
        encoder,
        layer=args.attention_layer,
        max_candidates=args.candidates,
        max_workspace_tokens=args.workspace_tokens,
    )
    candidates = [
        AssociativeMemoryCandidate(
            episode_id=f"candidate-{index}",
            text=text,
            score=1.0 - index / 10.0,
        )
        for index, text in enumerate(_CANDIDATES[: args.candidates])
    ]
    read_started = time.perf_counter()
    result = linker.inspect_coverage(
        "List every museum I visited.",
        candidates,
    )
    read_s = time.perf_counter() - read_started
    print(
        f"read_s={read_s:.3f} inspected={result.workspace_candidates} "
        f"workspace_tokens={result.workspace_tokens} passes={result.passes}",
        flush=True,
    )
    for hit in result.hits:
        signature = hit.transport_signature
        print(
            f"{hit.episode_id} qk={hit.qk_score:.6f} "
            f"ov={hit.ov_transport:.6f} signature={tuple(signature.shape)} "
            f"device={signature.device.type} dtype={signature.dtype}",
            flush=True,
        )

    del result, linker, encoder
    gc.collect()


if __name__ == "__main__":
    main()
