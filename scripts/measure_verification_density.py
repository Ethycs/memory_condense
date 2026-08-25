"""Measure verification-apparatus density per package.

Baseline for the verification-relocation charter
(``docs/06 - Roadmaps/03 - Verification Relocation Charter.md``).  Re-run
unchanged in V5 so the before/after rows in
``docs/08 - Analysis/12 - Verification Relocation Map.md`` are comparable.

A line is "verification-touching" if it mentions the apparatus: content
hashes, receipts, attestation, seals, lineage, or runtime certification.

    python scripts/measure_verification_density.py
"""

from __future__ import annotations

import pathlib
import re

SRC = pathlib.Path("src/memory_condense")

PACKAGES = (
    "eval",
    "search",
    "associations",
    "domain",
    "application",
    "modeling",
    "tooling",
)

# The two families the charter's roadmap targets, reported as a sub-row.
FAMILIES = (
    "_recall_guarded_cumulative_ops.py",
    "_recall_guarded_cumulative_contracts.py",
    "_recall_guarded_cumulative_result.py",
    "_recall_guarded_cumulative_validation_campaign.py",
    "_recall_guarded_cumulative_validation_shard.py",
    "_recall_guarded_cumulative_synthesis_artifacts.py",
    "_recall_guarded_cumulative_synthesis_contracts.py",
    "recall_guarded_cumulative.py",
    "recall_guarded_cumulative_1m.py",
    "recall_guarded_cumulative_runtime.py",
    "recall_guarded_cumulative_final_answer.py",
    "recall_guarded_cumulative_final_answer_semantic_judge.py",
    "_diffuse_replay_contracts.py",
    "_diffuse_replay_validation.py",
    "_diffuse_replay_reconstruction.py",
    "diffuse_longmemeval_route_v2.py",
    "diffuse_longmemeval_replay.py",
    "diffuse_longmemeval_analysis.py",
    "_diffuse_latent_training_corpus_codec.py",
    "_diffuse_latent_training_corpus_filesystem.py",
    "_diffuse_latent_training_corpus_io.py",
    "_diffuse_latent_training_corpus_models.py",
    "_diffuse_latent_training_corpus_route.py",
)

APPARATUS = re.compile(
    r"sha256|receipt|attest|identity_payload|\bseal|lineage|parent_hash"
    r"|fingerprint|certif",
    re.IGNORECASE,
)

# Packages that constitute the retrieval system eval is measuring.
SYSTEM = ("search", "associations", "domain")


def measure(paths):
    total = touched = 0
    count = 0
    for path in paths:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        total += len(lines)
        touched += sum(1 for line in lines if APPARATUS.search(line))
        count += 1
    return count, total, touched


def row(label, count, total, touched, indent=""):
    density = 100 * touched / total if total else 0.0
    print(f"{indent + label:<26}{count:>7}{total:>9}{touched:>9}{density:>8.1f}%")


def main() -> None:
    print(f"{'package':<26}{'files':>7}{'LOC':>9}{'verif':>9}{'density':>9}")
    print("-" * 60)

    sizes = {}
    for package in PACKAGES:
        root = SRC / package
        if not root.exists():
            continue
        count, total, touched = measure(sorted(root.rglob("*.py")))
        sizes[package] = total
        row(package, count, total, touched)
        if package == "eval":
            family_paths = [root / name for name in FAMILIES if (root / name).exists()]
            row("the two families", *measure(family_paths), indent="  ")

    print("-" * 60)
    system = sum(sizes.get(name, 0) for name in SYSTEM)
    evaluation = sizes.get("eval", 0)
    ratio = evaluation / system if system else 0.0
    print(f"retrieval system ({' + '.join(SYSTEM)}) = {system:,} LOC")
    print(f"eval = {evaluation:,} LOC  ->  eval is {ratio:.2f}x the system it measures")


if __name__ == "__main__":
    main()
