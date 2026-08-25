"""Behavior-preservation gate for the verification-relocation charter.

Replays the sealed dev10 gold-blind retrieval artifact and checks that the
cumulative ladder still has the behavior the arm is named for.  This is the
gate the charter requires after every V3 tranche
(``docs/06 - Roadmaps/03 - Verification Relocation Charter.md``).

Two seals, matching the charter's rule:

*Input seal* — ``retrieval.json`` hashes to its sealed sidecar, so we know
we are replaying the artifact Research Log 38 sealed
(``aa22f7c1...bd97``).

*Output seal* — a canonical behavior projection derived from the artifact's
**evidence lists**, hashed.  Per question and stage it records the evidence
identity sequence, what each stage added, the admission status, and the
cap arithmetic.  It deliberately does not hash receipt fields: V4 collapses
the receipt chain by design, and this gate must keep working across that
change.  What it seals is the recall guard itself —

  * each stage's evidence is an ordered superset of its parent's
    (nothing is ever dropped, so recall cannot fall),
  * no stage re-admits evidence a predecessor already held,
  * context and prompt proxies stay inside their hard caps.

Why not score against gold: ``score_published_retrieval`` needs the frozen
cleaned LongMemEval-S dataset (``d6f21ea9...a442``), which lives under the
gitignored ``/data/`` and is absent from this checkout.  When that file is
available, ``--with-gold-scoring`` adds the byte-identity check against the
sealed ``scores.json``.

    python scripts/dev10_replay_gate.py
    python scripts/dev10_replay_gate.py --with-gold-scoring

Exit 0 = behavior preserved.  Exit 1 = divergence (stop and report; the
charter forbids forcing the hash).  Exit 2 = the gate could not run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys

ARTIFACT_ROOT = pathlib.Path(
    "eval_results/longmemeval-1m-recall-guarded-cumulative-development-20260821"
)
DATASET = pathlib.Path("data/longmemeval_s_cleaned.json")
SPLIT = pathlib.Path("docs/10 - Research Log/data/longmemeval-95-target-split-v2.json")

# Recorded 2026-08-24 against the tree at charter V0, before any tranche.
BASELINE_BEHAVIOR_SHA256 = (
    "441ed735633a123d21f8990a57b22a0547f2d752ecbd634a031039b5aecf733c"
)


def sha256_of(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sealed_hash(path: pathlib.Path) -> str:
    """Read the digest field of a ``<name>.sha256`` sidecar."""
    return path.read_text(encoding="utf-8").split()[0]


def canonical_bytes(payload: object) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def stage_projection(stage, parent_ids):
    """Behavior of one ladder stage, plus its invariant verdicts."""
    receipt = stage["stage_receipt"]
    evidence_ids = [row["evidence_id"] for row in stage["evidence"]]
    added = [item for item in evidence_ids if item not in set(parent_ids)]

    nests = evidence_ids[: len(parent_ids)] == list(parent_ids)
    no_duplicates = len(set(evidence_ids)) == len(evidence_ids)
    readmits = bool(set(added) & set(parent_ids))
    within_context_cap = (
        receipt["context_token_proxy"] <= receipt["max_context_token_proxy"]
    )
    within_prompt_cap = (
        receipt["prompt_token_proxy"] <= receipt["max_prompt_token_proxy"]
    )

    return {
        "stage_id": stage["stage_id"],
        "admission_status": receipt["admission_status"],
        "evidence_count": len(evidence_ids),
        "evidence_ids": evidence_ids,
        "added_evidence_ids": added,
        "parent_count": len(parent_ids),
        "context_token_proxy": receipt["context_token_proxy"],
        "max_context_token_proxy": receipt["max_context_token_proxy"],
        "prompt_token_proxy": receipt["prompt_token_proxy"],
        "max_prompt_token_proxy": receipt["max_prompt_token_proxy"],
        "responder_output_token_reserve": receipt["responder_output_token_reserve"],
        "invariants": {
            "nests_parent_as_ordered_prefix": nests,
            "no_duplicate_evidence": no_duplicates,
            "no_readmitted_evidence": not readmits,
            "within_context_cap": within_context_cap,
            "within_prompt_cap": within_prompt_cap,
        },
    }, evidence_ids


def behavior_projection(artifact):
    """Canonical, receipt-free behavior of the whole ladder."""
    questions = []
    for question in artifact["questions"]:
        stages = []
        parent_ids: list[str] = []
        for stage in question["stages"]:
            projection, parent_ids = stage_projection(stage, parent_ids)
            stages.append(projection)
        questions.append(
            {
                "question_id": question["question_id"],
                "ordinal": question["ordinal"],
                "stage_ids": list(question["stage_ids"]),
                "provider_calls": question["provider_calls"],
                "stages": stages,
            }
        )
    return {
        "format": "dev10-behavior-projection-v1",
        "stage_ids": list(artifact["stage_ids"]),
        "question_count": artifact["question_count"],
        "provider_calls": artifact["provider_calls"],
        "gold_fields_present": artifact["gold_fields_present"],
        "questions": questions,
    }


def report_violations(projection) -> int:
    """Print every invariant that does not hold.  Returns the count."""
    violations = 0
    for question in projection["questions"]:
        for stage in question["stages"]:
            for name, ok in stage["invariants"].items():
                if not ok:
                    violations += 1
                    print(
                        f"  VIOLATION {question['question_id']}"
                        f"/{stage['stage_id']}: {name}"
                    )
    return violations


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=pathlib.Path, default=ARTIFACT_ROOT)
    parser.add_argument(
        "--expect",
        default=BASELINE_BEHAVIOR_SHA256,
        help="behavior-projection SHA-256 to require (default: the recorded baseline)",
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="print the behavior hash instead of checking it (V0 baseline only)",
    )
    parser.add_argument(
        "--with-gold-scoring",
        action="store_true",
        help="also re-derive scores.json and check it byte-for-byte (needs the dataset)",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dump",
        type=pathlib.Path,
        default=None,
        help="write the behavior projection here for diffing after a failure",
    )
    args = parser.parse_args()

    retrieval = args.artifact_root / "retrieval.json"
    sealed_retrieval = args.artifact_root / "retrieval.json.sha256"
    for path in (retrieval, sealed_retrieval):
        if not path.exists():
            print(f"gate cannot run: missing {path}", file=sys.stderr)
            return 2

    observed_input = sha256_of(retrieval)
    expected_input = sealed_hash(sealed_retrieval)
    if observed_input != expected_input:
        print("INPUT SEAL BROKEN — this is not the sealed retrieval artifact")
        print(f"  expected {expected_input}")
        print(f"  observed {observed_input}")
        return 2
    print(f"input seal    OK   retrieval.json {observed_input}")

    artifact = json.loads(retrieval.read_text(encoding="utf-8"))
    projection = behavior_projection(artifact)
    payload = canonical_bytes(projection)
    behavior = hashlib.sha256(payload).hexdigest()

    if args.dump:
        args.dump.write_bytes(payload)
        print(f"behavior projection written to {args.dump}")

    violations = report_violations(projection)
    if violations:
        print(f"\nGATE FAIL — {violations} recall-guard violations in the ladder")
        return 1
    print("recall guard  OK   ordered nesting, no re-admission, caps respected")

    if args.record:
        print(f"\nbehavior projection SHA-256: {behavior}")
        print("Record this as BASELINE_BEHAVIOR_SHA256.")
        return 0

    if not args.expect:
        print("\ngate cannot check: no baseline recorded; run with --record first")
        return 2
    if behavior != args.expect:
        print(f"behavior seal FAIL")
        print(f"  expected {args.expect}")
        print(f"  observed {behavior}")
        print("\nGATE FAIL — the ladder's behavior changed.")
        print("The charter forbids forcing the hash: stop and report.")
        return 1
    print(f"behavior seal OK   {behavior}")

    if args.with_gold_scoring:
        sealed_scores = args.artifact_root / "scores.json.sha256"
        if not (DATASET.exists() and SPLIT.exists() and sealed_scores.exists()):
            print(
                f"\ngold scoring skipped: need {DATASET} (the frozen cleaned "
                "LongMemEval-S), the split manifest, and the sealed scores sidecar"
            )
            return 2
        import tempfile

        from memory_condense.eval.recall_guarded_cumulative_1m import (
            load_original_population,
            score_published_retrieval,
        )

        scratch = pathlib.Path(tempfile.mkdtemp(prefix="dev10-gate-"))
        output = scratch / "scores.json"
        sample = load_original_population(DATASET, SPLIT)
        score_published_retrieval(
            sample=sample,
            retrieval_path=retrieval,
            output_path=output,
            source_embedding_device=args.device,
        )
        observed = sha256_of(output)
        expected = sealed_hash(sealed_scores)
        if observed != expected:
            print(f"gold seal     FAIL scores.json")
            print(f"  expected {expected}")
            print(f"  observed {observed}")
            print(f"\nGATE FAIL — re-derived scores kept at {output}")
            return 1
        print(f"gold seal     OK   scores.json {observed}")

    print("\nGATE PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
