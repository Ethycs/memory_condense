"""Classify every runtime check in the cumulative and diffuse-replay families.

Charter: docs/06 - Roadmaps/03 - Verification Relocation Charter.md
Emits docs/08 - Analysis/12 - Verification Relocation Map.csv
"""

import ast
import collections
import csv
import json
import pathlib
import re

CUM = """_recall_guarded_cumulative_ops.py _recall_guarded_cumulative_contracts.py
_recall_guarded_cumulative_result.py _recall_guarded_cumulative_validation_campaign.py
_recall_guarded_cumulative_validation_shard.py _recall_guarded_cumulative_synthesis_artifacts.py
_recall_guarded_cumulative_synthesis_contracts.py recall_guarded_cumulative_final_answer.py
recall_guarded_cumulative_final_answer_semantic_judge.py recall_guarded_cumulative.py
recall_guarded_cumulative_1m.py recall_guarded_cumulative_runtime.py""".split()

DIF = """_diffuse_replay_contracts.py _diffuse_replay_validation.py _diffuse_replay_reconstruction.py
diffuse_longmemeval_route_v2.py diffuse_longmemeval_replay.py diffuse_longmemeval_analysis.py
_diffuse_latent_training_corpus_codec.py _diffuse_latent_training_corpus_filesystem.py
_diffuse_latent_training_corpus_io.py _diffuse_latent_training_corpus_models.py
_diffuse_latent_training_corpus_route.py""".split()

FILES = CUM + DIF
ROOT = pathlib.Path("src/memory_condense/eval")
CORPUS_IO = {
    "_diffuse_latent_training_corpus_filesystem.py",
    "_diffuse_latent_training_corpus_io.py",
}
FENCED = {
    "recall_guarded_cumulative_fast_artifact.py",
    "fast_cav_feature_session.py",
    "run_fast_1m_cav.py",
    "consolidation_replay.py",
}

BEHAVIORAL = (
    r"prefix|cumulative union|duplicate|monoton|not an earlier stage"
    r"|immediate predecessor|no longer cumulative|complete evidence set"
)
IDENTITY = (
    r"sha-?256|_digest|identity_payload|fingerprint|lineage|parent_hash|attest"
    r"|\bseal|\bchanged\b|frozen original|bytes differ|mutated package files"
    r"|worktree is not clean|differs from|differ from the|does not match its"
    r"|cannot be reconstructed|not reproducible"
)
RECEIPT = (
    r"receipt|certified|certification|reservation|provenance|eligib"
    r"|is missing the|identity is missing|lacks a declared identity"
    r"|no mapping identity|omitted its identity|has no runtime identity"
    r"|no valid implementation digest|omitted its completion report"
)
TEST = (
    r"exceed|_cap\b| cap\b|budget|unique|sorted|one-to-one|zero .*state"
    r"|transformer state|token state|non-negative|disagree|accounting"
    r"|belongs to another|owned_|ownership|truncat"
    r"|population is incomplete|omitted its population|populations differ"
    r"|repeats a|repeated |is out of order|do(es)? not cover|cover the frozen"
    r"|exactly 100 questions|is incomplete|has no |no citations|unknown claim"
    r"|unknown evidence|exact evidence substring|must not cite"
    r"|cannot admit|cannot claim|admitted direct chunks|artifact-global"
    r"|gold firewall|gold-bearing|must own its|cannot mint|forbidden|firewall"
    r"|not part of the locked|locked 100q|omitted retrieval/evaluation"
    r"|not normalized|abstention retained|used different runtimes"
    r"|not authoritative|conflict"
)
INPUT_VALIDATION = (
    r"must be |must contain|must equal|must map|must return|must use|require"
    r"|cannot be empty|must not be empty|unsupported|invalid|non-empty"
    r"|non-finite|non-string|non-json|non-numeric|schema|wrong exact|non-exact"
    r"|malformed|noncanonical|not canonical|is not closed|not a canonical"
    r"|has no user|no user message|provider messages are missing|missing:"
)
OPERATIONAL = (
    r"does not exist|refusing to replace|no historical|undated turn"
    r"|unembedded chunk|lost its|not bound before|not produced"
    r"|could not be rehydrated|is unavailable|returned an empty|is empty"
    r"|absent from"
)
CORPUS_OPS = (
    r"corpus|staging|handle|clobber|collision|replaced|snapshot is closed"
    r"|short .* write"
    r"|cannot (read|write|seek|flush|open|close|create|inspect|enumerate"
    r"|resolve|atomically)"
)


def message_of(node):
    parts = []

    def walk(n):
        if isinstance(n, ast.Constant) and isinstance(n.value, str):
            parts.append(n.value)
            return
        if isinstance(n, ast.JoinedStr):
            for v in n.values:
                parts.append(str(v.value) if isinstance(v, ast.Constant) else "{}")
            return
        for child in ast.iter_child_nodes(n):
            walk(child)

    if node.exc is None:
        return "<re-raise>"
    walk(node.exc)
    return " ".join(parts).strip()


# Hand-assigned classes for checks the pattern rules cannot decide.
# Keyed "<file>:<line>"; reviewed one by one against the source.
OVERRIDES = {
    # --- cumulative ---------------------------------------------------
    "_recall_guarded_cumulative_ops.py:197": "InputValidation",
    "_recall_guarded_cumulative_ops.py:216": "Test",
    "_recall_guarded_cumulative_ops.py:232": "Operational",
    "_recall_guarded_cumulative_contracts.py:220": "Test",
    "_recall_guarded_cumulative_contracts.py:353": "Test",
    "_recall_guarded_cumulative_result.py:511": "Delete/identity",
    "_recall_guarded_cumulative_validation_campaign.py:654": "Test",
    "_recall_guarded_cumulative_synthesis_artifacts.py:172": "Test",
    "_recall_guarded_cumulative_synthesis_artifacts.py:202": "InputValidation",
    "_recall_guarded_cumulative_synthesis_artifacts.py:217": "InputValidation",
    "_recall_guarded_cumulative_synthesis_artifacts.py:591": "Delete/receipt",
    "_recall_guarded_cumulative_synthesis_artifacts.py:963": "InputValidation",
    "_recall_guarded_cumulative_synthesis_artifacts.py:1041": "Test",
    "_recall_guarded_cumulative_synthesis_artifacts.py:1156": "Delete/receipt",
    "_recall_guarded_cumulative_synthesis_contracts.py:197": "InputValidation",
    "_recall_guarded_cumulative_synthesis_contracts.py:338": "InputValidation",
    "_recall_guarded_cumulative_synthesis_contracts.py:340": "Test",
    "_recall_guarded_cumulative_synthesis_contracts.py:594": "InputValidation",
    "recall_guarded_cumulative_final_answer.py:77": "Test",
    "recall_guarded_cumulative_final_answer.py:218": "Delete/identity",
    "recall_guarded_cumulative_final_answer.py:267": "Test",
    "recall_guarded_cumulative_final_answer.py:306": "Test",
    "recall_guarded_cumulative_final_answer_semantic_judge.py:232": "Delete/receipt",
    "recall_guarded_cumulative_final_answer_semantic_judge.py:298": "Test",
    "recall_guarded_cumulative_1m.py:996": "Test",
    "recall_guarded_cumulative_runtime.py:631": "Operational",
    # --- diffuse-replay -----------------------------------------------
    "_diffuse_replay_contracts.py:213": "InputValidation",
    "_diffuse_replay_contracts.py:1045": "Test",
    "_diffuse_replay_validation.py:96": "Delete/identity",
    "_diffuse_replay_validation.py:145": "Delete/identity",
    "_diffuse_replay_reconstruction.py:524": "Test",
    "diffuse_longmemeval_route_v2.py:348": "Test",
    "diffuse_longmemeval_replay.py:231": "Delete/receipt",
    "diffuse_longmemeval_replay.py:241": "Delete/receipt",
    "diffuse_longmemeval_replay.py:314": "Test",
    "diffuse_longmemeval_replay.py:392": "Test",
    "diffuse_longmemeval_replay.py:396": "Operational",
    "diffuse_longmemeval_replay.py:456": "Test",
    "diffuse_longmemeval_analysis.py:578": "Test",
    "diffuse_longmemeval_analysis.py:638": "Test",
    "diffuse_longmemeval_analysis.py:995": "Operational",
    "_diffuse_latent_training_corpus_filesystem.py:341": "Operational",
    "_diffuse_latent_training_corpus_filesystem.py:439": "Operational",
    "_diffuse_latent_training_corpus_filesystem.py:843": "Operational",
    "_diffuse_latent_training_corpus_filesystem.py:1171": "Operational",
    "_diffuse_latent_training_corpus_io.py:631": "Test",
    "_diffuse_latent_training_corpus_io.py:722": "Test",
    "_diffuse_latent_training_corpus_io.py:748": "Delete/identity",
    "_diffuse_latent_training_corpus_io.py:915": "Operational",
    "_diffuse_latent_training_corpus_io.py:1046": "Delete/identity",
    "_diffuse_latent_training_corpus_io.py:1097": "Operational",
    "_diffuse_latent_training_corpus_models.py:213": "InputValidation",
    "_diffuse_latent_training_corpus_models.py:743": "InputValidation",
    "_diffuse_latent_training_corpus_models.py:837": "Delete/identity",
    "_diffuse_latent_training_corpus_route.py:729": "Test",
    "_diffuse_latent_training_corpus_route.py:828": "Test",
}


def classify(filename, line, condition, message, in_except):
    override = OVERRIDES.get(f"{filename}:{line}")
    if override:
        return override
    text = (condition + " " + message).lower()
    if in_except:
        return "Operational"
    if filename in CORPUS_IO and re.search(CORPUS_OPS, text):
        return "Operational"
    if re.search(BEHAVIORAL, text):
        return "Behavioral"
    if re.search(IDENTITY, text):
        return "Delete/identity"
    if re.search(RECEIPT, text):
        return "Delete/receipt"
    if re.search(TEST, text):
        return "Test"
    if re.search(INPUT_VALIDATION, text):
        return "InputValidation"
    if re.search(OPERATIONAL, text):
        return "Operational"
    return "Other"


# A Delete-class row is "suspect" when its message also carries substantive
# Test/Behavioral language — arity, type, cap arithmetic, ownership,
# coordinate agreement.  The pattern rules key on words like "changed",
# "receipt" and "parent", which appear in checks that are really asserting a
# property.  Such rows must be read individually before V3 deletes them:
# they are the ones where a blind delete would lose a real invariant.
SUSPECT_DELETE = re.compile(
    r"must be boolean|must be a |requires (three|exactly)|invalid .* status"
    r"|accounting|cannot name parent|non-negative|must be sorted"
    r"|values must be unique|exceeds|cap|budget|one-to-one|zero .*state"
    r"|prefix|union|duplicate|belongs to another|coordinates disagree"
    r"|must be non-empty",
    re.IGNORECASE,
)


DESTINATION = {
    "Delete/identity": "delete (interior identity cross-check)",
    "Delete/receipt": "delete (receipt/certification bookkeeping)",
    "Test": "move to pytest over the pure transformation",
    "Behavioral": "KEEP in-path (recall guard: monotonic nesting)",
    "InputValidation": "keep in place (ordinary type/format validation)",
    "Operational": "keep in place (OS / TOCTOU / precondition)",
    "Other": "MANUAL REVIEW",
}


def collect():
    rows = []
    for name in FILES:
        path = ROOT / name
        if not path.exists():
            continue
        source = path.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(source)
        parent = {}
        for node in ast.walk(tree):
            for child in ast.iter_child_nodes(node):
                parent[child] = node
        for node in ast.walk(tree):
            if not isinstance(node, ast.Raise):
                continue
            cursor, condition, span, in_except = node, None, 1, False
            while cursor in parent:
                up = parent[cursor]
                if isinstance(up, ast.ExceptHandler):
                    in_except = True
                    break
                if isinstance(up, ast.If) and any(cursor is s for s in up.body):
                    condition = ast.unparse(up.test)
                    span = (up.end_lineno or up.lineno) - up.lineno + 1
                    break
                cursor = up
            message = message_of(node)
            cls = classify(name, node.lineno, condition or "", message, in_except)
            suspect = cls.startswith("Delete") and bool(SUSPECT_DELETE.search(message))
            rows.append(
                {
                    "review_before_delete": "yes" if suspect else "",
                    "family": "cumulative" if name in CUM else "diffuse-replay",
                    "file": name,
                    "line": node.lineno,
                    "cls": cls,
                    "destination": DESTINATION[cls],
                    "check": message,
                    "condition": (condition or "")[:160],
                    "span": span,
                    "deferred": "yes" if name in FENCED else "",
                }
            )
    return rows


ORDER = [
    "Delete/identity",
    "Delete/receipt",
    "Test",
    "Behavioral",
    "InputValidation",
    "Operational",
    "Other",
]


def main():
    rows = collect()
    out = pathlib.Path("docs/08 - Analysis/12 - Verification Relocation Map.csv")
    fields = [
        "family", "file", "line", "cls", "review_before_delete", "destination",
        "check", "condition", "span", "deferred",
    ]
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in sorted(rows, key=lambda r: (r["family"], r["file"], r["line"])):
            writer.writerow({k: row[k] for k in fields})

    counts = collections.Counter((r["family"], r["cls"]) for r in rows)
    loc = collections.Counter()
    for row in rows:
        loc[(row["family"], row["cls"])] += row["span"]

    print(f"{'class':<18}{'cum#':>6}{'cumLOC':>8}{'dif#':>6}{'difLOC':>8}{'tot#':>7}{'totLOC':>8}")
    for key in ORDER:
        a, b = counts[("cumulative", key)], counts[("diffuse-replay", key)]
        la, lb = loc[("cumulative", key)], loc[("diffuse-replay", key)]
        print(f"{key:<18}{a:>6}{la:>8}{b:>6}{lb:>8}{a + b:>7}{la + lb:>8}")
    ca = sum(counts[("cumulative", k)] for k in ORDER)
    cb = sum(counts[("diffuse-replay", k)] for k in ORDER)
    lca = sum(loc[("cumulative", k)] for k in ORDER)
    lcb = sum(loc[("diffuse-replay", k)] for k in ORDER)
    print(f"{'TOTAL':<18}{ca:>6}{lca:>8}{cb:>6}{lcb:>8}{ca + cb:>7}{lca + lcb:>8}")

    print("\n=== per-file Delete counts ===")
    per_file = collections.Counter(
        r["file"] for r in rows if r["cls"].startswith("Delete")
    )
    for name, n in per_file.most_common():
        print(f"{n:>5}  {name}")

    suspects = [r for r in rows if r["review_before_delete"]]
    deletes = [r for r in rows if r["cls"].startswith("Delete")]
    print(
        "\n=== Delete rows needing individual review before V3: "
        f"{len(suspects)} of {len(deletes)} "
        f"({100 * len(suspects) / len(deletes):.1f}%) ==="
    )
    for row in suspects:
        print(f"  {row['file']}:{row['line']}  {row['check'][:66]}")

    leftover = [r for r in rows if r["cls"] == "Other"]
    print(f"\n=== remaining Other: {len(leftover)} ===")
    for row in leftover:
        print(f"  {row['file']}:{row['line']}  {row['check'][:68]}")

    scratch = pathlib.Path(
        r"C:/Users/Keytone/AppData/Local/Temp/claude"
        r"/f--Keytone-Documents-GitHub-memory-condense"
        r"/df76f76e-e421-442b-8bc3-cfe361373236/scratchpad/checks.json"
    )
    scratch.write_text(json.dumps(rows, indent=0), encoding="utf-8")


if __name__ == "__main__":
    main()
