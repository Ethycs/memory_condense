"""Classify every runtime check in the cumulative and diffuse-replay families.

Charter: ``docs/06 - Roadmaps/03 - Verification Relocation Charter.md``
Emits ``docs/08 - Analysis/12 - Verification Relocation Map.csv``.

**Classification is structural, not textual.**  An earlier revision matched
regexes against the raise message and produced a large false-Delete
population: any check whose message happened to contain "changed", "receipt"
or "parent" was swept into Delete regardless of what it asserted.
``self.prompt_workspace_token_proxy != self.prompt_token_proxy +
self.responder_output_token_reserve`` is addition; ``type(x) is not bool`` is
a type check.  Neither is a hash comparison, and no amount of rule reordering
fixes that — text about code is not code.

So each check is classified by the **shape of its guarding condition's AST**:

  recompute(...) != stored_digest      -> Delete/identity  (recomputation)
  stored_digest_a != stored_digest_b   -> Delete/identity  (cross-check)
  type(x) is not T                     -> InputValidation
  len(x) != N                          -> Test             (arity)
  x not in {literal, ...}              -> Test             (enum)
  a != b + c                           -> Test             (arithmetic)
  a[:n] != b                           -> Behavioral       (prefix nesting)
  set(a) & set(b)                      -> Behavioral       (re-admission)
  a <= b  on id collections            -> Test             (ownership)

The raise message is used only to break ties the AST leaves ambiguous, and
never to override a structural verdict.

    python scripts/classify_verification_checks.py
"""

from __future__ import annotations

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

FENCED = {
    "recall_guarded_cumulative_fast_artifact.py",
    "fast_cav_feature_session.py",
    "run_fast_1m_cav.py",
    "consolidation_replay.py",
}

# Functions that derive a digest from live data.  A comparison with one of
# these on either side is a recomputation cross-check.
HASH_CALLS = re.compile(
    r"^(identity_sha256|quote_sha256|sha256_digest|.*_sha256|.*_digest"
    r"|canonical_json_bytes|_canonical_json_bytes)$"
)

# Attributes/names that hold an already-stored digest.
DIGEST_NAME = re.compile(r"(sha256|sha_256|_digest|digest_|fingerprint)$", re.I)


# --------------------------------------------------------------------------
# AST feature extraction over one guarding condition
# --------------------------------------------------------------------------


def calls_hash(node) -> bool:
    for child in ast.walk(node):
        if not isinstance(child, ast.Call):
            continue
        func = child.func
        name = (
            func.id
            if isinstance(func, ast.Name)
            else func.attr
            if isinstance(func, ast.Attribute)
            else ""
        )
        if name and HASH_CALLS.match(name):
            return True
    return False


def names_digest(node) -> bool:
    for child in ast.walk(node):
        if isinstance(child, ast.Attribute) and DIGEST_NAME.search(child.attr):
            return True
        if isinstance(child, ast.Name) and DIGEST_NAME.search(child.id):
            return True
    return False


def is_type_check(node) -> bool:
    """``type(x) is not T`` / ``not isinstance(x, T)``."""
    for child in ast.walk(node):
        if isinstance(child, ast.Compare):
            left = child.left
            if (
                isinstance(left, ast.Call)
                and isinstance(left.func, ast.Name)
                and left.func.id == "type"
                and any(isinstance(op, (ast.Is, ast.IsNot)) for op in child.ops)
            ):
                return True
        if (
            isinstance(child, ast.Call)
            and isinstance(child.func, ast.Name)
            and child.func.id == "isinstance"
        ):
            return True
    return False


def has_len_call(node) -> bool:
    return any(
        isinstance(c, ast.Call)
        and isinstance(c.func, ast.Name)
        and c.func.id == "len"
        for c in ast.walk(node)
    )


def has_membership_literal(node) -> bool:
    """``x not in {'a', 'b'}`` — an enum check."""
    for child in ast.walk(node):
        if isinstance(child, ast.Compare) and any(
            isinstance(op, (ast.In, ast.NotIn)) for op in child.ops
        ):
            for comparator in child.comparators:
                if isinstance(comparator, (ast.Set, ast.Tuple, ast.List)):
                    if all(
                        isinstance(e, ast.Constant) for e in comparator.elts
                    ):
                        return True
    return False


def has_arithmetic(node) -> bool:
    return any(
        isinstance(c, ast.BinOp) and isinstance(c.op, (ast.Add, ast.Sub, ast.Mult))
        for c in ast.walk(node)
    )


def has_slice(node) -> bool:
    """``a[: len(b)] != b`` — the prefix/nesting shape."""
    return any(
        isinstance(c, ast.Subscript) and isinstance(c.slice, ast.Slice)
        for c in ast.walk(node)
    )


def has_set_op(node) -> bool:
    """set intersection or subset — re-admission / ownership."""
    for child in ast.walk(node):
        if isinstance(child, ast.BinOp) and isinstance(child.op, ast.BitAnd):
            return True
        if isinstance(child, ast.Compare) and any(
            isinstance(op, (ast.LtE, ast.GtE, ast.Lt, ast.Gt)) for op in child.ops
        ):
            if any(
                isinstance(c, ast.Call)
                and isinstance(c.func, ast.Name)
                and c.func.id == "set"
                for c in ast.walk(child)
            ):
                return True
    return False


def has_ordering_compare(node) -> bool:
    """``a > b`` on scalars — a cap or budget comparison."""
    for child in ast.walk(node):
        if isinstance(child, ast.Compare) and any(
            isinstance(op, (ast.Lt, ast.Gt, ast.LtE, ast.GtE)) for op in child.ops
        ):
            return True
    return False


STORED_ROOT = re.compile(r"^(receipt|compilation|artifact|manifest|snapshot)$", re.I)


def reaches_stored_field(operand) -> bool:
    """``self.receipt.x`` / ``obj.compilation.y`` — a persisted payload field."""
    while isinstance(operand, ast.Attribute):
        if STORED_ROOT.match(operand.attr):
            return True
        operand = operand.value
        if isinstance(operand, ast.Attribute) and STORED_ROOT.match(operand.attr):
            return True
    return False


def recomputes_against_stored(node) -> bool:
    """One side is a call over live data, the other reads a stored field."""
    for child in ast.walk(node):
        if not isinstance(child, ast.Compare):
            continue
        operands = (child.left, *child.comparators)
        has_call = any(isinstance(o, ast.Call) for o in operands)
        has_stored = any(reaches_stored_field(o) for o in operands)
        if has_call and has_stored:
            return True
    return False


def is_bare_truthiness(node) -> bool:
    """``not x`` / ``x`` / ``path.exists()`` — a presence test, no comparison."""
    return not any(isinstance(c, ast.Compare) for c in ast.walk(node))


def compares_module_constant(node) -> bool:
    """``x != SOME_CONSTANT`` — a format/schema check against a frozen literal."""
    for child in ast.walk(node):
        if not isinstance(child, ast.Compare):
            continue
        for operand in (child.left, *child.comparators):
            if isinstance(operand, ast.Name) and operand.id.isupper():
                return True
            if isinstance(operand, ast.Attribute) and operand.attr.isupper():
                return True
    return False


def compares_two_stored_chains(node) -> bool:
    """Both sides reach through a stored object (``.receipt.``/``.compilation.``).

    This is a cross-check between two persisted payloads, not a property of a
    live computation.
    """
    def chain_depth(operand):
        depth = 0
        while isinstance(operand, ast.Attribute):
            depth += 1
            operand = operand.value
        return depth

    for child in ast.walk(node):
        if not isinstance(child, ast.Compare):
            continue
        operands = (child.left, *child.comparators)
        if len(operands) < 2:
            continue
        if all(chain_depth(o) >= 2 for o in operands):
            return True
    return False


def is_identity_flag_check(node) -> bool:
    """``x is not False`` / ``x is not True`` — a policy flag, not an identity."""
    for child in ast.walk(node):
        if isinstance(child, ast.Compare) and any(
            isinstance(op, (ast.Is, ast.IsNot)) for op in child.ops
        ):
            if any(
                isinstance(c, ast.Constant) and isinstance(c.value, bool)
                for c in child.comparators
            ):
                return True
    return False


def is_none_check(node) -> bool:
    for child in ast.walk(node):
        if isinstance(child, ast.Compare) and any(
            isinstance(op, (ast.Is, ast.IsNot)) for op in child.ops
        ):
            if any(
                isinstance(c, ast.Constant) and c.value is None
                for c in child.comparators
            ):
                return True
    return False


# Message patterns, used ONLY where the AST is genuinely ambiguous.
BEHAVIORAL_MSG = re.compile(
    r"prefix|cumulative union|duplicate|monoton|not an earlier stage"
    r"|immediate predecessor|no longer cumulative|complete evidence set",
    re.I,
)
OWNERSHIP_MSG = re.compile(r"belongs to another|owned|ownership", re.I)
RECEIPT_MSG = re.compile(
    r"receipt|certif|reservation|provenance|eligib|attest|lineage", re.I
)
VALIDATION_MSG = re.compile(
    r"must be |must contain|must equal|must map|must return|must use|require"
    r"|cannot be empty|unsupported|invalid|non-empty|non-finite|non-string"
    r"|non-json|non-numeric|schema|malformed|noncanonical|not canonical",
    re.I,
)
OPERATIONAL_MSG = re.compile(
    r"does not exist|refusing to replace|cannot (read|write|seek|flush|open"
    r"|close|create|inspect|enumerate|resolve|atomically)|no historical"
    r"|undated turn|unembedded chunk|lost its|not bound before|not produced"
    r"|could not be rehydrated|is unavailable|corpus|staging|handle|clobber"
    r"|snapshot is closed|short .* write",
    re.I,
)


def classify(test_node, message: str, in_except: bool) -> tuple[str, str]:
    """Return ``(class, why)``.  Structure decides; message only breaks ties."""
    if in_except:
        return "Operational", "raised from an except handler"

    if test_node is None:
        if OPERATIONAL_MSG.search(message):
            return "Operational", "no condition; operational message"
        if RECEIPT_MSG.search(message):
            return "Delete/receipt", "no condition; receipt message"
        return "InputValidation", "no condition; presence check"

    # --- structural shapes, most specific first -------------------------
    if is_type_check(test_node):
        return "InputValidation", "type()/isinstance() check"

    if has_slice(test_node):
        return "Behavioral", "slice comparison (ordered-prefix nesting)"

    if has_set_op(test_node):
        if BEHAVIORAL_MSG.search(message):
            return "Behavioral", "set operation (re-admission)"
        return "Test", "set operation (ownership/subset)"

    if has_membership_literal(test_node):
        return "Test", "membership in a literal set (enum)"

    # A call on one side compared against a stored field on the other is a
    # recomputation cross-check, whether or not the callee is a hash. e.g.
    # count_chat_prompt_token_proxy(self.messages) != self.receipt.prompt_token_proxy
    if recomputes_against_stored(test_node):
        return "Delete/identity", "recomputes a value and compares to a stored field"

    recompute = calls_hash(test_node)
    digest = names_digest(test_node)

    if recompute and digest:
        return "Delete/identity", "recomputes a digest and compares to a stored one"
    if recompute:
        return "Delete/identity", "derives a digest for comparison"

    if has_arithmetic(test_node):
        return "Test", "arithmetic comparison"

    if has_len_call(test_node):
        return "Test", "len() arity comparison"

    if digest:
        # A stored digest is named, but nothing is recomputed. If the shape is
        # a presence or ordering test, it is structural, not an identity
        # cross-check — this is precisely where the textual rules failed.
        if is_none_check(test_node):
            return "Behavioral", "presence of a parent link (structural)"
        if has_ordering_compare(test_node):
            return "Test", "ordering/cap comparison"
        return "Delete/identity", "compares stored digest fields"

    if has_ordering_compare(test_node):
        return "Test", "ordering/cap comparison"

    if is_identity_flag_check(test_node):
        return "Test", "boolean policy-flag check"

    if is_bare_truthiness(test_node):
        if OPERATIONAL_MSG.search(message):
            return "Operational", "presence test; operational message"
        return "InputValidation", "presence test (no comparison)"

    if compares_module_constant(test_node):
        return "InputValidation", "compares against a frozen module constant"

    if compares_two_stored_chains(test_node):
        return "Delete/identity", "cross-check between two stored payloads"

    # --- no structural signal: fall back to the message -----------------
    if BEHAVIORAL_MSG.search(message):
        return "Behavioral", "message: nesting/monotonicity"
    if OPERATIONAL_MSG.search(message):
        return "Operational", "message: operational"
    if OWNERSHIP_MSG.search(message):
        return "Test", "message: ownership"
    if RECEIPT_MSG.search(message):
        return "Delete/receipt", "message: receipt/certification bookkeeping"
    if VALIDATION_MSG.search(message):
        return "InputValidation", "message: type/format validation"
    return "Test", "equality of two derived values"


DESTINATION = {
    "Delete/identity": "delete (interior identity cross-check)",
    "Delete/receipt": "delete (receipt/certification bookkeeping)",
    "Test": "move to pytest over the pure transformation",
    "Behavioral": "KEEP in-path (recall guard: monotonic nesting)",
    "InputValidation": "keep in place (ordinary type/format validation)",
    "Operational": "keep in place (OS / TOCTOU / precondition)",
}


def message_of(node) -> str:
    parts: list[str] = []

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


def collect():
    rows = []
    for name in FILES:
        path = ROOT / name
        if not path.exists():
            continue
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        parent = {}
        for node in ast.walk(tree):
            for child in ast.iter_child_nodes(node):
                parent[child] = node
        for node in ast.walk(tree):
            if not isinstance(node, ast.Raise):
                continue
            cursor, test_node, span, in_except = node, None, 1, False
            while cursor in parent:
                up = parent[cursor]
                if isinstance(up, ast.ExceptHandler):
                    in_except = True
                    break
                if isinstance(up, ast.If) and any(cursor is s for s in up.body):
                    test_node = up.test
                    span = (up.end_lineno or up.lineno) - up.lineno + 1
                    break
                cursor = up
            message = message_of(node)
            cls, why = classify(test_node, message, in_except)
            rows.append(
                {
                    "family": "cumulative" if name in CUM else "diffuse-replay",
                    "file": name,
                    "line": node.lineno,
                    "cls": cls,
                    "why": why,
                    "destination": DESTINATION[cls],
                    "check": message,
                    "condition": (
                        ast.unparse(test_node) if test_node is not None else ""
                    )[:160],
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
]


def main() -> None:
    rows = collect()
    out = pathlib.Path("docs/08 - Analysis/12 - Verification Relocation Map.csv")
    fields = [
        "family", "file", "line", "cls", "why", "destination",
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

    print("\n=== decided by AST shape vs by message fallback ===")
    fallback = sum(1 for r in rows if r["why"].startswith("message:"))
    print(f"  structural : {len(rows) - fallback:>5}  ({100 * (len(rows) - fallback) / len(rows):.1f}%)")
    print(f"  message    : {fallback:>5}  ({100 * fallback / len(rows):.1f}%)")

    print("\n=== why, by frequency ===")
    for why, n in collections.Counter(r["why"] for r in rows).most_common():
        print(f"{n:>5}  {why}")

    scratch = pathlib.Path(
        r"C:/Users/Keytone/AppData/Local/Temp/claude"
        r"/f--Keytone-Documents-GitHub-memory-condense"
        r"/df76f76e-e421-442b-8bc3-cfe361373236/scratchpad/checks.json"
    )
    if scratch.parent.exists():
        scratch.write_text(json.dumps(rows, indent=0), encoding="utf-8")


if __name__ == "__main__":
    main()
