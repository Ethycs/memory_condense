"""Claude Code session JSONL -> turns list + corpus stats (R6).

Mirrors what an MCP ingest hook would see: user text, assistant text,
tool calls and tool results as system turns. Excludes: thinking blocks,
meta/caveat records, command wrappers, sidechains, and non-message records.
Tool results truncated to 2000 chars (a hook would store a bounded record,
and untruncated dumps run to 30k chars). Truncation is flagged in stats.
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
DEFAULT_SNAPSHOT = ROOT / "data" / "build-session-8f7f7561.snapshot.jsonl"
DEFAULT_OUTPUT = ROOT / "data" / "build-session-8f7f7561.turns.json"
TOOL_RESULT_CAP = 2000

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
args = parser.parse_args()

def block_text(b):
    if isinstance(b, str):
        return b
    if isinstance(b, dict):
        if b.get("type") == "text":
            return b.get("text", "")
        if b.get("type") == "tool_result":
            c = b.get("content")
            if isinstance(c, str):
                return c
            if isinstance(c, list):
                return "\n".join(x.get("text", "") for x in c
                                 if isinstance(x, dict) and x.get("type") == "text")
    return ""

turns = []  # (role, kind, text)
truncated = 0
with args.snapshot.open(encoding="utf-8") as f:
    for line in f:
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("type") not in ("user", "assistant") or r.get("isSidechain"):
            continue
        if r.get("isMeta"):
            continue
        msg = r.get("message") or {}
        content = msg.get("content")
        if r["type"] == "user":
            if isinstance(content, str):
                t = content.strip()
                if t.startswith("<command-name>") or t.startswith("<local-command-stdout>"):
                    continue
                if t:
                    turns.append(("user", "user_text", t))
            elif isinstance(content, list):
                for b in content:
                    if isinstance(b, dict) and b.get("type") == "tool_result":
                        t = block_text(b).strip()
                        if t:
                            if len(t) > TOOL_RESULT_CAP:
                                t = t[:TOOL_RESULT_CAP]
                                truncated += 1
                            turns.append(("system", "tool_result", t))
                    else:
                        t = block_text(b).strip()
                        if t and not t.startswith("<command-name>"):
                            turns.append(("user", "user_text", t))
        else:  # assistant
            if not isinstance(content, list):
                continue
            texts = []
            for b in content:
                if not isinstance(b, dict):
                    continue
                if b.get("type") == "text":
                    texts.append(b.get("text", ""))
                elif b.get("type") == "tool_use":
                    inp = json.dumps(b.get("input", {}), ensure_ascii=False)[:300]
                    turns.append(("system", "tool_use",
                                  f"[tool_use {b.get('name','?')}] {inp}"))
            t = "\n".join(x for x in texts if x.strip()).strip()
            if t:
                turns.append(("assistant", "assistant_text", t))

args.output.parent.mkdir(parents=True, exist_ok=True)
with args.output.open("w", encoding="utf-8") as output_file:
    json.dump(turns, output_file, ensure_ascii=False)

from memory_condense._tokenizer import count_tokens
from collections import Counter
tok_by = Counter(); n_by = Counter(); total = 0
per_turn = []
for role, kind, text in turns:
    tk = count_tokens(text)
    tok_by[kind] += tk; n_by[kind] += 1; total += tk
    per_turn.append(tk)
per_turn.sort()
n = len(per_turn)
print(f"turns: {n}   total tokens: {total:,}   (tool_results truncated: {truncated})")
print(f"t per turn: mean={total/n:.0f}  p50={per_turn[n//2]}  p90={per_turn[int(n*.9)]}  max={per_turn[-1]}")
for k in tok_by:
    print(f"  {k:<15} n={n_by[k]:>5}  tokens={tok_by[k]:>9,}  ({100*tok_by[k]/total:.0f}%)")
conv = tok_by["user_text"] + tok_by["assistant_text"]
print(f"conversation (user+assistant) tokens: {conv:,}  ({100*conv/total:.0f}%)")
B = 6200
print(f"crossover N* = 2B/t = {2*B/(total/n):.0f} turns   (session has {n})")
print(f"full-context cumulative ~ N^2*t/2 = {n*n*(total/n)/2/1e6:.0f}M tokens; bounded ~ {n*B/1e6:.1f}M")
