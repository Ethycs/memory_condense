import json, sys
sys.path.insert(0, r"C:/Users/Keytone/AppData/Local/Temp/claude/f--Keytone-Documents-GitHub-memory-condense/8f7f7561-e2af-4bab-8f13-8fab1e2d71bb/scratchpad")
from cc_questions import QA
from memory_condense.eval.recall import contains_answer

turns = json.load(open(r"C:/Users/Keytone/AppData/Local/Temp/claude/f--Keytone-Documents-GitHub-memory-condense/8f7f7561-e2af-4bab-8f13-8fab1e2d71bb/scratchpad/session_turns.json", encoding="utf-8"))
texts = [t for _, _, t in turns]

kept, dropped = [], []
for q, a, cat in QA:
    hits = sum(contains_answer([t], a) for t in texts)
    if hits == 0:
        dropped.append((q, a, "absent"))
    elif hits > 25:
        dropped.append((q, a, f"ubiquitous({hits})"))
    else:
        kept.append((q, a, cat, hits))

print(f"authored {len(QA)}  kept {len(kept)}  dropped {len(dropped)}")
for q, a, why in dropped:
    print(f"  DROP [{why}] {a!r}")
from collections import Counter
print("kept by phase:", dict(Counter(c for _,_,c,_ in kept)))
print("turn-hit distribution:", sorted(h for _,_,_,h in kept))
json.dump(kept, open(r"C:/Users/Keytone/AppData/Local/Temp/claude/f--Keytone-Documents-GitHub-memory-condense/8f7f7561-e2af-4bab-8f13-8fab1e2d71bb/scratchpad/cc_probe.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
