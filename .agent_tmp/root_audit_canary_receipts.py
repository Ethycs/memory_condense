from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


def digest(value: object) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def walk(value: object, path: str = "$") -> tuple[int, list[str]]:
    checked = 0
    failures: list[str] = []
    if isinstance(value, dict):
        receipt = value.get("receipt_sha256")
        if isinstance(receipt, str) and isinstance(value.get("format"), str):
            checked += 1
            body = {key: child for key, child in value.items() if key != "receipt_sha256"}
            actual = digest(body)
            if actual != receipt:
                failures.append(f"{path}: expected={receipt} actual={actual}")
        for key, child in value.items():
            child_checked, child_failures = walk(child, f"{path}.{key}")
            checked += child_checked
            failures.extend(child_failures)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            child_checked, child_failures = walk(child, f"{path}[{index}]")
            checked += child_checked
            failures.extend(child_failures)
    return checked, failures


path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
count, errors = walk(payload)
print(json.dumps({"checked": count, "failures": errors}, sort_keys=True))
if errors:
    raise SystemExit(1)
