"""Reattach augment stores that the namespace-scheme change renamed.

One-time migration, for this beta only. Delete this file once the stores that
matter have been moved.

Why it exists. A conversation's store lives at ``<data-dir>/sessions/<namespace>``.
The namespace used to be ``sha256([provider, sha256(credentials), conversation])``
and is now ``sha256([provider, conversation])``. The credential hash is gone,
because a rotating OAuth token silently forked a live conversation's memory on
every refresh. Every existing directory therefore has the wrong name, and the old
name cannot be recomputed: it hashed a credential that has since rotated.

Why it is a script and not a proxy feature. The rename is a one-time event. A
permanent header on every request, for a migration that happens once, is not worth
its own surface. Nothing here is imported by the proxy.

Why it cannot be automatic. A store names itself ``proxy:<namespace>`` and never
records the conversation id, which arrives only in a request header. So a directory
cannot say which conversation it holds. Only its contents can. ``--list`` prints
those contents; you supply the conversation id for ``--rename``.

    # 1. see what the orphaned stores hold
    python -m tools.migrate_namespace_scheme --data-dir DIR --list

    # 2. compute the new name for a conversation, and check it
    python -m tools.migrate_namespace_scheme --data-dir DIR \
        --rename OLDNAME --conversation 'session:1f2e...' --provider anthropic

    # 3. do it
    python -m tools.migrate_namespace_scheme --data-dir DIR \
        --rename OLDNAME --conversation 'session:1f2e...' --provider anthropic --apply
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
import sys


def target_namespace(provider: str, conversation: str) -> str:
    """The new directory name, by the same rule the proxy now uses.

    Mirrors ``MemoryProxy.namespace``. Kept as its own two lines on purpose: this
    script must keep working after that function changes again, and it must not
    import the proxy to do a file rename.
    """
    payload = json.dumps([provider, conversation.strip()], ensure_ascii=False,
                         separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _first_messages(store: Path, limit: int = 3) -> list[str]:
    """The opening user text of a store, which is how you recognise it."""
    journal = store / "chat" / "chat.db"
    if not journal.is_file():
        for candidate in sorted(store.rglob("*.db")):
            journal = candidate
            break
        else:
            return []
    try:
        with sqlite3.connect(f"file:{journal}?mode=ro", uri=True) as db:
            rows = db.execute(
                "SELECT role, text FROM events WHERE role='user' ORDER BY sequence LIMIT ?",
                (limit,),
            ).fetchall()
    except sqlite3.Error as exc:
        return [f"<could not read {journal.name}: {exc}>"]
    return [" ".join(text.split())[:140] for _, text in rows]


def _stores(sessions: Path) -> list[Path]:
    if not sessions.is_dir():
        return []
    return sorted(p for p in sessions.iterdir() if p.is_dir())


def command_list(sessions: Path) -> int:
    stores = _stores(sessions)
    if not stores:
        print(f"no stores under {sessions}")
        return 1
    print(f"{len(stores)} store(s) under {sessions}\n")
    for store in stores:
        size = sum(f.stat().st_size for f in store.rglob("*") if f.is_file())
        print(f"{store.name}  ({size / 1e6:.1f} MB)")
        messages = _first_messages(store)
        if not messages:
            print("    (no user messages found)")
        for text in messages:
            print(f"    {text}")
        print()
    print("Pick the conversation id each store belongs to, then use --rename.")
    return 0


def command_rename(sessions: Path, old: str, provider: str, conversation: str, apply: bool) -> int:
    source = sessions / old
    if not source.is_dir():
        print(f"no store named {old} under {sessions}", file=sys.stderr)
        return 1
    new = target_namespace(provider, conversation)
    target = sessions / new
    print(f"provider     {provider}")
    print(f"conversation {conversation}")
    print(f"from         {source}")
    print(f"to           {target}")
    if target.exists():
        print("\nrefused: the target already exists. Move or delete it first, so that "
              "this never merges two stores.", file=sys.stderr)
        return 2
    if not apply:
        print("\nnothing changed. Add --apply to rename.")
        return 0
    source.rename(target)
    print("\nrenamed. Start the proxy and send one turn of that conversation to check it.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="migrate_namespace_scheme", description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", required=True,
                        help="the proxy's data directory, which holds sessions/")
    parser.add_argument("--list", action="store_true", help="show what each store holds")
    parser.add_argument("--rename", metavar="OLDNAME", help="the store directory to rename")
    parser.add_argument("--conversation", help="the conversation id, for --rename")
    parser.add_argument("--provider", choices=("openai", "anthropic"), default="anthropic",
                        help="the provider of that conversation, for --rename")
    parser.add_argument("--apply", action="store_true",
                        help="do the rename; without this the script only reports")
    args = parser.parse_args(argv)

    sessions = Path(args.data_dir).expanduser() / "sessions"
    if args.list:
        return command_list(sessions)
    if args.rename:
        if not args.conversation or not args.conversation.strip():
            parser.error("--rename needs --conversation")
        return command_rename(sessions, args.rename, args.provider,
                              args.conversation, args.apply)
    parser.error("give --list or --rename")


if __name__ == "__main__":
    raise SystemExit(main())
