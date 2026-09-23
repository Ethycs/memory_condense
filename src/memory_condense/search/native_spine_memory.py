"""Complete native summary bodies materialized at exact source occurrences."""
from dataclasses import dataclass
from datetime import datetime
from types import MappingProxyType

from memory_condense.domain._discourse_identity import identity_sha256, quote_sha256
from memory_condense.domain._tokenizer import count_tokens
from memory_condense.domain.schemas import Turn
from memory_condense.search.native_spine_summary import BodyFragment, body_identity, materialize


def validate_body_summaries(body, summaries):
    """Require ordered, gap-free coverage of every complete original raw turn."""
    sha = body_identity(body)
    cursor_turn = cursor_char = 0
    turn_hashes = [quote_sha256(t["text"]) for t in body["turns"]]
    for atom in summaries:
        if type(atom) is not dict or set(atom) != {"pointer", "summary"}:
            raise ValueError("only source-bound routing summaries are accepted")
        p = atom["pointer"]
        if (type(p) is not dict or any(type(p.get(k)) is not int for k in
                ("turn_ordinal", "start_char", "end_char", "token_count"))
                or cursor_turn >= len(body["turns"])
                or p.get("body_sha256") != sha or p.get("turn_ordinal") != cursor_turn
                or p.get("start_char") != cursor_char):
            raise ValueError("summary body coverage has a gap, overlap or wrong source")
        turn = body["turns"][cursor_turn]
        end = p.get("end_char")
        if type(end) is not int or not cursor_char < end <= len(turn["text"]):
            raise ValueError("summary fragment is outside its raw turn")
        fragment = BodyFragment(sha, cursor_turn, turn["role"], cursor_char, end,
                                turn_hashes[cursor_turn], turn["text"][cursor_char:end])
        if fragment.pointer() != p:
            raise ValueError("summary raw pointer, speaker, hash or token count changed")
        summary = atom["summary"]
        if (type(summary) is not str or not summary.strip() or len(summary.split()) > 96
                or count_tokens(summary) > 128):
            raise ValueError("invalid native routing summary")
        cursor_char = end
        if cursor_char == len(turn["text"]):
            cursor_turn += 1
            cursor_char = 0
    if cursor_turn != len(body["turns"]) or cursor_char:
        raise ValueError("summary body does not cover every complete raw turn")
    return sha


@dataclass(frozen=True)
class NativeHistory:
    atoms: tuple
    turns: object
    occurrence_ids: tuple

    def get_turn(self, turn_id):
        return self.turns.get(turn_id)


def materialize_history(sessions, *, load_body, load_summaries, compiler_identity):
    """Resolve one namespace only; dates and raw text never enter a model callback."""
    if not sessions:
        raise ValueError("native history requires actual source occurrences")
    bodies, summaries = {}, {}
    atoms, turns, seen = [], {}, set()
    source_fields = {"original_session_ordinal", "session_id", "created_at", "metadata_text",
                     "body_sha256", "dataset_origin", "occurrence_id"}
    for source in sessions:
        if type(source) is not dict or set(source) != source_fields:
            raise ValueError("native source occurrence fields changed")
        occurrence = source["occurrence_id"]
        if (occurrence != identity_sha256({k: v for k, v in source.items() if k != "occurrence_id"})
                or occurrence in seen):
            raise ValueError("source occurrence identity changed or is duplicated")
        created = datetime.fromisoformat(source["created_at"])
        if created.tzinfo is None or created.isoformat() != source["created_at"]:
            raise ValueError("a canonical actual occurrence timestamp is required")
        seen.add(occurrence)
        sha = source["body_sha256"]
        if sha not in bodies:
            body, cached = load_body(sha), tuple(load_summaries(sha))
            if validate_body_summaries(body, cached) != sha:
                raise ValueError("body loader returned another source")
            bodies[sha], summaries[sha] = body, cached
        body = bodies[sha]
        for cached in summaries[sha]:
            atom = materialize(cached, body, occurrence_id=occurrence,
                               created_at=source["created_at"], compiler_identity=compiler_identity)
            span = atom.spans[0]
            raw = body["turns"][cached["pointer"]["turn_ordinal"]]
            turn = Turn(turn_id=span.turn_id, source_id=span.source_id, role=raw["role"],
                        text=raw["text"], created_at=created)
            if span.turn_id in turns and turns[span.turn_id] != turn:
                raise ValueError("a raw turn identity aliases different evidence")
            turns[span.turn_id] = turn
            atoms.append(atom)
    return NativeHistory(tuple(atoms), MappingProxyType(turns),
                         tuple(source["occurrence_id"] for source in sessions))
