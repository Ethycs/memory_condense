"""Subdivide failed raw atoms without losing text or changing valid summaries."""
from memory_condense.search.native_spine_batch import admit
from memory_condense.search.native_spine_repair import partition
from memory_condense.search.native_spine_summary import BodyFragment, fragment_body


def subdivide(fragment):
    cap = min(512, max(1, fragment.pointer()["token_count"] // 2))
    pieces = fragment_body({"turns": [{"role": fragment.role, "text": fragment.text}]}, token_cap=cap)
    result = tuple(BodyFragment(fragment.body_sha256, fragment.turn_ordinal, fragment.role,
                               fragment.start_char + part.start_char, fragment.start_char + part.end_char,
                               fragment.turn_text_sha256, part.text) for part in pieces)
    if len(result) < 2 or "".join(part.text for part in result) != fragment.text:
        raise ValueError("subdivision must preserve the whole fragment in smaller parts")
    return result


def reconcile_sections(response, fragments, replacements):
    valid, bad = partition(response, fragments)
    if set(replacements) != set(bad):
        raise ValueError("replacement sections must cover exactly the invalid atoms")
    result = []
    for index, original in enumerate(fragments):
        if index in valid:
            result.append(valid[index])
            continue
        atoms = replacements[index]
        if not atoms:
            raise ValueError("an original fragment cannot disappear")
        cursor = original.start_char
        for atom in atoms:
            p = atom["pointer"]
            if (any(type(p.get(k)) is not int for k in ("start_char", "end_char", "turn_ordinal", "token_count"))
                    or p["start_char"] != cursor or p["end_char"] > original.end_char
                    or any(p[k] != original.pointer()[k] for k in
                           ("body_sha256", "turn_ordinal", "role", "turn_text_sha256"))):
                raise ValueError("replacement fragments changed original coverage or attribution")
            piece = BodyFragment(original.body_sha256, original.turn_ordinal, original.role,
                                 cursor, p["end_char"], original.turn_text_sha256,
                                 original.text[cursor-original.start_char:p["end_char"]-original.start_char])
            if piece.pointer() != p:
                raise ValueError("replacement raw pointer changed")
            # Reuse the same source-bound summary contract on each smaller part.
            from memory_condense.domain._discourse_identity import canonical_json
            accepted = admit(canonical_json({"atoms": [{"label": "T0", "summary": atom["summary"]}]}),
                             (piece,))[0]
            if accepted != atom:
                raise ValueError("replacement contains unsupported fields")
            result.append(atom)
            cursor = p["end_char"]
        if cursor != original.end_char:
            raise ValueError("replacement fragments do not cover the original suffix")
    return tuple(result)
