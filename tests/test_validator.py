from memory_condense.persistence.memory_store import MemoryStore
from memory_condense.domain.schemas import (
    CreateOp,
    DeleteOp,
    MemoryOps,
    MemoryType,
    PinOp,
    PinState,
    Provenance,
    SupersedeOp,
    UpdateOp,
)
from memory_condense.persistence.transcript_store import TranscriptStore
from memory_condense.ingest.validator import (
    REASON_EMPTY_CONTENT,
    REASON_MISSING_PROVENANCE,
    REASON_QUOTE_NOT_FOUND,
    REASON_UNKNOWN_MEM_ID,
    REASON_UNKNOWN_TURN,
    Validator,
    _normalize,
)


def _turn(db, text="I prefer dark mode for all my editors."):
    return TranscriptStore(db).append("user", text)


def _create(turn_id, quote, content="prefers dark mode"):
    return CreateOp(
        type=MemoryType.PREFERENCE,
        content=content,
        provenance=[Provenance(turn_id=turn_id, quote=quote)],
    )


# ----------------------------------------------------------------------
# _normalize
# ----------------------------------------------------------------------


def test_normalize_collapses_whitespace_runs():
    assert _normalize("  a\n\t b   c  ") == "a b c"


def test_normalize_empty():
    assert _normalize("   \n ") == ""


# ----------------------------------------------------------------------
# CreateOp
# ----------------------------------------------------------------------


def test_create_with_exact_quote_is_accepted(db):
    turn = _turn(db)
    report = Validator(db).validate(
        MemoryOps(create=[_create(turn.turn_id, "I prefer dark mode")])
    )
    assert report.ok
    assert len(report.accepted.create) == 1


def test_create_quote_not_found_is_rejected(db):
    turn = _turn(db)
    report = Validator(db).validate(
        MemoryOps(create=[_create(turn.turn_id, "I prefer light mode")])
    )
    assert not report.ok
    assert not report.accepted.create
    assert report.rejected[0].reason == REASON_QUOTE_NOT_FOUND
    assert report.rejected[0].op_kind == "create"
    assert "does not appear" in report.rejected[0].detail


def test_create_unknown_turn_is_rejected(db):
    _turn(db)
    report = Validator(db).validate(
        MemoryOps(create=[_create("no-such-turn", "I prefer dark mode")])
    )
    assert report.rejected[0].reason == REASON_UNKNOWN_TURN


def test_create_missing_provenance_is_rejected(db):
    op = CreateOp(type=MemoryType.DECISION, content="use postgres", provenance=[])
    report = Validator(db).validate(MemoryOps(create=[op]))
    assert report.rejected[0].reason == REASON_MISSING_PROVENANCE
    assert not report.accepted.create


def test_create_empty_content_is_rejected(db):
    turn = _turn(db)
    op = CreateOp(
        type=MemoryType.PREFERENCE,
        content="   ",
        provenance=[Provenance(turn_id=turn.turn_id, quote="I prefer dark mode")],
    )
    report = Validator(db).validate(MemoryOps(create=[op]))
    assert report.rejected[0].reason == REASON_EMPTY_CONTENT


def test_create_empty_quote_is_rejected(db):
    turn = _turn(db)
    report = Validator(db).validate(MemoryOps(create=[_create(turn.turn_id, "   ")]))
    assert report.rejected[0].reason == REASON_QUOTE_NOT_FOUND


def test_quote_matching_is_whitespace_insensitive(db):
    turn = TranscriptStore(db).append("user", "I prefer\n   dark    mode always")
    report = Validator(db).validate(
        MemoryOps(create=[_create(turn.turn_id, "I prefer dark mode")])
    )
    assert report.ok


def test_quote_matching_is_case_sensitive(db):
    turn = _turn(db)
    report = Validator(db).validate(
        MemoryOps(create=[_create(turn.turn_id, "i prefer dark mode")])
    )
    assert report.rejected[0].reason == REASON_QUOTE_NOT_FOUND


def test_all_provenance_entries_must_check_out(db):
    turn = _turn(db)
    op = CreateOp(
        type=MemoryType.PREFERENCE,
        content="prefers dark mode",
        provenance=[
            Provenance(turn_id=turn.turn_id, quote="I prefer dark mode"),
            Provenance(turn_id=turn.turn_id, quote="invented text"),
        ],
    )
    report = Validator(db).validate(MemoryOps(create=[op]))
    assert report.rejected[0].reason == REASON_QUOTE_NOT_FOUND


def test_mixed_batch_partially_accepted(db):
    turn = _turn(db)
    report = Validator(db).validate(
        MemoryOps(
            create=[
                _create(turn.turn_id, "I prefer dark mode"),
                _create(turn.turn_id, "hallucinated claim"),
            ]
        )
    )
    assert len(report.accepted.create) == 1
    assert len(report.rejected) == 1


def test_quote_matches_helper(db):
    turn = _turn(db)
    v = Validator(db)
    assert v.quote_matches(turn.turn_id, "dark mode")
    assert not v.quote_matches(turn.turn_id, "light mode")
    assert not v.quote_matches("nope", "dark mode")


# ----------------------------------------------------------------------
# mem_id-bearing ops
# ----------------------------------------------------------------------


def test_update_unknown_mem_id_is_rejected(db):
    report = Validator(db).validate(MemoryOps(update=[UpdateOp(mem_id="ghost")]))
    assert report.rejected[0].reason == REASON_UNKNOWN_MEM_ID
    assert report.rejected[0].op_kind == "update"


def test_update_existing_mem_id_is_accepted(db):
    turn = _turn(db)
    item = MemoryStore(db).create(_create(turn.turn_id, "I prefer dark mode"))
    report = Validator(db).validate(
        MemoryOps(update=[UpdateOp(mem_id=item.mem_id, content="new content")])
    )
    assert report.ok
    assert len(report.accepted.update) == 1


def test_update_with_bad_quote_is_rejected(db):
    """Provenance is optional on an update, but anything supplied must be real."""
    turn = _turn(db)
    item = MemoryStore(db).create(_create(turn.turn_id, "I prefer dark mode"))
    report = Validator(db).validate(
        MemoryOps(
            update=[
                UpdateOp(
                    mem_id=item.mem_id,
                    provenance=[Provenance(turn_id=turn.turn_id, quote="made up")],
                )
            ]
        )
    )
    assert report.rejected[0].reason == REASON_QUOTE_NOT_FOUND


def test_delete_and_pin_unknown_mem_id_rejected(db):
    report = Validator(db).validate(
        MemoryOps(
            delete=[DeleteOp(mem_id="ghost")],
            pin=[PinOp(mem_id="ghost", pin=PinState.USER)],
        )
    )
    reasons = {(e.op_kind, e.reason) for e in report.rejected}
    assert ("delete", REASON_UNKNOWN_MEM_ID) in reasons
    assert ("pin", REASON_UNKNOWN_MEM_ID) in reasons


def test_supersede_validates_replacement_like_a_create(db):
    turn = _turn(db)
    item = MemoryStore(db).create(_create(turn.turn_id, "I prefer dark mode"))

    bad = SupersedeOp(
        mem_id=item.mem_id,
        replacement=_create(turn.turn_id, "never said this", content="light mode"),
    )
    report = Validator(db).validate(MemoryOps(supersede=[bad]))
    assert report.rejected[0].op_kind == "supersede"
    assert report.rejected[0].reason == REASON_QUOTE_NOT_FOUND

    good = SupersedeOp(
        mem_id=item.mem_id,
        replacement=_create(turn.turn_id, "dark mode", content="dark mode confirmed"),
    )
    assert Validator(db).validate(MemoryOps(supersede=[good])).ok


def test_supersede_unknown_mem_id_rejected_before_replacement(db):
    turn = _turn(db)
    op = SupersedeOp(
        mem_id="ghost", replacement=_create(turn.turn_id, "I prefer dark mode")
    )
    report = Validator(db).validate(MemoryOps(supersede=[op]))
    assert report.rejected[0].reason == REASON_UNKNOWN_MEM_ID


def test_supersede_replacement_missing_provenance_rejected(db):
    turn = _turn(db)
    item = MemoryStore(db).create(_create(turn.turn_id, "I prefer dark mode"))
    op = SupersedeOp(
        mem_id=item.mem_id,
        replacement=CreateOp(type=MemoryType.PREFERENCE, content="x", provenance=[]),
    )
    report = Validator(db).validate(MemoryOps(supersede=[op]))
    assert report.rejected[0].reason == REASON_MISSING_PROVENANCE


# ----------------------------------------------------------------------
# General behaviour
# ----------------------------------------------------------------------


def test_empty_ops_validate_clean(db):
    report = Validator(db).validate(MemoryOps())
    assert report.ok
    assert report.accepted.is_empty()


def test_validator_never_raises_on_garbage(db):
    ops = MemoryOps(
        create=[CreateOp(type=MemoryType.TASK, content="", provenance=[])],
        update=[UpdateOp(mem_id="")],
        delete=[DeleteOp(mem_id="")],
        pin=[PinOp(mem_id="")],
    )
    report = Validator(db).validate(ops)
    assert len(report.rejected) == 4
    assert report.accepted.is_empty()


def test_validator_does_not_mutate_state(db):
    turn = _turn(db)
    store = MemoryStore(db)
    Validator(db).validate(MemoryOps(create=[_create(turn.turn_id, "dark mode")]))
    assert store.count() == 0
