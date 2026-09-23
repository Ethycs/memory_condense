"""Schema fragment for durable post-index automatic enrichment."""

from __future__ import annotations

PENDING_ENRICHMENT_SCHEMA_V14 = """
CREATE UNIQUE INDEX IF NOT EXISTS idx_pending_ingests_turn_manifest
ON pending_ingests(turn_id, manifest_sha256);

CREATE TABLE IF NOT EXISTS pending_enrichments (
    turn_id TEXT PRIMARY KEY REFERENCES pending_ingests(turn_id),
    ingest_manifest_sha256 TEXT NOT NULL
        CHECK(length(ingest_manifest_sha256) = 64
              AND ingest_manifest_sha256 NOT GLOB '*[^0-9a-f]*'),
    status TEXT NOT NULL CHECK(status IN ('pending', 'enriched')),
    created_at TEXT NOT NULL,
    enriched_at TEXT,
    CHECK((status = 'pending' AND enriched_at IS NULL)
          OR (status = 'enriched' AND enriched_at IS NOT NULL)),
    FOREIGN KEY(turn_id, ingest_manifest_sha256)
        REFERENCES pending_ingests(turn_id, manifest_sha256)
);

CREATE INDEX IF NOT EXISTS idx_pending_enrichments_status
ON pending_enrichments(status, created_at, turn_id);

CREATE TRIGGER IF NOT EXISTS trg_pending_enrichments_no_delete
BEFORE DELETE ON pending_enrichments
BEGIN
    SELECT RAISE(ABORT, 'enrichment receipts are durable');
END;

CREATE TRIGGER IF NOT EXISTS trg_pending_enrichments_guard_update
BEFORE UPDATE ON pending_enrichments
WHEN NOT (
    OLD.status = 'pending' AND NEW.status = 'enriched'
    AND NEW.turn_id = OLD.turn_id
    AND NEW.ingest_manifest_sha256 = OLD.ingest_manifest_sha256
    AND NEW.created_at = OLD.created_at
    AND OLD.enriched_at IS NULL AND NEW.enriched_at IS NOT NULL
    AND EXISTS (SELECT 1 FROM pending_ingests AS p
                WHERE p.turn_id = OLD.turn_id
                  AND p.manifest_sha256 = OLD.ingest_manifest_sha256
                  AND p.status = 'indexed')
)
BEGIN
    SELECT RAISE(ABORT, 'enrichment receipts allow only indexed pending-to-enriched');
END;
"""

__all__ = ["PENDING_ENRICHMENT_SCHEMA_V14"]
