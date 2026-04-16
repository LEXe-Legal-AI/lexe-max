-- sprint27_001_legacy_delete.sql
-- Sprint 27 (S6.1): hard delete of the 1,229 legacy articles with
-- canonical_source IS NULL (OCR-degraded imports) for the 12 legacy codes.
--
-- STRATEGY — accept FK CASCADE, rebuild children from OpenData bundle
-- ---------------------------------------------------------------------------
-- T2 verdict (docs/sprint27_fk_check.md) originally proposed strategy A
-- (ALTER FK ON DELETE SET NULL to preserve vigenza/chunk rows). However:
--   - kb.normativa_vigenza.normativa_id  is NOT NULL → SET NULL blocked
--   - kb.normativa_modification.target_normativa_id is NOT NULL → same
-- T2-BIS (docs/sprint27_t4_smoke_zip.md) verified the "Testi Unici" ZIP
-- bundle contains full multi-version vigenza (TUIR: 244 versions, TUEL: 146,
-- TUSG: 102, TUDA: 43, TUESP: 36). The Sprint 16 Platinum ingest script
-- `ingest_opendata_codici.py` already rebuilds all 6 child tables
-- (normativa_chunk, _embeddings, _vigenza, _modification, _citations,
-- _updates) from the bundle.
-- → Clean cut + clean rebuild is simpler and equivalent.
--
-- RUNBOOK (all steps in maintenance window):
--   1. psql < sprint27_001_legacy_delete.sql            ← this file
--   2. python scripts/legacy_sanitizer.py --all --execute
--      (orchestrates ingest_opendata_codici.py per code)
--   3. psql < sprint27_002_post_ingest_vacuum.sql
--
-- ROLLBACK: restore from kb.normativa_legacy_backup_20260416 via INSERT …
-- SELECT. Cascade rows are lost but equivalent re-ingest can repeat step 2.

BEGIN;

SET LOCAL lock_timeout = '30s';
SET LOCAL statement_timeout = '15min';

-- ---------------------------------------------------------------------------
-- 0. Pre-conditions (fail loudly if state drifted)
-- ---------------------------------------------------------------------------
DO $$
DECLARE
    v_pending_legacy INTEGER;
    v_missing_codes  INTEGER;
    v_expected_codes TEXT[] := ARRAY[
        -- Sprint 27 scope 9 (COST/TUSL/CMED/CGS excluded — bundle misses, Sprint 28 backlog)
        'TUB','TUF','TUIR','TUEL','TUI','TUSG','TUIST','TUDA','TUESP'
    ];
BEGIN
    -- 0a. All 13 code slots must exist in kb.work (else ingest hasn't seeded)
    SELECT COUNT(*) INTO v_missing_codes
    FROM UNNEST(v_expected_codes) AS expected(code)
    WHERE NOT EXISTS (SELECT 1 FROM kb.work w WHERE w.code = expected.code);

    IF v_missing_codes > 0 THEN
        RAISE EXCEPTION
            'sprint27_001: % expected work codes are missing from kb.work — seed Sprint 16 Platinum ingest first',
            v_missing_codes;
    END IF;

    -- 0b. Count legacy candidates to be deleted (informational)
    SELECT COUNT(*) INTO v_pending_legacy
    FROM kb.normativa n
    JOIN kb.work w ON w.id = n.work_id
    WHERE w.code = ANY(v_expected_codes)
      AND n.canonical_source IS NULL
      AND n.validation_status IS DISTINCT FROM 'verified';

    RAISE NOTICE 'sprint27_001: % legacy rows identified for delete', v_pending_legacy;
END $$;

-- ---------------------------------------------------------------------------
-- 1. Backups (normativa + chunk + chunk_embeddings — the only 3 tables with
--    rows linked to legacy per baseline 2026-04-16; vigenza/citations/updates/
--    modification contain ZERO rows linked to legacy on stage so no backup).
-- ---------------------------------------------------------------------------
DROP TABLE IF EXISTS kb.normativa_legacy_backup_20260416;
DROP TABLE IF EXISTS kb.normativa_chunk_legacy_backup_20260416;
DROP TABLE IF EXISTS kb.normativa_chunk_embeddings_legacy_backup_20260416;

-- 1a. Parent table backup
CREATE TABLE kb.normativa_legacy_backup_20260416 AS
SELECT
    n.*,
    w.code AS _backup_work_code,
    now() AS _backup_at
FROM kb.normativa n
JOIN kb.work w ON w.id = n.work_id
WHERE w.code IN (
        -- Sprint 27 scope 9 (COST/TUSL/CMED/CGS excluded — bundle misses, Sprint 28 backlog)
        'TUB','TUF','TUIR','TUEL','TUI','TUSG','TUIST','TUDA','TUESP'
      )
  AND n.canonical_source IS NULL
  AND n.validation_status IS DISTINCT FROM 'verified';

CREATE INDEX ON kb.normativa_legacy_backup_20260416 (_backup_work_code);
CREATE INDEX ON kb.normativa_legacy_backup_20260416 (urn_nir);

-- 1b. Child: normativa_chunk (baseline: 9,385 rows on stage)
CREATE TABLE kb.normativa_chunk_legacy_backup_20260416 AS
SELECT c.*, now() AS _backup_at
FROM kb.normativa_chunk c
WHERE c.normativa_id IN (SELECT id FROM kb.normativa_legacy_backup_20260416);

CREATE INDEX ON kb.normativa_chunk_legacy_backup_20260416 (normativa_id);
CREATE INDEX ON kb.normativa_chunk_legacy_backup_20260416 (id);

-- 1c. Child: normativa_chunk_embeddings (baseline: 9,385 rows on stage)
CREATE TABLE kb.normativa_chunk_embeddings_legacy_backup_20260416 AS
SELECT e.*, c.normativa_id AS _backup_normativa_id, now() AS _backup_at
FROM kb.normativa_chunk_embeddings e
JOIN kb.normativa_chunk c ON c.id = e.chunk_id
WHERE c.normativa_id IN (SELECT id FROM kb.normativa_legacy_backup_20260416);

CREATE INDEX ON kb.normativa_chunk_embeddings_legacy_backup_20260416 (chunk_id);
CREATE INDEX ON kb.normativa_chunk_embeddings_legacy_backup_20260416 (_backup_normativa_id);

-- ---------------------------------------------------------------------------
-- 2. Delete (FK CASCADE removes rows in 6 child tables). The outer BEGIN/
--    COMMIT, combined with RAISE EXCEPTION in the post-conditions DO block,
--    guarantees atomic rollback of both backups + delete if residuals > 0.
-- ---------------------------------------------------------------------------
DELETE FROM kb.normativa n
USING kb.work w
WHERE w.id = n.work_id
  AND w.code IN (
        -- Sprint 27 scope 9 (COST/TUSL/CMED/CGS excluded — bundle misses, Sprint 28 backlog)
        'TUB','TUF','TUIR','TUEL','TUI','TUSG','TUIST','TUDA','TUESP'
      )
  AND n.canonical_source IS NULL
  AND n.validation_status IS DISTINCT FROM 'verified';

-- ---------------------------------------------------------------------------
-- 3. Post-conditions
-- ---------------------------------------------------------------------------
DO $$
DECLARE
    v_residual INTEGER;
    v_deleted  INTEGER;
BEGIN
    SELECT COUNT(*) INTO v_deleted FROM kb.normativa_legacy_backup_20260416;

    -- There must be zero legacy residuals under the 13 codes
    SELECT COUNT(*) INTO v_residual
    FROM kb.normativa n
    JOIN kb.work w ON w.id = n.work_id
    WHERE w.code IN (
            -- Sprint 27 scope 9 (COST/TUSL/CMED/CGS excluded, Sprint 28 backlog)
            'TUB','TUF','TUIR','TUEL','TUI','TUSG','TUIST','TUDA','TUESP'
          )
      AND n.canonical_source IS NULL
      AND n.validation_status IS DISTINCT FROM 'verified';

    IF v_residual > 0 THEN
        -- RAISE EXCEPTION inside the DO block aborts the outer transaction
        -- (ROLLBACK TO SAVEPOINT is not allowed in PL/pgSQL — must rely on
        -- transaction-level rollback triggered by the exception).
        RAISE EXCEPTION
            'sprint27_001: % residual legacy rows after delete — rolling back',
            v_residual;
    END IF;

    RAISE NOTICE 'sprint27_001: deleted % legacy rows (cascade handled in 6 child tables)',
                 v_deleted;
END $$;

COMMIT;

-- ---------------------------------------------------------------------------
-- Post-commit checklist (manual, documented here for the DBA)
-- ---------------------------------------------------------------------------
-- [ ] Launch scripts/legacy_sanitizer.py --all --execute
-- [ ] Verify kb.normativa counts per code match T2-BIS bundle expected:
--       COST~139, TUB~162, TUF~214, TUIR~235, TUEL~295, TUSL~306,
--       TUI~46, TUSG~316, TUIST~676, TUDA~92, TUESP~70, CMED~158, CGS~152
-- [ ] Run sprint27_002_post_ingest_vacuum.sql
-- [ ] Gate A: bench COS-F-002 → ≥75 PASS, 65-74 WARN, <65 ROLLBACK
