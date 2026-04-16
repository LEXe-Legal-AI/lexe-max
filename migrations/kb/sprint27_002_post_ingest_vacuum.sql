-- sprint27_002_post_ingest_vacuum.sql
-- Sprint 27 (S6.3): post-delete + post-ingest VACUUM + REINDEX runbook.
--
-- Run AFTER `sprint27_001_legacy_delete.sql` AND after
-- `scripts/legacy_sanitizer.py --all --execute` has completed successfully.
--
-- VACUUM FULL takes an ACCESS EXCLUSIVE lock → schedule inside the
-- maintenance window. Children benefit from VACUUM ANALYZE alone (no FULL)
-- since cascade + re-ingest already removed dead tuples in bulk.
--
-- REINDEX: the HNSW index on embeddings is the most expensive by order of
-- magnitude. Rebuild it last so intermediate queries still work.
--
-- Execution time on staging (~11 GB KB, 348K embeddings): ~6–12 min total.
-- Run outside a single transaction (each statement auto-commits).

-- ---------------------------------------------------------------------------
-- 1. VACUUM FULL on normativa (rewrites table, reclaims disk)
-- ---------------------------------------------------------------------------
VACUUM (FULL, VERBOSE, ANALYZE) kb.normativa;

-- ---------------------------------------------------------------------------
-- 2. VACUUM ANALYZE on 6 child tables (dead tuples from cascade)
-- ---------------------------------------------------------------------------
VACUUM (ANALYZE, VERBOSE) kb.normativa_chunk;
VACUUM (ANALYZE, VERBOSE) kb.normativa_chunk_embeddings;
VACUUM (ANALYZE, VERBOSE) kb.normativa_vigenza;
VACUUM (ANALYZE, VERBOSE) kb.normativa_citations;
VACUUM (ANALYZE, VERBOSE) kb.normativa_updates;
VACUUM (ANALYZE, VERBOSE) kb.normativa_modification;

-- ---------------------------------------------------------------------------
-- 3. REINDEX — HNSW first, then FTS, then regular B-tree set
-- ---------------------------------------------------------------------------
-- Use CONCURRENTLY where possible to avoid table-level locks (read traffic
-- keeps flowing; DBA monitors and retries any aborted concurrent index.)

REINDEX INDEX CONCURRENTLY kb.idx_normativa_chunk_emb_hnsw;

-- FTS (sparse search) — rebuild if present
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM pg_indexes
        WHERE schemaname = 'kb'
          AND tablename = 'normativa_chunk_fts'
          AND indexname = 'idx_normativa_chunk_fts_tsv'
    ) THEN
        EXECUTE 'REINDEX INDEX CONCURRENTLY kb.idx_normativa_chunk_fts_tsv';
    END IF;
END $$;

-- B-tree indexes on normativa (join, lookup, filter)
REINDEX INDEX CONCURRENTLY kb.idx_normativa_work;
REINDEX INDEX CONCURRENTLY kb.idx_normativa_sort;
REINDEX INDEX CONCURRENTLY kb.idx_normativa_validation;

-- ---------------------------------------------------------------------------
-- 4. Post-conditions — catalog sanity check
-- ---------------------------------------------------------------------------
DO $$
DECLARE
    v_bloat_pct NUMERIC;
    v_total_articles INTEGER;
    v_total_embeddings INTEGER;
BEGIN
    SELECT COUNT(*) INTO v_total_articles FROM kb.normativa;
    SELECT COUNT(*) INTO v_total_embeddings FROM kb.normativa_chunk_embeddings;

    RAISE NOTICE 'sprint27_002: post-vacuum stats';
    RAISE NOTICE '  kb.normativa articles : %', v_total_articles;
    RAISE NOTICE '  kb.chunk_embeddings   : %', v_total_embeddings;
    RAISE NOTICE 'Expected per T2-BIS (12 Sprint 27 codes only):';
    RAISE NOTICE '  COST~139  TUB~162  TUF~214  TUSL~306  TUI~46   CMED~158';
    RAISE NOTICE '  TUIR~235  TUEL~295 TUSG~316 TUDA~92  TUIST~676 TUESP~70';
    RAISE NOTICE '  CGS~152 pending Sprint 28 (FIGC ingest)';
END $$;

-- ---------------------------------------------------------------------------
-- Next step (manual)
-- ---------------------------------------------------------------------------
-- [ ] Run bench Gate A: COS-F-002 scenario
--     python lexe-core/scripts/benchmarks/run_bench_single.py --scenario COS-F-002
--     Policy: score ≥75 = PASS, 65-74 = WARN + Sprint 28 fix, <65 = ROLLBACK
--     Rollback path:
--       1. psql -c 'TRUNCATE kb.normativa RESTART IDENTITY CASCADE'
--          (yes cascade — wipes all kb.normativa* children; acceptable post-gate-fail)
--       2. psql -c 'INSERT INTO kb.normativa
--                   SELECT id, work_id, articolo, articolo_num, articolo_suffix,
--                          identity_class, quality, lifecycle, articolo_sort_key,
--                          urn_nir, rubrica, testo, is_current, created_at, updated_at
--                   FROM kb.normativa_legacy_backup_20260416'
--       (adjust columns to match current schema at time of rollback.)
