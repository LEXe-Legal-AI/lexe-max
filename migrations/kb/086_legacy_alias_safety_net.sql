-- 086_legacy_alias_safety_net.sql
-- Sprint 27 (S2.2): defensive alias table so retrieval SQL can JOIN on
-- legacy short-form codes even when kb.work.code has drifted to the
-- OpenData canonical form (e.g. TUE→TUESP, TUS→TUSL).
--
-- Mirror of lexe-core/src/lexe_core/agent/alias_resolver.py CANONICAL_ALIASES.
-- Seed rows are idempotent via ON CONFLICT DO NOTHING and a WHERE EXISTS
-- filter: if kb.work does not yet have a row for a given code, the alias
-- row is simply skipped — the re-ingest pass (S6.2) will repopulate later.

BEGIN;

CREATE TABLE IF NOT EXISTS kb.work_alias (
    alias        TEXT PRIMARY KEY,
    work_id      UUID NOT NULL REFERENCES kb.work(id) ON DELETE CASCADE,
    alias_type   TEXT NOT NULL DEFAULT 'canonical_short',
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT chk_alias_type CHECK (alias_type IN (
        'canonical_short',   -- 12 canonical Sprint 27 aliases (TUB, TUF, …)
        'legacy_deprecated', -- old KB codes rewritten to canonical
        'user_synonym'       -- free-form user aliases (reserved, not used S27)
    ))
);

CREATE INDEX IF NOT EXISTS idx_work_alias_work_id
    ON kb.work_alias(work_id);

CREATE INDEX IF NOT EXISTS idx_work_alias_type
    ON kb.work_alias(alias_type);

-- ---------------------------------------------------------------------------
-- Seed: 12 canonical short-form aliases → kb.work.id
-- ---------------------------------------------------------------------------
-- Each INSERT picks the work_id by code lookup; skipped if code missing.
INSERT INTO kb.work_alias (alias, work_id, alias_type)
SELECT v.alias, w.id, 'canonical_short'
FROM (VALUES
    ('TUB'),
    ('TUF'),
    ('TUIR'),
    ('TUEL'),
    ('TUSL'),
    ('TUI'),
    ('TUSG'),
    ('TUIST'),
    ('TUDA'),
    ('TUESP'),
    ('CMED'),
    ('CGS')
) AS v(alias)
JOIN kb.work w ON w.code = v.alias
ON CONFLICT (alias) DO NOTHING;

-- ---------------------------------------------------------------------------
-- Reserved: legacy rewrites (populate once T0d / DB audit confirms drift)
-- ---------------------------------------------------------------------------
-- Example (kept as comment until confirmed):
--   INSERT INTO kb.work_alias (alias, work_id, alias_type)
--   SELECT 'TUEDIL', w.id, 'legacy_deprecated'
--   FROM kb.work w WHERE w.code = 'TUESP'
--   ON CONFLICT (alias) DO NOTHING;

COMMIT;
