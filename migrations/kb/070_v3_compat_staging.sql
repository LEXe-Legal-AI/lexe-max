-- ============================================================================
-- 070_v3_compat_staging.sql
-- Align staging schema (V1 + migration 060) to V3 compatibility
-- ============================================================================
-- PRESERVES: massimari (46K), graph_edges (58K), embeddings (41K), norms (4K)
-- ADDS: 3 V3 enums, 3 classification columns, 64 new works, UNIQUE constraint
-- SAFE: Idempotent, re-runnable, wrapped in transaction
-- ============================================================================

BEGIN;

-- ============================================================================
-- STEP 1: Create V3 enums (idempotent)
-- ============================================================================

DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_catalog.pg_type t
    JOIN pg_catalog.pg_namespace n ON n.oid = t.typnamespace
    WHERE n.nspname = 'kb' AND t.typname = 'article_identity_class'
  ) THEN
    CREATE TYPE kb.article_identity_class AS ENUM ('BASE', 'SUFFIX', 'SPECIAL');
    RAISE NOTICE 'Created enum kb.article_identity_class';
  ELSE
    RAISE NOTICE 'Enum kb.article_identity_class already exists — skipped';
  END IF;

  IF NOT EXISTS (
    SELECT 1 FROM pg_catalog.pg_type t
    JOIN pg_catalog.pg_namespace n ON n.oid = t.typnamespace
    WHERE n.nspname = 'kb' AND t.typname = 'article_quality_class'
  ) THEN
    CREATE TYPE kb.article_quality_class AS ENUM ('VALID_STRONG', 'VALID_SHORT', 'WEAK', 'EMPTY', 'INVALID');
    RAISE NOTICE 'Created enum kb.article_quality_class';
  ELSE
    RAISE NOTICE 'Enum kb.article_quality_class already exists — skipped';
  END IF;

  IF NOT EXISTS (
    SELECT 1 FROM pg_catalog.pg_type t
    JOIN pg_catalog.pg_namespace n ON n.oid = t.typnamespace
    WHERE n.nspname = 'kb' AND t.typname = 'lifecycle_status'
  ) THEN
    CREATE TYPE kb.lifecycle_status AS ENUM ('CURRENT', 'HISTORICAL', 'REPEALED', 'UNKNOWN');
    RAISE NOTICE 'Created enum kb.lifecycle_status';
  ELSE
    RAISE NOTICE 'Enum kb.lifecycle_status already exists — skipped';
  END IF;
END$$;

-- ============================================================================
-- STEP 2: Add V3 columns to kb.normativa (nullable first for backfill)
-- ============================================================================

ALTER TABLE kb.normativa ADD COLUMN IF NOT EXISTS identity_class kb.article_identity_class;
ALTER TABLE kb.normativa ADD COLUMN IF NOT EXISTS quality kb.article_quality_class;
ALTER TABLE kb.normativa ADD COLUMN IF NOT EXISTS lifecycle kb.lifecycle_status;
ALTER TABLE kb.normativa ADD COLUMN IF NOT EXISTS urn_nir TEXT;

-- ============================================================================
-- STEP 3: Add 'notes' column to kb.work (V3 has it, 060 doesn't)
-- ============================================================================

ALTER TABLE kb.work ADD COLUMN IF NOT EXISTS notes TEXT;

-- ============================================================================
-- STEP 4: Backfill V3 columns for existing rows
-- ============================================================================

UPDATE kb.normativa SET
  identity_class = CASE
    WHEN articolo_suffix IS NOT NULL AND articolo_suffix != ''
      THEN 'SUFFIX'::kb.article_identity_class
    WHEN articolo !~ '^\d'
      THEN 'SPECIAL'::kb.article_identity_class
    ELSE 'BASE'::kb.article_identity_class
  END,
  quality = CASE
    WHEN length(coalesce(testo, '')) >= 150
      THEN 'VALID_STRONG'::kb.article_quality_class
    WHEN length(coalesce(testo, '')) >= 10
      THEN 'VALID_SHORT'::kb.article_quality_class
    ELSE 'EMPTY'::kb.article_quality_class
  END,
  lifecycle = 'UNKNOWN'::kb.lifecycle_status
WHERE identity_class IS NULL;

DO $$
DECLARE v_backfilled BIGINT;
BEGIN
  GET DIAGNOSTICS v_backfilled = ROW_COUNT;
  RAISE NOTICE 'Backfilled % rows with V3 classification', v_backfilled;
END$$;

-- ============================================================================
-- STEP 5: Safety pass + SET NOT NULL
-- ============================================================================

-- Catch any remaining NULLs
UPDATE kb.normativa SET identity_class = 'BASE'::kb.article_identity_class
WHERE identity_class IS NULL;

UPDATE kb.normativa SET quality = 'VALID_STRONG'::kb.article_quality_class
WHERE quality IS NULL;

UPDATE kb.normativa SET lifecycle = 'UNKNOWN'::kb.lifecycle_status
WHERE lifecycle IS NULL;

-- Now safe to enforce NOT NULL
ALTER TABLE kb.normativa ALTER COLUMN identity_class SET NOT NULL;
ALTER TABLE kb.normativa ALTER COLUMN quality SET NOT NULL;
ALTER TABLE kb.normativa ALTER COLUMN lifecycle SET NOT NULL;
ALTER TABLE kb.normativa ALTER COLUMN lifecycle SET DEFAULT 'UNKNOWN'::kb.lifecycle_status;

-- ============================================================================
-- STEP 6: Ensure articolo_sort_key is computed for all rows
-- ============================================================================

UPDATE kb.normativa
SET articolo_sort_key = CASE
  WHEN articolo ~ '^\d+' THEN
    LPAD((regexp_match(articolo, '^(\d+)'))[1], 6, '0') || '.' ||
    CASE
      WHEN articolo_suffix IS NOT NULL AND articolo_suffix != ''
      THEN kb.fn_suffix_ordinal(articolo_suffix)
      ELSE '00'
    END
  ELSE articolo
END
WHERE articolo_sort_key IS NULL;

-- ============================================================================
-- STEP 7: Deduplicate on (work_id, articolo_sort_key) — keep newest row
-- ============================================================================

-- Pass 1: different created_at → keep newer
DELETE FROM kb.normativa a
USING kb.normativa b
WHERE a.work_id = b.work_id
  AND a.articolo_sort_key = b.articolo_sort_key
  AND a.work_id IS NOT NULL
  AND a.articolo_sort_key IS NOT NULL
  AND a.id != b.id
  AND a.created_at < b.created_at;

-- Pass 2: same created_at → keep higher UUID (deterministic)
DELETE FROM kb.normativa a
USING kb.normativa b
WHERE a.work_id = b.work_id
  AND a.articolo_sort_key = b.articolo_sort_key
  AND a.work_id IS NOT NULL
  AND a.articolo_sort_key IS NOT NULL
  AND a.id != b.id
  AND a.created_at = b.created_at
  AND a.id < b.id;

DO $$
DECLARE v_remaining BIGINT;
BEGIN
  SELECT COUNT(*) INTO v_remaining FROM kb.normativa;
  RAISE NOTICE 'After dedup: % rows in kb.normativa', v_remaining;
END$$;

-- ============================================================================
-- STEP 8: Add UNIQUE constraint on (work_id, articolo_sort_key)
-- Required for import_to_staging.py ON CONFLICT clause
-- ============================================================================

DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_constraint c
    JOIN pg_namespace n ON n.oid = c.connamespace
    WHERE n.nspname = 'kb' AND c.conname = 'normativa_work_sort_key'
  ) THEN
    ALTER TABLE kb.normativa
      ADD CONSTRAINT normativa_work_sort_key UNIQUE (work_id, articolo_sort_key);
    RAISE NOTICE 'Added UNIQUE constraint normativa_work_sort_key';
  ELSE
    RAISE NOTICE 'UNIQUE constraint normativa_work_sort_key already exists — skipped';
  END IF;
END$$;

-- V3 indexes
CREATE INDEX IF NOT EXISTS idx_normativa_quality ON kb.normativa(work_id, quality);
CREATE INDEX IF NOT EXISTS idx_normativa_lifecycle ON kb.normativa(work_id, lifecycle);
CREATE INDEX IF NOT EXISTS idx_normativa_sort ON kb.normativa(work_id, articolo_sort_key);

-- ============================================================================
-- STEP 9: Expand kb.work from 5 to 69 codes (from V3 seed 052)
-- ============================================================================

INSERT INTO kb.work (code, title, notes) VALUES
-- Costituzione e 4 codici (gia presenti, ON CONFLICT skips)
('COST', 'Costituzione della Repubblica Italiana', '139 articoli'),
('CC', 'Codice Civile', '2969 articoli'),
('CP', 'Codice Penale', '734 articoli'),
('CPC', 'Codice di Procedura Civile', '840 articoli'),
('CPP', 'Codice di Procedura Penale', '746 articoli'),
-- Commerciale
('CCII', 'Codice della Crisi d''Impresa e dell''Insolvenza', 'D.Lgs. 14/2019, 391 articoli'),
('CPI', 'Codice della Proprieta Industriale', 'D.Lgs. 30/2005, 245 articoli'),
('LF', 'Legge Fallimentare', 'R.D. 267/1942 - abrogata, 264 articoli'),
('TUB', 'Testo Unico Bancario', 'D.Lgs. 385/1993, 162 articoli'),
('TUF', 'Testo Unico della Finanza', 'D.Lgs. 58/1998, 214 articoli'),
-- Civile
('CCONS', 'Codice del Consumo', 'D.Lgs. 206/2005, 170 articoli'),
('CTUR', 'Codice del Turismo', 'D.Lgs. 79/2011, 74 articoli'),
('CGS', 'Codice Giustizia Sportiva', 'FIGC, 152 articoli'),
('CPRIV', 'Codice Privacy', 'D.Lgs. 196/2003, 186 articoli'),
('CTS', 'Codice del Terzo Settore', 'D.Lgs. 117/2017, 104 articoli'),
('GDPR', 'Regolamento Generale Protezione Dati', 'Reg. UE 2016/679, 99 articoli'),
('LDA', 'Legge Diritto d''Autore', 'L. 633/1941, 206 articoli'),
('LDIV', 'Legge Divorzio', 'L. 898/1970, 12 articoli'),
('LLOC', 'Legge Locazioni Abitative', 'L. 431/1998, 14 articoli'),
('DMED', 'Mediazione Civile', 'D.Lgs. 28/2010, 24 articoli'),
('TUSG', 'Testo Unico Spese di Giustizia', 'D.P.R. 115/2002, 302 articoli'),
-- Amministrativo
('CAD', 'Codice Amministrazione Digitale', 'D.Lgs. 82/2005, 92 articoli'),
('CAM', 'Codice Antimafia', 'D.Lgs. 159/2011, 144 articoli'),
('CAPP', 'Codice degli Appalti', 'D.Lgs. 36/2023, 229 articoli'),
('CGC', 'Codice Giustizia Contabile', 'D.Lgs. 174/2016, 219 articoli'),
('CMED', 'Codice dei Medicinali', 'D.Lgs. 219/2006, 158 articoli'),
('CPA', 'Codice Processo Amministrativo', 'D.Lgs. 104/2010, 138 articoli'),
('LPA', 'Legge Procedimento Amministrativo', 'L. 241/1990, 31 articoli'),
('DLAS', 'Responsabilita Amministrativa Societa', 'D.Lgs. 231/2001, 83 articoli'),
('DSIC', 'Sicurezza Urbana', 'D.L. 14/2017, 17 articoli'),
('TUE', 'Testo Unico Edilizia', 'D.P.R. 380/2001, 138 articoli'),
('TUEL', 'Testo Unico Enti Locali', 'D.Lgs. 267/2000, 275 articoli'),
('TUESP', 'Testo Unico Espropriazioni', 'D.P.R. 327/2001, 59 articoli'),
('TUI', 'Testo Unico Immigrazione', 'D.Lgs. 286/1998, 46 articoli'),
('TUIST', 'Testo Unico Istruzione', 'D.Lgs. 297/1994, 676 articoli'),
('TUPI', 'Testo Unico Pubblico Impiego', 'D.Lgs. 165/2001, 73 articoli'),
('TUSP', 'Testo Unico Societa Partecipate', 'D.Lgs. 175/2016, 26 articoli'),
('TUDA', 'Testo Unico Documentazione Amministrativa', 'D.P.R. 445/2000, 78 articoli'),
-- Ambiente
('CAMB', 'Codice dell''Ambiente', 'D.Lgs. 152/2006, 318 articoli'),
('CBCP', 'Codice Beni Culturali e Paesaggio', 'D.Lgs. 42/2004, 184 articoli'),
('TUFOR', 'Testo Unico Foreste', 'D.Lgs. 34/2018, 20 articoli'),
-- Circolazione
('CAP', 'Codice Assicurazioni Private', 'D.Lgs. 209/2005, 355 articoli'),
('CDS', 'Codice della Strada', 'D.Lgs. 285/1992, 245 articoli'),
('CND', 'Codice Nautica da Diporto', 'D.Lgs. 171/2005, 66 articoli'),
('REGCDS', 'Regolamento CdS', 'D.P.R. 495/1992, 408 articoli'),
-- Comunicazioni
('CCE', 'Codice Comunicazioni Elettroniche', 'D.Lgs. 259/2003, 98 articoli'),
('TUSMAR', 'Testo Unico Radiotelevisione', 'D.Lgs. 177/2005, 63 articoli'),
-- Fisco
('CPT', 'Codice Processo Tributario', 'D.Lgs. 546/1992, 72 articoli'),
('LRT', 'Legge Reati Tributari', 'D.Lgs. 74/2000, 25 articoli'),
('STATC', 'Statuto del Contribuente', 'L. 212/2000, 21 articoli'),
('TUIR', 'Testo Unico Imposte sui Redditi', 'D.P.R. 917/1986, 188 articoli'),
('DIVA', 'DPR IVA', 'D.P.R. 633/1972, 74 articoli'),
-- Internazionale
('DUDU', 'Dichiarazione Universale Diritti Uomo', 'ONU 1948, 30 articoli'),
('LDIP', 'Legge Diritto Internazionale Privato', 'L. 218/1995, 74 articoli'),
-- Lavoro
('CPO', 'Codice Pari Opportunita', 'D.Lgs. 198/2006, 60 articoli'),
('LSSP', 'Legge Sciopero Servizi Pubblici', 'L. 146/1990, 13 articoli'),
('DBIAGI', 'Riforma Lavoro Biagi', 'D.Lgs. 276/2003, 86 articoli'),
('LFORN', 'Riforma Lavoro Fornero', 'L. 92/2012, 4 articoli'),
('SL', 'Statuto dei Lavoratori', 'L. 300/1970, 41 articoli'),
('TUPC', 'Testo Unico Previdenza Complementare', 'D.Lgs. 252/2005, 23 articoli'),
('TUMP', 'Testo Unico Maternita Paternita', 'D.Lgs. 151/2001, 88 articoli'),
('TUSL', 'Testo Unico Sicurezza Lavoro', 'D.Lgs. 81/2008, 306 articoli'),
-- Penale
('CPPM', 'Codice Procedura Penale Minorile', 'D.P.R. 448/1988, 41 articoli'),
('LDEP', 'Legge Depenalizzazione', 'D.Lgs. 8/2016, 42 articoli'),
('OP', 'Ordinamento Penitenziario', 'L. 354/1975, 91 articoli'),
('TUCG', 'Testo Unico Casellario Giudiziale', 'D.P.R. 313/2002, 48 articoli'),
('TUS', 'Testo Unico Stupefacenti', 'D.P.R. 309/1990, 127 articoli'),
-- Professioni
('CDF', 'Codice Deontologico Forense', 'CNF 2014, 73 articoli'),
('LPF', 'Legge Professionale Forense', 'L. 247/2012, 67 articoli')
ON CONFLICT (code) DO NOTHING;

DO $$
DECLARE v_work_count BIGINT;
BEGIN
  SELECT COUNT(*) INTO v_work_count FROM kb.work;
  RAISE NOTICE 'kb.work now has % codes', v_work_count;
END$$;

-- ============================================================================
-- STEP 10: Backfill work_id for any orphaned normativa rows
-- ============================================================================

UPDATE kb.normativa n
SET work_id = w.id
FROM kb.work w
WHERE UPPER(n.codice) = w.code
  AND n.work_id IS NULL;

-- ============================================================================
-- STEP 11: Final report
-- ============================================================================

DO $$
DECLARE
  v_normativa_count BIGINT;
  v_work_count BIGINT;
  v_massime_count BIGINT := -1;
  v_graph_edges_count BIGINT := -1;
  v_embeddings_count BIGINT := -1;
  v_chunks_count BIGINT := -1;
  v_chunk_emb_count BIGINT := -1;
  v_orphans BIGINT;
  v_identity RECORD;
  v_quality RECORD;
BEGIN
  SELECT COUNT(*) INTO v_normativa_count FROM kb.normativa;
  SELECT COUNT(*) INTO v_work_count FROM kb.work;
  SELECT COUNT(*) INTO v_orphans FROM kb.normativa WHERE work_id IS NULL;

  -- Safe counts for tables that may not exist
  BEGIN SELECT COUNT(*) INTO v_massime_count FROM kb.massime; EXCEPTION WHEN undefined_table THEN NULL; END;
  BEGIN SELECT COUNT(*) INTO v_graph_edges_count FROM kb.graph_edges; EXCEPTION WHEN undefined_table THEN NULL; END;
  BEGIN SELECT COUNT(*) INTO v_embeddings_count FROM kb.embeddings; EXCEPTION WHEN undefined_table THEN NULL; END;
  BEGIN SELECT COUNT(*) INTO v_chunks_count FROM kb.normativa_chunk; EXCEPTION WHEN undefined_table THEN NULL; END;
  BEGIN SELECT COUNT(*) INTO v_chunk_emb_count FROM kb.normativa_chunk_embeddings; EXCEPTION WHEN undefined_table THEN NULL; END;

  RAISE NOTICE '';
  RAISE NOTICE '============================================';
  RAISE NOTICE '  070 V3 COMPAT STAGING — FINAL REPORT';
  RAISE NOTICE '============================================';
  RAISE NOTICE '  kb.work:                   % codes', v_work_count;
  RAISE NOTICE '  kb.normativa:              % articles', v_normativa_count;
  RAISE NOTICE '  kb.normativa (orphans):    %', v_orphans;
  RAISE NOTICE '  kb.normativa_chunk:        %', v_chunks_count;
  RAISE NOTICE '  kb.normativa_chunk_emb:    %', v_chunk_emb_count;
  RAISE NOTICE '  --------------------------------------------';
  RAISE NOTICE '  PRESERVED DATA (massimari):';
  RAISE NOTICE '  kb.massime:                %', v_massime_count;
  RAISE NOTICE '  kb.graph_edges:            %', v_graph_edges_count;
  RAISE NOTICE '  kb.embeddings:             %', v_embeddings_count;
  RAISE NOTICE '============================================';

  -- Identity class distribution
  RAISE NOTICE '';
  RAISE NOTICE '  Identity class distribution:';
  FOR v_identity IN
    SELECT identity_class, COUNT(*) as cnt
    FROM kb.normativa GROUP BY identity_class ORDER BY cnt DESC
  LOOP
    RAISE NOTICE '    %: %', v_identity.identity_class, v_identity.cnt;
  END LOOP;

  -- Quality distribution
  RAISE NOTICE '';
  RAISE NOTICE '  Quality distribution:';
  FOR v_quality IN
    SELECT quality, COUNT(*) as cnt
    FROM kb.normativa GROUP BY quality ORDER BY cnt DESC
  LOOP
    RAISE NOTICE '    %: %', v_quality.quality, v_quality.cnt;
  END LOOP;

  RAISE NOTICE '';
  RAISE NOTICE '  V3 enums: OK';
  RAISE NOTICE '  UNIQUE(work_id, articolo_sort_key): OK';
  RAISE NOTICE '  Ready for import_to_staging.py --all';
  RAISE NOTICE '============================================';
END$$;

COMMIT;
