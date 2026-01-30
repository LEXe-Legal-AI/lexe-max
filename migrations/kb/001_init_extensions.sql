-- 001_init_extensions.sql
-- LEXE Knowledge Base - Inizializzazione estensioni
-- Eseguito automaticamente da docker-entrypoint-initdb.d

-- ============================================================
-- ESTENSIONI
-- ============================================================

-- pgvector per vector similarity search
CREATE EXTENSION IF NOT EXISTS vector;

-- Apache AGE per graph database
CREATE EXTENSION IF NOT EXISTS age;

-- pg_trgm per fuzzy matching e typo catch
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- pg_search (ParadeDB) per BM25 - può fallire se non installato
DO $$
BEGIN
    CREATE EXTENSION IF NOT EXISTS pg_search;
    RAISE NOTICE 'pg_search extension created successfully';
EXCEPTION WHEN OTHERS THEN
    RAISE WARNING 'pg_search extension not available: %', SQLERRM;
END $$;

-- uuid-ossp per UUID generation (fallback se gen_random_uuid non disponibile)
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================================
-- CONFIGURAZIONE AGE
-- ============================================================

-- Aggiungi ag_catalog al search_path
DO $$
BEGIN
    ALTER DATABASE lexe_kb SET search_path TO ag_catalog, kb, public;
EXCEPTION WHEN OTHERS THEN
    RAISE WARNING 'Could not set search_path: %', SQLERRM;
END $$;

-- Carica AGE
LOAD 'age';

-- Crea il grafo per la giurisprudenza
SELECT * FROM ag_catalog.create_graph('lexe_jurisprudence');

-- ============================================================
-- LOG
-- ============================================================

DO $$
DECLARE
    ext_record RECORD;
BEGIN
    RAISE NOTICE '============================================================';
    RAISE NOTICE 'LEXE Knowledge Base - Extensions initialized';
    RAISE NOTICE '============================================================';
    FOR ext_record IN SELECT extname, extversion FROM pg_extension LOOP
        RAISE NOTICE 'Extension: % (v%)', ext_record.extname, ext_record.extversion;
    END LOOP;
    RAISE NOTICE '============================================================';
END $$;
