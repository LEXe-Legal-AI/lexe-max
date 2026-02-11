-- migrations/kb/054_annotation_fts.sql
-- FTS per annotations Brocardi (note, commenti, massime)
-- Complementa kb.annotation_embeddings per hybrid search

BEGIN;

-- Tabella FTS separata (stesso pattern di normativa_fts)
CREATE TABLE IF NOT EXISTS kb.annotation_fts (
    annotation_id UUID PRIMARY KEY REFERENCES kb.annotation(id) ON DELETE CASCADE,
    tsv_it TSVECTOR
);

-- Indice GIN per full-text search
CREATE INDEX IF NOT EXISTS idx_annotation_fts ON kb.annotation_fts USING GIN(tsv_it);

-- Funzione trigger per mantenere tsvector aggiornato
CREATE OR REPLACE FUNCTION kb.fn_annotation_fts_update() RETURNS trigger AS $$
BEGIN
    INSERT INTO kb.annotation_fts(annotation_id, tsv_it)
    VALUES(
        NEW.id,
        to_tsvector('italian', unaccent(
            COALESCE(TRIM(NEW.title), '') || ' ' || COALESCE(TRIM(NEW.content), '')
        ))
    )
    ON CONFLICT (annotation_id) DO UPDATE SET tsv_it = EXCLUDED.tsv_it;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger su INSERT e UPDATE di title/content
DROP TRIGGER IF EXISTS trg_annotation_fts ON kb.annotation;
CREATE TRIGGER trg_annotation_fts
    AFTER INSERT OR UPDATE OF title, content ON kb.annotation
    FOR EACH ROW EXECUTE FUNCTION kb.fn_annotation_fts_update();

-- Popolamento iniziale (idempotente)
INSERT INTO kb.annotation_fts (annotation_id, tsv_it)
SELECT
    id,
    to_tsvector('italian', unaccent(
        COALESCE(TRIM(title), '') || ' ' || COALESCE(TRIM(content), '')
    ))
FROM kb.annotation
WHERE content IS NOT NULL
ON CONFLICT (annotation_id) DO UPDATE SET tsv_it = EXCLUDED.tsv_it;

COMMIT;
