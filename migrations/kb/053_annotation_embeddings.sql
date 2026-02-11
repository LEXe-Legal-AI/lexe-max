-- migrations/kb/053_annotation_embeddings.sql
-- Tabella embeddings per annotations Brocardi
-- Separata da normativa_embeddings per canali/filtri/index diversi

BEGIN;

CREATE TABLE IF NOT EXISTS kb.annotation_embeddings (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  annotation_id UUID NOT NULL REFERENCES kb.annotation(id) ON DELETE CASCADE,
  model TEXT NOT NULL,
  channel TEXT NOT NULL DEFAULT 'content',
  dims INTEGER NOT NULL,
  embedding vector(1536) NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE(annotation_id, model, channel, dims)
);

-- Indice HNSW per cosine similarity
CREATE INDEX IF NOT EXISTS idx_ann_emb_1536_content
ON kb.annotation_embeddings USING hnsw (embedding vector_cosine_ops)
WHERE dims = 1536 AND channel = 'content';

COMMIT;
