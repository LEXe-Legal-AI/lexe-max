-- 085_library_chunk_fascicolo.sql
-- Add fascicolo_id to library_chunk for scoped search.

BEGIN;

ALTER TABLE kb.library_chunk
    ADD COLUMN IF NOT EXISTS fascicolo_id UUID;

CREATE INDEX IF NOT EXISTS idx_lib_chunk_fascicolo
    ON kb.library_chunk(fascicolo_id);

CREATE INDEX IF NOT EXISTS idx_lib_chunk_tenant_fascicolo
    ON kb.library_chunk(tenant_id, fascicolo_id);

COMMIT;
