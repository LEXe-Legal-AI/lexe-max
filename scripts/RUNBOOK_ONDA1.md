# Runbook — Sprint 22 Onda 1 KB Fixes

## Pre-requisiti

```bash
# Ambiente
cd C:/PROJECTS/lexe-genesis/lexe-max
pip install asyncpg httpx beautifulsoup4 playwright

# API key per embeddings
export OPENROUTER_API_KEY=sk-or-...
```

## Step 1: Fetch articoli -bis (GIA FATTO)

```bash
# Known-only: ~40 articoli noti, ~3 min
python scripts/fetch_bis_articles.py --known-only
# Output: data/bis_articles/ (37 JSON)

# Discovery completa (opzionale, ~2-4 ore):
# python scripts/fetch_bis_articles.py --resume
```

## Step 2: Download EUR-Lex HTML (Playwright)

```bash
python scripts/fetch_eurlex_playwright.py
# Output: data/eurlex/<celex>.html (5 files)

# Se Playwright fallisce, download manuale:
# Apri browser -> https://eur-lex.europa.eu/legal-content/IT/TXT/HTML/?uri=CELEX:32022R1925
# Salva come: data/eurlex/32022R1925.html
```

## Step 3: Deploy su Staging

```bash
# SSH tunnel (terminale separato)
ssh -i ~/.ssh/id_stage_new -L 5436:localhost:5436 root@91.99.229.111

# Dry-run
python scripts/ingest_sprint22_onda1.py --env staging --dry-run

# Ingest reale (bis + eurlex + embed + verify)
OPENROUTER_API_KEY=sk-or-... python scripts/ingest_sprint22_onda1.py --env staging

# Solo una fase
python scripts/ingest_sprint22_onda1.py --env staging --phase bis
python scripts/ingest_sprint22_onda1.py --env staging --phase eurlex
python scripts/ingest_sprint22_onda1.py --env staging --phase embed
python scripts/ingest_sprint22_onda1.py --env staging --phase verify
```

## Step 4: Verifica Staging

```bash
# Via tunnel SSH o docker exec
docker exec lexe-max psql -U lexe_kb -d lexe_kb -c "
  SELECT w.code, count(n.id) as articles,
         count(n.id) FILTER (WHERE n.articolo_suffix IS NOT NULL) as bis
  FROM kb.work w
  JOIN kb.normativa n ON n.work_id = w.id
  WHERE w.code IN ('CCII','TUSL','L212','L241','CDS','CCONS','CPRIV','CAMB','CAD','COST',
                    'DMA','DORA','DSA','NIS2','TFUE')
  GROUP BY w.code ORDER BY w.code;
"

# Check embeddings
docker exec lexe-max psql -U lexe_kb -d lexe_kb -c "
  SELECT count(*) as chunks_missing_emb
  FROM kb.normativa_chunk c
  WHERE NOT EXISTS (
    SELECT 1 FROM kb.normativa_chunk_embeddings e
    WHERE e.chunk_id = c.id AND e.dims = 1536
  );
"
```

## Step 5: Benchmark Staging

```bash
# Singola query di verifica
# EU-ANA-101 (TFUE), COS-F-002 (art. 13 Cost), target >= 85

# Full worst16 bench via admin panel:
# https://stage-chat.lexe.pro/admin -> Benchmark -> worst16
```

## Step 6: Deploy su Prod

```bash
# SSH tunnel (terminale separato)
ssh -i ~/.ssh/hetzner_leo_key -L 5436:localhost:5436 root@49.12.85.92

# ATTENZIONE: conferma manuale richiesta
OPENROUTER_API_KEY=sk-or-... python scripts/ingest_sprint22_onda1.py --env prod
```

## Step 7: Verifica Prod

Stesse query di Step 4, su prod.

## Rollback

Non serve rollback — tutti gli upsert sono idempotenti con `ON CONFLICT DO UPDATE`.
Per rimuovere articoli aggiunti erroneamente:

```sql
-- Esempio: rimuovi articoli -bis CCII
DELETE FROM kb.normativa_chunk_embeddings
WHERE chunk_id IN (
  SELECT c.id FROM kb.normativa_chunk c
  JOIN kb.normativa n ON c.normativa_id = n.id
  WHERE n.codice = 'CCII' AND n.articolo_suffix IS NOT NULL
  AND n.canonical_source = 'opendata_api_urn'
);
DELETE FROM kb.normativa_chunk
WHERE normativa_id IN (
  SELECT id FROM kb.normativa
  WHERE codice = 'CCII' AND articolo_suffix IS NOT NULL
  AND canonical_source = 'opendata_api_urn'
);
DELETE FROM kb.normativa
WHERE codice = 'CCII' AND articolo_suffix IS NOT NULL
AND canonical_source = 'opendata_api_urn';
```

## Checklist

- [ ] Step 1: Fetch -bis articles (37 JSON)
- [ ] Step 2: Download EUR-Lex HTML (5 files)
- [ ] Step 3: Deploy staging (bis + eurlex + embed)
- [ ] Step 4: Verifica staging (counts OK)
- [ ] Step 5: Bench staging (EU-ANA-101 >= 85, COS-F-002 >= 75)
- [ ] Step 6: Deploy prod
- [ ] Step 7: Verifica prod
- [ ] Memory update (STATUS.md + project memory)
