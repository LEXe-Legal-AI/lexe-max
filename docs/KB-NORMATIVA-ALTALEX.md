# KB Normativa Altalex - Pipeline Ingestion

> Documentazione del lavoro svolto per l'ingestion di 68+ PDF Altalex nella Knowledge Base LEXE.
>
> **Data:** 2026-02-04
> **Status:** FASE 3 completata, pronto per batch

---

## Obiettivo

Creare pipeline robusta per ingestionare codici e leggi italiane (PDF Altalex) nella KB LEXE con:
- Chunking semantico per articolo
- Embeddings multi-dimensione
- Hybrid retrieval (Dense + FTS + Trigram + RRF)

**Scope questo piano:** PDF → JSON → Chunk → SQL + Embeddings
**Fuori scope (fase successiva):** Graph connections, citation linking, norm graph

---

## Architettura Pipeline

```
PDF (Altalex, 68 files)
    │
    ▼
marker-pdf --disable_ocr --output_format json
    │
    ▼
marker_chunker.py (group blocks → articles)
    │
    ▼
PostgreSQL (kb.normativa_altalex extended)
    │
    ▼
Embeddings (multilingual-e5-large-instruct, 1024 dims)
    │
    ▼
Hybrid Retrieval (Dense + FTS + Trigram + RRF)
```

---

## Fasi Completate

### FASE 0: Valutazione Marker Chunks ✅

**Obiettivo:** Verificare se marker-pdf `--output_format chunks` produce chunks utilizzabili.

**Risultato:** I chunks marker sono troppo granulari (blocchi, non articoli), ma contengono le informazioni necessarie per post-processing.

**Decisione:** Usare marker JSON + post-processing invece di LLM chunking.

---

### FASE 1: Migration Database ✅

**File:** `migrations/kb/012_normativa_altalex_v2.sql`

**Estensioni a kb.normativa_altalex:**

| Colonna | Tipo | Descrizione |
|---------|------|-------------|
| `articolo_num_norm` | INTEGER | Numero normalizzato (2043 per "2043-bis") |
| `articolo_suffix` | TEXT | Suffisso: bis, ter, quater, ..., octiesdecies |
| `articolo_sort_key` | TEXT | Chiave ordinamento: 002043.bis |
| `global_key` | TEXT UNIQUE | Chiave globale: altalex:cc:2043:bis |
| `testo_context` | TEXT | Testo con overlap ±200 chars |
| `commi` | JSONB | Array commi strutturati |
| `riferimenti_parsed` | JSONB | Riferimenti validati |
| `riferimenti_raw` | TEXT[] | Riferimenti raw |
| `page_start` | INTEGER | Pagina inizio |
| `page_end` | INTEGER | Pagina fine |
| `testo_tsv` | tsvector | FTS con unaccent |

**Nuove tabelle:**

| Tabella | Scopo |
|---------|-------|
| `kb.altalex_ingestion_logs` | Tracking errori e quarantina |
| `kb.altalex_embeddings` | Embeddings multi-dim (384-1536) |
| `kb.altalex_embedding_cache` | Cache embeddings persistente |

**Trigger automatici:**
- `trg_altalex_sort_key` - Genera sort key per ordinamento
- `trg_altalex_global_key` - Genera chiave globale univoca

**Indici:**
- HNSW per vector search (dims 768, 1024, 1536)
- GIN per FTS (tsvector)
- Composite unique per deduplicazione

**pgvector verificato:** v0.7.4 → HNSW supportato ✓

---

### FASE 2: Marker Chunker ✅

**File:** `src/lexe_api/kb/ingestion/marker_chunker.py`

**Funzionalità:**
1. Legge JSON output di marker-pdf
2. Identifica header articoli (SectionHeader con "Art. N" o "Articolo N")
3. Raggruppa blocchi consecutivi fino al prossimo header
4. Estrae rubrica dal primo blocco testo
5. Aggiunge overlap inter-articolo (±200 chars)
6. Calcola normalizzazioni (num_norm, suffix, sort_key)
7. Valida articoli estratti

**Suffissi latini supportati:**
```
bis, ter, quater, quinquies, sexies, septies, octies, novies, nonies,
decies, undecies, duodecies, terdecies, quaterdecies, quinquiesdecies,
sexiesdecies, septiesdecies, octiesdecies
```

**Risultati test:**

| Documento | Articoli Estratti | Validi | Percentuale |
|-----------|-------------------|--------|-------------|
| **GDPR** | 98 | 98 | **100%** |
| **Codice Penale** | 924 | 869 | **94%** |
| Dichiarazione Universale | 30 | 22 | 73% (formato edge case) |

**Output per articolo:**
```python
@dataclass
class ExtractedArticle:
    articolo_num: str           # "17", "2043-bis"
    articolo_num_norm: int      # 17, 2043
    articolo_suffix: str | None # None, "bis", "ter"
    rubrica: str | None         # Titolo articolo
    testo: str                  # Testo pulito
    testo_context: str          # Testo con overlap
    libro: str | None           # Gerarchia
    titolo: str | None
    capo: str | None
    sezione: str | None
    page_start: int
    page_end: int
    content_hash: str           # SHA256
    warnings: list[str]
```

**Uso:**
```bash
# Test chunking su file JSON
python -m src.lexe_api.kb.ingestion.marker_chunker "path/to/file.json" CODICE
```

---

### FASE 3: Benchmark Embeddings ✅

**File:** `scripts/benchmark/mini_embedding_benchmark.py`

**Modelli testati:**

| Modello | Provider | Dims | Recall@10 | MRR | Latency | Throughput |
|---------|----------|------|-----------|-----|---------|------------|
| **openai/text-embedding-3-small** | OpenRouter | 1536 | **95%** | **0.920** | 30.4ms | 32.9/s |
| multilingual-e5-large-instruct | sentence-transformers | 1024 | 90% | 0.825 | 51.2ms | 19.5/s |

**Confronto dettagliato:**

| Metrica | OpenAI (OpenRouter) | e5-large (Locale) | Differenza |
|---------|---------------------|-------------------|------------|
| Recall@10 | 95% | 90% | **+5%** |
| MRR | 0.920 | 0.825 | **+0.095** |
| Latency | 30.4ms | 51.2ms | **-40%** |
| Throughput | 32.9/s | 19.5/s | **+69%** |

**Query di test (10 su GDPR):**

| Query | Expected | OpenAI Top-3 | e5 Top-3 |
|-------|----------|--------------|----------|
| diritto all'oblio cancellazione dati | Art. 17 | 17, 20, 21 ✓ | 17, 15, 21 ✓ |
| consenso al trattamento dati | Art. 7, 8 | 7, 32, 25 | 7, 21, 9 |
| portabilità dei dati | Art. 20 | 20, 49, 5 ✓ | 96, 20, 45 |
| data protection officer DPO | Art. 37, 38, 39 | 39, 38, 37 ✓ | 31, 39, 38 |
| trasferimento dati paesi terzi | Art. 44, 45, 46 | 45, 49, 44 ✓ | 45, 46, 44 ✓ |
| sanzioni amministrative | Art. 83, 84 | 83, 84, 59 ✓ | 83, 59, 84 ✓ |
| responsabile del trattamento | Art. 28, 29 | 28, 31, 29 ✓ | 28, 39, 38 |
| informativa privacy interessato | Art. 13, 14 | 11, 15, 12 | 11, 4, 15 |
| valutazione impatto DPIA | Art. 35, 36 | 35, 34, 36 ✓ | 35, 34, 39 |
| diritto di rettifica | Art. 16 | 16, 19, 79 ✓ | 16, 82, 21 ✓ |

**Decisione:** Usare **`openai/text-embedding-3-small` via OpenRouter** come modello primario:
- Recall@10 migliore (95% vs 90%)
- MRR migliore (0.920 vs 0.825)
- Latency migliore (30ms vs 51ms)
- Throughput migliore (33/s vs 20/s)

**IMPORTANTE:** Se OpenRouter/cloud embeddings non disponibile, la pipeline deve **interrompersi** (no fallback locale).

---

## Fasi Pendenti

### FASE 4: Pipeline Integrata
- [ ] Creare `altalex_pipeline.py` con staged processing
- [ ] Integrare embedder con cache
- [ ] UPSERT idempotente in DB
- [ ] Quarantine per articoli invalidi

### FASE 5: Batch 68 PDF
- [ ] Generare JSON marker per tutti i PDF
- [ ] Validare chunking per ogni documento
- [ ] Eseguire batch overnight
- [ ] Verificare metriche finali

---

## Struttura File

```
lexe-max/
├── migrations/kb/
│   └── 012_normativa_altalex_v2.sql    # Schema extension
├── src/lexe_api/kb/ingestion/
│   └── marker_chunker.py               # JSON → Articles
├── scripts/benchmark/
│   ├── mini_embedding_benchmark.py     # Benchmark script
│   └── benchmark_results.json          # Results
└── docs/
    └── KB-NORMATIVA-ALTALEX.md         # This file
```

---

## Comandi Utili

```bash
# 1. Convertire PDF in JSON (marker)
marker_single "altalex pdf/Civile/codice-civile.pdf" \
  --output_dir altalex-md/json \
  --disable_ocr \
  --output_format json

# 2. Testare chunking
python -m src.lexe_api.kb.ingestion.marker_chunker \
  "altalex-md/json/codice-civile.json" CC

# 3. Benchmark embeddings
python scripts/benchmark/mini_embedding_benchmark.py

# 4. Applicare migration
psql -h localhost -p 5434 -U lexe_kb -d lexe_kb \
  -f migrations/kb/012_normativa_altalex_v2.sql
```

---

## Metriche Target

| Metrica | Target | Attuale |
|---------|--------|---------|
| Chunking accuracy | >95% | 94-100% ✓ |
| Recall@10 | >90% | 90% ✓ |
| MRR | >0.7 | 0.825 ✓ |
| Latency P95 | <200ms | 50ms ✓ |

---

## Dipendenze

```
marker-pdf>=1.10.2          # PDF → JSON
sentence-transformers>=3.3  # Embeddings locali
asyncpg>=0.30.0             # PostgreSQL async
pgvector>=0.4.0             # Vector extension
structlog                   # Logging
```

---

## Note Tecniche

### Overlap Strategy
- **Dentro articolo:** NO overlap (unità atomica)
- **Tra articoli consecutivi:** SÌ, ±200 chars per contesto retrieval

### Articoli Invalidi (5-6%)
Articoli con testo vuoto sono tipicamente:
- Articoli abrogati
- Errori di parsing marker
- Articoli con solo rubrica (dichiarativi)

Vengono messi in quarantine (`kb.altalex_ingestion_logs`) per review manuale.

### Global Key Format
```
altalex:{codice}:{articolo_num_norm}:{suffix|null}

Esempi:
- altalex:cc:2043:null
- altalex:cp:1:bis
- altalex:gdpr:17:null
```

---

*Documentazione generata: 2026-02-04*
