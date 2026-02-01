# KB Massimari - Documento Definitivo

> Knowledge Base verticale per i Massimari della Corte di Cassazione
> Periodo: 27-28 gennaio 2026
> Autori: Francesco + Claude Code

---

## 1. Obiettivo

Costruire un sistema di retrieval ibrido per la giurisprudenza italiana (massimari della Corte di Cassazione), partendo dai PDF e arrivando a un vertical knowledge base con ricerca full-text, semantica e (futuro) graph-based.

---

## 2. Cronologia e Decisioni

### Fase 1 — Setup e Estrazione (27 gennaio, mattina)

**Ambiente:** locale, container `lexe-kb` (PostgreSQL 17.7, porta 5434)

**Test estrazione PDF:** confronto tra due metodi su 3 PDF campione.

| PDF               | PyMuPDF | Unstructured API | Ratio    |
| ----------------- | ------- | ---------------- | -------- |
| 2021 PENALE Vol.1 | 645     | 2,114            | 3.3x     |
| 2023 PENALE Vol.1 | 433     | 1,633            | 3.8x     |
| 2018 CIVILE Vol.1 | 189     | 800              | 4.2x     |
| **Totale**        | 1,267   | **4,547**        | **3.6x** |

**Decisione:** Unstructured API (strategy `fast`, ~25s per PDF).

**Timing alternativo:** `hi_res` ~60s, scartato perché il gain non giustifica il costo.

### Fase 2 — Gate Policy e QA (27 gennaio, pomeriggio)

**Problema:** la pipeline grezza estraeva 1,615 massime con il 24% di falsi positivi (frammenti, indici, liste di citazioni).

**Gate policy implementata:**

| Filtro              | Valore                        | Effetto                 |
| ------------------- | ----------------------------- | ----------------------- |
| Lunghezza minima    | 150 char                      | Elimina frammenti       |
| Max citation ratio  | 3% citazioni/parole           | Elimina liste citazioni |
| Bad starts          | ", del", "INDICE"...          | Elimina metadata        |
| SKIP_PAGES dinamico | Calcolato dalla prima massima | Evita match nel TOC     |

**Risultato:** 1,615 → **772 massime** (-52% falsi positivi, 0% short, 100% sezione linked).

**Sezioni estratte:** 336 (8 parti, 35 capitoli, 265 sezioni) con backfill pagine al 93-100%.

### Fase 3 — Benchmark Embedding (27 gennaio, sera)

**Approccio "Rally Car":** prima il tracciato (dati), poi il motore (embedding), poi i tempi (retrieval).

**Piano originale (Decision Log v1):** 4 modelli locali/cloud (OpenAI 3-Large, BGE-M3, Distil-ITA-Legal-BERT, Qwen3) × 2 retrieval variants = 8 corse, 200 query.

**Pivot:** passaggio a OpenRouter API per semplificare. Modelli testati diversi dal piano.

**Round 1 — 4 modelli via OpenRouter** (sample: 22 massime)

| Modello           | Dim      | Latency   | Avg Similarity |
| ----------------- | -------- | --------- | -------------- |
| Qwen3 8B          | 4096     | 1678ms    | 0.455          |
| OpenAI 3-Large    | 3072     | 1054ms    | 0.451          |
| **Mistral Embed** | **1024** | **782ms** | **0.793**      |
| Gemini Embed      | 3072     | 944ms     | 0.636          |

**Round 2 — Conferma** (Codestral 0.666, Gemini 0.662)

**Vincitore: Mistral Embed** (`mistralai/mistral-embed-2312`, 1024 dim).

### Fase 4 — Benchmark Retrieval (28 gennaio, mattina)

**Query set ridotto:** 14 query (5 istituto, 5 avversaria, 4 citazione) vs le 200 pianificate.

**Reranker API benchmark:**

| Reranker   | Accuracy | Latency |
| ---------- | -------- | ------- |
| Cohere 3.5 | 72.3%    | 230ms   |
| Jina v2    | 71.4%    | 467ms   |
| Voyage 2   | 60.0%    | 310ms   |

**Benchmark finale R1/R2:**

| Metodo                                 | Accuracy  | Latency   | Costo          |
| -------------------------------------- | --------- | --------- | -------------- |
| R1 Hybrid (BM25+Dense+RRF)             | 58.6%     | 55ms      | Solo embedding |
| R2-Cohere (R1+Cohere rerank)           | 74.3%     | 290ms     | API ($)        |
| **R2-Local (R1+cross-encoder locale)** | **78.6%** | **403ms** | **Zero**       |

**Vincitore: R2-Local** — reranker `cross-encoder/ms-marco-MiniLM-L-6-v2` locale, gratuito.

### Fase 5 — Deploy Staging (28 gennaio, pomeriggio)

**Server:** 91.99.229.111 (LEXe Staging)

**Primo tentativo: 51 PDF** dalla cartella `Massimario_PDF`.
Risultato: solo 9 documenti e 541 massime.

**Correzione di rotta:** l'utente nota "perché 9 documenti?" quando ne aspettava 51. Analisi rivela che **42 PDF erano duplicati** (stesso SHA256 hash, nomi diversi — es. 2010, 2011, 2012 civile vol.1 tutti identici con hash `dea996a8...`). Problema nella copia dei file sorgente.

**Secondo tentativo:** l'utente indica la cartella corretta `new/New folder` con **63 file unici** (verificati via SHA256: 63 hash univoci, 1 duplicato escluso).

**Ingestion definitiva:**

- 52 documenti processati direttamente (filename con anno)
- 11 PDF senza anno nel filename → recovery automatico anno dalle prime pagine
- 2 PDF "Approfondimenti Tematici" → fix manuale tipo="civile" (constraint DB impediva tipo="unknown")
- **Risultato: 63 documenti, 3,127 massime**

**Embeddings:** 3,127/3,127 (100%), 142 batch, ~3 minuti, zero errori.

**Test retrieval su staging:**

| Query                                   | R1 Score | R2 Similarity |
| --------------------------------------- | -------- | ------------- |
| responsabilità medica danno alla salute | 0.98     | 0.81          |
| risarcimento danni contratto            | 1.00     | 0.79          |
| prescrizione crediti lavoro             | 0.76     | 0.82          |
| licenziamento giusta causa              | 0.95     | 0.83          |
| proprietà immobiliare usucapione        | 0.53     | 0.80          |

---

## 3. Stato Attuale

| Componente                       | Stato                        |
| -------------------------------- | ---------------------------- |
| PostgreSQL 17.7 + pgvector 0.8.1 | Deployato                    |
| Apache AGE 1.6.0                 | Installato, grafo vuoto      |
| Unstructured API :8500           | Deployato                    |
| Ingestion pipeline               | Completa (63 docs)           |
| Embeddings Mistral               | Completi (3,127)             |
| R1 Full-Text                     | Funzionante                  |
| R2 Vector Search                 | Funzionante                  |
| Hybrid RRF                       | Funzionante                  |
| Reranker locale                  | **NON deployato su staging** |
| Graph (citazioni)                | **NON implementato**         |
| API endpoints                    | **NON implementati**         |

---

## 4. Contraddizioni e Deviazioni dal Piano

### 4.1 Modelli embedding: piano vs realtà

| Piano originale                    | Realtà                           |
| ---------------------------------- | -------------------------------- |
| OpenAI 3-Large (gold standard)     | Scores bassi su italiano (0.451) |
| BGE-M3 (best OSS multilingua)      | Mai testato via OpenRouter       |
| Distil-ITA-Legal-BERT (specialist) | Mai testato via OpenRouter       |
| Qwen3 (baseline)                   | Scores bassi (0.455), lento      |

**Vincitore non pianificato:** Mistral Embed, non era nemmeno nella lista originale del Decision Log.

### 4.2 Infrastruttura: piano vs staging

| Piano (Architecture doc)                    | Staging reale                                |
| ------------------------------------------- | -------------------------------------------- |
| Container dedicato `lexe-kb` porta 5434     | Postgres condiviso `lexe-postgres` porta 5432 |
| User `lexe_kb`, DB `lexe_kb`                | User `lexe`, DB `lexe`, schema `kb`          |
| pg_search (ParadeDB) per BM25               | **NON disponibile**, usa tsvector nativo     |
| Tabella singola `kb.embeddings` multi-model | Tabelle separate `kb.emb_mistral`            |

### 4.3 BM25 vs tsvector

Il piano architetturale prevedeva pg_search (ParadeDB) per BM25 nativo. Su staging non è installato. Il retrieval usa `ts_rank()` con tsvector, che ha un ranking diverso da BM25 (penalizza meno i documenti lunghi). Il benchmark R1 fu fatto con pg_search, ma staging usa tsvector.

### 4.4 Reranker assente su staging

Il benchmark R2-Local (78.6% accuracy) include un cross-encoder locale. Su staging il reranker **non è deployato**. Il test retrieval su staging usa solo vector search senza reranking. La "R2" su staging non è la stessa "R2-Local" del benchmark.

### 4.5 Dimensione del benchmark

| Piano (Decision Log)                                 | Eseguito        |
| ---------------------------------------------------- | --------------- |
| 200 query (80 istituto, 80 avversaria, 40 citazione) | 14 query totali |
| 22 massime per embedding benchmark                   | 22 massime      |
| Cross-validation                                     | Nessuna         |

I risultati hanno validità indicativa, non statistica.

---

## 5. Punti Deboli

### 5.1 QA non eseguita sul dataset staging

La QA dettagliata (0% short, 100% linked, 0 out-of-range) fu fatta sui 772 massime originali (3 PDF). Le 3,127 massime su staging **non hanno metriche QA**. Non sappiamo:

- Percentuale di falsi positivi tra le nuove 2,355 massime
- Se la gate policy calibrata su 3 PDF funziona bene su 63 PDF diversi
- Se i PDF con anno recuperato hanno dati corretti

### 5.2 Sezioni non presenti su staging

Il sistema di sezioni (336 sezioni, 100% linking) esiste solo sul dataset locale. Su staging `kb.sections` è vuoto. Il linking massime-sezioni non è stato fatto.

### 5.3 Anno recuperato senza verifica

11 PDF avevano l'anno estratto dalle prime pagine. Non è stato verificato manualmente se l'anno è corretto. Alcuni potrebbero avere default (2014).

### 5.4 Gate policy potenzialmente troppo permissiva

La gate policy fu calibrata su PDF ben strutturati (2018, 2021, 2023). I PDF 2010-2013 ("Rassegna civile/penale") hanno layout diverso. La policy potrebbe lasciar passare falsi positivi o filtrare massime valide.

### 5.5 Grafo non implementato

Apache AGE è installato e il grafo `lexe_jurisprudence` esiste, ma è completamente vuoto. Nessun nodo, nessun edge. Il valore aggiunto del graph retrieval (R3) non è ancora disponibile.

### 5.6 Nessuna API esposta

Non ci sono endpoint REST. Il KB è accessibile solo via script Python diretti e query SQL.

---

## 6. Stack Approvato (Decisioni Finali)

| Componente        | Scelta                                        | Motivazione                            |
| ----------------- | --------------------------------------------- | -------------------------------------- |
| **Extraction**    | Unstructured API (`fast`)                     | 3.6x vs PyMuPDF                        |
| **Embedding**     | Mistral Embed 1024d via OpenRouter            | 0.793 avg similarity, best su italiano |
| **Sparse search** | PostgreSQL tsvector (italian + simple)        | Nativo, zero overhead                  |
| **Dense search**  | pgvector HNSW (cosine, m=16, ef=64)           | Standard                               |
| **Fusion**        | RRF k=60                                      | Standard                               |
| **Reranker**      | cross-encoder/ms-marco-MiniLM-L-6-v2 (locale) | +20% accuracy, zero costi              |
| **Graph**         | Apache AGE (Cypher)                           | Installato, da popolare                |
| **Gate policy**   | min_length=150, citation_ratio<3%, bad_starts | Filtro 52% falsi positivi              |

---

## 7. Evoluzione Grafo: Piano Tecnico

### 7.1 Estrazione Citazioni

Le massime contengono citazioni a sentenze e articoli. Pattern regex:

```python
# Citazione sentenza completa
# "Sez. 3, n. 12345/2020, Rossi, Rv. 654321-01"
CITATION_PATTERN = r"""
    Sez\.?\s*(?:Un\.|U\.|(\d|[IVX]+))\s*,?\s*
    (?:ord\.\s*)?
    n\.?\s*(\d+)/(\d{4})\s*,?\s*
    ([A-Z][a-z]+)?\s*,?\s*
    Rv\.?\s*(\d+(?:-\d+)?)?
"""
```

### 7.2 Disambiguazione Citazioni

**Problema:** "Sez. 3, n. 123" è ambigua senza anno.

**Strategia a cascata:**

1. **Match Rv** — Il numero Rv (Repertorio) è univoco nel sistema della Cassazione. Se presente, la citazione è risolta.
2. **Match esatto** — anno + numero + sezione identifica univocamente.
3. **Match fuzzy** — senza anno, cerca in finestra ±2 anni dal documento citante.
4. **Fallback** — nodo "citazione non risolta" per review manuale.

```sql
CREATE TABLE kb.citation_resolution (
    id UUID PRIMARY KEY,
    raw_citation TEXT,
    resolved_massima_id UUID,
    confidence FLOAT,            -- 0.0-1.0
    resolution_method VARCHAR,   -- 'rv', 'exact', 'fuzzy', 'manual'
    created_at TIMESTAMPTZ
);
```

### 7.3 Calcolo Pesi degli Edges

Non tutte le citazioni hanno lo stesso peso. Una Sez. Un. citata in apertura vale più di un "v. anche" in chiusura.

**4 fattori:**

| Fattore              | Range   | Logica                              |
| -------------------- | ------- | ----------------------------------- |
| Posizione nel testo  | 0.1-0.4 | Inizio testo=0.4, fine=0.1          |
| Tipo citazione       | 0.1-0.3 | Sez.Un.=0.3, Sez.=0.2, generico=0.1 |
| Contesto linguistico | 0.1-0.3 | "conferma"=0.3, "v. anche"=0.1      |
| Recency              | 0.0-0.2 | Citazione recente=0.2, vecchia=0.0  |

Formula: `weight = sum(fattori)`, capped a 1.0.

### 7.4 Inferenza CONFERMA vs CONTRASTA

Determinare se una citazione conferma o contrasta richiede comprensione semantica.

**Pipeline ibrida:**

**Step 1 — Rule-based (atteso ~70% coverage):**

```python
CONFERMA = ["conferma", "ribadisce", "costante orientamento", "pacifico", "consolidato"]
CONTRASTA = ["contrasto con", "diversamente da", "superando", "overruling", "disattende"]
# Default: CITA (neutro, confidence 0.5)
```

**Step 2 — LLM per ambigui (confidence < 0.7):**

Prompt strutturato → classificazione CONFERMA/CONTRASTA/INTERPRETA/CITA con confidence.

**Costo stimato:**

- ~5 citazioni/massima × 3,127 massime = ~15,635 citazioni
- 70% risolte rule-based = ~4,690 chiamate LLM
- A ~$0.0002/citazione = **~$1 totale**

---

## 8. Comandi Operativi

```bash
# SSH staging
ssh -i ~/.ssh/id_stage_new root@91.99.229.111

# Database
docker exec -it lexe-postgres psql -U lexe -d lexe

# Pipeline
/opt/lexe-platform/lexe-api/run_ingest.sh
/opt/lexe-platform/lexe-api/run_embeddings.sh
/opt/lexe-platform/lexe-api/run_retrieval_test.sh
```

---

## 9. Next Steps (priorità)

| #   | Task                                     | Note                                     |
| --- | ---------------------------------------- | ---------------------------------------- |
| 1   | QA sulle 3,127 massime staging           | Verificare falsi positivi, anno recovery |
| 2   | Deploy reranker cross-encoder su staging | Per ottenere la vera R2-Local (78.6%)    |
| 3   | Estrazione citazioni + grafo base        | Regex → nodi → edges CITA                |
| 4   | API endpoint `/api/v1/kb/search`         | Esporre il retrieval                     |
| 5   | Disambiguazione citazioni                | Rv match, exact, fuzzy                   |
| 6   | Pesi edges + inferenza LLM               | CONFERMA vs CONTRASTA                    |
| 7   | Integrazione LEXE TRIDENT                | Tool per ricerca massime                 |

---

## 10. File di Riferimento

**Questa documentazione sostituisce:**

- `KB-MASSIMARI-ARCHITECTURE.md` — architettura iniziale (locale, 772 massime)
- `KB-EMBEDDING-BENCHMARK-DECISION-LOG.md` — piano benchmark (parzialmente eseguito)
- `KB-MASSIMARI-BENCHMARK-REPORT.md` — risultati benchmark (validi ma su campione ridotto)
- `KB-MASSIMARI-STAGING-DEPLOY.md` — deploy staging (superseded da questo doc)

**Script su staging:**

```
/opt/lexe-platform/lexe-api/
├── data/massimari/           # 63 PDF
├── scripts/
│   ├── ingest_staging.py
│   ├── ingest_recover_anno.py
│   ├── fix_approfondimenti.py
│   ├── generate_embeddings_staging.py
│   └── test_retrieval_staging.py
├── run_ingest.sh
├── run_embeddings.sh
├── run_retrieval_test.sh
└── run_recover.sh
```

---

*Ultimo aggiornamento: 2026-01-28*
