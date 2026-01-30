# KB Embedding Benchmark - Decision Log v1

> Data: 2026-01-27
> Status: **APPROVED**
> Autori: Francesco + Claude Code

---

## Decisione Strategica

**Approccio "Rally Car"**: tracciato -> motore -> tempi sul giro.

Partenza lean con configurazione controllabile, espansione dopo primi risultati.

---

## 1. Unita di Verita

| Livello | Definizione | Uso |
|---------|-------------|-----|
| **Unita canonica** | Massima intera | Storage, linking, display |
| **Unita di retrieval** | Dipende dal modello | Search, ranking |

### Regola Chunking per Embedding

| Modello | Max Tokens | Strategia |
|---------|------------|-----------|
| BGE-M3 | 8192 | Massima intera |
| E5-Large | 512 | Sottochunk se > 512 |
| Distil-ITA-Legal-BERT | 512 | Sottochunk se > 512 |

**Sottochunk spec:**
- Size: 250-350 tokens
- Overlap: 40-80 tokens
- Retrieval: a livello sottochunk
- Aggregazione: risalita a `massima_id`, score = max(chunk_scores)

---

## 2. Candidati al Ring

### Embedding Models (4)

| Modello | Dim | Ruolo | Note |
|---------|-----|-------|------|
| **OpenAI text-embedding-3-large** | 2000 | Gold Standard | Top cloud model, max quality |
| **BGE-M3** | 1024 | Riferimento OSS | Best-in-class multilingua, 8192 tokens |
| **Distil-ITA-Legal-BERT** | 768 | Challenger | Specialist italiano legale |
| **Qwen3** | 1536 | Baseline | Via LiteLLM (placeholder) |

> **Nota**: KB one-shot con aggiornamenti annuali. Investimento su qualita embedding giustificato.

### Retrieval Variants (2)

| Variante | Componenti | Note |
|----------|------------|------|
| **R1 Hybrid** | BM25/FTS + pgvector + RRF | Senza reranker |
| **R2 Hybrid+Rerank** | R1 + reranker top-30 | Con reranker |

### Reranker (1)

| Modello | Note |
|---------|------|
| **bge-reranker-v2-m3** | Multilingua robusto, hostable |

### Totale Corse: 8

```
OpenAI-Large   x R1 = 1  (gold standard)
OpenAI-Large   x R2 = 2  (gold standard + rerank)
BGE-M3         x R1 = 3
BGE-M3         x R2 = 4
ITA-Legal-BERT x R1 = 5
ITA-Legal-BERT x R2 = 6
Qwen3          x R1 = 7  (optional baseline)
Qwen3          x R2 = 8  (optional baseline)
```

---

## 3. Schema Database

### Tabelle Embeddings (per modello)

```sql
kb.emb_openai_large     -- VECTOR(2000) - Gold standard
kb.emb_bge_m3           -- VECTOR(1024) - Best OSS
kb.emb_ita_legal_bert   -- VECTOR(768)  - Italian legal
kb.emb_e5_large         -- VECTOR(1024) [placeholder]
kb.emb_qwen3            -- VECTOR(1536) [placeholder]
```

### Struttura Comune

| Colonna | Tipo | Note |
|---------|------|------|
| id | UUID | PK |
| massima_id | UUID | FK -> kb.massime |
| chunk_idx | SMALLINT | 0 = full, 1+ = sottochunk |
| embedding | VECTOR(dim) | Dimensione fissa per modello |
| created_at | TIMESTAMPTZ | - |

### Indici HNSW

```sql
-- Cosine similarity, m=16, ef_construction=64
CREATE INDEX idx_emb_{model}_hnsw
ON kb.emb_{model} USING hnsw (embedding vector_cosine_ops);
```

---

## 4. Query Set

| Categoria | Count | Descrizione |
|-----------|-------|-------------|
| **Istituto** | 80 | Query su concetti giuridici (es. "responsabilita medica") |
| **Avversaria** | 80 | Query che tentano di confondere (es. negazioni, casi limite) |
| **Citazione** | 40 | Query su riferimenti normativi specifici |
| **TOTALE** | **200** | - |

### Breakdown Addizionale

- Per tipo documento: civile vs penale
- Per profilo ingestion: default vs toc_collision

---

## 5. Metriche

### Primarie

| Metrica | Descrizione | Soglia Minima |
|---------|-------------|---------------|
| **Recall@20** | % relevant trovati in top 20 | >= 0.92 |
| **MRR@10** | Mean Reciprocal Rank top 10 | - |
| **Precision@5** | Precisione top 5 | - |

### Qualita

| Metrica | Descrizione | Soglia |
|---------|-------------|--------|
| **Citation Completeness** | % citazioni correttamente estratte | >= 0.85 |
| **Contradiction Rate** | % risposte contradditorie | <= 0.03 |

### Performance

| Metrica | Descrizione |
|---------|-------------|
| **Latency p50** | Mediana tempo risposta |
| **Latency p95** | 95th percentile |

---

## 6. Criteri di Vittoria

### Soglie Minime (Gate)

```
Recall@20 >= 0.92
Citation Completeness >= 0.85
Contradiction Rate <= 0.03
```

### Formula Vittoria

```
score = 0.6 * avg(istituto_metrics) + 0.4 * avg(avversaria_metrics)
```

**Peso maggiore alle avversarie** perche sono quelle che fanno inciampare un assistant.

### Vincitore

Sistema che:
1. Passa TUTTE le soglie minime
2. Ha il miglior `score` combinato

---

## 7. Piano Esecuzione

### Fase 1: Generazione Embeddings

```bash
# Per ogni modello
python scripts/generate_embeddings.py --model bge_m3
python scripts/generate_embeddings.py --model ita_legal_bert
python scripts/generate_embeddings.py --model qwen3
```

### Fase 2: Query Set Creation

```bash
# Genera query set con categorie
python scripts/create_query_set.py --output data/benchmark_queries.json
```

### Fase 3: Benchmark Run

```bash
# 6 corse
python scripts/run_benchmark.py --model bge_m3 --retrieval hybrid
python scripts/run_benchmark.py --model bge_m3 --retrieval hybrid_rerank
# ... etc
```

### Fase 4: Analisi

```bash
# Report comparativo
python scripts/analyze_benchmark.py --runs all --output reports/benchmark_v1.md
```

---

## 8. Espansioni Future

Dopo v1, se i numeri reggono:

1. **+1 modello**: multilingual-e5-large (generalista)
2. **+1 retrieval**: ColBERT-style con BGE-M3
3. **+1 reranker**: italiano-specifico se disponibile
4. **Graph expansion**: Cypher per citazioni incrociate

---

## Changelog

| Data | Versione | Note |
|------|----------|------|
| 2026-01-27 | v1 | Decision Log iniziale |

---

*"Prima il tracciato, poi il motore, poi i tempi sul giro."*
