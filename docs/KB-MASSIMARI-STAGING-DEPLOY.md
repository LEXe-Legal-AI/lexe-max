# KB Massimari - Staging Deployment

> Knowledge Base verticale per i Massimari della Corte di Cassazione
> Deploy: 2026-01-28 | Server: 91.99.229.111 (LEXe Staging)

---

## 1. Overview

Il KB Massimari è un sistema di retrieval ibrido (full-text + vector + graph) per la giurisprudenza italiana, deployato su staging come vertical knowledge base indipendente.

### Statistiche Finali

| Metrica                   | Valore       |
| ------------------------- | ------------ |
| **PDF processati**        | 63           |
| **Documenti nel DB**      | 63           |
| **Massime totali**        | 3,127        |
| **Embeddings Mistral**    | 3,127 (100%) |
| **Copertura temporale**   | 2010-2024    |
| **Dimensione embeddings** | 1024 dim     |

---

## 2. Infrastruttura

### 2.1 Stack Tecnologico

```
91.99.229.111 (LEXE Staging)
├── PostgreSQL 17.7
│   ├── pgvector 0.8.1      # HNSW vector search
│   ├── Apache AGE 1.6.0    # Graph database (Cypher)
│   └── pg_trgm 1.6         # Fuzzy text search
├── Unstructured API :8500  # PDF extraction
└── Python 3.12 + uv        # Scripts runtime
```

### 2.2 Schema Database

```sql
-- Schema: kb

kb.documents        -- 63 PDF sorgente
kb.massime          -- 3,127 massime estratte
kb.emb_mistral      -- 3,127 embeddings 1024-dim
kb.sections         -- Sezioni gerarchiche (future)
kb.citations        -- Citazioni estratte (future)
kb.edge_weights     -- Pesi per graph edges (future)
```

### 2.3 Indici

```sql
-- Full-text search (tsvector)
idx_massime_tsv_italian    -- Stemming italiano
idx_massime_tsv_simple     -- Match esatto

-- Vector search (HNSW)
idx_emb_mistral_hnsw       -- m=16, ef_construction=64

-- Fuzzy search (trigram)
idx_massime_trgm           -- gin_trgm_ops
```

---

## 3. Pipeline di Ingestion

### 3.1 Flusso

```
PDF → Unstructured API → Elements → Gate Policy → Massime → DB
                                         ↓
                              OpenRouter API → Embeddings → DB
```

### 3.2 Gate Policy

Filtri applicati per escludere contenuto non-massima:

| Filtro               | Valore               | Scopo                  |
| -------------------- | -------------------- | ---------------------- |
| `MIN_LENGTH`         | 150 chars            | Esclude frammenti      |
| `MAX_CITATION_RATIO` | 3%                   | Esclude indici/sommari |
| `BAD_STARTS`         | ", del", "INDICE"... | Esclude metadata       |

### 3.3 Script Utilizzati

| Script                           | Funzione                                  |
| -------------------------------- | ----------------------------------------- |
| `ingest_staging.py`              | Ingestion principale con parsing filename |
| `ingest_recover_anno.py`         | Recovery anno da prime pagine PDF         |
| `fix_approfondimenti.py`         | Fix per file con tipo "unknown"           |
| `generate_embeddings_staging.py` | Generazione embeddings Mistral            |
| `test_retrieval_staging.py`      | Test R1/R2/Hybrid                         |

### 3.4 Parsing Filename

Il parser riconosce pattern multipli:

```python
# Pattern supportati:
"Volume I_2024_Massimario_Civile..."     → 2024, civile, vol.1
"Rassegna Civile 2012 - I volume.pdf"    → 2012, civile, vol.1
"2023_MASSIMARIO PENALE VOL. 1..."       → 2023, penale, vol.1
"rassegna civile 2020 vol_IV.pdf"        → 2020, civile, vol.4

# Recovery automatico anno:
# Se filename senza anno, estrae da prime 30 pagine
```

---

## 4. Retrieval System

### 4.1 Metodi Disponibili

| Metodo              | Latenza | Descrizione                 |
| ------------------- | ------- | --------------------------- |
| **R1** Full-Text    | ~20ms   | tsvector italiano + simple  |
| **R2-Local** Vector | ~200ms  | Mistral embeddings + cosine |
| **Hybrid** RRF      | ~200ms  | Reciprocal Rank Fusion k=60 |

### 4.2 Query R1 (Full-Text)

```sql
SELECT m.id, m.anno, m.tipo, LEFT(m.testo, 200) as snippet,
       ts_rank(m.tsv_italian, plainto_tsquery('italian', $1)) as score
FROM kb.massime m
WHERE m.tsv_italian @@ plainto_tsquery('italian', $1)
ORDER BY score DESC
LIMIT 5;
```

### 4.3 Query R2-Local (Vector)

```sql
SELECT m.id, m.anno, m.tipo, LEFT(m.testo, 200) as snippet,
       1 - (e.embedding <=> $1::vector) as similarity
FROM kb.emb_mistral e
JOIN kb.massime m ON m.id = e.massima_id
ORDER BY e.embedding <=> $1::vector
LIMIT 5;
```

### 4.4 Query Hybrid (RRF)

```sql
WITH r1_results AS (
    SELECT m.id, ROW_NUMBER() OVER (ORDER BY ts_rank(...) DESC) as rank
    FROM kb.massime m WHERE m.tsv_italian @@ ...
    LIMIT 20
),
r2_results AS (
    SELECT m.id, ROW_NUMBER() OVER (ORDER BY e.embedding <=> ...) as rank
    FROM kb.emb_mistral e JOIN kb.massime m ON ...
    LIMIT 20
),
combined AS (
    SELECT COALESCE(r1.id, r2.id) as id,
           COALESCE(1.0/(60+r1.rank), 0) + COALESCE(1.0/(60+r2.rank), 0) as rrf_score
    FROM r1_results r1 FULL OUTER JOIN r2_results r2 ON r1.id = r2.id
)
SELECT * FROM combined ORDER BY rrf_score DESC LIMIT 5;
```

### 4.5 Benchmark Risultati

Query testate:

| Query                                   | R1 Score | R2 Similarity |
| --------------------------------------- | -------- | ------------- |
| responsabilità medica danno alla salute | 0.98     | 0.81          |
| risarcimento danni contratto            | 1.00     | 0.79          |
| prescrizione crediti lavoro             | 0.76     | 0.82          |
| licenziamento giusta causa              | 0.95     | 0.83          |
| proprietà immobiliare usucapione        | 0.53     | 0.80          |

**Osservazione:** R2 (vector) trova risultati semantici anche quando R1 (lessicale) ha score basso.

---

## 5. Graph Database (Apache AGE)

### 5.1 Stato Attuale

- **Installato:** Apache AGE 1.6.0
- **Grafo creato:** `lexe_jurisprudence`
- **Popolato:** NO (next step)

### 5.2 Schema Grafo Previsto

```cypher
-- Nodi
(:Massima {id, anno, tipo, sezione, numero})
(:Articolo {codice, numero})  -- es. "c.c. art. 2043"
(:Sentenza {sezione, numero, anno, rv})

-- Edges
[:CITA]        -- Massima cita Articolo/Sentenza
[:CONFERMA]    -- Massima conferma orientamento
[:CONTRASTA]   -- Massima contrasta orientamento
[:INTERPRETA]  -- Massima interpreta norma
```

---

## 6. Evoluzione Grafo: Roadmap Tecnica

### 6.1 Disambiguazione Citazioni

**Problema:** Una citazione come "Sez. 3, n. 123" è ambigua senza anno.

**Soluzione proposta:**

```python
# Pattern di citazione completo
CITATION_PATTERN = r"""
    Sez\.?\s*(?:Un\.|U\.|(\d|[IVX]+))\s*,?\s*
    (?:ord\.\s*)?
    n\.?\s*(\d+)/(\d{4})\s*,?\s*
    ([A-Z][a-z]+)?\s*,?\s*      # Relatore (opzionale)
    Rv\.?\s*(\d+(?:-\d+)?)?     # Rv (opzionale)
"""

# Esempio match:
# "Sez. 3, n. 12345/2020, Rossi, Rv. 654321-01"
# → sezione=3, numero=12345, anno=2020, relatore=Rossi, rv=654321-01
```

**Strategia di disambiguazione:**

1. **Match esatto:** Se anno + numero + sezione → univoco
2. **Match fuzzy:** Se manca anno, cerca in finestra ±2 anni dal documento citante
3. **Match Rv:** Il numero Rv è sempre univoco (identificatore Cassazione)
4. **Fallback:** Crea nodo "citazione non risolta" per review manuale

```sql
-- Tabella per tracking disambiguazione
CREATE TABLE kb.citation_resolution (
    id UUID PRIMARY KEY,
    raw_citation TEXT,           -- "Sez. 3, n. 123"
    resolved_massima_id UUID,    -- FK se risolta
    confidence FLOAT,            -- 0.0-1.0
    resolution_method VARCHAR,   -- 'exact', 'fuzzy', 'rv', 'manual'
    created_at TIMESTAMPTZ
);
```

### 6.2 Calcolo Pesi degli Edges

**Problema:** Non tutte le citazioni hanno lo stesso peso. Una citazione in apertura ("Come affermato da Sez. Un. n. X...") è più rilevante di una in nota.

**Fattori per il peso:**

| Fattore                  | Peso    | Logica                              |
| ------------------------ | ------- | ----------------------------------- |
| **Posizione**            | 0.1-0.4 | Inizio=0.4, metà=0.2, fine=0.1      |
| **Tipo citazione**       | 0.1-0.3 | Sez.Un.=0.3, Sez.=0.2, generico=0.1 |
| **Contesto linguistico** | 0.1-0.3 | "conferma"=0.3, "v. anche"=0.1      |
| **Recency**              | 0.0-0.2 | Citazione recente=0.2, vecchia=0.0  |

**Formula:**

```python
def calculate_edge_weight(citation: Citation, source: Massima, target: Massima) -> float:
    weight = 0.0

    # Posizione nel testo (0-1 normalizzato)
    position_ratio = citation.char_offset / len(source.testo)
    if position_ratio < 0.2:
        weight += 0.4  # Inizio
    elif position_ratio < 0.5:
        weight += 0.2  # Metà
    else:
        weight += 0.1  # Fine

    # Tipo sezione
    if "Sez. Un." in citation.raw or "Sezioni Unite" in citation.raw:
        weight += 0.3
    elif "Sez." in citation.raw:
        weight += 0.2
    else:
        weight += 0.1

    # Contesto linguistico
    context = source.testo[max(0, citation.char_offset-50):citation.char_offset]
    if any(w in context.lower() for w in ["conferma", "ribadisce", "costante"]):
        weight += 0.3
    elif any(w in context.lower() for w in ["v. anche", "cfr.", "si veda"]):
        weight += 0.1
    else:
        weight += 0.2

    # Recency (anni di distanza)
    years_diff = abs(source.anno - target.anno)
    recency_bonus = max(0, 0.2 - (years_diff * 0.02))
    weight += recency_bonus

    return min(1.0, weight)  # Cap a 1.0
```

**Storage:**

```sql
CREATE TABLE kb.edge_weights (
    id UUID PRIMARY KEY,
    source_id UUID NOT NULL,      -- Massima citante
    target_id UUID NOT NULL,      -- Massima/Articolo citato
    edge_type VARCHAR(50),        -- CITA, CONFERMA, etc.
    weight FLOAT NOT NULL,        -- 0.0-1.0
    factors JSONB,                -- {position: 0.4, tipo: 0.3, ...}
    created_at TIMESTAMPTZ
);
```

### 6.3 Inferenza CONFERMA vs CONTRASTA

**Problema:** Determinare se una citazione è di conferma o contrasto richiede comprensione semantica.

**Approccio ibrido:**

#### Step 1: Pattern linguistici (rule-based)

```python
CONFERMA_PATTERNS = [
    r"conferma(ndo|to|re)?",
    r"ribadisce|ribadito",
    r"in linea con",
    r"costante (orientamento|giurisprudenza)",
    r"consolidato",
    r"pacifico",
    r"come già (affermato|statuito)",
]

CONTRASTA_PATTERNS = [
    r"contrasto con",
    r"diversamente da",
    r"superando",
    r"non (condivide|può condividersi)",
    r"in senso contrario",
    r"disattende|disatteso",
    r"overruling",
]

def infer_relation_rule_based(context: str) -> tuple[str, float]:
    """Returns (relation_type, confidence)"""
    context_lower = context.lower()

    for pattern in CONFERMA_PATTERNS:
        if re.search(pattern, context_lower):
            return "CONFERMA", 0.7

    for pattern in CONTRASTA_PATTERNS:
        if re.search(pattern, context_lower):
            return "CONTRASTA", 0.7

    return "CITA", 0.5  # Default neutro
```

#### Step 2: LLM per casi ambigui (confidence < 0.7)

```python
async def infer_relation_llm(
    source_text: str,
    target_text: str,
    citation_context: str,
    client: LiteLLMClient
) -> tuple[str, float]:
    """Use LLM for semantic inference."""

    prompt = f"""Analizza questa citazione giurisprudenziale e classifica la relazione.

MASSIMA CITANTE:
{source_text[:500]}

MASSIMA CITATA:
{target_text[:500]}

CONTESTO CITAZIONE:
{citation_context}

Classifica la relazione come:
- CONFERMA: la massima citante conferma/ribadisce l'orientamento
- CONTRASTA: la massima citante si discosta/contrasta l'orientamento
- INTERPRETA: la massima citante interpreta/chiarisce
- CITA: citazione neutra/di riferimento

Rispondi SOLO con: RELAZIONE|CONFIDENCE (es: CONFERMA|0.9)
"""

    response = await client.complete(
        model="mistral-small",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=20,
        temperature=0.1
    )

    # Parse response
    parts = response.strip().split("|")
    relation = parts[0] if parts[0] in ["CONFERMA", "CONTRASTA", "INTERPRETA", "CITA"] else "CITA"
    confidence = float(parts[1]) if len(parts) > 1 else 0.8

    return relation, confidence
```

#### Step 3: Pipeline completa

```python
async def classify_citation(
    source: Massima,
    target: Massima,
    citation: Citation,
    llm_client: LiteLLMClient
) -> EdgeClassification:
    """Full classification pipeline."""

    # Extract context around citation
    context_start = max(0, citation.char_offset - 100)
    context_end = min(len(source.testo), citation.char_offset + 100)
    context = source.testo[context_start:context_end]

    # Step 1: Rule-based
    relation, confidence = infer_relation_rule_based(context)

    # Step 2: LLM if uncertain
    if confidence < 0.7:
        relation_llm, confidence_llm = await infer_relation_llm(
            source.testo, target.testo, context, llm_client
        )
        # Ensemble: prefer LLM if more confident
        if confidence_llm > confidence:
            relation, confidence = relation_llm, confidence_llm

    # Calculate edge weight
    weight = calculate_edge_weight(citation, source, target)

    return EdgeClassification(
        source_id=source.id,
        target_id=target.id,
        edge_type=relation,
        weight=weight,
        confidence=confidence,
        method="rule_based" if confidence >= 0.7 else "llm_ensemble"
    )
```

### 6.4 Costi Stimati per LLM Inference

| Modello       | Costo/1K tokens | Tokens/citazione | Costo/citazione |
| ------------- | --------------- | ---------------- | --------------- |
| Mistral Small | $0.0002         | ~800             | $0.00016        |
| GPT-4o-mini   | $0.00015        | ~800             | $0.00012        |
| Claude Haiku  | $0.00025        | ~800             | $0.0002         |

**Stima per 3,127 massime:**

- ~5 citazioni/massima media = 15,635 citazioni
- Con rule-based al 70% = 4,690 chiamate LLM
- Costo totale: ~$0.75 - $1.00

---

## 7. Comandi Operativi

### 7.1 Connessione Staging

```bash
# SSH
ssh -i ~/.ssh/id_stage_new root@91.99.229.111

# Database
docker exec -it lexe-postgres psql -U lexe -d lexe
```

### 7.2 Script Runner

```bash
# Ingestion
/opt/lexe-platform/lexe-api/run_ingest.sh

# Embeddings
/opt/lexe-platform/lexe-api/run_embeddings.sh

# Retrieval test
/opt/lexe-platform/lexe-api/run_retrieval_test.sh
```

### 7.3 Query Utili

```sql
-- Conteggi
SELECT COUNT(*) FROM kb.documents;
SELECT COUNT(*) FROM kb.massime;
SELECT COUNT(*) FROM kb.emb_mistral;

-- Distribuzione per anno
SELECT anno, tipo, COUNT(*)
FROM kb.massime
GROUP BY anno, tipo
ORDER BY anno, tipo;

-- Massime senza embedding
SELECT COUNT(*) FROM kb.massime m
LEFT JOIN kb.emb_mistral e ON e.massima_id = m.id
WHERE e.id IS NULL;
```

---

## 8. Next Steps

| Priorità | Task                               | Effort |
| -------- | ---------------------------------- | ------ |
| 1        | Estrazione citazioni (regex)       | 2h     |
| 2        | Popolamento grafo base (CITA)      | 2h     |
| 3        | Query Cypher per retrieval         | 1h     |
| 4        | Disambiguazione citazioni          | 4h     |
| 5        | Calcolo pesi edges                 | 2h     |
| 6        | Inferenza LLM (CONFERMA/CONTRASTA) | 4h     |
| 7        | API endpoints per retrieval        | 4h     |
| 8        | Integrazione con LEXE TRIDENT      | 8h     |

---

## 9. Appendice: File Structure

```
/opt/lexe-platform/lexe-api/
├── data/
│   └── massimari/           # 63 PDF files
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

*Documento creato: 2026-01-28*
*Autore: Claude Code*
*Server: 91.99.229.111 (LEXE Staging)*
