# Piano Implementazione Grafi KB Massimari

> Progressive Graph Implementation with Tests, Benchmarks & Visualization
> Status: **PLANNING** | Created: 2026-01-31

---

## Stato Attuale KB (Baseline)

| Metrica          | Valore         |
| ---------------- | -------------- |
| Massime attive   | 41,437         |
| Embeddings       | 41,437         |
| Recall@10 hybrid | 97.5%          |
| MRR              | 0.756          |
| RV popolati      | 16,002 (38.6%) |

**Obiettivo:** Aggiungere Citation Graph + Thematic Graph + GraphRAG reranking

---

## Stack Disponibile

| Componente    | Container    | Stato         | Note          |
| ------------- | ------------ | ------------- | ------------- |
| Apache AGE    | lexe-postgres | ✅ Disponibile | Cypher su PG  |
| pgvector      | lexe-kb      | ✅ Attivo      | HNSW 1536dim  |
| Grafana       | lexe-grafana  | ✅ Deployed    | Dashboard     |
| PostgreSQL 17 | lexe-kb      | ✅ Attivo      | Needs AGE ext |

**Decisione:** Usare lexe-kb con AGE extension (non container separato)

---

## Fase 0: Setup AGE Extension

### 0.1 Aggiungere AGE a lexe-kb

**File:** `docker-compose.kb.yml`

```yaml
lexe-kb:
  image: apache/age:PG17_latest  # Cambia da postgres:17
  # ... rest config invariato
```

**File:** `scripts/graph/migrations/003_age_setup.sql`

```sql
CREATE EXTENSION IF NOT EXISTS age;
LOAD 'age';
SET search_path = ag_catalog, "$user", public, kb;

SELECT create_graph('lexe_citations');
```

### 0.2 Test Setup

```python
# tests/kb/graph/test_age_setup.py
async def test_age_extension():
    assert await conn.fetchval("SELECT extname FROM pg_extension WHERE extname='age'")

async def test_graph_created():
    result = await conn.fetchrow("SELECT * FROM ag_graph WHERE name='lexe_citations'")
    assert result is not None
```

### Benchmark 0

| Check        | Target |
| ------------ | ------ |
| AGE loaded   | ✅      |
| Graph exists | ✅      |
| Cypher works | < 10ms |

---

## Fase 1: Citation Graph Extraction

### 1.1 Schema Nodi/Edges

```sql
-- Nodo Massima (dentro AGE graph)
(:Massima {id, rv, sezione, numero, anno})

-- Tipi di edge
(:Massima)-[:CITES {confidence, context}]->(:Massima)
(:Massima)-[:CONFIRMS]->(:Massima)
(:Massima)-[:DISTINGUISHES]->(:Massima)
(:Massima)-[:OVERRULES {is_turning_point}]->(:Massima)
```

### 1.2 Citation Extractor

**File:** `src/lexe_api/kb/graph/citation_extractor.py`

```python
CITATION_PATTERN = r"Rv\.?[\s\u00a0]+(\d{5,7})(?:-\d+)?"

RELATION_INDICATORS = {
    "CONFIRMS": ["conforme", "nello stesso senso", "v. anche"],
    "DISTINGUISHES": ["diversamente", "va distinto"],
    "OVERRULES": ["contra", "in senso contrario", "superando"],
    "CITES": []  # default
}

async def extract_citations(massima_id: str, testo: str) -> list[CitationEdge]:
    """Estrae citazioni con tipo relazione."""
```

### 1.3 Batch Builder

**File:** `scripts/graph/build_citation_graph.py`

```bash
uv run python scripts/graph/build_citation_graph.py --batch-size 1000 --dry-run
uv run python scripts/graph/build_citation_graph.py --batch-size 1000 --commit
```

### 1.4 Tests

```python
def test_rv_extraction():
    text = "cfr. Cass. Sez. Un., Rv. 639966"
    citations = extract_citations_from_text(text)
    assert citations[0].target_rv == "639966"

def test_relation_detection():
    text = "in senso contrario: Rv. 654321"
    citations = extract_citations_from_text(text)
    assert citations[0].relation_type == "OVERRULES"
```

### Benchmark 1

| Metrica                | Target   |
| ---------------------- | -------- |
| Edges estratti         | > 5,000  |
| Precision (sample 100) | > 90%    |
| Build time full        | < 30 min |

### 1.5 Visualization: Grafana Dashboard

**Dashboard:** "KB Citation Graph Stats"

```sql
-- Panel 1: Network Stats
SELECT 'Nodes' as metric, count(*) as value
FROM cypher('lexe_citations', $$MATCH (n:Massima) RETURN count(n)$$) as (c agtype)
UNION ALL
SELECT 'Edges', count(*)
FROM cypher('lexe_citations', $$MATCH ()-[r]->() RETURN count(r)$$) as (c agtype);

-- Panel 2: Top Cited (Table)
SELECT rv, anno, citations
FROM cypher('lexe_citations', $$
    MATCH (m:Massima)<-[:CITES]-()
    RETURN m.rv, m.anno, count(*) as citations
    ORDER BY citations DESC LIMIT 20
$$) as (rv agtype, anno agtype, citations agtype);

-- Panel 3: Relation Types (Pie)
SELECT type, count
FROM cypher('lexe_citations', $$
    MATCH ()-[r]->()
    RETURN type(r) as type, count(*) as count
$$) as (type agtype, count agtype);
```

---

## Fase 2: Topic Classification

### 2.1 Macro-Categorie (~30-50)

**File:** `src/lexe_api/kb/graph/categories.py`

```python
MACRO_CATEGORIES = {
    # Civile
    "OBBLIGAZIONI": {"keywords": ["obbligazione", "debito", "adempimento"], "parent": None},
    "CONTRATTI": {"keywords": ["contratto", "consenso", "risoluzione"], "parent": None},
    "RESPONSABILITA_CIVILE": {"keywords": ["risarcimento", "danno", "colpa"], "parent": None},
    "PROPRIETA": {"keywords": ["proprietà", "possesso", "usucapione"], "parent": None},
    "FAMIGLIA": {"keywords": ["matrimonio", "separazione", "divorzio"], "parent": None},

    # Processuale
    "COMPETENZA": {"keywords": ["competenza", "giurisdizione"], "parent": None},
    "PROVE": {"keywords": ["prova", "testimone", "documento"], "parent": None},
    "IMPUGNAZIONI": {"keywords": ["appello", "ricorso", "cassazione"], "parent": None},

    # Penale
    "REATI_PERSONA": {"keywords": ["omicidio", "lesioni"], "parent": None},
    "REATI_PATRIMONIO": {"keywords": ["furto", "rapina", "truffa"], "parent": None},

    # ... ~20 altre
}
```

### 2.2 Classification Pipeline

```python
async def classify_massima(
    testo: str,
    embedding: list[float],
    method: str = "hybrid"  # keyword | embedding | hybrid
) -> list[tuple[str, float]]:
    """
    Returns: [(category_id, confidence), ...]
    Solo categorie con confidence >= 0.6
    Max 3 categorie per massima
    """
```

### 2.3 Schema

```sql
(:Category {id, name, description, parent_id})
(:Massima)-[:HAS_TOPIC {confidence, method}]->(:Category)
(:Category)-[:SUBCATEGORY_OF]->(:Category)
```

### 2.4 Tests

```python
def test_keyword_classification():
    text = "risarcimento del danno per responsabilità"
    cats = classify_by_keywords(text)
    assert "RESPONSABILITA_CIVILE" in [c[0] for c in cats]

def test_max_categories():
    cats = classify_massima(text, embedding)
    assert len([c for c in cats if c[1] >= 0.6]) <= 3
```

### Benchmark 2

| Metrica                | Target       |
| ---------------------- | ------------ |
| Coverage               | 100% massime |
| Precision sample       | > 85%        |
| Avg categories/massima | 1-3          |

---

## Fase 3: Norm Graph

### 3.1 Norm Extraction

```python
NORM_PATTERNS = [
    (r"art\.?\s*(\d+(?:\s*bis)?)\s*c\.?\s*c\.?", "CC"),
    (r"art\.?\s*(\d+)\s*c\.?\s*p\.?\s*c\.?", "CPC"),
    (r"(?:l\.|legge)\s*n?\.?\s*(\d+)/(\d{4})", "LEGGE"),
]

async def extract_norms(testo: str) -> list[NormRef]:
    """Returns: [(type, code, article)]"""
```

### 3.2 Schema

```sql
(:Norm {id, type, code, article, full_ref})
(:Massima)-[:CITES_NORM {context}]->(:Norm)
```

### 3.3 Tests

```python
def test_cc_extraction():
    text = "ai sensi dell'art. 2043 c.c."
    norms = extract_norms(text)
    assert norms[0] == ("CC", "2043", None)
```

### Benchmark 3

| Metrica      | Target   |
| ------------ | -------- |
| Norm refs    | > 50,000 |
| Unique norms | > 500    |
| Precision    | > 95%    |

---

## Fase 4: Turning Points Detection

### 4.1 Overrule Signals

```python
TURNING_POINT_SIGNALS = [
    r"superando\s+(?:il\s+)?(?:precedente\s+)?orientamento",
    r"Sez\.?\s*Un\.?\s*.*(?:risolve|compone)\s+(?:il\s+)?contrasto",
    r"muta\s+(?:il\s+)?(?:proprio\s+)?orientamento",
]

async def detect_turning_points() -> list[TurningPoint]:
    """Identifica massime che cambiano orientamento."""
```

### 4.2 Schema

```sql
(:Massima)-[:OVERRULES {
    is_turning_point: true,
    contrast_resolved: text
}]->(:Massima)
```

### Benchmark 4

| Metrica                 | Target |
| ----------------------- | ------ |
| Turning points detected | > 50   |
| Precision sample        | > 80%  |

---

## Fase 5: GraphRAG Reranking

### 5.1 Graph Expansion

**File:** `src/lexe_api/kb/retrieval/graph_reranker.py`

```python
async def graph_expand(
    seed_ids: list[str],
    depth: int = 2,
    min_weight: float = 0.5
) -> list[GraphNode]:
    """
    Cypher: MATCH path = (seed)-[*1..depth]-(related)
    Returns: related nodes with path weight
    """
```

### 5.2 Hybrid + Graph Pipeline

```python
async def hybrid_search_with_graph(
    query: str,
    embedding: list[float],
    top_k: int = 10,
    graph_boost: float = 0.2
) -> list[SearchResult]:
    """
    1. Hybrid search → top 50
    2. Graph expand from top 10
    3. Boost graph-connected results
    4. Re-rank → top K
    """
```

### 5.3 Tests

```python
async def test_graph_boost():
    results_base = await hybrid_search(query, embedding, 10)
    results_graph = await hybrid_search_with_graph(query, embedding, 10)

    # Graph deve avere effetto (ordine diverso o graph_connected flag)
    assert any(r.graph_connected for r in results_graph) or \
           results_base[0].massima_id != results_graph[0].massima_id
```

### Benchmark 5

| Metrica        | Baseline | With Graph | Target     |
| -------------- | -------- | ---------- | ---------- |
| Recall@10      | 97.5%    | ≥ 97.5%    | maintain   |
| MRR            | 0.756    | ≥ 0.76     | improve    |
| Latency p95    | 78ms     | < 150ms    | acceptable |
| Graph hit rate | 0%       | > 30%      | new metric |

---

## Fase 6: API Endpoints + Visualization

### 6.1 REST Endpoints

**File:** `src/lexe_api/api/graph.py`

```python
@router.get("/api/v1/kb/graph/explore/{massima_id}")
async def explore_graph(massima_id: str, depth: int = 2)

@router.get("/api/v1/kb/graph/category/{category_id}")
async def category_cluster(category_id: str, limit: int = 100)

@router.get("/api/v1/kb/graph/turning-points")
async def list_turning_points(from_date: date = None)
```

### 6.2 Grafana Full Dashboard

Panels:

1. **Network Stats** - nodes, edges, avg degree
2. **Top Cited** - table with RV, anno, citation count
3. **Category Distribution** - pie chart
4. **Turning Points Timeline** - time series
5. **Relation Types** - bar chart (CITES, CONFIRMS, etc.)
6. **Most Connected Norms** - table CC/CPC articles

### 6.3 React Component (future)

```typescript
// lexe-frontend/src/components/kb/GraphExplorer.tsx
// D3 force-directed graph visualization
```

---

## Timeline Riepilogo

| Fase                    | Durata     | Output                 |
| ----------------------- | ---------- | ---------------------- |
| 0. Setup AGE            | 1 giorno   | Extension + graph      |
| 1. Citation Graph       | 3-5 giorni | ~10k edges             |
| 2. Topic Classification | 5-7 giorni | 30-50 categorie        |
| 3. Norm Graph           | 3-4 giorni | ~500 norme             |
| 4. Turning Points       | 2-3 giorni | Eventi overrule        |
| 5. GraphRAG             | 3-4 giorni | Reranking integrato    |
| 6. API + Viz            | 3-4 giorni | Endpoints + dashboards |

**Totale:** ~20-28 giorni

---

## Files da Creare

| File                                          | Scopo                |
| --------------------------------------------- | -------------------- |
| `scripts/graph/migrations/003_age_setup.sql`  | Setup AGE            |
| `src/lexe_api/kb/graph/citation_extractor.py` | Pattern extraction   |
| `src/lexe_api/kb/graph/edge_builder.py`       | Build edges          |
| `src/lexe_api/kb/graph/classifier.py`         | Topic classification |
| `src/lexe_api/kb/graph/categories.py`         | Category definitions |
| `src/lexe_api/kb/graph/norm_extractor.py`     | Norm extraction      |
| `src/lexe_api/kb/graph/overrule_detector.py`  | Turning points       |
| `src/lexe_api/kb/retrieval/graph_reranker.py` | GraphRAG             |
| `scripts/graph/build_citation_graph.py`       | Batch script         |
| `scripts/graph/classify_massime.py`           | Batch script         |
| `tests/kb/graph/test_*.py`                    | Test suite           |

---

## Verification Plan

### Per ogni Fase

1. **Run tests**: `pytest tests/kb/graph/test_*.py -v`
2. **Check benchmarks**: confronta con target table
3. **Grafana check**: verifica dashboard si popola
4. **Sample review**: manuale su 20-50 risultati

### End-to-End

```bash
# 1. Retrieval eval con graph
uv run python scripts/qa/run_retrieval_eval.py \
  --mode hybrid_graph --top-k 10 --log-results

# 2. Verifica metriche
# Recall >= 97.5%, MRR >= 0.76, Latency < 150ms

# 3. Grafana
# Open http://localhost:3000/d/kb-graph-stats
```

---

## Note Implementative

1. **Visualization first** - Grafana dashboards fin da Fase 1
2. **Test-driven** - Ogni fase con test completi
3. **Benchmark gates** - Non procedere senza validazione
4. **Feature flag** - GraphRAG behind FF_KB_GRAPH_RERANK
5. **Hybrid fallback** - Graph boost additivo, non sostitutivo
6. **Incremental** - Ogni fase deployabile indipendentemente

---

*Piano creato: 2026-01-31*
*Prossimo step: Fase 0 - Setup AGE*
