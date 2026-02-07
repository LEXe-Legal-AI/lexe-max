# Evaluation Scripts

> Scripts for LEXE KB Normativa hybrid search evaluation

---

## Scripts

| Script | Purpose | Usage |
|--------|---------|-------|
| `hybrid_search.py` | Full hybrid search implementation | `OPENROUTER_API_KEY=... python hybrid_search.py "query"` |
| `hybrid_test.py` | Test with source indicator (BOTH/DENSE/SPARSE) | `OPENROUTER_API_KEY=... python hybrid_test.py "query"` |
| `embed_normativa.py` | Generate embeddings for normativa chunks | `OPENROUTER_API_KEY=... python embed_normativa.py` |

---

## Requirements

```bash
pip install psycopg2-binary requests
```

## Environment Variables

```bash
export OPENROUTER_API_KEY="sk-or-..."
```

## Database Connection

Scripts connect to lexe-max staging:
- Host: localhost
- Port: 5436 (requires SSH tunnel)
- User: lexe_max
- Database: lexe_max

### SSH Tunnel Setup

```bash
ssh -i ~/.ssh/id_stage_new -L 5436:localhost:5436 root@91.99.229.111
```

---

## Example Usage

### 1. Run Hybrid Search

```bash
OPENROUTER_API_KEY="sk-or-..." python hybrid_test.py "sequestro preventivo beni"
```

Output:
```
Query: sequestro preventivo beni
Generating embedding...
===================================================================================================================
Code  Art        #  Source RRF     Dense  Sparse  Preview
-------------------------------------------------------------------------------------------------------------------
CPP   104-bis    0  BOTH   0.0293  0.606  0.0048  ...
CPC   670        0  DENSE  0.0164  0.635  0.0000  ...
```

### 2. Generate Missing Embeddings

```bash
# Estimate cost
python embed_normativa.py --estimate

# Generate embeddings
OPENROUTER_API_KEY="sk-or-..." python embed_normativa.py
```

---

*Scripts copied from lexe-max/scripts/ on 2026-02-07*
