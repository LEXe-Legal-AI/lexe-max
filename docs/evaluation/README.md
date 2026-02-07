# LEXE KB Normativa - Evaluation Package

> **ORCHIDEA Framework Support Documentation**
> Knowledge Base for Italian Legal Codes - Hybrid Retrieval System

---

## Overview

This folder contains evaluation evidence and documentation for the **LEXE KB Normativa** system, a hybrid retrieval architecture for Italian legal codes supporting the ORCHIDEA pipeline.

**Last Updated:** 2026-02-07 18:30 UTC

---

## Index

| Document | Description |
|----------|-------------|
| [TIMELINE.md](./TIMELINE.md) | Complete development timeline with milestones |
| [HYBRID-SEARCH-EVALUATION.md](./HYBRID-SEARCH-EVALUATION.md) | Hybrid search (Dense+Sparse+RRF) evaluation report |
| [ARCHITECTURE.md](./ARCHITECTURE.md) | System architecture overview |

### Scripts

| Script | Purpose |
|--------|---------|
| [hybrid_search.py](./scripts/hybrid_search.py) | Hybrid search implementation |
| [hybrid_test.py](./scripts/hybrid_test.py) | Test script with source indicator |
| [embed_normativa.py](./scripts/embed_normativa.py) | Embedding generation script |

### Evidence

| File | Description |
|------|-------------|
| [test_results_2026-02-07.txt](./evidence/test_results_2026-02-07.txt) | Hybrid search test outputs |

---

## Quick Stats

| Metric | Value |
|--------|-------|
| **Total Articles** | 6,335 |
| **Total Chunks** | 10,246 |
| **Embeddings Coverage** | 100% |
| **FTS Coverage** | 100% |
| **Codes Indexed** | CC, CP, CPC, CPP, COST |

---

## Related Documentation

| Document | Location |
|----------|----------|
| KB Schema Overview | [../SCHEMA_KB_OVERVIEW.md](../SCHEMA_KB_OVERVIEW.md) |
| KB Massimari Handoff | [../KB-HANDOFF.md](../KB-HANDOFF.md) |
| KB Normativa Ingestion | [../KB-NORMATIVA-INGESTION.md](../KB-NORMATIVA-INGESTION.md) |
| Norm Graph | [../KB-NORM-GRAPH-HANDOFF.md](../KB-NORM-GRAPH-HANDOFF.md) |

---

## ORCHIDEA Integration

This KB serves as the **Legal Knowledge Layer** in the ORCHIDEA architecture:

```
┌─────────────────────────────────────────────────────────────────┐
│                      ORCHIDEA PIPELINE                          │
├─────────────────────────────────────────────────────────────────┤
│  User Query                                                      │
│      │                                                           │
│      ▼                                                           │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              LEXE KB NORMATIVA (this system)              │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐       │   │
│  │  │   Dense     │  │   Sparse    │  │    RRF      │       │   │
│  │  │  (pgvector) │  │  (tsvector) │  │   Fusion    │       │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘       │   │
│  └──────────────────────────────────────────────────────────┘   │
│      │                                                           │
│      ▼                                                           │
│  Retrieved Legal Context → LLM Response                          │
└─────────────────────────────────────────────────────────────────┘
```

---

## Authors

- LEXE Development Team
- Generated: 2026-02-07

---

*This documentation supports ORCHIDEA papers and evaluation.*
