# Mappa LEXE: lexe-max

> Generato automaticamente - 2026-02-01
> Repo path: C:\PROJECTS\lexe-genesis\lexe-max
> **Ambiente di riferimento: stage**
> Altri ambienti: prod (stabile attuale), local (sviluppo)
> Generato con: APIMaster + 1 Explore subagent

---

## 1. Architettura

### Stack Tecnologico

| Aspetto     | Valore                                                                 |
| ----------- | ---------------------------------------------------------------------- |
| Framework   | FastAPI 0.109.0+                                                       |
| Linguaggio  | Python 3.12                                                            |
| Database    | PostgreSQL 17 (main :5433) + PostgreSQL 17 KB (:5434 + pgvector 0.7.4) |
| Cache       | Valkey (Redis-compatible) :6380                                        |
| Porta       | 8020                                                                   |
| ASGI Server | Uvicorn 0.27.0+                                                        |

### Struttura Principale

```
lexe-max/
├── src/lexe_api/
│   ├── main.py                  (143 LL) [FastAPI entry]
│   ├── config.py                 (73 LL) [Pydantic Settings]
│   ├── database.py              (310 LL) [AsyncPG pool]
│   ├── cache.py                 (167 LL) [Valkey client]
│   ├── api/
│   │   ├── health.py             (98 LL) [Health endpoints]
│   │   └── tools.py             (202 LL) [Legal tools routes]
│   ├── tools/
│   │   ├── base.py              (202 LL) [BaseLegalTool + circuit breaker]
│   │   ├── normattiva.py        (215 LL) [Normattiva.it scraper]
│   │   ├── eurlex.py            (199 LL) [EUR-Lex SPARQL + scraper]
│   │   ├── infolex.py           (270 LL) [Brocardi.it scraper]
│   │   └── health_monitor.py    (200 LL) [Tool health + alerting]
│   ├── scrapers/
│   │   ├── http_client.py       (145 LL) [Throttled HTTP + SPARQL]
│   │   └── selectors.py         (146 LL) [CSS selectors]
│   ├── models/
│   │   └── schemas.py           (282 LL) [Pydantic models]
│   └── kb/                               [Knowledge Base]
│       ├── config.py            (158 LL)
│       ├── models.py            (385 LL)
│       ├── graph/                        [Citation/Norm graphs]
│       ├── ingestion/                    [Pipeline ingestion]
│       └── retrieval/                    [Hybrid search]
├── migrations/kb/                         [DB schema + pgvector]
├── docker-compose.kb.yml                  [KB infrastructure]
├── Dockerfile                             [Multi-stage build]
├── Dockerfile.kb                          [KB PostgreSQL 17]
└── pyproject.toml                         [Python 3.12 deps]
```

### File Critici

| File                    | Funzione                        | Righe | Priorita Analisi |
| ----------------------- | ------------------------------- | ----- | ---------------- |
| main.py                 | FastAPI app, lifespan, routers  | 143   | Alta             |
| database.py             | AsyncPG pool, CRUD              | 310   | Alta             |
| api/tools.py            | Legal tools endpoints           | 202   | Alta             |
| tools/base.py           | BaseLegalTool + circuit breaker | 202   | Alta             |
| tools/normattiva.py     | Normattiva URN + scraping       | 215   | Alta             |
| tools/eurlex.py         | SPARQL + CELEX builder          | 199   | Alta             |
| tools/infolex.py        | Brocardi scraper (massime)      | 270   | Alta             |
| kb/models.py            | KB domain models                | 385   | Media            |
| scrapers/http_client.py | Throttled HTTP + retry          | 145   | Media            |
| models/schemas.py       | Pydantic request/response       | 282   | Media            |

**Totale Core Code**: 2,609 righe

---

## 2. Riferimenti LEO (Legacy)

### Critici (COMPLETATI)

| File                        | Riga | Contenuto                                                          | Azione                                |
| --------------------------- | ---- | ------------------------------------------------------------------ | ------------------------------------- |
| [x] tools/health_monitor.py | 144  | `# TODO: Implement email notification via lexe-core email service` | ✅ Rinominato "leo-core" → "lexe-core" |

### Condivisi (OK - mantenere)

- [x] `docker-compose.kb.yml` linee 56-59: `lexe-platform-network` (commentato, opzionale)
- [x] Network `shared_public` per Traefik (condiviso)

### Statistiche

- [x] File con riferimenti LEO: 2 → **0 (tutti rinominati)**
- [x] Riferimenti critici: 1 → **0 (completati)**
- [x] Riferimenti condivisi: 1 → **0 (rinominati a lexe-*)**

**Nota Storica**: ⚠️ Tutti i riferimenti LEO sono stati completamente rinominati a LEXE (2026-02-01).

- `leo-core` → `lexe-core`
- `leo-platform-network` → `lexe-platform-network`

**Verdict**: LEXE-MAX è **completamente indipendente da LEO**. Nessun import diretto, nessuna dipendenza runtime. Tutti i riferimenti legacy sono stati migrati.

---

## 3. Interazioni Servizi

### Questo Servizio E Chiamato Da:

| Chiamante         | Endpoint        | Metodo | Auth | Scopo                                    |
| ----------------- | --------------- | ------ | ---- | ---------------------------------------- |
| lexe-orchestrator | /api/v1/tools/* | POST   | No   | Legal research durante pipeline ORCHIDEA |
| lexe-core         | /health/*       | GET    | No   | Health monitoring                        |

### Questo Servizio Chiama:

| Servizio Target      | Base URL                       | Endpoint                        | Metodo   | Scopo                |
| -------------------- | ------------------------------ | ------------------------------- | -------- | -------------------- |
| Normattiva.it        | https://www.normattiva.it      | /uri-res/N2Ls                   | GET      | Leggi italiane       |
| EUR-Lex (SPARQL)     | https://publications.europa.eu | /webapi/rdf/sparql              | GET/POST | Metadata EU          |
| EUR-Lex (HTML)       | https://eur-lex.europa.eu      | /legal-content/{lang}/TXT/HTML/ | GET      | Testo atti EU        |
| Brocardi.it          | https://www.brocardi.it        | /codice-*/art*.html             | GET      | Commentari + massime |
| OpenAI (via LiteLLM) | Configurable                   | /embeddings                     | POST     | Embedding massime    |

### Diagramma Flusso

```
                    ┌─────────────────┐
                    │ lexe-orchestrator│
                    └────────┬────────┘
                             │ POST /api/v1/tools/*
                    ┌────────▼────────┐
                    │    lexe-max     │
                    │     :8020       │
                    └────────┬────────┘
          ┌──────────────────┼──────────────────┐
          │                  │                  │
    ┌─────▼─────┐     ┌─────▼─────┐     ┌─────▼─────┐
    │Normattiva │     │  EUR-Lex  │     │ Brocardi  │
    │   .it     │     │ .europa.eu│     │   .it     │
    └───────────┘     └───────────┘     └───────────┘
```

---

## 4. API

### API-INGRESSO (Endpoint Esposti)

#### Health & Status

| Endpoint                        | Metodo | Auth | Response           | Note                      |
| ------------------------------- | ------ | ---- | ------------------ | ------------------------- |
| /health/live                    | GET    | No   | 200 OK             | Liveness probe            |
| /health/ready                   | GET    | No   | 200 OK             | Readiness (DB+cache)      |
| /health/status                  | GET    | No   | Full health JSON   | Version + tools state     |
| /health/tools                   | GET    | No   | All tools health   | Circuit breaker state     |
| /health/tools/{tool_name}       | GET    | No   | Single tool health | normattiva/eurlex/infolex |
| /health/tools/{tool_name}/reset | POST   | No*  | Admin reset        | Reset to healthy          |

#### Normattiva Endpoints

| Endpoint                         | Metodo | Auth | Request Body      | Response           |
| -------------------------------- | ------ | ---- | ----------------- | ------------------ |
| /api/v1/tools/normattiva/search  | POST   | No   | NormattivaRequest | NormattivaResponse |
| /api/v1/tools/normattiva/vigenza | POST   | No   | NormattivaRequest | VigenzaResponse    |

**NormattivaRequest**:

```json
{
    "act_type": "legge",           // Required: legge, decreto legislativo, codice civile...
    "date": "1990-08-07",          // Optional: YYYY-MM-DD or YYYY
    "act_number": "241",           // Optional
    "article": "1",                // Optional
    "version": "vigente"           // Default: vigente | originale
}
```

**NormattivaResponse**:

```json
{
    "success": true,
    "urn": "urn:nir:stato:legge:1990-08-07;241",
    "title": "Legge sulla trasparenza",
    "text": "L'amministrazione pubblica...",
    "vigente": true,
    "abrogato_da": null,
    "source": "normattiva",
    "cached": false
}
```

#### EUR-Lex Endpoints

| Endpoint                    | Metodo | Auth | Request Body  | Response       |
| --------------------------- | ------ | ---- | ------------- | -------------- |
| /api/v1/tools/eurlex/search | POST   | No   | EurLexRequest | EurLexResponse |

**EurLexRequest**:

```json
{
    "act_type": "regolamento",     // Required: regolamento, direttiva, decisione
    "year": 2016,                  // Required
    "number": 679,                 // Required
    "article": "1",                // Optional
    "language": "ita"              // Default: ita | eng | fra | deu | spa
}
```

**EurLexResponse**:

```json
{
    "success": true,
    "celex": "32016R0679",
    "eli": "http://data.europa.eu/eli/regulation/2016/679/oj",
    "title": "GDPR - Regolamento sulla protezione dei dati",
    "text": "...",
    "in_force": true,
    "source": "eurlex",
    "language": "ita"
}
```

#### InfoLex (Brocardi) Endpoints

| Endpoint                     | Metodo | Auth | Request Body   | Response        |
| ---------------------------- | ------ | ---- | -------------- | --------------- |
| /api/v1/tools/infolex/search | POST   | No   | InfoLexRequest | InfoLexResponse |

**InfoLexRequest**:

```json
{
    "act_type": "codice civile",   // Required
    "article": "2043",             // Required
    "include_massime": true,       // Default: true
    "include_relazioni": false,
    "include_footnotes": false
}
```

**InfoLexResponse**:

```json
{
    "success": true,
    "article_title": "Risarcimento per fatto illecito",
    "article_text": "Qualunque fatto doloso o colposo...",
    "massime": [
        {
            "autorita": "Cass. civ.",
            "numero": "1234/2023",
            "data": "2023-01-15",
            "testo": "...",
            "materia": "Diritto civile"
        }
    ],
    "spiegazione": "Commento Brocardi...",
    "brocardi_url": "https://www.brocardi.it/codice-civile/art2043.html",
    "source": "brocardi"
}
```

#### Tools Status

| Endpoint             | Metodo | Auth | Response               |
| -------------------- | ------ | ---- | ---------------------- |
| /api/v1/tools/status | GET    | No   | Status di tutti i tool |

### API-USCITA (Chiamate Esterne)

| Servizio Target | Base URL                       | Endpoints Usati                 | Frequenza  |
| --------------- | ------------------------------ | ------------------------------- | ---------- |
| Normattiva.it   | https://www.normattiva.it      | /uri-res/N2Ls?urn:...           | Alta       |
| EUR-Lex SPARQL  | https://publications.europa.eu | /webapi/rdf/sparql              | Media      |
| EUR-Lex HTML    | https://eur-lex.europa.eu      | /legal-content/{lang}/TXT/HTML/ | Media      |
| Brocardi.it     | https://www.brocardi.it        | /codice-*/art*.html             | Alta       |
| OpenAI/LiteLLM  | Configurable                   | /embeddings                     | Batch (KB) |

---

## 5. Configurazione

### Environment Variables Richieste

| Variabile                  | Default | Obbligatoria | Descrizione                  |
| -------------------------- | ------- | ------------ | ---------------------------- |
| LEXE_DATABASE_URL          | -       | Si           | PostgreSQL connection string |
| LEXE_REDIS_URL             | -       | Si           | Valkey/Redis URL             |
| LEXE_KB_DATABASE_URL       | -       | No           | KB PostgreSQL (se abilitato) |
| LEXE_LOG_LEVEL             | INFO    | No           | DEBUG/INFO/WARNING/ERROR     |
| LEXE_HTTP_TIMEOUT_SECONDS  | 30      | No           | HTTP timeout                 |
| LEXE_HTTP_MAX_RETRIES      | 3       | No           | Retry attempts               |
| LEXE_RATE_LIMIT_NORMATTIVA | 30      | No           | Req/min Normattiva           |
| LEXE_RATE_LIMIT_EURLEX     | 60      | No           | Req/min EUR-Lex              |
| LEXE_RATE_LIMIT_BROCARDI   | 30      | No           | Req/min Brocardi             |

### Feature Flags

| Flag                       | Default | Ambiente | Scopo                  |
| -------------------------- | ------- | -------- | ---------------------- |
| LEXE_FF_NORMATTIVA_ENABLED | true    | all      | Enable Normattiva tool |
| LEXE_FF_EURLEX_ENABLED     | true    | all      | Enable EUR-Lex tool    |
| LEXE_FF_INFOLEX_ENABLED    | true    | all      | Enable Brocardi tool   |

---

## 6. Note e Problemi

### Da Fixare (P1 - Urgente)

- (nessuno)

### Da Fixare (P2 - Importante)

- [ ] Rimuovere riferimento "leo-core" in health_monitor.py linea 144

### Suggerimenti Miglioramento

- Configurare alerting webhook (Slack/Discord) per circuit breaker
- Abilitare KB retrieval endpoints (attualmente solo backend)

---

## 7. Environments

| Ambiente | Base URL                     | Health Check | Note            |
| -------- | ---------------------------- | ------------ | --------------- |
| local    | http://localhost:8020        | /health/live | Dev             |
| stage    | https://tools-stage.lexe.pro | /health/live | **Riferimento** |
| prod     | - (via lexe-orchestrator)    | /health/live | Stabile         |

> **stage** = ambiente canonico per documentazione e test contrattuali
> **prod** = prima stabile, NON riferimento per nuovi contratti

---

## 8. Knowledge Base (KB) - Status

### Metriche Attuali (2026-02-01)

| Metrica              | Valore       |
| -------------------- | ------------ |
| Active Massime       | 38,718       |
| Embeddings           | 41,437 total |
| Citation Graph Edges | 58,737       |
| Norm Graph Edges     | 42,338       |
| Unique Norms         | 4,128        |
| Norm Coverage        | 60.3%        |
| Recall@10            | 97.5%        |
| MRR                  | 0.756        |

### Retrieval Architecture

```
Query → Router → Citation Query? → Direct Lookup
              → Norm Query? → Norm Lookup
              → Semantic → Hybrid Search (Dense + Sparse + Trigram + RRF)
                              → Norm Boost → Top-10
```

---

## 9. Circuit Breaker Pattern

```
CLOSED (OK) ──[5 failures]──> OPEN (Failing)
    ↑                              │
    │                        [5 min timeout]
    └──────[Success]─────── HALF_OPEN
```

- Failure threshold: 5
- Retry timeout: 5 minutes
- Alert cooldown: 1 hour

---

*Generato: 2026-02-01 | Terminale 4 | APIMaster + Explore*
