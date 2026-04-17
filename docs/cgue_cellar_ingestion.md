# CGUE EUR-Lex CELLAR — Nightly Ingestion Spec

> Sprint 30 P1.1 — CGUE sentences 2020-2026 (GDPR / biometria / profilazione /
> trasferimenti extra-UE). Estimated ~3-5K judgments.

Ref: `C:/Users/Fra/.claude/plans/analizza-le-inferenze-produzione-indexed-wren.md`
Sections `P1 — Ingestion dedicati CGUE + Garante + EDPB` and
`P1.1 — CGUE via EUR-Lex CELLAR nightly`.

Status: **SKELETON** (stub only). Not wired to CELLAR yet.

---

## 1. Sources

### 1.1 Primary — EUR-Lex CELLAR SPARQL endpoint

- **Endpoint**: `http://publications.europa.eu/webapi/rdf/sparql`
- **Dataset**: EU Publications Office linked-data store (CELLAR).
- **Auth**: None (public endpoint).
- **Format**: SPARQL 1.1, JSON results (`Accept: application/sparql-results+json`).
- **Usage**: discovery query → returns list of CELEX identifiers published or
  updated in the last N hours, filtered by CELEX pattern and EUROVOC concepts.

### 1.2 Secondary — CELLAR REST (document fetch)

- **Base**: `https://publications.europa.eu/resource/cellar/{uuid}`
- **Alt**: `https://eur-lex.europa.eu/legal-content/IT/TXT/HTML/?uri=CELEX:{celex}`
- **Purpose**: given a CELEX, fetch the Italian HTML (+ optional XML manifest).
- **Content negotiation**: `Accept-Language: it`, `Accept: text/html`.
- **Parsing**: HTML via `lxml` / `beautifulsoup4` tree navigation.
  **No regex** — structured selectors only.

---

## 2. CELEX Filter Pattern

CELEX (Communauté Européenne Lex) identifiers encode:

```
  6 1989 CJ 0262
  │ │    │  │
  │ │    │  └── doc number within series
  │ │    └───── series code (CJ = Court of Justice judgment)
  │ └────────── year
  └──────────── sector (6 = Case-law)
```

**Prefix filter**: `6*CJ*`

- `6` → sector 6 (case-law).
- `CJ` → series "Judgment of the Court of Justice".
- Related series intentionally **excluded** from Sprint 30 P1.1:
  - `CO` (Orders), `CC` (Opinions of the AG), `CB` (Opinions), `TJ` (General Court).
  - These may be added in P1.2 if coverage insufficient.

**Year range**: 2020-2026 (publication year segment of CELEX).

**Topic filter** (applied in SPARQL query via EUROVOC concepts or subject-matter codes):

| Topic | EUROVOC / subject | Notes |
|-------|-------------------|-------|
| GDPR / data protection | `http://eurovoc.europa.eu/4708` (protection of privacy) | primary |
| Biometria | keyword match in title/keywords (narrow) | complementary |
| Profilazione / automated decisions | `http://eurovoc.europa.eu/5428` (information processing) | complementary |
| Transfers extra-UE | subject-matter code re: international data transfers | GDPR ch. V |

At ingest time a wider net is cast (all `6*CJ*` in date range) and topic filter
is applied post-fetch against metadata + keywords, so the SPARQL query stays
simple and resilient to EUROVOC reshuffles.

---

## 3. Target Schema

Migration: `migrations/kb/088_cgue_cellar.sql`.

### 3.1 `kb.cgue_sentences`

| Column | Type | Notes |
|--------|------|-------|
| `celex` | `text PRIMARY KEY` | natural PK, e.g. `62022CJ0300` |
| `ecli` | `text` | e.g. `ECLI:EU:C:2023:537` |
| `data` | `date` | judgment date (delivery) |
| `parti` | `text` | party line (e.g. "Meta Platforms Ireland v Bundeskartellamt") |
| `testo_integrale` | `text` | full Italian text, HTML-stripped |
| `massime` | `text` | operative part / summary if available |
| `lingua` | `text` default `'it'` | language of `testo_integrale` |
| `source_url` | `text` | CELLAR or EUR-Lex URL |
| `metadata` | `jsonb` default `{}` | eurovoc, keywords, subject-matter, reporter |
| `tsv_italian` | `tsvector` | GIN index, Italian FTS |
| `ingested_at` | `timestamptz` default `NOW()` | first ingest time |
| `updated_at` | `timestamptz` default `NOW()` | last upsert time |

### 3.2 `kb.cgue_citations` (schema only — P1.5 populates)

Parallel edge table, shape-compatible with `kb.sentenze_cc_edges`. Holds
CITES / OVERRULES / FOLLOWS / DISTINGUISHES edges from CGUE → {CGUE, norm,
ECLI ext.}. **Not populated by the nightly sync**; populated by the P1.5
citation graph builder (separate job).

---

## 4. Freshness SLA

| Metric | Target | Method |
|--------|--------|--------|
| Discovery latency | ≤ 48h from CELLAR publication | nightly job at 02:30 UTC |
| Ingest success rate | ≥ 95% over 7-day window | monitor `ingested_at` deltas |
| Backfill completeness | 100% of `6*CJ*` 2020-2026 within 30d of first run | see §5 |
| Retry budget | 3 attempts, exponential backoff 2s→8s→32s | `tenacity` |

### 4.1 Runtime budgets

- **Per-document fetch**: 30s hard timeout (httpx `timeout=30`).
- **SPARQL discovery**: 60s hard timeout.
- **Rate limit**: 2 req/s (0.5s sleep between requests), per netiquette below.
- **Job budget**: 2h total wall clock for nightly delta run. Backfill runs
  in separate multi-day batched job.

---

## 5. Backfill Strategy

1. **Seed run** (T0): SPARQL query with broad window `2020-01-01..<today>`,
   paginated in 3-month windows to avoid SPARQL result caps. Fetch CELEX list
   only (no bodies).
2. **Store in `kb.cgue_sentences` with NULL body fields** (celex + ecli + data
   + source_url populated from SPARQL metadata).
3. **Body hydration job**: iterates rows WHERE `testo_integrale IS NULL`,
   fetches HTML, parses, upserts. Respects rate limit. Target: ~3-5K rows
   at 2 req/s = ~25-42 min of fetch (plus parse overhead).
4. **Daily delta** (steady state): SPARQL query with `?date >= NOW() - 48h`,
   upsert each hit (metadata + body). Idempotent on `celex`.

---

## 6. Scheduling

Two options evaluated; **recommendation: Temporal** (same as Normattiva
nightly sync, prod-proven).

### 6.1 Temporal workflow (recommended)

- New workflow: `CgueNightlySync` in `lexe_api/temporal/workflows/`.
- Activities:
  - `cgue_discover_celex_activity` — SPARQL discovery, returns CELEX list.
  - `cgue_fetch_and_upsert_activity` — fetch + parse + upsert single CELEX.
  - `cgue_backfill_seed_activity` — one-shot seed run.
- Schedule: cron `30 02 * * *` UTC (staggered from Normattiva at `00 03`).
- Idempotency via Temporal workflow_id = `cgue-nightly-{YYYYMMDD}`.

### 6.2 Cron fallback

If Temporal is unavailable (dev / smoke), a plain cron hitting
`python -m lexe_api.ingestion.cgue_cellar --mode=daily` is acceptable.
The skeleton exposes `run_daily_sync()` as a module-level coroutine for
exactly this purpose.

---

## 7. Netiquette

Publications Office / EUR-Lex has no hard-published rate limit for CELLAR REST,
but community consensus and `robots.txt` indicate:

- **Rate**: 2 req/s maximum (0.5s min between requests).
- **User-Agent**: `LEXe-Legal-AI/1.0 contact@lexe.pro` (MUST include contact).
- **Retry**: `tenacity` with exponential backoff on 429/5xx: 2s → 8s → 32s,
  max 3 attempts. Give up on 4xx (except 429) and log.
- **Caching**: respect ETag / Last-Modified if provided; bypass cache only
  on explicit refresh.
- **Off-peak preference**: schedule nightly job at 02:30 UTC (low EU traffic).

---

## 8. Module Skeleton

**File**: `src/lexe_api/ingestion/cgue_cellar.py`

Public entry points:

- `async def run_daily_sync(*, db_dsn: str | None = None, since_hours: int = 48) -> SyncReport`
  — nightly delta: SPARQL discovery → fetch → upsert. Returns counts.
- `async def run_backfill_seed(*, db_dsn: str | None = None, start: date, end: date) -> SyncReport`
  — one-shot seed, stores metadata rows without bodies.
- `async def hydrate_pending_bodies(*, db_dsn: str | None = None, batch: int = 500) -> SyncReport`
  — iterates rows with `testo_integrale IS NULL`, fetches + upserts.

Internals (all async, all dependency-injected for testability):

- `CgueCellarClient(httpx.AsyncClient, rate_limiter)` — wraps SPARQL + REST.
- `RateLimiter(rate_per_sec=2.0)` — token-bucket async gate.
- `parse_cellar_html(html: str) -> CgueDocument` — lxml tree nav (no regex).
- `upsert_sentence(conn: asyncpg.Connection, doc: CgueDocument) -> str` —
  `INSERT ... ON CONFLICT (celex) DO UPDATE` with `updated_at = NOW()`.

---

## 9. Testing

Unit tests at `tests/unit/test_cgue_cellar_ingestion.py`:

1. CELEX filter query built correctly (SPARQL string contains `6*CJ*` pattern
   and correct date bindings).
2. Malformed response (truncated HTML, missing fields) handled gracefully —
   returns `CgueDocument` with nullable fields, logs warning, does not crash.
3. Upsert is idempotent — two consecutive calls with the same CELEX produce
   one row; `ingested_at` stays the same, `updated_at` advances.
4. Rate limiter respects 2 req/s budget — 5 sequential calls take ≥ 2.0s
   (measured via monotonic clock in test).

---

## 10. Out of Scope (Sprint 30 P1.1)

- Embedding generation (handled by existing `embed_eu_batch.py` pattern,
  Sprint 30 P1.4 or later).
- Citation graph extraction (P1.5).
- EUROVOC concept mapping refinement (best-effort in P1.1, refined in P1.6).
- Multi-language mirroring (IT only for Sprint 30).
- Admin dashboard wiring (P2.3).

---

## 11. Open Questions / TODOs

- [ ] Finalize EUROVOC concept URIs with legal-domain expert (P1.6).
- [ ] Decide whether to ingest `CO` (orders) in P1.2 based on coverage gap analysis.
- [ ] Evaluate ECLI de-duplication across EUR-Lex vs. curia.europa.eu
      (same judgment, two sources).
- [ ] Prometheus metrics: expose `cgue_ingest_duration_seconds`,
      `cgue_ingest_rows_total`, `cgue_fetch_failures_total`.
- [ ] Prod deploy gate: confirm CELLAR endpoint reachable from staging
      (no IP-ban history like Normattiva).

---

*Created: 2026-04-17 — Sprint 30 P1.1 skeleton*
