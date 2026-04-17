"""CGUE EUR-Lex CELLAR ingestion — Sprint 30 P1.1 skeleton.

Nightly sync of Court of Justice judgments (CELEX prefix `6*CJ*`) from the
EUR-Lex CELLAR SPARQL endpoint + REST, targeting `kb.cgue_sentences`.

NOTE (Sprint 30 P1.1): this is a **skeleton stub**. Public entry points are
callable and tested, but they are NOT wired to the real CELLAR endpoint yet.
Actual wiring lands in P1.2 after the migration is applied on staging.

Design:
- HTML/XML parsing uses lxml tree navigation. **No regex**.
- HTTP via httpx.AsyncClient + tenacity for retries with exponential backoff.
- Rate limit: 2 req/s, enforced by an async token-bucket `RateLimiter`.
- User-Agent: "LEXe-Legal-AI/1.0 contact@lexe.pro" (netiquette).
- DB access via asyncpg. Upserts idempotent on `celex`.
- Public surface: `run_daily_sync`, `run_backfill_seed`, `hydrate_pending_bodies`.

Spec: docs/cgue_cellar_ingestion.md
Migration: migrations/kb/088_cgue_cellar.sql
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any
from urllib.parse import quote, urlencode

import httpx
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SPARQL_ENDPOINT = "http://publications.europa.eu/webapi/rdf/sparql"
CELLAR_RESOURCE_BASE = "https://publications.europa.eu/resource/cellar"
EURLEX_HTML_BASE = "https://eur-lex.europa.eu/legal-content/IT/TXT/HTML/"

USER_AGENT = "LEXe-Legal-AI/1.0 contact@lexe.pro"

# CELEX prefix used by SPARQL FILTER: sector 6 (case-law), series CJ (Court of Justice).
# See docs/cgue_cellar_ingestion.md §2 for pattern decomposition.
CELEX_PREFIX_PATTERN = "6*CJ*"

DEFAULT_RATE_PER_SEC = 2.0
DEFAULT_FETCH_TIMEOUT_S = 30.0
DEFAULT_SPARQL_TIMEOUT_S = 60.0
DEFAULT_RETRY_ATTEMPTS = 3

RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CgueDocument:
    """Parsed CGUE document ready for upsert into kb.cgue_sentences."""

    celex: str
    ecli: str | None = None
    data: date | None = None
    parti: str | None = None
    testo_integrale: str | None = None
    massime: str | None = None
    lingua: str = "it"
    source_url: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SyncReport:
    """Counters emitted by the public entry points."""

    discovered: int = 0
    fetched: int = 0
    upserted: int = 0
    skipped: int = 0
    errors: int = 0
    elapsed_s: float = 0.0
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# Rate limiter (async token bucket, simple interval enforcement)
# ---------------------------------------------------------------------------


class RateLimiter:
    """Minimal async rate limiter — enforces a minimum interval between acquires.

    For the low rates required by CELLAR netiquette (2 req/s), a simple
    inter-call interval gate is sufficient and deterministic (easier to test
    than a probabilistic token bucket).
    """

    def __init__(self, rate_per_sec: float = DEFAULT_RATE_PER_SEC) -> None:
        if rate_per_sec <= 0:
            raise ValueError("rate_per_sec must be > 0")
        self._min_interval = 1.0 / rate_per_sec
        self._lock = asyncio.Lock()
        self._last: float = 0.0

    async def acquire(self) -> None:
        async with self._lock:
            now = time.monotonic()
            wait = self._min_interval - (now - self._last)
            if wait > 0:
                await asyncio.sleep(wait)
            self._last = time.monotonic()


# ---------------------------------------------------------------------------
# SPARQL query builder
# ---------------------------------------------------------------------------


def build_discovery_sparql(
    *,
    since: datetime,
    until: datetime | None = None,
    celex_prefix: str = CELEX_PREFIX_PATTERN,
    limit: int = 1000,
) -> str:
    """Build the SPARQL query that discovers CGUE CELEX identifiers.

    The query filters by CELEX pattern (sector 6 + series CJ) and by the
    date range. Binds are inlined (SPARQL has no prepared statements) —
    values come from a trusted source (datetime arithmetic on the caller
    side), NOT from user input.
    """
    if until is None:
        until = datetime.now(timezone.utc)

    since_iso = since.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    until_iso = until.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # Convert the glob-style "6*CJ*" into a regex anchor. CELLAR's SPARQL endpoint
    # supports REGEX() on string literals.
    regex = _celex_glob_to_regex(celex_prefix)

    query = f"""
PREFIX cdm: <http://publications.europa.eu/ontology/cdm#>
PREFIX dcterms: <http://purl.org/dc/terms/>

SELECT DISTINCT ?celex ?ecli ?date ?title WHERE {{
  ?work cdm:resource_legal_id_celex ?celex ;
        cdm:work_date_document ?date .
  OPTIONAL {{ ?work cdm:case-law_ecli ?ecli }}
  OPTIONAL {{ ?work dcterms:title ?title . FILTER(LANG(?title) = "it") }}
  FILTER(REGEX(STR(?celex), "{regex}"))
  FILTER(?date >= "{since_iso}"^^xsd:dateTime)
  FILTER(?date <= "{until_iso}"^^xsd:dateTime)
}}
ORDER BY DESC(?date)
LIMIT {int(limit)}
""".strip()
    return query


def _celex_glob_to_regex(glob: str) -> str:
    """Convert a `6*CJ*`-style glob to a SPARQL-safe regex anchor.

    Only `*` is supported as wildcard. No regex chars are allowed in `glob`
    beyond that (CELEX alphabet is [0-9A-Z]).
    """
    allowed = set("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ*")
    bad = [c for c in glob if c not in allowed]
    if bad:
        raise ValueError(f"celex glob contains disallowed chars: {bad!r}")
    # Replace * with .* and anchor to start.
    return "^" + glob.replace("*", ".*")


# ---------------------------------------------------------------------------
# HTTP client wrapper
# ---------------------------------------------------------------------------


class CgueCellarClient:
    """Thin async wrapper around httpx + tenacity + rate limiter."""

    def __init__(
        self,
        *,
        http: httpx.AsyncClient | None = None,
        rate_limiter: RateLimiter | None = None,
        retry_attempts: int = DEFAULT_RETRY_ATTEMPTS,
    ) -> None:
        self._http = http or httpx.AsyncClient(
            headers={"User-Agent": USER_AGENT, "Accept-Language": "it"},
            timeout=DEFAULT_FETCH_TIMEOUT_S,
        )
        self._owns_http = http is None
        self._rate = rate_limiter or RateLimiter(DEFAULT_RATE_PER_SEC)
        self._retry_attempts = retry_attempts

    async def close(self) -> None:
        if self._owns_http:
            await self._http.aclose()

    async def __aenter__(self) -> CgueCellarClient:
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    # ------------------------------------------------------------------
    # SPARQL discovery
    # ------------------------------------------------------------------

    async def discover(
        self,
        *,
        since: datetime,
        until: datetime | None = None,
        celex_prefix: str = CELEX_PREFIX_PATTERN,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        """Run the SPARQL discovery query and return a list of result bindings."""
        query = build_discovery_sparql(
            since=since, until=until, celex_prefix=celex_prefix, limit=limit,
        )
        params = urlencode({"query": query, "format": "application/sparql-results+json"})
        url = f"{SPARQL_ENDPOINT}?{params}"

        resp = await self._request("GET", url, timeout=DEFAULT_SPARQL_TIMEOUT_S)
        data = resp.json()
        bindings = data.get("results", {}).get("bindings", [])

        out: list[dict[str, Any]] = []
        for b in bindings:
            row = {k: v.get("value") for k, v in b.items() if isinstance(v, dict)}
            if "celex" in row:
                out.append(row)
        return out

    # ------------------------------------------------------------------
    # Document fetch
    # ------------------------------------------------------------------

    async def fetch_html(self, celex: str) -> str:
        """Fetch the Italian HTML body of a document by CELEX."""
        url = f"{EURLEX_HTML_BASE}?{urlencode({'uri': f'CELEX:{celex}'})}"
        resp = await self._request("GET", url)
        return resp.text

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(self._retry_attempts),
            wait=wait_exponential(multiplier=2, min=2, max=32),
            retry=retry_if_exception_type((httpx.HTTPError, RetryableStatusError)),
            reraise=True,
        ):
            with attempt:
                await self._rate.acquire()
                resp = await self._http.request(method, url, **kwargs)
                if resp.status_code in RETRYABLE_STATUS:
                    raise RetryableStatusError(
                        f"retryable status {resp.status_code} for {url}",
                    )
                resp.raise_for_status()
                return resp
        # Unreachable — AsyncRetrying raises on exhaustion.
        raise RuntimeError("AsyncRetrying exited without result")


class RetryableStatusError(Exception):
    """Raised to signal tenacity that the HTTP status code is retryable."""


# ---------------------------------------------------------------------------
# HTML parser (lxml / beautifulsoup4 — NO regex)
# ---------------------------------------------------------------------------


def parse_cellar_html(html: str, *, celex: str, source_url: str | None = None) -> CgueDocument:
    """Parse an EUR-Lex HTML page into a CgueDocument.

    Uses BeautifulSoup4 with lxml backend. Missing fields yield None; the
    function NEVER raises on malformed input (logs a warning, returns
    best-effort partial doc). Downstream upsert tolerates NULLs.
    """
    # Import lazily to keep module import cheap for tests that stub parse.
    try:
        from bs4 import BeautifulSoup  # type: ignore[import-not-found]
    except ImportError as e:
        raise ImportError(
            "beautifulsoup4 is required for parse_cellar_html; install via "
            "`uv sync` (lexe-max pyproject declares bs4).",
        ) from e

    soup = BeautifulSoup(html or "", "lxml")

    ecli: str | None = None
    data_val: date | None = None
    parti: str | None = None

    # ECLI is exposed in a <meta name="ECLI"> or as part of a metadata table.
    meta_ecli = soup.find("meta", attrs={"name": "ECLI"})
    if meta_ecli and meta_ecli.get("content"):
        ecli = meta_ecli["content"].strip()

    # Judgment date: <meta name="WKF-CASE-DATE"> or a <time datetime="..."> tag.
    meta_date = soup.find("meta", attrs={"name": "WKF-CASE-DATE"})
    if meta_date and meta_date.get("content"):
        data_val = _safe_parse_date(meta_date["content"])
    if data_val is None:
        time_tag = soup.find("time")
        if time_tag and time_tag.get("datetime"):
            data_val = _safe_parse_date(time_tag["datetime"])

    # Party line: the first <p class="sum-title-1"> or an <h1> inside .judgment-header.
    parti_tag = soup.find("p", class_="sum-title-1") or soup.find(
        "div", class_="judgment-parties",
    )
    if parti_tag:
        parti = parti_tag.get_text(strip=True) or None

    # Full text: the main body container. EUR-Lex uses #text or .eli-container.
    body_tag = (
        soup.find(id="text")
        or soup.find("div", class_="eli-container")
        or soup.find("div", class_="WordSection1")
    )
    testo = body_tag.get_text("\n", strip=True) if body_tag else None

    # Operative part / massime: EUR-Lex labels it "Dispositivo" in IT.
    massime: str | None = None
    disp_tag = soup.find(id="dispositivo") or soup.find(
        "div", class_="operative-part",
    )
    if disp_tag:
        massime = disp_tag.get_text("\n", strip=True) or None

    return CgueDocument(
        celex=celex,
        ecli=ecli,
        data=data_val,
        parti=parti,
        testo_integrale=testo,
        massime=massime,
        lingua="it",
        source_url=source_url,
        metadata={},
    )


def _safe_parse_date(value: str) -> date | None:
    """Parse ISO-8601 dates without regex. Returns None on failure."""
    if not value:
        return None
    value = value.strip()
    # Try plain date first, then datetime.
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%SZ"):
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            continue
    # Last resort: fromisoformat (handles tz offsets).
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).date()
    except ValueError:
        logger.warning("could not parse date: %r", value)
        return None


# ---------------------------------------------------------------------------
# Database upsert (asyncpg)
# ---------------------------------------------------------------------------


UPSERT_SQL = """
INSERT INTO kb.cgue_sentences (
    celex, ecli, data, parti, testo_integrale, massime,
    lingua, source_url, metadata, updated_at
) VALUES (
    $1, $2, $3, $4, $5, $6, $7, $8, $9::jsonb, NOW()
)
ON CONFLICT (celex) DO UPDATE SET
    ecli            = COALESCE(EXCLUDED.ecli, kb.cgue_sentences.ecli),
    data            = COALESCE(EXCLUDED.data, kb.cgue_sentences.data),
    parti           = COALESCE(EXCLUDED.parti, kb.cgue_sentences.parti),
    testo_integrale = COALESCE(EXCLUDED.testo_integrale, kb.cgue_sentences.testo_integrale),
    massime         = COALESCE(EXCLUDED.massime, kb.cgue_sentences.massime),
    lingua          = EXCLUDED.lingua,
    source_url      = COALESCE(EXCLUDED.source_url, kb.cgue_sentences.source_url),
    metadata        = kb.cgue_sentences.metadata || EXCLUDED.metadata,
    updated_at      = NOW()
RETURNING celex, (xmax = 0) AS inserted
"""


async def upsert_sentence(conn: Any, doc: CgueDocument) -> bool:
    """Upsert a CgueDocument. Returns True if a new row was inserted, False if updated.

    `conn` is an `asyncpg.Connection`; typed as Any to avoid a hard import
    in module-load path (tests use AsyncMock).
    """
    import json

    row = await conn.fetchrow(
        UPSERT_SQL,
        doc.celex,
        doc.ecli,
        doc.data,
        doc.parti,
        doc.testo_integrale,
        doc.massime,
        doc.lingua,
        doc.source_url,
        json.dumps(doc.metadata or {}),
    )
    if row is None:
        return False
    return bool(row.get("inserted") if isinstance(row, dict) else row["inserted"])


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def _resolve_dsn(db_dsn: str | None) -> str:
    dsn = db_dsn or os.getenv("LEXE_KB_DSN") or os.getenv("LEXE_DATABASE_URL")
    if not dsn:
        raise RuntimeError(
            "No DB DSN: pass db_dsn= or set LEXE_KB_DSN / LEXE_DATABASE_URL",
        )
    return dsn


async def run_daily_sync(
    *,
    db_dsn: str | None = None,
    since_hours: int = 48,
    client: CgueCellarClient | None = None,
    conn: Any = None,
) -> SyncReport:
    """Nightly delta sync: SPARQL discovery → fetch → upsert.

    Sprint 30 P1.1 STUB: executes the full flow if `client` and `conn` are
    provided (for tests); otherwise is intentionally inert and returns an
    empty SyncReport with a warning log. Real wiring lands in P1.2.
    """
    report = SyncReport()
    t0 = time.monotonic()

    if client is None or conn is None:
        logger.warning(
            "run_daily_sync called without injected client/conn — stub returning "
            "empty report. Sprint 30 P1.1 skeleton does NOT hit CELLAR yet. "
            "dsn-resolved=%s",
            _resolve_dsn(db_dsn) if (db_dsn or os.getenv("LEXE_KB_DSN")
                                     or os.getenv("LEXE_DATABASE_URL")) else "<none>",
        )
        report.elapsed_s = time.monotonic() - t0
        return report

    since = datetime.now(timezone.utc) - timedelta(hours=since_hours)
    try:
        bindings = await client.discover(since=since)
    except Exception:
        logger.exception("cgue discovery failed")
        report.errors += 1
        report.elapsed_s = time.monotonic() - t0
        return report

    report.discovered = len(bindings)

    for b in bindings:
        celex = b.get("celex")
        if not celex:
            report.skipped += 1
            continue
        try:
            html = await client.fetch_html(celex)
            report.fetched += 1
        except Exception:
            logger.exception("cgue fetch failed celex=%s", celex)
            report.errors += 1
            continue

        doc = parse_cellar_html(
            html,
            celex=celex,
            source_url=f"{EURLEX_HTML_BASE}?uri=CELEX:{quote(celex)}",
        )
        try:
            await upsert_sentence(conn, doc)
            report.upserted += 1
        except Exception:
            logger.exception("cgue upsert failed celex=%s", celex)
            report.errors += 1

    report.elapsed_s = time.monotonic() - t0
    return report


async def run_backfill_seed(
    *,
    start: date,
    end: date,
    db_dsn: str | None = None,
    client: CgueCellarClient | None = None,
    conn: Any = None,
) -> SyncReport:
    """Seed run: SPARQL-only discovery, inserts metadata rows without bodies.

    STUB: Sprint 30 P1.1 provides signature only; real implementation in P1.2.
    """
    _ = (start, end, db_dsn, client, conn)
    report = SyncReport()
    logger.warning("run_backfill_seed is a stub — returns empty report")
    return report


async def hydrate_pending_bodies(
    *,
    db_dsn: str | None = None,
    batch: int = 500,
    client: CgueCellarClient | None = None,
    conn: Any = None,
) -> SyncReport:
    """Hydrate rows where testo_integrale IS NULL.

    STUB: Sprint 30 P1.1 provides signature only; real implementation in P1.2.
    """
    _ = (db_dsn, batch, client, conn)
    report = SyncReport()
    logger.warning("hydrate_pending_bodies is a stub — returns empty report")
    return report


__all__ = [
    "CELEX_PREFIX_PATTERN",
    "CELLAR_RESOURCE_BASE",
    "CgueCellarClient",
    "CgueDocument",
    "DEFAULT_RATE_PER_SEC",
    "EURLEX_HTML_BASE",
    "RateLimiter",
    "RetryableStatusError",
    "SPARQL_ENDPOINT",
    "SyncReport",
    "USER_AGENT",
    "UPSERT_SQL",
    "build_discovery_sparql",
    "hydrate_pending_bodies",
    "parse_cellar_html",
    "run_backfill_seed",
    "run_daily_sync",
    "upsert_sentence",
]
