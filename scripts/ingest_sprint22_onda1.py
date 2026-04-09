#!/usr/bin/env python3
"""
ingest_sprint22_onda1.py — Orchestratore Onda 1: KB Fixes completo

Esegue in sequenza:
  Phase 1: Ingest articoli -bis da fetch_bis_articles.py output
  Phase 2: Ingest EUR-Lex 5 atti (DMA, DORA, DSA, NIS2, TFUE)
  Phase 3: Ingest art. 13 Costituzione
  Phase 4: Genera embeddings per tutti i nuovi chunks
  Phase 5: Aggiorna norm graph (articoli referenziati nelle massime)
  Phase 6: Verifica finale (count articles, chunks, embeddings)

Targeting:
  - STAGING: ssh tunnel a 91.99.229.111:5436
  - PROD:    ssh tunnel a 49.12.85.92:5436
  - LOCAL:   docker exec lexe-max (porta 5436)

Usage:
    # Dry-run locale (nessuna modifica DB)
    python ingest_sprint22_onda1.py --env local --dry-run

    # Staging reale (richiede tunnel SSH o exec nel container)
    python ingest_sprint22_onda1.py --env staging

    # Prod (richiede tunnel SSH)
    python ingest_sprint22_onda1.py --env prod

    # Solo una fase specifica
    python ingest_sprint22_onda1.py --env staging --phase bis
    python ingest_sprint22_onda1.py --env staging --phase eurlex
    python ingest_sprint22_onda1.py --env staging --phase embed
    python ingest_sprint22_onda1.py --env staging --phase verify

Requisiti:
    pip install asyncpg httpx beautifulsoup4
    OPENROUTER_API_KEY=sk-or-... (per embeddings)
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import re
import sys
import time
import uuid
from datetime import date, datetime, timezone
from html import unescape
from pathlib import Path
from typing import Any, Optional

try:
    import asyncpg
except ImportError:
    print("ERROR: asyncpg required. pip install asyncpg")
    sys.exit(1)

try:
    import httpx
except ImportError:
    httpx = None

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

log = logging.getLogger("onda1")

ENV_CONFIG = {
    "local": {
        "dsn": "postgresql://lexe_kb:lexe_kb_secret@localhost:5436/lexe_kb",
        "label": "LOCAL (docker)",
    },
    "staging": {
        "dsn": "postgresql://lexe_kb:lexe_kb_secret@localhost:5436/lexe_kb",
        "label": "STAGING (tunnel 91.99.229.111:5436 -> localhost:5436)",
    },
    "prod": {
        "dsn": "postgresql://lexe_kb:lexe_kb_secret@localhost:5436/lexe_kb",
        "label": "PRODUCTION (tunnel 49.12.85.92:5436 -> localhost:5436)",
    },
}

EMBED_MODEL = "openai/text-embedding-3-small"
EMBED_DIMS = 1536
EMBED_CHANNEL = "testo"
EMBED_BATCH_SIZE = 50
OPENROUTER_URL = "https://openrouter.ai/api/v1/embeddings"

# Chunk config (match ingest_opendata_codici.py)
CHUNK_MAX_CHARS = 2000
CHUNK_OVERLAP_CHARS = 200
MIN_CHUNK_LENGTH = 30

# EUR-Lex acts to re-ingest
EURLEX_TARGETS = {
    "dma":  {"celex": "32022R1925", "title": "DMA (Reg. UE 2022/1925)", "code": "DMA"},
    "dora": {"celex": "32022R2554", "title": "DORA (Reg. UE 2022/2554)", "code": "DORA"},
    "dsa":  {"celex": "32022R2065", "title": "DSA (Reg. UE 2022/2065)", "code": "DSA"},
    "nis2": {"celex": "32022L2555", "title": "NIS2 (Dir. UE 2022/2555)", "code": "NIS2"},
    "tfue": {"celex": "12012E/TXT", "title": "TFUE", "code": "TFUE"},
}

# Suffix ordering (match ingest_opendata_codici.py)
_SUFFIX_ORDER = {
    None: "00",
    "bis": "01", "ter": "02", "quater": "03", "quinquies": "04",
    "sexies": "05", "septies": "06", "octies": "07",
    "novies": "08", "nonies": "08",
    "decies": "09", "undecies": "10", "duodecies": "11",
    "terdecies": "12", "quaterdecies": "13", "quinquiesdecies": "14",
    "sexiesdecies": "15", "septiesdecies": "16", "octiesdecies": "17",
}

# BIS code mapping (fetch_bis codice -> kb.work code, title)
BIS_CODE_MAP = {
    "CCII":       ("CCII",  "Codice della crisi d'impresa e dell'insolvenza"),
    "TUSL":       ("TUSL",  "Testo Unico Sicurezza Lavoro"),
    "L212":       ("L212",  "Statuto del Contribuente"),
    "L241":       ("L241",  "Procedimento Amministrativo"),
    "CDS_STRADA": ("CDS",   "Codice della Strada"),
    "CONSUMO":    ("CCONS", "Codice del Consumo"),
    "PRIVACY":    ("CPRIV", "Codice Privacy"),
    "AMBIENTE":   ("CAMB",  "Codice dell'Ambiente"),
    "CAD":        ("CAD",   "Codice Amministrazione Digitale"),
    "COST":       ("COST",  "Costituzione della Repubblica Italiana"),
}

BIS_URN_BASE = {
    "CCII":       "urn:nir:stato:decreto.legislativo:2019-01-12;14",
    "TUSL":       "urn:nir:stato:decreto.legislativo:2008-04-09;81",
    "L212":       "urn:nir:stato:legge:2000-07-27;212",
    "L241":       "urn:nir:stato:legge:1990-08-07;241",
    "CDS_STRADA": "urn:nir:stato:decreto.legislativo:1992-04-30;285",
    "CONSUMO":    "urn:nir:stato:decreto.legislativo:2005-09-06;206",
    "PRIVACY":    "urn:nir:stato:decreto.legislativo:2003-06-30;196",
    "AMBIENTE":   "urn:nir:stato:decreto.legislativo:2006-04-03;152",
    "CAD":        "urn:nir:stato:decreto.legislativo:2005-03-07;82",
    "COST":       "urn:nir:stato:costituzione:1947-12-27",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_sort_key(art_num: int, suffix: str | None) -> str:
    suffix_code = _SUFFIX_ORDER.get(suffix, "00")
    return f"{art_num:06d}.{suffix_code}"


def chunk_text(text: str) -> list[dict]:
    if not text or len(text.strip()) < MIN_CHUNK_LENGTH:
        return []
    text = text.strip()
    if len(text) <= CHUNK_MAX_CHARS:
        return [{"idx": 0, "text": text, "start": 0, "end": len(text)}]

    chunks = []
    start = 0
    idx = 0
    while start < len(text):
        end = min(start + CHUNK_MAX_CHARS, len(text))
        if end < len(text):
            para = text.rfind("\n\n", start + CHUNK_MAX_CHARS // 2, end)
            if para > start:
                end = para + 2
            else:
                sent = text.rfind(". ", start + CHUNK_MAX_CHARS // 2, end)
                if sent > start:
                    end = sent + 2
        chunk_str = text[start:end].strip()
        if chunk_str and len(chunk_str) >= MIN_CHUNK_LENGTH:
            chunks.append({"idx": idx, "text": chunk_str, "start": start, "end": end})
            idx += 1
        start = end - CHUNK_OVERLAP_CHARS if end < len(text) else len(text)
    return chunks


def compute_hash(testo: str, rubrica: str | None) -> str:
    parts = [testo or "", rubrica or ""]
    return hashlib.sha256("|".join(parts).encode()).hexdigest()


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

_work_cache: dict[str, uuid.UUID] = {}


async def ensure_work(conn: asyncpg.Connection, code: str, title: str) -> uuid.UUID:
    if code in _work_cache:
        return _work_cache[code]
    row = await conn.fetchval("SELECT id FROM kb.work WHERE code = $1", code)
    if row:
        _work_cache[code] = row
        return row
    work_id = uuid.uuid5(uuid.NAMESPACE_URL, f"onda1:work:{code}")
    await conn.execute("""
        INSERT INTO kb.work (id, code, title, title_short, created_at)
        VALUES ($1, $2, $3, $4, NOW())
        ON CONFLICT (code) DO UPDATE SET title = EXCLUDED.title
        RETURNING id
    """, work_id, code, title[:200], code)
    _work_cache[code] = work_id
    log.info("  Created work entry: %s (%s)", code, title[:60])
    return work_id


async def upsert_article(
    conn: asyncpg.Connection, work_id: uuid.UUID, code: str,
    art_str: str, art_num: int, suffix: str | None,
    testo: str, rubrica: str | None, urn_nir: str,
    urn_base: str, source: str,
) -> uuid.UUID | None:
    sort_key = make_sort_key(art_num, suffix)
    identity = "SUFFIX" if suffix else "BASE"
    content_hash = compute_hash(testo, rubrica)
    art_id = uuid.uuid5(uuid.NAMESPACE_URL, f"onda1:{code}:{art_str}")

    return await conn.fetchval("""
        INSERT INTO kb.normativa (
            id, work_id, codice, articolo, articolo_num, articolo_suffix,
            articolo_sort_key, identity_class, quality, lifecycle,
            urn_nir, urn_base, rubrica, testo,
            canonical_source, content_hash, is_current,
            created_at, updated_at
        ) VALUES (
            $1, $2, $3, $4, $5, $6,
            $7, $8::article_identity_class, 'VALID_STRONG'::article_quality_class,
            'CURRENT'::lifecycle_status,
            $9, $10, $11, $12,
            $13, $14, TRUE,
            NOW(), NOW()
        )
        ON CONFLICT (work_id, articolo_sort_key) DO UPDATE SET
            testo = EXCLUDED.testo,
            rubrica = EXCLUDED.rubrica,
            urn_nir = EXCLUDED.urn_nir,
            canonical_source = EXCLUDED.canonical_source,
            content_hash = EXCLUDED.content_hash,
            updated_at = NOW()
        RETURNING id
    """,
        art_id, work_id, code, art_str, art_num, suffix,
        sort_key, identity,
        urn_nir, urn_base, rubrica, testo,
        source, content_hash,
    )


async def upsert_chunk(
    conn: asyncpg.Connection, normativa_id: uuid.UUID, work_id: uuid.UUID,
    art_num: int, suffix: str | None, chunk: dict, code: str,
) -> uuid.UUID | None:
    sort_key = make_sort_key(art_num, suffix)
    chunk_id = uuid.uuid5(uuid.NAMESPACE_URL, f"onda1:chunk:{code}:{normativa_id}:{chunk['idx']}")
    token_est = max(1, len(chunk["text"]) // 4)

    return await conn.fetchval("""
        INSERT INTO kb.normativa_chunk (
            id, normativa_id, work_id,
            articolo_sort_key, articolo_num, articolo_suffix,
            chunk_no, char_start, char_end, text, token_est,
            created_at
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, NOW())
        ON CONFLICT (normativa_id, chunk_no) DO UPDATE SET
            text = EXCLUDED.text,
            char_start = EXCLUDED.char_start,
            char_end = EXCLUDED.char_end,
            token_est = EXCLUDED.token_est
        RETURNING id
    """,
        chunk_id, normativa_id, work_id,
        sort_key, art_num, suffix,
        chunk["idx"], chunk["start"], chunk["end"], chunk["text"], token_est,
    )


async def insert_embedding(
    conn: asyncpg.Connection, chunk_id: uuid.UUID,
    embedding: list[float], model: str,
):
    emb_id = uuid.uuid5(uuid.NAMESPACE_URL, f"onda1:emb:{chunk_id}")
    emb_str = "[" + ",".join(str(v) for v in embedding) + "]"
    await conn.execute("""
        INSERT INTO kb.normativa_chunk_embeddings (
            id, chunk_id, model, channel, dims, embedding, created_at
        ) VALUES ($1, $2, $3, $4, $5, $6::vector, NOW())
        ON CONFLICT (chunk_id, model, channel, dims) DO UPDATE SET
            embedding = EXCLUDED.embedding
    """, emb_id, chunk_id, model, EMBED_CHANNEL, EMBED_DIMS, emb_str)


# ---------------------------------------------------------------------------
# Embedding client
# ---------------------------------------------------------------------------

async def embed_batch(client: httpx.AsyncClient, texts: list[str], api_key: str) -> list[list[float]]:
    resp = await client.post(
        OPENROUTER_URL,
        headers={"Authorization": f"Bearer {api_key}"},
        json={"model": EMBED_MODEL, "input": texts},
        timeout=120.0,
    )
    resp.raise_for_status()
    data = resp.json()
    items = sorted(data["data"], key=lambda x: x["index"])
    return [item["embedding"] for item in items]


# ═══════════════════════════════════════════════════════════════════════
# PHASE 1: Ingest -bis articles
# ═══════════════════════════════════════════════════════════════════════

async def phase_bis(
    conn: asyncpg.Connection, bis_dir: Path, dry_run: bool,
) -> dict[str, int]:
    log.info("")
    log.info("=" * 70)
    log.info("PHASE 1: Ingest -bis articles from %s", bis_dir)
    log.info("=" * 70)

    json_files = sorted(f for f in bis_dir.glob("*.json") if not f.name.startswith("_"))
    log.info("  Found %d JSON files", len(json_files))

    stats = {"articles": 0, "chunks": 0, "skipped": 0}

    for f in json_files:
        data = json.loads(f.read_text(encoding="utf-8"))
        codice = data.get("codice", "UNKNOWN")

        if codice not in BIS_CODE_MAP:
            log.warning("  Skip unknown codice: %s (%s)", codice, f.name)
            stats["skipped"] += 1
            continue

        kb_code, work_title = BIS_CODE_MAP[codice]
        urn_base = BIS_URN_BASE.get(codice, "")

        art_display = data.get("articolo", "")
        testo = data.get("testo_plain", "").strip()
        rubrica = (data.get("rubrica") or "").strip() or None
        urn_nir = data.get("urn", "")

        if not testo or len(testo) < 10:
            stats["skipped"] += 1
            continue

        # Parse article
        parts = art_display.split("-", 1)
        try:
            art_num = int(parts[0])
        except (ValueError, IndexError):
            art_num = 0
        suffix = parts[1].lower() if len(parts) > 1 and parts[1] else None
        art_str = f"{art_num}{suffix}" if suffix else str(art_num)

        sort_key = make_sort_key(art_num, suffix)
        log.info("  [%s] art.%-12s %s  %d chars  %s",
                 kb_code, art_display, sort_key, len(testo),
                 rubrica[:40] if rubrica else "-")

        if dry_run:
            stats["articles"] += 1
            continue

        work_id = await ensure_work(conn, kb_code, work_title)
        norm_id = await upsert_article(
            conn, work_id, kb_code, art_str, art_num, suffix,
            testo, rubrica, urn_nir, urn_base, "opendata_api_urn",
        )
        if not norm_id:
            continue
        stats["articles"] += 1

        # Build enriched text for embedding
        embed_prefix = f"{work_title} | Art. {art_display}"
        if rubrica:
            embed_prefix += f" — {rubrica}"
        embed_text = f"{embed_prefix} | {testo}"

        chunks = chunk_text(embed_text)
        for ch in chunks:
            await upsert_chunk(conn, norm_id, work_id, art_num, suffix, ch, kb_code)
            stats["chunks"] += 1

    log.info("  Phase 1 done: %d articles, %d chunks, %d skipped",
             stats["articles"], stats["chunks"], stats["skipped"])
    return stats


# ═══════════════════════════════════════════════════════════════════════
# PHASE 2: Ingest EUR-Lex acts
# ═══════════════════════════════════════════════════════════════════════

async def fetch_eurlex_html(celex: str) -> str | None:
    """Fetch EUR-Lex HTML. Falls back to local file if WAF blocks."""
    local_dir = Path(__file__).parent.parent / "data" / "eurlex"
    safe_name = celex.replace("/", "_").replace("\\", "_")
    # Try exact match first, then glob
    candidates = []
    if local_dir.exists():
        exact = local_dir / f"{safe_name}.html"
        if exact.exists():
            candidates = [exact]
        else:
            candidates = list(local_dir.glob(f"*{safe_name}*.html"))
    if candidates:
        log.info("  Loading %s from local file: %s", celex, candidates[0].name)
        return candidates[0].read_text(encoding="utf-8")

    url = f"https://eur-lex.europa.eu/legal-content/IT/TXT/HTML/?uri=CELEX:{celex}"
    log.info("  Fetching %s from EUR-Lex...", celex)

    async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
        try:
            resp = await client.get(url, headers={
                "Accept-Language": "it",
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
            })
            if resp.status_code == 200:
                # Save locally for future runs
                local_dir.mkdir(parents=True, exist_ok=True)
                (local_dir / f"{celex}.html").write_text(resp.text, encoding="utf-8")
                return resp.text
            log.warning("  HTTP %d for %s — try downloading manually", resp.status_code, celex)
            return None
        except Exception as e:
            log.error("  Fetch failed for %s: %s — try downloading manually", celex, e)
            return None


def extract_eurlex_articles(html: str, celex: str) -> list[dict]:
    """Parse EUR-Lex HTML to extract articles (sibling-walking)."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    articles = []

    art_headers = soup.find_all("p", class_="oj-ti-art")
    if not art_headers:
        art_headers = soup.find_all("div", class_="eli-subdivision")

    if not art_headers:
        # Regex fallback
        text = soup.get_text(separator="\n")
        for m in re.finditer(r"Articolo\s+(\d+[a-z]*)\s*\n(.*?)(?=Articolo\s+\d|$)",
                             text, re.DOTALL | re.IGNORECASE):
            art_text = m.group(2).strip()[:50000]
            if len(art_text) > 20:
                articles.append({"article": m.group(1), "title": "", "text": art_text})
        return articles

    for header in art_headers:
        header_text = header.get_text(strip=True)
        m = re.search(r"(\d+[a-z]*)", header_text)
        if not m:
            art_id = header.get("id", "")
            m = re.search(r"(\d+[a-z]*)", art_id)
            if not m:
                continue

        art_num = m.group(1)
        title = ""
        next_el = header.find_next_sibling()
        if next_el and next_el.name == "p" and "oj-sti-art" in next_el.get("class", []):
            title = next_el.get_text(strip=True)
            next_el = next_el.find_next_sibling()

        text_parts = []
        element = next_el
        while element:
            if element.name == "p" and "oj-ti-art" in element.get("class", []):
                break
            if element.name == "div" and "eli-subdivision" in element.get("class", []):
                break
            if element.name in ("p", "div"):
                t = element.get_text(strip=True)
                if t:
                    text_parts.append(t)
            elif element.name == "table":
                for row in element.find_all("tr"):
                    cells = row.find_all("td")
                    row_text = " ".join(c.get_text(strip=True) for c in cells)
                    if row_text:
                        text_parts.append(row_text)
            elif element.name == "li":
                t = element.get_text(strip=True)
                if t:
                    text_parts.append(t)
            element = element.find_next_sibling()

        art_text = "\n".join(text_parts)[:50000]
        if len(art_text) > 20:
            articles.append({"article": art_num, "title": title, "text": art_text})

    return articles


async def phase_eurlex(
    conn: asyncpg.Connection, dry_run: bool,
) -> dict[str, int]:
    log.info("")
    log.info("=" * 70)
    log.info("PHASE 2: Ingest EUR-Lex acts (%d targets)", len(EURLEX_TARGETS))
    log.info("=" * 70)

    stats = {"articles": 0, "chunks": 0, "acts_ok": 0, "acts_fail": 0}

    for key, info in EURLEX_TARGETS.items():
        celex = info["celex"]
        code = info["code"]
        title = info["title"]
        log.info("")
        log.info("  --- %s (%s) CELEX: %s ---", key, title, celex)

        html = await fetch_eurlex_html(celex)
        if not html:
            log.warning("  SKIP %s — no HTML available. Download manually to data/eurlex/%s.html",
                        key, celex)
            stats["acts_fail"] += 1
            continue

        articles = extract_eurlex_articles(html, celex)
        log.info("  Parsed %d articles from %s", len(articles), celex)

        if not articles:
            stats["acts_fail"] += 1
            continue

        if not dry_run:
            work_id = await ensure_work(conn, code, title)

        for art in articles:
            try:
                art_num = int(re.match(r"(\d+)", art["article"]).group(1))
            except (AttributeError, ValueError):
                art_num = 0
            # EU articles don't have Latin suffixes
            suffix = None
            art_str = str(art_num)
            urn_nir = f"celex:{celex}#art{art['article']}"
            testo = art["text"]
            rubrica = art.get("title") or None

            log.info("    art.%-5s  %d chars  %s",
                     art["article"], len(testo),
                     rubrica[:40] if rubrica else "-")

            if dry_run:
                stats["articles"] += 1
                continue

            norm_id = await upsert_article(
                conn, work_id, code, art_str, art_num, suffix,
                testo, rubrica, urn_nir, f"celex:{celex}", "eurlex_online",
            )
            if not norm_id:
                continue
            stats["articles"] += 1

            embed_prefix = f"{title} | Art. {art['article']}"
            if rubrica:
                embed_prefix += f" — {rubrica}"
            embed_text = f"{embed_prefix} | {testo}"

            chunks = chunk_text(embed_text)
            for ch in chunks:
                await upsert_chunk(conn, norm_id, work_id, art_num, suffix, ch, code)
                stats["chunks"] += 1

        stats["acts_ok"] += 1
        # Save JSON
        out_dir = Path(__file__).parent.parent / "data" / "eurlex"
        out_dir.mkdir(parents=True, exist_ok=True)
        safe_celex = celex.replace("/", "_").replace("\\", "_")
        with open(out_dir / f"{key}_{safe_celex}.json", "w", encoding="utf-8") as f:
            json.dump({"key": key, "celex": celex, "articles_count": len(articles),
                       "articles": articles}, f, ensure_ascii=False, indent=2)

    log.info("")
    log.info("  Phase 2 done: %d articles, %d chunks, %d/%d acts OK",
             stats["articles"], stats["chunks"],
             stats["acts_ok"], stats["acts_ok"] + stats["acts_fail"])
    return stats


# ═══════════════════════════════════════════════════════════════════════
# PHASE 3: Generate embeddings for new chunks
# ═══════════════════════════════════════════════════════════════════════

async def phase_embed(
    conn: asyncpg.Connection, api_key: str, dry_run: bool,
) -> dict[str, int]:
    log.info("")
    log.info("=" * 70)
    log.info("PHASE 3: Generate embeddings for chunks missing them")
    log.info("=" * 70)

    # Count missing embeddings
    missing = await conn.fetchval("""
        SELECT count(*)
        FROM kb.normativa_chunk c
        WHERE NOT EXISTS (
            SELECT 1 FROM kb.normativa_chunk_embeddings e
            WHERE e.chunk_id = c.id
            AND e.model = $1
            AND e.channel = $2
            AND e.dims = $3
        )
    """, EMBED_MODEL, EMBED_CHANNEL, EMBED_DIMS)

    log.info("  Missing embeddings: %d chunks", missing)

    if missing == 0:
        log.info("  Nothing to embed!")
        return {"embedded": 0, "batches": 0}

    if dry_run:
        est_tokens = missing * 500  # rough estimate
        est_cost = est_tokens / 1000 * 0.00002
        log.info("  [DRY RUN] Would embed %d chunks (~%d tokens, ~$%.4f)",
                 missing, est_tokens, est_cost)
        return {"embedded": 0, "batches": 0}

    if not api_key:
        log.error("  OPENROUTER_API_KEY not set — skipping embeddings")
        return {"embedded": 0, "batches": 0}

    stats = {"embedded": 0, "batches": 0}

    async with httpx.AsyncClient() as client:
        offset = 0
        while True:
            rows = await conn.fetch("""
                SELECT c.id, c.text
                FROM kb.normativa_chunk c
                WHERE NOT EXISTS (
                    SELECT 1 FROM kb.normativa_chunk_embeddings e
                    WHERE e.chunk_id = c.id
                    AND e.model = $1
                    AND e.channel = $2
                    AND e.dims = $3
                )
                ORDER BY c.created_at, c.id
                LIMIT $4
            """, EMBED_MODEL, EMBED_CHANNEL, EMBED_DIMS, EMBED_BATCH_SIZE)

            if not rows:
                break

            texts = [r["text"] for r in rows]
            chunk_ids = [r["id"] for r in rows]

            try:
                embeddings = await embed_batch(client, texts, api_key)
            except Exception as e:
                log.error("  Embedding API error: %s", e)
                break

            for chunk_id, emb in zip(chunk_ids, embeddings):
                await insert_embedding(conn, chunk_id, emb, EMBED_MODEL)

            stats["embedded"] += len(rows)
            stats["batches"] += 1
            log.info("  Batch %d: embedded %d chunks (total: %d/%d)",
                     stats["batches"], len(rows), stats["embedded"], missing)

    log.info("  Phase 3 done: %d embeddings in %d batches",
             stats["embedded"], stats["batches"])
    return stats


# ═══════════════════════════════════════════════════════════════════════
# PHASE 4: Verification
# ═══════════════════════════════════════════════════════════════════════

async def phase_verify(conn: asyncpg.Connection) -> dict[str, Any]:
    log.info("")
    log.info("=" * 70)
    log.info("PHASE 4: Verification")
    log.info("=" * 70)

    # Overall stats
    total_art = await conn.fetchval("SELECT count(*) FROM kb.normativa")
    total_chunks = await conn.fetchval("SELECT count(*) FROM kb.normativa_chunk")
    total_embs = await conn.fetchval("SELECT count(*) FROM kb.normativa_chunk_embeddings")
    missing_embs = await conn.fetchval("""
        SELECT count(*) FROM kb.normativa_chunk c
        WHERE NOT EXISTS (
            SELECT 1 FROM kb.normativa_chunk_embeddings e
            WHERE e.chunk_id = c.id AND e.dims = $1
        )
    """, EMBED_DIMS)

    log.info("  Total articles:   %d", total_art)
    log.info("  Total chunks:     %d", total_chunks)
    log.info("  Total embeddings: %d", total_embs)
    log.info("  Missing embeddings: %d", missing_embs)

    # Per-code stats for Onda 1 targets
    log.info("")
    log.info("  Per-code stats (Onda 1 targets):")
    target_codes = ["CCII", "TUSL", "L212", "L241", "CDS", "CCONS", "CPRIV",
                    "CAMB", "CAD", "COST", "DMA", "DORA", "DSA", "NIS2", "TFUE"]
    for code in target_codes:
        row = await conn.fetchrow("""
            SELECT
                w.code,
                count(DISTINCT n.id) as articles,
                count(DISTINCT n.id) FILTER (WHERE n.articolo_suffix IS NOT NULL) as bis_articles,
                count(DISTINCT c.id) as chunks,
                count(DISTINCT e.id) as embeddings
            FROM kb.work w
            LEFT JOIN kb.normativa n ON n.work_id = w.id
            LEFT JOIN kb.normativa_chunk c ON c.work_id = w.id
            LEFT JOIN kb.normativa_chunk_embeddings e ON e.chunk_id = c.id
            WHERE w.code = $1
            GROUP BY w.code
        """, code)
        if row:
            log.info("    %-8s  %4d art (%3d -bis)  %5d chunks  %5d embs",
                     row["code"], row["articles"], row["bis_articles"],
                     row["chunks"], row["embeddings"])
        else:
            log.info("    %-8s  NOT IN DB", code)

    return {
        "total_articles": total_art,
        "total_chunks": total_chunks,
        "total_embeddings": total_embs,
        "missing_embeddings": missing_embs,
    }


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

async def main():
    parser = argparse.ArgumentParser(
        description="Sprint 22 Onda 1 — KB Fixes orchestrator"
    )
    parser.add_argument(
        "--env", choices=["local", "staging", "prod"], default="local",
        help="Target environment (default: local)",
    )
    parser.add_argument(
        "--dsn", default=None,
        help="Override DB connection string",
    )
    parser.add_argument(
        "--bis-dir", default="data/bis_articles",
        help="Directory with -bis article JSON files (default: data/bis_articles)",
    )
    parser.add_argument(
        "--phase", choices=["bis", "eurlex", "embed", "verify", "all"],
        default="all",
        help="Run only a specific phase (default: all)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Parse only, no DB writes or API calls",
    )
    parser.add_argument(
        "--skip-embed", action="store_true",
        help="Skip embedding generation phase",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    env = ENV_CONFIG[args.env]
    dsn = args.dsn or env["dsn"]
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    bis_dir = Path(args.bis_dir)

    log.info("=" * 70)
    log.info("SPRINT 22 ONDA 1 — KB FIXES ORCHESTRATOR")
    log.info("=" * 70)
    log.info("  Environment: %s", env["label"])
    log.info("  Phase:       %s", args.phase)
    log.info("  Dry run:     %s", args.dry_run)
    log.info("  Bis dir:     %s (%d files)", bis_dir,
             len(list(bis_dir.glob("*.json"))) if bis_dir.exists() else 0)
    log.info("  Embed:       %s", "skip" if args.skip_embed else EMBED_MODEL)
    log.info("  API key:     %s", "set" if api_key else "NOT SET")
    log.info("=" * 70)

    if args.env == "prod" and not args.dry_run:
        log.warning("")
        log.warning("  *** PRODUCTION MODE — press Enter to confirm, Ctrl+C to abort ***")
        input()

    conn = None
    try:
        if not args.dry_run:
            log.info("Connecting to DB...")
            conn = await asyncpg.connect(dsn)
            log.info("Connected.")
        else:
            log.info("[DRY RUN] Skipping DB connection")

        all_stats = {}
        t0 = time.monotonic()

        # Phase 1: Bis articles
        if args.phase in ("bis", "all"):
            if bis_dir.exists():
                all_stats["bis"] = await phase_bis(conn, bis_dir, args.dry_run)
            else:
                log.warning("Bis dir not found: %s — skipping Phase 1", bis_dir)

        # Phase 2: EUR-Lex
        if args.phase in ("eurlex", "all"):
            if args.dry_run:
                # Dry-run EUR-Lex still tries to fetch HTML
                all_stats["eurlex"] = await phase_eurlex(conn, args.dry_run)
            elif conn:
                all_stats["eurlex"] = await phase_eurlex(conn, args.dry_run)

        # Phase 3: Embeddings
        if args.phase in ("embed", "all") and not args.skip_embed:
            if conn and not args.dry_run:
                all_stats["embed"] = await phase_embed(conn, api_key, args.dry_run)
            elif args.dry_run and conn:
                all_stats["embed"] = await phase_embed(conn, api_key, args.dry_run)
            else:
                log.info("  Skipping embedding phase (no connection or dry-run)")

        # Phase 4: Verification
        if args.phase in ("verify", "all"):
            if conn and not args.dry_run:
                all_stats["verify"] = await phase_verify(conn)

        elapsed = time.monotonic() - t0

        # Final summary
        log.info("")
        log.info("=" * 70)
        log.info("ONDA 1 COMPLETE in %.1f seconds", elapsed)
        log.info("=" * 70)
        for phase_name, stats in all_stats.items():
            log.info("  %s: %s", phase_name, json.dumps(stats, default=str))

    finally:
        if conn:
            await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
