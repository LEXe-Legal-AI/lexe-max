#!/usr/bin/env python3
"""
Recovery metadata citazionali per kb.massime (Sprint 27 — S3).

Problema: 14,275 massime (30.5%) con anno NULL, 17,064 con numero NULL.
Rompe filtering, ItalGiure resolver e citation graph.

Strategia cascade a due stadi:
  1. REGEX PASS  — zero cost, pattern standard massimari Cassazione.
  2. LLM FALLBACK — solo su residui (dopo regex), Gemini Flash Lite via
     OpenRouter (o LiteLLM proxy) con asyncio.Semaphore(10) per concorrenza.

Target (Gate B Sprint 27):
  anno IS NULL    < 2,000  (da 14,275)
  numero IS NULL  < 4,000  (da 17,064)

Usage:
    cd lexe-max

    # Self-test contro fixture (nessun DB/LLM richiesto)
    uv run python scripts/massima_metadata_recovery.py --self-test

    # Dry-run regex, report, nessun UPDATE
    uv run python scripts/massima_metadata_recovery.py --only-regex --limit 200

    # Regex --execute: aggiorna DB
    uv run python scripts/massima_metadata_recovery.py --only-regex --execute

    # LLM fallback su residui (dopo regex)
    export OPENROUTER_API_KEY=sk-or-...
    uv run python scripts/massima_metadata_recovery.py --only-llm --execute

    # Full cascade: regex + LLM
    uv run python scripts/massima_metadata_recovery.py --execute

Env vars:
    QA_DB_URL             postgresql://lexe_kb:...@host:port/lexe_kb
    OPENROUTER_API_KEY    sk-or-...
    LLM_API_URL           (opzionale) override OpenRouter con LiteLLM proxy
    LLM_API_KEY           (opzionale) key per LiteLLM proxy
    LLM_MODEL             (opzionale) override modello default
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import asyncpg
import httpx

# ──────────────────────────────────────────────────────────────────
# Configurazione
# ──────────────────────────────────────────────────────────────────

DB_URL_DEFAULT = os.getenv(
    "QA_DB_URL",
    "postgresql://lexe_kb:lexe_kb_dev_password@localhost:5434/lexe_kb",
)

LLM_API_URL = os.getenv("LLM_API_URL", "https://openrouter.ai/api/v1/chat/completions")
LLM_API_KEY = os.getenv("LLM_API_KEY") or os.getenv("OPENROUTER_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "google/gemini-2.5-flash-lite-preview-09-2025")

SCRIPT_DIR = Path(__file__).resolve().parent
FIXTURE_SAMPLE = SCRIPT_DIR / "qa" / "sample_100_massime.json"

# ──────────────────────────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────────────────────────


@dataclass
class ExtractedMetadata:
    anno: int | None = None
    numero: str | None = None
    sezione: str | None = None
    rv: str | None = None  # "NNNNNN-NN" preferito, fallback "NNNNNN"
    confidence: float = 0.0
    source: str = "none"  # 'regex' | 'llm' | 'llm_failed' | 'none'

    def filled_count(self) -> int:
        return sum(1 for x in (self.anno, self.numero, self.sezione, self.rv) if x)


# ──────────────────────────────────────────────────────────────────
# REGEX PASS
# ──────────────────────────────────────────────────────────────────

# ANNO (5 pattern)
ANNO_PATTERNS: list[re.Pattern] = [
    # P1: "Sez. X, n. NNNN/YYYY"  (pattern cartiglio principale)
    re.compile(
        r"Sez\.?\s*[A-Z0-9]+(?:[-/][0-9]+)?\s*,?\s*n\.?\s*\d{3,6}\s*/\s*(?P<anno>\d{4})",
        re.IGNORECASE,
    ),
    # P2: "n. NNNN/YYYY"  (senza prefisso Sez.)
    re.compile(r"\bn\.?\s*\d{3,6}\s*/\s*(?P<anno>\d{4})", re.IGNORECASE),
    # P3: "n. NNNN del YYYY"
    re.compile(r"\bn\.?\s*\d{3,6}\s+del\s+(?P<anno>\d{4})", re.IGNORECASE),
    # P4: "NNNNN del YYYY" (narrativo, senza "n.")
    re.compile(r"\b\d{3,6}\s+del\s+(?P<anno>\d{4})\b"),
    # P5: "(ud. DD/MM/YYYY)" o "(dep. DD/MM/YYYY)"
    re.compile(
        r"\((?:ud|dep)\.?\s*\d{1,2}[/.]\d{1,2}[/.](?P<anno>\d{4})\)",
        re.IGNORECASE,
    ),
]

# NUMERO (3 pattern)
NUMERO_PATTERNS: list[re.Pattern] = [
    # N1: "Sez. X, n. NNNN/YYYY"
    re.compile(
        r"Sez\.?\s*[A-Z0-9]+(?:[-/][0-9]+)?\s*,?\s*n\.?\s*(?P<num>\d{3,6})\s*/\s*\d{4}",
        re.IGNORECASE,
    ),
    # N2: "n. NNNN/YYYY"
    re.compile(r"\bn\.?\s*(?P<num>\d{3,6})\s*/\s*\d{4}", re.IGNORECASE),
    # N3: "n. NNNN del YYYY"
    re.compile(r"\bn\.?\s*(?P<num>\d{3,6})\s+del\s+\d{4}", re.IGNORECASE),
]

# SEZIONE (2 pattern)
SEZIONE_PATTERNS: list[re.Pattern] = [
    # S1: "Sez. X[-M], n. ..." — forma abbreviata
    re.compile(
        r"Sez\.?\s*(?P<sez>U|L|[1-7](?:[-/][1-7])?)\s*,?\s*n\.?\s*\d",
        re.IGNORECASE,
    ),
    # S2: "Sezioni Unite" / "Sezione Lavoro" / "Sezione Prima…Settima" — forma estesa
    re.compile(
        r"Sezion[ei]\s+(?P<sez>Unite|Lavoro|Prima|Seconda|Terza|Quarta|Quinta|Sesta|Settima)",
        re.IGNORECASE,
    ),
]

# RV (2 pattern)
RV_PATTERNS: list[re.Pattern] = [
    # R1: "Rv. NNNNNN-NN" (con sub)
    re.compile(
        r"Rv\.?\s*(?P<rv>\d{5,7})\s*[-/]\s*(?P<sub>\d{1,3})",
        re.IGNORECASE,
    ),
    # R2: "Rv. NNNNNN" (senza sub)
    re.compile(r"Rv\.?\s*(?P<rv>\d{5,7})(?!\s*[-/]\s*\d)", re.IGNORECASE),
]

# Normalizzazione sezione: qualsiasi rappresentazione → codice canonico
SEZ_NORMALIZE: dict[str, str] = {
    # Unite
    "unite": "U", "un": "U", "u": "U", "s.u.": "U", "s.s.u.u.": "U", "su": "U",
    # Lavoro
    "lavoro": "L", "l": "L",
    # Numeriche
    "prima": "1", "seconda": "2", "terza": "3", "quarta": "4",
    "quinta": "5", "sesta": "6", "settima": "7",
    "i": "1", "ii": "2", "iii": "3", "iv": "4", "v": "5", "vi": "6", "vii": "7",
    "1": "1", "2": "2", "3": "3", "4": "4", "5": "5", "6": "6", "7": "7",
}


def normalize_sezione(raw: str | None) -> str | None:
    """Normalizza 'Prima' → '1', 'Unite' → 'U', '6-1' → '6-1' (invariato)."""
    if not raw:
        return None
    s = raw.strip().lower()
    if s in SEZ_NORMALIZE:
        return SEZ_NORMALIZE[s]
    # Formati composti tipo "6-1", "6/1"
    m = re.fullmatch(r"([1-7])[-/]([1-7])", s)
    if m:
        return f"{m.group(1)}-{m.group(2)}"
    # Fallback: uppercase trim (caso "L" già matchato, o forme ignote)
    return raw.strip().upper()


def normalize_numero(raw: str | None) -> str | None:
    """Rimuove zero-padding: '04008' → '4008'."""
    if not raw:
        return None
    stripped = raw.lstrip("0")
    return stripped or raw  # se era "000" → "0"


def regex_extract(testo: str) -> ExtractedMetadata:
    """Cascade regex: anno, numero, sezione, rv. Primo match vince per ogni campo."""
    meta = ExtractedMetadata(source="regex")
    if not testo:
        return meta

    # ANNO
    for p in ANNO_PATTERNS:
        m = p.search(testo)
        if m:
            try:
                year = int(m.group("anno"))
                if 1950 <= year <= 2030:
                    meta.anno = year
                    break
            except (ValueError, IndexError):
                continue

    # NUMERO
    for p in NUMERO_PATTERNS:
        m = p.search(testo)
        if m:
            meta.numero = normalize_numero(m.group("num"))
            break

    # SEZIONE
    for p in SEZIONE_PATTERNS:
        m = p.search(testo)
        if m:
            meta.sezione = normalize_sezione(m.group("sez"))
            break

    # RV (full format preferito, fallback su no-sub)
    for p in RV_PATTERNS:
        m = p.search(testo)
        if m:
            rv_num = m.group("rv")
            try:
                sub = m.group("sub")
                meta.rv = f"{rv_num}-{sub.zfill(2)}"
            except (IndexError, AttributeError):
                meta.rv = rv_num
            break

    meta.confidence = meta.filled_count() / 4.0
    return meta


# ──────────────────────────────────────────────────────────────────
# LLM FALLBACK
# ──────────────────────────────────────────────────────────────────

LLM_PROMPT = """Sei un esperto di massime della Corte di Cassazione italiana.

Dal testo seguente estrai i metadati citazionali della sentenza citata (se presente).

TESTO:
{testo}

Rispondi SOLO con un JSON nel formato esatto:
{{"anno": <int o null>, "numero": <str o null>, "sezione": <str o null>, "rv_number": <str o null>, "confidence": <float 0.0-1.0>}}

Regole:
- anno: anno sentenza (int 1950-2030), null se non presente
- numero: SOLO cifre, es "17104" (no "n.", no "/YYYY"). Rimuovi zero-padding
- sezione: "U" (Unite), "L" (Lavoro), "1"..."7", o forme composte "6-1". Null se non chiaro
- rv_number: formato "NNNNNN-NN" (6-7 cifre + dash + 2 cifre); se manca sub: solo "NNNNNN"
- confidence: 0.25 per ogni campo estratto con certezza. 1.0 se tutti 4, 0.0 se nessuno

Se il testo è narrativo e NON contiene info strutturate, restituisci:
{{"anno": null, "numero": null, "sezione": null, "rv_number": null, "confidence": 0.0}}

JSON:"""


def _parse_llm_json(raw: str) -> dict | None:
    """Parse robusto di JSON da risposta LLM, tollera markdown fences."""
    if not raw:
        return None
    s = raw.strip()
    if s.startswith("```"):
        # Rimuovi ```json ... ``` o ``` ... ```
        parts = s.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("{"):
                s = part
                break
    # Ritaglia al primo blocco { ... } valido
    match = re.search(r"\{.*\}", s, re.DOTALL)
    if match:
        s = match.group(0)
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return None


async def llm_extract(
    client: httpx.AsyncClient,
    api_key: str,
    testo: str,
    model: str = LLM_MODEL,
    timeout: float = 15.0,
    max_retries: int = 1,
) -> ExtractedMetadata:
    """Chiama LLM per estrazione. Truncation a 3000 char per evitare token bloat."""
    if not testo:
        return ExtractedMetadata(source="llm_failed")

    # Cap testo per contenere costi: 3000 char ≈ 750 token
    trunc = testo[:3000]
    prompt = LLM_PROMPT.format(testo=trunc)

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 256,
        "response_format": {"type": "json_object"},
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://lexe.pro",
        "X-Title": "LEXE Metadata Recovery",
    }

    for attempt in range(max_retries + 1):
        try:
            resp = await client.post(LLM_API_URL, headers=headers, json=payload, timeout=timeout)
            if resp.status_code == 429:
                # Rate limit: backoff e retry
                await asyncio.sleep(2 ** attempt * 1.5)
                continue
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            data = _parse_llm_json(content)
            if not data:
                return ExtractedMetadata(source="llm_failed")

            numero = data.get("numero")
            if numero is not None:
                numero = str(numero)
            meta = ExtractedMetadata(
                anno=data.get("anno") if isinstance(data.get("anno"), int) else None,
                numero=normalize_numero(numero),
                sezione=normalize_sezione(data.get("sezione")) if data.get("sezione") else None,
                rv=str(data["rv_number"]) if data.get("rv_number") else None,
                confidence=float(data.get("confidence", 0.0)),
                source="llm",
            )
            # Sanity check sull'anno
            if meta.anno is not None and not (1950 <= meta.anno <= 2030):
                meta.anno = None
            return meta
        except (httpx.TimeoutException, httpx.HTTPStatusError) as e:
            if attempt < max_retries:
                await asyncio.sleep(1.5)
                continue
            return ExtractedMetadata(source="llm_failed")
        except Exception:
            return ExtractedMetadata(source="llm_failed")

    return ExtractedMetadata(source="llm_failed")


# ──────────────────────────────────────────────────────────────────
# DB operations
# ──────────────────────────────────────────────────────────────────

UPDATE_SQL = """
    UPDATE kb.massime
    SET anno    = COALESCE(anno, $1::integer),
        numero  = COALESCE(numero, $2),
        sezione = COALESCE(sezione, $3),
        rv      = COALESCE(rv, $4),
        citation_extracted = TRUE,
        updated_at = NOW()
    WHERE id = $5::uuid
      AND (anno IS NULL OR numero IS NULL OR sezione IS NULL OR rv IS NULL)
"""


async def fetch_rows(
    conn: asyncpg.Connection,
    limit: int | None,
    only_anno_null: bool,
) -> list[asyncpg.Record]:
    """Seleziona massime con metadati incompleti. Usa testo_con_contesto se disponibile."""
    where = "anno IS NULL" if only_anno_null else (
        "anno IS NULL OR numero IS NULL OR sezione IS NULL OR rv IS NULL"
    )
    query = f"""
        SELECT id,
               COALESCE(testo_con_contesto, testo) AS testo_ctx,
               testo
        FROM kb.massime
        WHERE {where}
        ORDER BY id
    """
    if limit and limit > 0:
        query += f" LIMIT {int(limit)}"
    return await conn.fetch(query)


async def apply_update(conn: asyncpg.Connection, row_id, meta: ExtractedMetadata) -> None:
    await conn.execute(
        UPDATE_SQL,
        meta.anno,
        meta.numero,
        meta.sezione,
        meta.rv,
        row_id,
    )


async def final_report(conn: asyncpg.Connection) -> dict:
    row = await conn.fetchrow(
        """
        SELECT
            COUNT(*)                                           AS total,
            COUNT(*) FILTER (WHERE anno IS NULL)               AS anno_null,
            COUNT(*) FILTER (WHERE numero IS NULL)             AS numero_null,
            COUNT(*) FILTER (WHERE sezione IS NULL)            AS sezione_null,
            COUNT(*) FILTER (WHERE rv IS NULL)                 AS rv_null,
            COUNT(*) FILTER (WHERE citation_extracted = TRUE)  AS extracted
        FROM kb.massime
        """
    )
    return dict(row)


# ──────────────────────────────────────────────────────────────────
# Pipeline
# ──────────────────────────────────────────────────────────────────


async def run_regex_pass(
    dsn: str,
    limit: int | None,
    min_confidence: float,
    execute: bool,
) -> dict:
    conn = await asyncpg.connect(dsn)
    try:
        rows = await fetch_rows(conn, limit, only_anno_null=False)
        stats = {
            "phase": "regex",
            "processed": len(rows),
            "matched_at_threshold": 0,
            "below_threshold": 0,
            "filled_breakdown": {"anno": 0, "numero": 0, "sezione": 0, "rv": 0},
            "updated": 0,
            "execute": execute,
        }
        for row in rows:
            testo = row["testo_ctx"] or row["testo"] or ""
            meta = regex_extract(testo)
            if meta.anno is not None:
                stats["filled_breakdown"]["anno"] += 1
            if meta.numero is not None:
                stats["filled_breakdown"]["numero"] += 1
            if meta.sezione is not None:
                stats["filled_breakdown"]["sezione"] += 1
            if meta.rv is not None:
                stats["filled_breakdown"]["rv"] += 1

            if meta.confidence >= min_confidence:
                stats["matched_at_threshold"] += 1
                if execute:
                    await apply_update(conn, row["id"], meta)
                    stats["updated"] += 1
            else:
                stats["below_threshold"] += 1
        return stats
    finally:
        await conn.close()


async def run_llm_pass(
    dsn: str,
    limit: int | None,
    min_confidence: float,
    concurrency: int,
    execute: bool,
    api_key: str,
    model: str,
) -> dict:
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY (o LLM_API_KEY) non impostata. "
            "Export la variabile prima di eseguire --only-llm o cascade completa."
        )

    conn = await asyncpg.connect(dsn)
    try:
        # LLM opera solo sui residui post-regex: anno IS NULL
        rows = await fetch_rows(conn, limit, only_anno_null=True)
        if not rows:
            return {"phase": "llm", "processed": 0, "updated": 0, "execute": execute}

        stats = {
            "phase": "llm",
            "processed": len(rows),
            "concurrency": concurrency,
            "llm_ok": 0,
            "llm_failed": 0,
            "matched_at_threshold": 0,
            "below_threshold": 0,
            "updated": 0,
            "execute": execute,
            "model": model,
            "wall_clock_sec": 0.0,
        }
        sem = asyncio.Semaphore(concurrency)
        results: list[tuple[object, ExtractedMetadata]] = []

        async with httpx.AsyncClient() as client:
            async def worker(row):
                async with sem:
                    testo = row["testo_ctx"] or row["testo"] or ""
                    meta = await llm_extract(client, api_key, testo, model=model)
                    return row["id"], meta

            t0 = time.monotonic()
            results = await asyncio.gather(*[worker(r) for r in rows])
            stats["wall_clock_sec"] = round(time.monotonic() - t0, 1)

        for row_id, meta in results:
            if meta.source == "llm_failed":
                stats["llm_failed"] += 1
                continue
            stats["llm_ok"] += 1
            if meta.confidence >= min_confidence:
                stats["matched_at_threshold"] += 1
                if execute:
                    await apply_update(conn, row_id, meta)
                    stats["updated"] += 1
            else:
                stats["below_threshold"] += 1

        return stats
    finally:
        await conn.close()


# ──────────────────────────────────────────────────────────────────
# Self-test (no DB, no LLM)
# ──────────────────────────────────────────────────────────────────


def run_self_test() -> int:
    """Valida i regex contro fixture sample_100_massime.json."""
    if not FIXTURE_SAMPLE.exists():
        print(f"FIXTURE not found: {FIXTURE_SAMPLE}", file=sys.stderr)
        return 1
    with open(FIXTURE_SAMPLE, "r", encoding="utf-8") as f:
        data = json.load(f)

    tot = len(data)
    filled = {"anno": 0, "numero": 0, "sezione": 0, "rv": 0}
    high_conf = 0

    # Agreement: quando il regex estrae un valore per una massima che HA quel
    # valore in gold, confronta.
    agree = {"anno": [0, 0], "numero": [0, 0], "sezione": [0, 0]}

    for m in data:
        testo = m.get("testo") or ""
        meta = regex_extract(testo)
        if meta.confidence >= 0.50:
            high_conf += 1

        if meta.anno:
            filled["anno"] += 1
            if m.get("anno"):
                agree["anno"][1] += 1
                if meta.anno == m["anno"]:
                    agree["anno"][0] += 1
        if meta.numero:
            filled["numero"] += 1
            if m.get("numero"):
                agree["numero"][1] += 1
                # Normalizza gold (può avere zero-padding)
                gold_num = str(m["numero"]).lstrip("0") or str(m["numero"])
                if meta.numero == gold_num:
                    agree["numero"][0] += 1
        if meta.sezione:
            filled["sezione"] += 1
            if m.get("sezione"):
                agree["sezione"][1] += 1
                gold_sez = normalize_sezione(str(m["sezione"]))
                if meta.sezione == gold_sez:
                    agree["sezione"][0] += 1
        if meta.rv:
            filled["rv"] += 1

    print(f"=== SELF-TEST regex on {tot} sample massime ===")
    for k, v in filled.items():
        print(f"  {k:8s} estratto:   {v:3d}/{tot} ({100*v/tot:.0f}%)")
    print(f"  confidence >= 0.50:  {high_conf}/{tot} ({100*high_conf/tot:.0f}%)")
    print()
    print("Agreement (solo dove gold ha valore):")
    for k, (ok, tot_k) in agree.items():
        if tot_k == 0:
            print(f"  {k:8s} no gold values to compare")
        else:
            pct = 100 * ok / tot_k
            print(f"  {k:8s} {ok}/{tot_k}  ({pct:.0f}%)")

    # Passa se agreement anno+numero entrambi >= 70% e almeno 50% high_conf
    anno_ok = agree["anno"][1] == 0 or agree["anno"][0] / agree["anno"][1] >= 0.70
    num_ok = agree["numero"][1] == 0 or agree["numero"][0] / agree["numero"][1] >= 0.70
    hc_ok = high_conf / tot >= 0.50

    if anno_ok and num_ok and hc_ok:
        print("\n[OK] Self-test PASS")
        return 0
    print("\n[FAIL] Self-test FAIL (anno_ok={}, num_ok={}, hc_ok={})".format(anno_ok, num_ok, hc_ok))
    return 2


# ──────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--db-url", default=DB_URL_DEFAULT,
        help="PostgreSQL DSN. Default: env QA_DB_URL o local dev",
    )
    p.add_argument(
        "--min-confidence", type=float, default=0.80,
        help="Soglia minima confidence per applicare update (default: 0.80)",
    )
    p.add_argument("--only-regex", action="store_true", help="Esegui SOLO regex pass")
    p.add_argument("--only-llm", action="store_true", help="Esegui SOLO LLM pass (su anno IS NULL)")
    p.add_argument(
        "--execute", action="store_true",
        help="Applica UPDATE al DB (default: dry-run, nessuna scrittura)",
    )
    p.add_argument("--limit", type=int, default=0, help="Limita numero righe (0 = nessun limit)")
    p.add_argument(
        "--concurrency", type=int, default=10,
        help="Parallelismo LLM via asyncio.Semaphore (default: 10)",
    )
    p.add_argument("--model", default=LLM_MODEL, help=f"Modello LLM (default: {LLM_MODEL})")
    p.add_argument("--self-test", action="store_true", help="Valida regex contro fixture, no DB/LLM")
    p.add_argument(
        "--report-json", type=Path, default=None,
        help="Scrivi report finale come JSON nel path indicato",
    )
    return p.parse_args()


async def async_main(args: argparse.Namespace) -> int:
    results: list[dict] = []

    # Inizio / fine stats DB
    start_stats = None
    end_stats = None
    dsn = args.db_url

    if not args.only_llm:
        # Regex pass
        try:
            conn_probe = await asyncpg.connect(dsn)
            try:
                start_stats = await final_report(conn_probe)
            finally:
                await conn_probe.close()
        except Exception as e:
            print(f"[WARN] Impossibile ottenere baseline stats: {e}", file=sys.stderr)

        print(f"\n--- REGEX PASS (execute={args.execute}, min_conf={args.min_confidence}) ---")
        regex_stats = await run_regex_pass(
            dsn=dsn,
            limit=args.limit or None,
            min_confidence=args.min_confidence,
            execute=args.execute,
        )
        results.append(regex_stats)
        print(json.dumps(regex_stats, indent=2, default=str))

    if not args.only_regex:
        # LLM pass
        print(f"\n--- LLM PASS (execute={args.execute}, conc={args.concurrency}, model={args.model}) ---")
        llm_stats = await run_llm_pass(
            dsn=dsn,
            limit=args.limit or None,
            min_confidence=args.min_confidence,
            concurrency=args.concurrency,
            execute=args.execute,
            api_key=LLM_API_KEY,
            model=args.model,
        )
        results.append(llm_stats)
        print(json.dumps(llm_stats, indent=2, default=str))

    # Stats finali
    try:
        conn_probe = await asyncpg.connect(dsn)
        try:
            end_stats = await final_report(conn_probe)
        finally:
            await conn_probe.close()
    except Exception as e:
        print(f"[WARN] Impossibile ottenere stats finali: {e}", file=sys.stderr)

    # Gate B check
    gate_b_status = None
    if end_stats:
        anno_null = end_stats["anno_null"]
        numero_null = end_stats["numero_null"]
        if anno_null < 2000 and numero_null < 4000:
            gate_b_status = "PASS"
        elif anno_null < 3500:
            gate_b_status = "WARN"
        else:
            gate_b_status = "FAIL"

    print("\n=== FINAL REPORT ===")
    if start_stats:
        print(f"Baseline:  anno_null={start_stats['anno_null']}  numero_null={start_stats['numero_null']}")
    if end_stats:
        print(f"Finale:    anno_null={end_stats['anno_null']}  numero_null={end_stats['numero_null']}  "
              f"sezione_null={end_stats['sezione_null']}  rv_null={end_stats['rv_null']}")
        print(f"Citation extracted: {end_stats['extracted']}/{end_stats['total']}")
    if gate_b_status:
        print(f"GATE B: {gate_b_status}  (target: anno_null<2000 AND numero_null<4000)")

    if args.report_json:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "args": {
                "execute": args.execute,
                "min_confidence": args.min_confidence,
                "only_regex": args.only_regex,
                "only_llm": args.only_llm,
                "limit": args.limit,
                "concurrency": args.concurrency,
                "model": args.model,
            },
            "baseline": start_stats,
            "final": end_stats,
            "gate_b": gate_b_status,
            "phases": results,
        }
        with open(args.report_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)
        print(f"Report scritto: {args.report_json}")

    if gate_b_status == "FAIL":
        return 3
    if gate_b_status == "WARN":
        return 1
    return 0


def main() -> int:
    args = parse_args()
    if args.self_test:
        return run_self_test()
    if args.only_regex and args.only_llm:
        print("ERROR: --only-regex e --only-llm sono mutuamente esclusivi", file=sys.stderr)
        return 2
    try:
        return asyncio.run(async_main(args))
    except KeyboardInterrupt:
        print("\nInterrotto.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
