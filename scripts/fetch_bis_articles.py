#!/usr/bin/env python3
"""
fetch_bis_articles.py — Fetch articoli -bis/-ter da Normattiva OpenData API

Usa l'endpoint POST /api/v1/atto/dettaglio-atto-urn per scaricare articoli
con suffissi latini (-bis, -ter, -quater, ...) che mancano nella KB LEXE.

IMPORTANTE: delay conservativo 2s base + 0.1-3.2s random jitter per evitare
ban IP da parte di Normattiva. Connessione locale ambiente dev.

Usage:
    python fetch_bis_articles.py                          # Tutti i codici, discovery mode
    python fetch_bis_articles.py --codice CCII            # Solo CCII
    python fetch_bis_articles.py --codice CCII --dry-run  # Solo stampa URN senza fetch
    python fetch_bis_articles.py --known-only             # Solo articoli da lista nota
    python fetch_bis_articles.py --resume                 # Riprende da ultimo checkpoint

Output:
    data/bis_articles/                    # Directory con JSON per articolo
    data/bis_articles/_manifest.jsonl     # Log di ogni fetch (per resume)
    data/bis_articles/_summary.json       # Riepilogo finale

Autore: lexe-improve-lab
Data: 2026-04-09
"""

import argparse
import json
import logging
import random
import re
import sys
import time
from html import unescape
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.error import HTTPError
from urllib.request import Request, urlopen

# ──────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────

API_URL = "https://api.normattiva.it/t/normattiva.api/bff-opendata/v1/api/v1/atto/dettaglio-atto-urn"

# Delay: 2s fisso + random 0.1–3.2s = totale 2.1–5.2s tra richieste
DELAY_BASE_S = 2.0
DELAY_JITTER_MIN_S = 0.1
DELAY_JITTER_MAX_S = 3.2

# Dopo N richieste consecutive, pausa lunga per simulare umano
BATCH_PAUSE_EVERY = 25
BATCH_PAUSE_MIN_S = 15.0
BATCH_PAUSE_MAX_S = 45.0

# Retry su errori transient (500, timeout)
MAX_RETRIES = 2
RETRY_BACKOFF_S = 10.0

# Timeout per singola richiesta
REQUEST_TIMEOUT_S = 15

SUFFISSI = [
    "bis", "ter", "quater", "quinquies", "sexies",
    "septies", "octies", "novies", "decies",
    "undecies", "duodecies", "terdecies",
    "quaterdecies", "quinquiesdecies", "sexiesdecies",
]

# Codici con article-level access confermato sull'API OpenData (09/04/2026)
CODICI = {
    "CCII": {
        "urn_base": "urn:nir:stato:decreto.legislativo:2019-01-12;14",
        "nome": "Codice della crisi d'impresa e dell'insolvenza",
        "max_art": 391,
        "priorita": "P0",  # crisi d'impresa, art. 64-bis critico
    },
    "TUSL": {
        "urn_base": "urn:nir:stato:decreto.legislativo:2008-04-09;81",
        "nome": "Testo Unico Sicurezza Lavoro",
        "max_art": 306,
        "priorita": "P1",
    },
    "L212": {
        "urn_base": "urn:nir:stato:legge:2000-07-27;212",
        "nome": "Statuto del Contribuente",
        "max_art": 21,
        "priorita": "P0",  # art. 10-bis (abuso del diritto) critico per tributario
    },
    "L241": {
        "urn_base": "urn:nir:stato:legge:1990-08-07;241",
        "nome": "Procedimento Amministrativo",
        "max_art": 31,
        "priorita": "P1",
    },
    "CDS_STRADA": {
        "urn_base": "urn:nir:stato:decreto.legislativo:1992-04-30;285",
        "nome": "Codice della Strada",
        "max_art": 245,
        "priorita": "P2",
    },
    "CONSUMO": {
        "urn_base": "urn:nir:stato:decreto.legislativo:2005-09-06;206",
        "nome": "Codice del Consumo",
        "max_art": 146,
        "priorita": "P1",
    },
    "PRIVACY": {
        "urn_base": "urn:nir:stato:decreto.legislativo:2003-06-30;196",
        "nome": "Codice Privacy",
        "max_art": 186,
        "priorita": "P1",
    },
    "AMBIENTE": {
        "urn_base": "urn:nir:stato:decreto.legislativo:2006-04-03;152",
        "nome": "Codice dell'Ambiente",
        "max_art": 318,
        "priorita": "P2",
    },
    "CAD": {
        "urn_base": "urn:nir:stato:decreto.legislativo:2005-03-07;82",
        "nome": "Codice Amministrazione Digitale",
        "max_art": 92,
        "priorita": "P2",
    },
}

# Articoli -bis noti che DEVONO esistere (per validazione rapida)
# Usabili con --known-only per fetch mirato senza discovery
KNOWN_BIS = {
    "CCII": [
        "25bis", "25ter", "25quater", "25quinquies", "25sexies",
        "25septies", "25octies", "25novies", "25decies", "25undecies",
        "64bis", "84bis", "84ter", "84quater", "120bis",
    ],
    "TUSL": [
        "3bis", "14bis", "301bis", "302bis",
    ],
    "L212": [
        "10bis",
    ],
    "L241": [
        "10bis", "21bis", "21ter", "21quater", "21quinquies",
        "21sexies", "21septies", "21octies", "21novies",
    ],
    "CDS_STRADA": [
        "186bis", "187bis",
    ],
    "PRIVACY": [
        "2ter", "2quater", "2quinquies", "2sexies",
        "58bis", "110bis", "111bis", "132ter", "132quater",
    ],
    "CONSUMO": [
        "49bis",
    ],
}

# ──────────────────────────────────────────────────────────────────────
# MODELS
# ──────────────────────────────────────────────────────────────────────

@dataclass
class ArticleResult:
    codice: str
    articolo: str  # es. "64-bis"
    rubrica: str
    testo_html: str
    testo_plain: str
    urn: str
    data_vigenza_inizio: Optional[str] = None
    data_vigenza_fine: Optional[str] = None
    chars: int = 0
    data_fetch: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FetchLog:
    urn: str
    codice: str
    articolo: str
    status: str  # "ok", "404", "error", "skip"
    chars: int = 0
    error_msg: str = ""
    timestamp: str = ""
    elapsed_ms: int = 0

# ──────────────────────────────────────────────────────────────────────
# CORE FETCH
# ──────────────────────────────────────────────────────────────────────

def _random_delay():
    """Delay conservativo: 2s base + 0.1-3.2s jitter."""
    delay = DELAY_BASE_S + random.uniform(DELAY_JITTER_MIN_S, DELAY_JITTER_MAX_S)
    return delay


def _batch_pause(request_count: int, logger: logging.Logger):
    """Pausa lunga ogni BATCH_PAUSE_EVERY richieste."""
    if request_count > 0 and request_count % BATCH_PAUSE_EVERY == 0:
        pause = random.uniform(BATCH_PAUSE_MIN_S, BATCH_PAUSE_MAX_S)
        logger.info(f"  Pausa batch dopo {request_count} richieste: {pause:.1f}s")
        time.sleep(pause)


def fetch_article(urn: str, logger: logging.Logger) -> tuple[Optional[dict], str]:
    """
    Fetch singolo articolo da Normattiva OpenData API.

    Returns:
        (atto_dict, status) dove status e "ok", "404", "empty", "error"
    """
    body = json.dumps({"urn": urn, "formato": "JSON"}).encode("utf-8")
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Origin": "https://dati.normattiva.it",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
        "Accept-Language": "it-IT,it;q=0.9",
    }

    req = Request(url=API_URL, data=body, headers=headers, method="POST")

    for attempt in range(MAX_RETRIES + 1):
        try:
            with urlopen(req, timeout=REQUEST_TIMEOUT_S) as resp:
                data = json.loads(resp.read().decode("utf-8"))

            atto = data.get("data", {}).get("atto")
            if atto and atto.get("articoloHtml"):
                return atto, "ok"
            else:
                return None, "empty"

        except HTTPError as e:
            if e.code == 404:
                return None, "404"
            elif e.code == 400:
                return None, "400"
            elif e.code in (429, 503):
                # Rate limited o server sovraccarico — backoff pesante
                wait = RETRY_BACKOFF_S * (attempt + 1) + random.uniform(5, 15)
                logger.warning(f"  HTTP {e.code} su {urn} — retry {attempt+1}/{MAX_RETRIES} dopo {wait:.0f}s")
                time.sleep(wait)
            elif e.code >= 500:
                wait = RETRY_BACKOFF_S * (attempt + 1)
                logger.warning(f"  HTTP {e.code} su {urn} — retry {attempt+1}/{MAX_RETRIES} dopo {wait:.0f}s")
                time.sleep(wait)
            else:
                return None, f"http_{e.code}"
        except Exception as e:
            if attempt < MAX_RETRIES:
                wait = RETRY_BACKOFF_S * (attempt + 1)
                logger.warning(f"  {type(e).__name__} su {urn} — retry dopo {wait:.0f}s")
                time.sleep(wait)
            else:
                return None, f"error_{type(e).__name__}"

    return None, "max_retries"


def extract_text(html: str) -> tuple[str, str]:
    """Estrai testo plain e rubrica da HTML articolo."""
    text = re.sub(r'<[^>]+>', ' ', html)
    text = unescape(text)  # decode HTML entities (e.g. &agrave; -> à)
    text = re.sub(r'\s+', ' ', text).strip()

    # Estrai rubrica: "Art. 64-bis (Piano di ristrutturazione soggetto a omologazione)"
    rubrica = ""
    m = re.search(r'Art\.\s*[\d\w-]+\s*(?:\(([^)]+)\))?', text)
    if m and m.group(1):
        rubrica = m.group(1).strip()

    return text, rubrica

# ──────────────────────────────────────────────────────────────────────
# DISCOVERY: trova tutti i -bis per un codice
# ──────────────────────────────────────────────────────────────────────

def discover_bis_articles(
    codice: str,
    config: dict,
    output_dir: Path,
    manifest_path: Path,
    logger: logging.Logger,
    dry_run: bool = False,
    already_fetched: set[str] = None,
) -> list[ArticleResult]:
    """
    Scopre e scarica tutti gli articoli -bis/-ter per un codice.

    Strategia: per ogni articolo base (1..max_art), prova ogni suffisso.
    Appena un suffisso non esiste (404), passa al prossimo articolo base
    SOLO se il suffisso e "bis". Per suffissi successivi (ter, quater, ...),
    continua solo se il precedente esisteva.

    Esempio: se art.25-bis esiste, prova 25-ter. Se 25-ter esiste, prova 25-quater.
    Se 25-quater e 404, smetti con la serie 25-*.
    Ma per art.26: riparti da 26-bis.
    """
    urn_base = config["urn_base"]
    max_art = config["max_art"]
    results = []
    request_count = 0
    already_fetched = already_fetched or set()

    logger.info(f"\n{'='*60}")
    logger.info(f"  {codice} — {config['nome']}")
    logger.info(f"   URN base: {urn_base}")
    logger.info(f"   Articoli: 1-{max_art}, suffissi: {len(SUFFISSI)}")
    logger.info(f"   Max richieste stimate: {max_art * 1} (solo -bis) + cascata")
    logger.info(f"{'='*60}")

    for art_num in range(1, max_art + 1):
        # Per ogni articolo base, prova i suffissi in ordine
        prev_suffix_exists = True  # Il base esiste sempre

        for suffix in SUFFISSI:
            # Ottimizzazione: se il suffisso precedente non esisteva,
            # e improbabile che questo esista (salvo eccezioni rare)
            # Ma per "bis" proviamo sempre
            if suffix != "bis" and not prev_suffix_exists:
                break

            art_label = f"{art_num}{suffix}"
            art_display = f"{art_num}-{suffix}"
            urn = f"{urn_base}~art{art_label}"

            # Skip se gia fetchato (resume mode)
            if urn in already_fetched:
                logger.debug(f"  {codice} art.{art_display} — gia fetchato")
                continue

            if dry_run:
                logger.info(f"  [DRY] {codice} art.{art_display} -> {urn}")
                continue

            # Delay anti-ban
            delay = _random_delay()
            time.sleep(delay)
            _batch_pause(request_count, logger)

            # Fetch
            t0 = time.monotonic()
            atto, status = fetch_article(urn, logger)
            elapsed_ms = int((time.monotonic() - t0) * 1000)
            request_count += 1

            # Log
            log_entry = FetchLog(
                urn=urn,
                codice=codice,
                articolo=art_display,
                status=status,
                timestamp=datetime.now(timezone.utc).isoformat(),
                elapsed_ms=elapsed_ms,
            )

            if status == "ok" and atto:
                html = atto["articoloHtml"]
                text, rubrica = extract_text(html)

                result = ArticleResult(
                    codice=codice,
                    articolo=art_display,
                    rubrica=rubrica,
                    testo_html=html,
                    testo_plain=text,
                    urn=urn,
                    data_vigenza_inizio=atto.get("articoloDataInizioVigenza"),
                    data_vigenza_fine=atto.get("articoloDataFineVigenza"),
                    chars=len(text),
                    data_fetch=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                )
                results.append(result)
                log_entry.chars = len(text)

                # Salva JSON articolo
                safe_name = f"{codice}_art_{art_num}_{suffix}.json"
                out_path = output_dir / safe_name
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)

                logger.info(f"  OK {codice} art.{art_display}: {len(text)} chars — {rubrica[:50]}")
                prev_suffix_exists = True

            elif status == "404":
                logger.debug(f"  .  {codice} art.{art_display}: 404")
                prev_suffix_exists = False

            elif status == "empty":
                logger.debug(f"  .  {codice} art.{art_display}: empty response")
                prev_suffix_exists = False

            elif status.startswith("http_"):
                logger.warning(f"  WARN {codice} art.{art_display}: {status}")
                prev_suffix_exists = False

            else:
                logger.error(f"  ERR {codice} art.{art_display}: {status}")
                log_entry.error_msg = status
                prev_suffix_exists = False

            # Append al manifest (per resume)
            with open(manifest_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(log_entry), ensure_ascii=False) + "\n")

    logger.info(f"\n  {codice}: {len(results)} articoli -bis trovati in {request_count} richieste")
    return results


def fetch_known_only(
    codice: str,
    config: dict,
    known_articles: list[str],
    output_dir: Path,
    manifest_path: Path,
    logger: logging.Logger,
    dry_run: bool = False,
    already_fetched: set[str] = None,
) -> list[ArticleResult]:
    """Fetch solo articoli da lista nota (piu veloce, no discovery)."""
    urn_base = config["urn_base"]
    results = []
    request_count = 0
    already_fetched = already_fetched or set()

    logger.info(f"\n{'='*60}")
    logger.info(f"  {codice} — {config['nome']} (known-only: {len(known_articles)} articoli)")
    logger.info(f"{'='*60}")

    for art_label in known_articles:
        # Converti "25bis" -> display "25-bis"
        m = re.match(r'^(\d+)(\w+)$', art_label)
        if not m:
            logger.warning(f"  WARN Formato articolo non valido: {art_label}")
            continue
        art_display = f"{m.group(1)}-{m.group(2)}"
        urn = f"{urn_base}~art{art_label}"

        if urn in already_fetched:
            logger.debug(f"  {art_display} — gia fetchato")
            continue

        if dry_run:
            logger.info(f"  [DRY] {codice} art.{art_display} -> {urn}")
            continue

        delay = _random_delay()
        time.sleep(delay)
        _batch_pause(request_count, logger)

        t0 = time.monotonic()
        atto, status = fetch_article(urn, logger)
        elapsed_ms = int((time.monotonic() - t0) * 1000)
        request_count += 1

        log_entry = FetchLog(
            urn=urn, codice=codice, articolo=art_display,
            status=status, timestamp=datetime.now(timezone.utc).isoformat(),
            elapsed_ms=elapsed_ms,
        )

        if status == "ok" and atto:
            html = atto["articoloHtml"]
            text, rubrica = extract_text(html)
            result = ArticleResult(
                codice=codice, articolo=art_display, rubrica=rubrica,
                testo_html=html, testo_plain=text, urn=urn,
                data_vigenza_inizio=atto.get("articoloDataInizioVigenza"),
                data_vigenza_fine=atto.get("articoloDataFineVigenza"),
                chars=len(text),
                data_fetch=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            )
            results.append(result)
            log_entry.chars = len(text)

            safe_name = f"{codice}_art_{m.group(1)}_{m.group(2)}.json"
            with open(output_dir / safe_name, "w", encoding="utf-8") as f:
                json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)

            logger.info(f"  OK {codice} art.{art_display}: {len(text)} chars — {rubrica[:50]}")
        else:
            logger.warning(f"  {'ERR' if status != '404' else '.'} {codice} art.{art_display}: {status}")
            if status != "404":
                log_entry.error_msg = status

        with open(manifest_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(log_entry), ensure_ascii=False) + "\n")

    logger.info(f"\n  {codice}: {len(results)} articoli trovati in {request_count} richieste")
    return results

# ──────────────────────────────────────────────────────────────────────
# RESUME
# ──────────────────────────────────────────────────────────────────────

def load_manifest(manifest_path: Path) -> set[str]:
    """Carica URN gia processati dal manifest per resume."""
    fetched = set()
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    fetched.add(entry["urn"])
                except (json.JSONDecodeError, KeyError):
                    continue
    return fetched

# ──────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Fetch articoli -bis/-ter da Normattiva OpenData API"
    )
    parser.add_argument(
        "--codice", type=str, default=None,
        help="Solo un codice specifico (es. CCII, TUSL, L212)"
    )
    parser.add_argument(
        "--known-only", action="store_true",
        help="Fetch solo articoli dalla lista nota (no discovery)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Stampa URN senza fare fetch"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Riprende da ultimo checkpoint (legge manifest)"
    )
    parser.add_argument(
        "--output", type=str, default="data/bis_articles",
        help="Directory output (default: data/bis_articles)"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Log dettagliato (include 404)"
    )
    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger("fetch_bis")

    # Setup output
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "_manifest.jsonl"

    # Resume
    already_fetched = set()
    if args.resume:
        already_fetched = load_manifest(manifest_path)
        logger.info(f"Resume mode: {len(already_fetched)} URN gia processati")

    # Seleziona codici
    if args.codice:
        if args.codice not in CODICI:
            logger.error(f"Codice '{args.codice}' non trovato. Disponibili: {', '.join(CODICI.keys())}")
            sys.exit(1)
        codici_to_process = {args.codice: CODICI[args.codice]}
    else:
        # Ordina per priorita
        codici_to_process = dict(
            sorted(CODICI.items(), key=lambda x: x[1]["priorita"])
        )

    # Header
    logger.info(f"\n{'='*60}")
    logger.info(f"Normattiva OpenData — Fetch articoli -bis/-ter")
    logger.info(f"   Codici: {', '.join(codici_to_process.keys())}")
    logger.info(f"   Mode: {'known-only' if args.known_only else 'discovery'}")
    logger.info(f"   Delay: {DELAY_BASE_S}s + {DELAY_JITTER_MIN_S}-{DELAY_JITTER_MAX_S}s jitter")
    logger.info(f"   Output: {output_dir}")
    if args.dry_run:
        logger.info(f"   DRY RUN — nessun fetch reale")
    logger.info(f"{'='*60}")

    # Stima tempo
    if not args.known_only and not args.dry_run:
        total_arts = sum(c["max_art"] for c in codici_to_process.values())
        avg_delay = DELAY_BASE_S + (DELAY_JITTER_MIN_S + DELAY_JITTER_MAX_S) / 2
        est_minutes = (total_arts * avg_delay) / 60
        logger.info(f"\n   Stima: {total_arts} articoli base x ~{avg_delay:.1f}s = ~{est_minutes:.0f} min")
        logger.info(f"   (In realta meno, perche per ogni art. base proviamo solo -bis prima)")

    # Execute
    all_results = []
    t_start = time.monotonic()

    for codice, config in codici_to_process.items():
        if args.known_only:
            known = KNOWN_BIS.get(codice, [])
            if not known:
                logger.info(f"\n  {codice}: nessun articolo noto nella lista, skip")
                continue
            results = fetch_known_only(
                codice, config, known, output_dir, manifest_path,
                logger, args.dry_run, already_fetched,
            )
        else:
            results = discover_bis_articles(
                codice, config, output_dir, manifest_path,
                logger, args.dry_run, already_fetched,
            )
        all_results.extend(results)

    elapsed = time.monotonic() - t_start

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info(f"COMPLETATO in {elapsed/60:.1f} minuti")
    logger.info(f"   Articoli -bis trovati: {len(all_results)}")
    logger.info(f"   Chars totali: {sum(r.chars for r in all_results):,}")
    logger.info(f"{'='*60}")

    # Per codice
    by_codice = {}
    for r in all_results:
        by_codice.setdefault(r.codice, []).append(r)
    for codice, arts in sorted(by_codice.items()):
        logger.info(f"   {codice}: {len(arts)} articoli — {', '.join(a.articolo for a in arts[:5])}{'...' if len(arts)>5 else ''}")

    # Salva summary
    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": "known-only" if args.known_only else "discovery",
        "dry_run": args.dry_run,
        "total_articles": len(all_results),
        "total_chars": sum(r.chars for r in all_results),
        "elapsed_seconds": round(elapsed, 1),
        "by_codice": {
            codice: {
                "count": len(arts),
                "chars": sum(a.chars for a in arts),
                "articles": [a.articolo for a in arts],
            }
            for codice, arts in by_codice.items()
        },
    }
    summary_path = output_dir / "_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logger.info(f"\n   Summary salvato in: {summary_path}")


if __name__ == "__main__":
    main()
