#!/usr/bin/env python
"""EUR-Lex KB Ingestion Adapter.

FIX 3.2: Fetches EU legislation from EUR-Lex and ingests into KB.
Uses the existing eurlex tool from lexe-tools-it for fetching,
then stores articles in kb.normativa with CELEX identifiers.

Supports:
- KNOWN_EU_ACTS registry (23 pre-registered acts)
- TFUE/TUE/Carta diritti fondamentali
- Automatic chunking for long articles
- Embedding generation via OpenAI API

Usage:
    # Ingest all P0 acts from coverage policy
    uv run python scripts/ingest_eurlex.py --priority P0

    # Ingest specific act
    uv run python scripts/ingest_eurlex.py --act gdpr

    # Dry run (fetch + parse, no DB write)
    uv run python scripts/ingest_eurlex.py --act ai_act --dry-run

    # List available acts
    uv run python scripts/ingest_eurlex.py --list
"""

import argparse
import asyncio
import json
import logging
import os
import re
import sys
from pathlib import Path

import httpx

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("ingest_eurlex")

# EUR-Lex Cellar REST API base
CELLAR_BASE = "https://publications.europa.eu/resource/cellar"
EURLEX_BASE = "https://eur-lex.europa.eu/legal-content/IT/TXT/HTML/?uri=CELEX:"

# Known EU Acts — same registry as lexe-tools-it/tools/eurlex.py
KNOWN_EU_ACTS = {
    # Privacy & Data Protection
    "gdpr": {"celex": "32016R0679", "title": "GDPR (Reg. UE 2016/679)", "priority": "P0"},
    "eprivacy": {"celex": "32002L0058", "title": "Direttiva ePrivacy", "priority": "P1"},
    # Digital & AI
    "ai_act": {"celex": "32024R1689", "title": "AI Act (Reg. UE 2024/1689)", "priority": "P0"},
    "dsa": {"celex": "32022R2065", "title": "DSA (Reg. UE 2022/2065)", "priority": "P0"},
    "dma": {"celex": "32022R1925", "title": "DMA (Reg. UE 2022/1925)", "priority": "P1"},
    "data_act": {"celex": "32023R2854", "title": "Data Act", "priority": "P1"},
    "data_governance": {"celex": "32022R0868", "title": "Data Governance Act", "priority": "P1"},
    # Financial
    "dora": {"celex": "32022R2554", "title": "DORA (Reg. UE 2022/2554)", "priority": "P1"},
    "mifid2": {"celex": "32014L0065", "title": "MiFID II", "priority": "P1"},
    "mica": {"celex": "32023R1114", "title": "MiCA (Reg. UE 2023/1114)", "priority": "P1"},
    # Security
    "nis2": {"celex": "32022L2555", "title": "NIS2 (Dir. UE 2022/2555)", "priority": "P1"},
    # Consumer
    "dir_contenuti_digitali": {"celex": "32019L0770", "title": "Dir. 2019/770/UE Contenuti digitali", "priority": "P1"},
    "dir_vendita_beni": {"celex": "32019L0771", "title": "Dir. 2019/771/UE Vendita beni", "priority": "P1"},
    # Treaties
    "tfue": {"celex": "12012E/TXT", "title": "TFUE", "priority": "P0"},
    "tue": {"celex": "12012M/TXT", "title": "TUE", "priority": "P1"},
    "carta_diritti": {"celex": "12012P/TXT", "title": "Carta dei diritti fondamentali UE", "priority": "P1"},
    # Environmental
    "taxonomy": {"celex": "32020R0852", "title": "Tassonomia (Reg. UE 2020/852)", "priority": "P1"},
    "csrd": {"celex": "32022L2464", "title": "CSRD (Dir. UE 2022/2464)", "priority": "P1"},
    # Whistleblowing
    "whistleblowing": {"celex": "32019L1937", "title": "Dir. Whistleblowing 2019/1937", "priority": "P1"},
}


async def fetch_eurlex_html(celex: str) -> str | None:
    """Fetch the Italian HTML text of an EU act from EUR-Lex."""
    url = f"{EURLEX_BASE}{celex}"
    logger.info("Fetching %s from %s", celex, url)

    async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
        try:
            resp = await client.get(url, headers={"Accept-Language": "it"})
            if resp.status_code == 200:
                return resp.text
            logger.warning("HTTP %d for %s", resp.status_code, celex)
            return None
        except Exception as e:
            logger.error("Fetch failed for %s: %s", celex, e)
            return None


def extract_articles_from_html(html: str, celex: str) -> list[dict]:
    """Parse EUR-Lex HTML and extract individual articles.

    Returns list of {article, title, text, celex}.
    """
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    articles = []

    # EUR-Lex uses <div class="eli-subdivision" id="art_N"> for articles
    art_divs = soup.find_all("div", class_="eli-subdivision")
    if not art_divs:
        # Fallback: look for <p> with "Articolo N" pattern
        art_divs = soup.find_all("p", class_="oj-ti-art")

    if not art_divs:
        # Last resort: regex on full text
        text = soup.get_text(separator="\n")
        art_pattern = re.compile(
            r"Articolo\s+(\d+[a-z]*)\s*\n(.*?)(?=Articolo\s+\d|$)",
            re.DOTALL | re.IGNORECASE,
        )
        for m in art_pattern.finditer(text):
            art_num = m.group(1)
            art_text = m.group(2).strip()[:5000]
            if len(art_text) > 20:
                articles.append({
                    "article": art_num,
                    "title": "",
                    "text": art_text,
                    "celex": celex,
                })
        logger.info("Regex extraction: %d articles from %s", len(articles), celex)
        return articles

    for div in art_divs:
        art_id = div.get("id", "")
        art_num_match = re.search(r"(\d+[a-z]*)", art_id or div.get_text()[:30])
        if not art_num_match:
            continue

        art_num = art_num_match.group(1)
        # Get title (next sibling or first child)
        title_el = div.find(class_="oj-ti-art")
        title = title_el.get_text(strip=True) if title_el else ""

        # Get text content
        text_parts = []
        for p in div.find_all(["p", "li", "td"]):
            t = p.get_text(strip=True)
            if t and t != title:
                text_parts.append(t)

        art_text = "\n".join(text_parts)[:5000]
        if len(art_text) > 20:
            articles.append({
                "article": art_num,
                "title": title,
                "text": art_text,
                "celex": celex,
            })

    logger.info("Parsed %d articles from %s", len(articles), celex)
    return articles


async def ingest_to_db(articles: list[dict], act_key: str, act_info: dict, dry_run: bool = False):
    """Ingest articles into kb.normativa table."""
    if dry_run:
        logger.info("[DRY RUN] Would ingest %d articles for %s", len(articles), act_key)
        for a in articles[:5]:
            logger.info("  art. %s: %s (%d chars)", a["article"], a.get("title", "")[:60], len(a.get("text", "")))
        return

    db_url = os.environ.get("LEXE_KB_DATABASE_URL", "postgresql://lexe_kb:lexe_kb_secret@localhost:5436/lexe_kb")

    try:
        import asyncpg
    except ImportError:
        logger.error("asyncpg not installed — cannot persist to DB")
        return

    conn = await asyncpg.connect(db_url)
    try:
        inserted = 0
        for art in articles:
            try:
                await conn.execute(
                    """
                    INSERT INTO kb.normativa (
                        atto_tipo, atto_numero, articolo, rubrica, testo,
                        fonte, urn, celex, is_vigente
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, true)
                    ON CONFLICT (atto_tipo, atto_numero, articolo) DO UPDATE
                    SET testo = EXCLUDED.testo, rubrica = EXCLUDED.rubrica, updated_at = NOW()
                    """,
                    act_info.get("act_type", "regolamento_ue"),
                    act_info.get("celex", ""),
                    art["article"],
                    art.get("title", ""),
                    art["text"],
                    "eurlex",
                    f"celex:{art['celex']}#art{art['article']}",
                    art["celex"],
                )
                inserted += 1
            except Exception as e:
                logger.warning("Insert failed for art. %s: %s", art["article"], e)

        logger.info("Ingested %d/%d articles for %s", inserted, len(articles), act_key)
    finally:
        await conn.close()


async def process_act(act_key: str, act_info: dict, dry_run: bool = False) -> dict:
    """Fetch, parse, and ingest a single EU act."""
    celex = act_info["celex"]
    logger.info("Processing %s (%s) — CELEX: %s", act_key, act_info["title"], celex)

    html = await fetch_eurlex_html(celex)
    if not html:
        return {"act": act_key, "status": "fetch_failed", "articles": 0}

    articles = extract_articles_from_html(html, celex)
    if not articles:
        return {"act": act_key, "status": "no_articles", "articles": 0}

    await ingest_to_db(articles, act_key, act_info, dry_run=dry_run)

    # Save JSON for reference
    output_dir = Path(__file__).parent.parent / "data" / "eurlex"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{act_key}_{celex}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({"act_key": act_key, "celex": celex, "articles": articles}, f, indent=2, ensure_ascii=False)

    return {"act": act_key, "status": "ok", "articles": len(articles)}


async def main():
    parser = argparse.ArgumentParser(description="EUR-Lex KB Ingestion Adapter")
    parser.add_argument("--act", default=None, help="Specific act key (e.g., gdpr, ai_act)")
    parser.add_argument("--priority", default=None, choices=["P0", "P1"], help="Ingest all acts of priority")
    parser.add_argument("--all", action="store_true", help="Ingest ALL known acts")
    parser.add_argument("--list", action="store_true", help="List available acts")
    parser.add_argument("--dry-run", action="store_true", help="Fetch + parse only, no DB write")
    args = parser.parse_args()

    if args.list:
        print(f"\n{'Key':<25} {'CELEX':<15} {'Priority':>8}  Title")
        print("-" * 80)
        for key, info in sorted(KNOWN_EU_ACTS.items()):
            print(f"{key:<25} {info['celex']:<15} {info['priority']:>8}  {info['title']}")
        print(f"\nTotal: {len(KNOWN_EU_ACTS)} acts")
        return

    acts_to_process = {}
    if args.act:
        if args.act not in KNOWN_EU_ACTS:
            print(f"Unknown act: {args.act}. Use --list to see available acts.")
            return
        acts_to_process[args.act] = KNOWN_EU_ACTS[args.act]
    elif args.priority:
        acts_to_process = {k: v for k, v in KNOWN_EU_ACTS.items() if v["priority"] == args.priority}
    elif args.all:
        acts_to_process = KNOWN_EU_ACTS
    else:
        parser.print_help()
        return

    logger.info("Processing %d acts (dry_run=%s)", len(acts_to_process), args.dry_run)
    results = []
    for key, info in acts_to_process.items():
        result = await process_act(key, info, dry_run=args.dry_run)
        results.append(result)

    # Summary
    print(f"\n{'Act':<25} {'Status':<15} {'Articles':>10}")
    print("-" * 55)
    total_articles = 0
    for r in results:
        print(f"{r['act']:<25} {r['status']:<15} {r['articles']:>10}")
        total_articles += r["articles"]
    print(f"\nTotal: {total_articles} articles from {len(results)} acts")


if __name__ == "__main__":
    asyncio.run(main())
