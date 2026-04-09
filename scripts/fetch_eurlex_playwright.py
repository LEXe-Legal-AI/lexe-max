#!/usr/bin/env python3
"""
fetch_eurlex_playwright.py — Download EUR-Lex HTML with Playwright (WAF bypass)

EUR-Lex uses CloudFront WAF that returns HTTP 202 to curl/httpx.
Playwright with a real Chromium browser bypasses this.

Downloads each act's Italian HTML and saves to data/eurlex/<celex>.html
for subsequent ingestion by ingest_sprint22_onda1.py.

Usage:
    python fetch_eurlex_playwright.py                     # All 5 targets
    python fetch_eurlex_playwright.py --act dma           # Single act
    python fetch_eurlex_playwright.py --act dma --headed  # Visible browser
"""

import argparse
import logging
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("eurlex_pw")

EURLEX_BASE = "https://eur-lex.europa.eu/legal-content/IT/TXT/HTML/?uri=CELEX:"

TARGETS = {
    "dma":  {"celex": "32022R1925", "title": "DMA (Reg. UE 2022/1925)"},
    "dora": {"celex": "32022R2554", "title": "DORA (Reg. UE 2022/2554)"},
    "dsa":  {"celex": "32022R2065", "title": "DSA (Reg. UE 2022/2065)"},
    "nis2": {"celex": "32022L2555", "title": "NIS2 (Dir. UE 2022/2555)"},
    "tfue": {"celex": "12012E/TXT", "title": "TFUE"},
}

OUTPUT_DIR = Path(__file__).parent.parent / "data" / "eurlex"


def fetch_act(page, key: str, info: dict, output_dir: Path) -> bool:
    celex = info["celex"]
    url = f"{EURLEX_BASE}{celex}"
    safe_name = celex.replace("/", "_").replace("\\", "_")
    out_path = output_dir / f"{safe_name}.html"

    if out_path.exists() and out_path.stat().st_size > 10000:
        log.info("  [%s] Already downloaded: %s (%d bytes)", key, out_path.name, out_path.stat().st_size)
        return True

    log.info("  [%s] Navigating to %s", key, url)
    try:
        page.goto(url, wait_until="networkidle", timeout=60000)
    except Exception as e:
        log.warning("  [%s] Navigation timeout, trying domcontentloaded: %s", key, e)
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=30000)
            time.sleep(5)  # wait for dynamic content
        except Exception as e2:
            log.error("  [%s] FAILED: %s", key, e2)
            return False

    # Wait for article content to appear
    try:
        page.wait_for_selector("p.oj-ti-art, div.eli-subdivision, .texte", timeout=15000)
    except Exception:
        log.warning("  [%s] No article selectors found, saving anyway", key)

    html = page.content()
    if len(html) < 5000:
        log.warning("  [%s] HTML suspiciously short: %d chars", key, len(html))
        return False

    out_path.write_text(html, encoding="utf-8")
    log.info("  [%s] Saved: %s (%d chars)", key, out_path.name, len(html))

    # Be polite
    time.sleep(3)
    return True


def main():
    parser = argparse.ArgumentParser(description="Download EUR-Lex HTML with Playwright")
    parser.add_argument("--act", default=None, help="Single act key (dma, dora, dsa, nis2, tfue)")
    parser.add_argument("--headed", action="store_true", help="Run browser in visible mode")
    args = parser.parse_args()

    targets = TARGETS
    if args.act:
        if args.act not in TARGETS:
            log.error("Unknown act: %s. Available: %s", args.act, ", ".join(TARGETS.keys()))
            return
        targets = {args.act: TARGETS[args.act]}

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    log.info("EUR-Lex Playwright Downloader")
    log.info("  Targets: %s", ", ".join(targets.keys()))
    log.info("  Output:  %s", OUTPUT_DIR)
    log.info("  Headed:  %s", args.headed)

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=not args.headed)
        context = browser.new_context(
            locale="it-IT",
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                       "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
        )
        page = context.new_page()

        ok = 0
        fail = 0
        for key, info in targets.items():
            if fetch_act(page, key, info, OUTPUT_DIR):
                ok += 1
            else:
                fail += 1

        browser.close()

    log.info("")
    log.info("Done: %d/%d downloaded, %d failed", ok, ok + fail, fail)
    if fail:
        log.info("Failed acts need manual download to %s/<celex>.html", OUTPUT_DIR)


if __name__ == "__main__":
    main()
