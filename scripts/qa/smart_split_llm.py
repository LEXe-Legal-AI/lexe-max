#!/usr/bin/env python3
"""
Smart PDF Split with LLM - Find optimal chapter boundaries using Gemini.

Usa LLM per identificare i veri inizi di capitolo/sezione nelle pagine
attorno alla metà del PDF.
"""

import json
import os
import re
from pathlib import Path

import fitz
import httpx

from qa_config import OPENROUTER_API_KEY, OPENROUTER_URL, LLM_MODEL

# Fallback API key se non in env
API_KEY = OPENROUTER_API_KEY or os.getenv("OPENROUTER_API_KEY", "")


def extract_pages_text(pdf_path: Path, start_page: int, end_page: int) -> dict[int, str]:
    """Estrae testo da un range di pagine."""
    doc = fitz.open(pdf_path)
    texts = {}

    for pg in range(start_page - 1, min(end_page, len(doc))):
        text = doc[pg].get_text()[:800]  # Prime 800 chars per pagina
        texts[pg + 1] = text.strip()

    doc.close()
    return texts


def find_split_with_llm(pdf_path: Path, search_range: int = 30) -> dict | None:
    """
    Usa LLM per trovare il miglior punto di split.

    Args:
        pdf_path: Path del PDF
        search_range: Quante pagine cercare attorno alla metà

    Returns:
        Dict con: page, title, reasoning
    """
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    doc.close()

    if total_pages <= 300:
        print(f"  {pdf_path.name}: {total_pages} pag, no split needed")
        return None

    middle = total_pages // 2
    start = max(1, middle - search_range)
    end = min(total_pages, middle + search_range)

    print(f"  Analizzando pagine {start}-{end} (middle={middle})...")

    # Estrai testo dalle pagine
    pages_text = extract_pages_text(pdf_path, start, end)

    # Prepara prompt per LLM
    pages_content = "\n\n".join([
        f"=== PAGINA {pg} ===\n{text[:500]}"
        for pg, text in pages_text.items()
    ])

    prompt = f"""Analizza queste pagine di un documento legale italiano e trova la pagina che inizia un nuovo CAPITOLO o SEZIONE principale.

Cerco un punto di split per dividere un PDF di {total_pages} pagine in due parti.
La metà ideale è pagina {middle}.

{pages_content}

Trova la pagina che:
1. Inizia con un titolo come "CAPITOLO", "SEZIONE", "PARTE", "TITOLO" o simile
2. È il più vicino possibile alla metà ({middle})
3. Rappresenta un vero confine di sezione (non header ripetuto)

Rispondi SOLO in JSON:
{{"page": numero_pagina, "title": "titolo della sezione", "reasoning": "breve spiegazione"}}
"""

    # Chiama LLM
    try:
        response = httpx.post(
            OPENROUTER_URL,
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": LLM_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
                "max_tokens": 200,
            },
            timeout=60.0,
        )

        if response.status_code != 200:
            print(f"  [ERROR] LLM API: {response.status_code}")
            return None

        result = response.json()
        content = result["choices"][0]["message"]["content"]

        # Parse JSON dalla risposta
        json_match = re.search(r'\{[^}]+\}', content, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            return data
        else:
            print(f"  [WARN] No JSON in response: {content[:100]}")
            return None

    except Exception as e:
        print(f"  [ERROR] LLM call failed: {e}")
        return None


def verify_all_splits_with_llm(pdf_dir: Path, current_splits: dict) -> dict:
    """
    Verifica e corregge tutti gli split usando LLM.

    Returns:
        Dict aggiornato con split corretti
    """
    updated_splits = current_splits.copy()

    for pdf in sorted(pdf_dir.glob("*.pdf")):
        doc = fitz.open(pdf)
        total = len(doc)
        doc.close()

        if total <= 300:
            continue

        print(f"\n{pdf.name} ({total} pag)")

        result = find_split_with_llm(pdf)

        if result:
            page = result.get("page")
            title = result.get("title", "")

            # Calcola parti
            part1 = page - 1
            part2 = total - page + 1

            status = "OK" if part1 <= 300 and part2 <= 300 else "PROBLEMA"

            print(f"  LLM suggerisce: pag {page} '{title[:40]}...'")
            print(f"  Parti: {part1} + {part2} [{status}]")

            if status == "OK":
                updated_splits[pdf.name] = [page]

    return updated_splits


def test_single_pdf(pdf_path: Path):
    """Test su singolo PDF."""
    print(f"Analisi LLM per: {pdf_path.name}")
    print("=" * 60)

    result = find_split_with_llm(pdf_path)

    if result:
        print(f"\nRisultato LLM:")
        print(f"  Pagina: {result.get('page')}")
        print(f"  Titolo: {result.get('title')}")
        print(f"  Motivo: {result.get('reasoning')}")

    return result


if __name__ == "__main__":
    import sys

    if not API_KEY:
        print("ERROR: OPENROUTER_API_KEY not set")
        sys.exit(1)

    if len(sys.argv) > 1:
        pdf_path = Path(sys.argv[1])
        if pdf_path.is_file():
            test_single_pdf(pdf_path)
        else:
            print(f"File not found: {pdf_path}")
    else:
        # Test su Rassegna Penale 2011
        test_pdf = Path(r"C:\PROJECTS\LEO-ITC\raccolta\Rassegna Penale 2011.pdf")
        if test_pdf.exists():
            test_single_pdf(test_pdf)
