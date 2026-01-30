#!/usr/bin/env python3
"""Classify missing documents with simplified prompt."""

import asyncio
import re
from pathlib import Path

import asyncpg
import fitz
import httpx

import os

from qa_config import DB_URL, PDF_DIR

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "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")
LLM_MODEL = "google/gemini-2.5-flash-lite-preview-09-2025"

SIMPLE_PROMPT = """Classifica questo documento legale italiano.

Rispondi SOLO con una riga nel formato:
TYPE|PROFILE

Dove TYPE: list_only, massima_plus_commentary, mixed, toc_heavy
Dove PROFILE: structured_parent_child, baseline_toc_filter, mixed_hybrid

Campione:
{sample}

Risposta (una sola riga TYPE|PROFILE):"""


async def main():
    conn = await asyncpg.connect(DB_URL)

    # Find missing
    missing = await conn.fetch("""
        SELECT m.id, m.filename
        FROM kb.pdf_manifest m
        LEFT JOIN kb.doc_intel_ab_results r ON r.manifest_id = m.id AND r.run_name = 'A'
        WHERE r.id IS NULL
    """)

    print(f"Documenti mancanti: {len(missing)}")

    for doc in missing:
        filename = doc["filename"]
        manifest_id = doc["id"]

        pdf_path = PDF_DIR / filename
        if not pdf_path.exists():
            pdf_path = PDF_DIR / "new" / filename
        if not pdf_path.exists():
            print(f"  {filename[:45]:45} NOT FOUND")
            continue

        # Extract sample from first content pages
        pdf = fitz.open(pdf_path)
        sample = ""
        for p in range(min(10, len(pdf))):
            text = pdf[p].get_text()
            if len(text) > 300:
                sample = text[:1500]
                break
        pdf.close()

        if not sample:
            print(f"  {filename[:45]:45} NO TEXT")
            continue

        prompt = SIMPLE_PROMPT.format(sample=sample)

        async with httpx.AsyncClient() as client:
            try:
                resp = await client.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"},
                    json={
                        "model": LLM_MODEL,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.0,
                        "max_tokens": 50,
                    },
                    timeout=30.0,
                )

                if resp.status_code != 200:
                    print(f"  {filename[:45]:45} HTTP {resp.status_code}")
                    continue

                content = resp.json()["choices"][0]["message"]["content"].strip()

                # Parse simple format: TYPE|PROFILE
                parts = content.split("|")
                if len(parts) >= 2:
                    doc_type = parts[0].strip().lower()
                    profile = parts[1].strip().lower()

                    # Normalize
                    if doc_type not in ["list_only", "massima_plus_commentary", "mixed", "toc_heavy"]:
                        doc_type = "mixed"
                    if profile not in ["structured_parent_child", "baseline_toc_filter", "mixed_hybrid", "structured_by_title"]:
                        profile = "structured_parent_child"
                else:
                    doc_type = "mixed"
                    profile = "structured_parent_child"

                await conn.execute(
                    """
                    INSERT INTO kb.doc_intel_ab_results
                    (run_name, manifest_id, filename, doc_type, profile, chunking_strategy, confidence)
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                    ON CONFLICT (run_name, manifest_id) DO UPDATE SET
                    doc_type = EXCLUDED.doc_type, profile = EXCLUDED.profile
                    """,
                    "A", manifest_id, filename, doc_type, profile, "by_title", 0.8,
                )

                print(f"  {filename[:45]:45} OK: {doc_type} / {profile}")

            except Exception as e:
                print(f"  {filename[:45]:45} ERROR: {str(e)[:40]}")

    # Final count
    count = await conn.fetchval("SELECT count(*) FROM kb.doc_intel_ab_results WHERE run_name = 'A'")
    print(f"\nTotale classificati: {count}/63")

    await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
