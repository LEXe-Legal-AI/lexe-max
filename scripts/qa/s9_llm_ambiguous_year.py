"""
QA Protocol - Phase 9: LLM Trigger - Ambiguous Year

For PDFs with year_resolution.has_conflict=true, asks LLM to resolve
based on front matter text.

Model: Mistral Small via OpenRouter
Max calls: ~5 per doc with conflict
Cache: by content_hash of input text

Usage (on staging server):
    cd /opt/leo-platform/lexe-api
    export OPENROUTER_API_KEY='sk-or-...'
    uv run python scripts/qa/s9_llm_ambiguous_year.py
"""

import asyncio
import hashlib
import json
import os
import re
import time
from pathlib import Path

import asyncpg
import httpx

from qa_config import DB_URL, PDF_DIR, OPENROUTER_URL, OPENROUTER_API_KEY, LLM_MODEL

PROMPT_TEMPLATE = """Analizza il seguente testo estratto dalle prime pagine di un PDF del massimario della Corte di Cassazione.
Il filename del PDF è: {filename}

Ci sono indicazioni contrastanti sull'anno di pubblicazione:
{conflict_details}

Testo prime pagine:
---
{text}
---

Rispondi SOLO con JSON:
{{"anno": XXXX, "confidence": 0.0-1.0, "reasoning": "breve spiegazione"}}"""


def get_front_text(pdf_path: Path, max_pages: int = 3) -> str:
    """Get text from first pages via PyMuPDF."""
    try:
        import pymupdf
        doc = pymupdf.open(str(pdf_path))
        texts = []
        for i in range(min(max_pages, doc.page_count)):
            texts.append(doc[i].get_text())
        doc.close()
        return " ".join(texts)[:2000]
    except Exception:
        return ""


async def main():
    print("=" * 70)
    print("QA PROTOCOL - PHASE 9: LLM AMBIGUOUS YEAR")
    print("=" * 70)

    if not OPENROUTER_API_KEY:
        print("[ERROR] Set OPENROUTER_API_KEY")
        return

    conn = await asyncpg.connect(DB_URL)
    print("[OK] Database connected")

    qa_run_id = await conn.fetchval(
        "SELECT id FROM kb.qa_runs ORDER BY started_at DESC LIMIT 1"
    )

    # Get PDFs with year conflicts
    conflicts = await conn.fetch(
        """
        SELECT yr.manifest_id, yr.conflict_details,
               pm.filename
        FROM kb.pdf_year_resolution yr
        JOIN kb.pdf_manifest pm ON pm.id = yr.manifest_id
        WHERE yr.has_conflict = true
        """,
    )
    print(f"[OK] Found {len(conflicts)} PDFs with year conflicts")

    if not conflicts:
        print("[DONE] No conflicts to resolve")
        await conn.close()
        return

    cache = {}
    total_calls = 0
    total_cost = 0.0

    async with httpx.AsyncClient() as http_client:
        for c in conflicts:
            manifest_id = c["manifest_id"]
            filename = c["filename"]

            pdf_path = PDF_DIR / filename
            if not pdf_path.exists():
                pdf_path = PDF_DIR / "new" / filename
            if not pdf_path.exists():
                continue

            front_text = get_front_text(pdf_path)
            if not front_text:
                continue

            content_hash = hashlib.sha256(front_text.encode()).hexdigest()

            # Check cache
            if content_hash in cache:
                print(f"  [CACHE] {filename}")
                continue

            prompt = PROMPT_TEMPLATE.format(
                filename=filename,
                conflict_details=c["conflict_details"] or "unknown",
                text=front_text,
            )

            start = time.time()
            try:
                response = await http_client.post(
                    OPENROUTER_URL,
                    headers={
                        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": LLM_MODEL,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.1,
                        "max_tokens": 200,
                    },
                    timeout=30.0,
                )
                latency = int((time.time() - start) * 1000)

                if response.status_code != 200:
                    print(f"  [ERROR] {filename}: HTTP {response.status_code}")
                    continue

                data = response.json()
                content = data["choices"][0]["message"]["content"]
                usage = data.get("usage", {})
                tokens_in = usage.get("prompt_tokens", 0)
                tokens_out = usage.get("completion_tokens", 0)
                cost = (tokens_in * 0.1 + tokens_out * 0.3) / 1_000_000  # Mistral Small pricing

                # Parse JSON
                json_match = re.search(r"\{[^}]+\}", content)
                parsed = json.loads(json_match.group()) if json_match else {}
                confidence = float(parsed.get("confidence", 0.5))

                await conn.execute(
                    """
                    INSERT INTO kb.llm_decisions
                      (qa_run_id, manifest_id, trigger_type,
                       input_text, model, prompt_template,
                       raw_response, parsed_output,
                       confidence, tokens_input, tokens_output,
                       cost_usd, latency_ms)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8::jsonb, $9, $10, $11, $12, $13)
                    """,
                    qa_run_id, manifest_id, "ambiguous_year",
                    front_text[:500], LLM_MODEL, "s9_ambiguous_year_v1",
                    content, json.dumps(parsed), confidence,
                    tokens_in, tokens_out, round(cost, 6), latency,
                )

                cache[content_hash] = parsed
                total_calls += 1
                total_cost += cost

                print(f"  [LLM] {filename}: anno={parsed.get('anno')}, conf={confidence:.2f}")
                await asyncio.sleep(0.5)

            except Exception as e:
                print(f"  [ERROR] {filename}: {e}")

    print(f"\n{'=' * 70}")
    print(f"LLM AMBIGUOUS YEAR COMPLETE")
    print(f"{'=' * 70}")
    print(f"Total LLM calls: {total_calls}")
    print(f"Total cost: ${total_cost:.4f}")

    await conn.close()
    print("[DONE]")


if __name__ == "__main__":
    asyncio.run(main())
