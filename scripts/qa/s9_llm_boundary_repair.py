"""
QA Protocol - Phase 9: LLM Trigger - Boundary Repair

For reference units with fragment_count >= 3 (split across 3+ massime),
asks LLM to determine if fragments should be merged.

Model: Mistral Small via OpenRouter
Input: 3+ fragments + surrounding context (2000 chars)

Usage (on staging server):
    cd /opt/leo-platform/lexe-api
    export OPENROUTER_API_KEY='sk-or-...'
    uv run python scripts/qa/s9_llm_boundary_repair.py
"""

import asyncio
import hashlib
import json
import os
import re
import time

import asyncpg
import httpx

from qa_config import DB_URL, OPENROUTER_URL, OPENROUTER_API_KEY, LLM_MODEL

PROMPT_TEMPLATE = """Sei un esperto di massimari della Corte di Cassazione.

Un'unità di riferimento (potenziale massima) è stata frammentata in {n} pezzi durante l'estrazione.
Analizza i frammenti e determina se dovrebbero essere riuniti in un'unica massima.

Frammenti:
{fragments}

Rispondi SOLO con JSON:
{{"should_merge": true|false, "confidence": 0.0-1.0, "merge_groups": [[0,1,2]], "reasoning": "spiegazione"}}

Note:
- merge_groups indica quali frammenti vanno uniti (indici 0-based)
- Se i frammenti sono massime indipendenti, should_merge = false
- Se sono parti della stessa massima frammentata, should_merge = true"""


async def main():
    print("=" * 70)
    print("QA PROTOCOL - PHASE 9: LLM BOUNDARY REPAIR")
    print("=" * 70)

    if not OPENROUTER_API_KEY:
        print("[ERROR] Set OPENROUTER_API_KEY")
        return

    conn = await asyncpg.connect(DB_URL)
    print("[OK] Database connected")

    qa_run_id = await conn.fetchval(
        "SELECT id FROM kb.qa_runs ORDER BY started_at DESC LIMIT 1"
    )
    batch_id = await conn.fetchval(
        "SELECT id FROM kb.ingest_batches WHERE batch_name = 'standard_v1'"
    )

    # Get fragmented ref units (fragment_count >= 3)
    fragmented = await conn.fetch(
        """
        SELECT ra.ref_unit_id, ra.manifest_id, ra.fragment_count,
               ru.testo as ref_testo,
               pm.filename
        FROM kb.reference_alignment ra
        JOIN kb.qa_reference_units ru ON ru.id = ra.ref_unit_id
        JOIN kb.pdf_manifest pm ON pm.id = ra.manifest_id
        WHERE ra.ingest_batch_id = $1
          AND ra.match_type = 'split'
          AND ra.fragment_count >= 3
        """,
        batch_id,
    )
    print(f"[OK] Found {len(fragmented)} fragmented ref units (fragment_count >= 3)")

    cache = {}
    total_calls = 0
    total_cost = 0.0

    async with httpx.AsyncClient() as http_client:
        for frag in fragmented:
            manifest_id = frag["manifest_id"]
            ref_testo = frag["ref_testo"] or ""
            filename = frag["filename"]

            # Get the matching massime fragments
            matching = await conn.fetch(
                """
                SELECT m.id, m.testo
                FROM kb.reference_alignment ra2
                JOIN kb.massime m ON m.id = ra2.matched_massima_id
                WHERE ra2.ref_unit_id = $1 AND ra2.ingest_batch_id = $2
                ORDER BY m.id
                """,
                frag["ref_unit_id"], batch_id,
            )

            if len(matching) < 2:
                continue

            # Build fragments text
            fragments_text = ""
            for i, m in enumerate(matching):
                text_preview = (m["testo"] or "")[:500]
                fragments_text += f"\n--- Frammento {i} ---\n{text_preview}\n"

            content_hash = hashlib.sha256(fragments_text.encode()).hexdigest()

            if content_hash in cache:
                print(f"  [CACHE] {filename}")
                continue

            prompt = PROMPT_TEMPLATE.format(
                n=len(matching),
                fragments=fragments_text[:2000],
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
                        "max_tokens": 300,
                    },
                    timeout=30.0,
                )
                latency = int((time.time() - start) * 1000)

                if response.status_code != 200:
                    continue

                data = response.json()
                content = data["choices"][0]["message"]["content"]
                usage = data.get("usage", {})
                tokens_in = usage.get("prompt_tokens", 0)
                tokens_out = usage.get("completion_tokens", 0)
                cost = (tokens_in * 0.1 + tokens_out * 0.3) / 1_000_000

                json_match = re.search(r"\{[^}]+\}", content, re.DOTALL)
                parsed = json.loads(json_match.group()) if json_match else {}
                confidence = float(parsed.get("confidence", 0.5))

                cache[content_hash] = parsed

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
                    qa_run_id, manifest_id, "boundary_repair",
                    fragments_text[:500], LLM_MODEL, "s9_boundary_repair_v1",
                    content, json.dumps(parsed), confidence,
                    tokens_in, tokens_out, round(cost, 6), latency,
                )

                total_calls += 1
                total_cost += cost

                should_merge = parsed.get("should_merge", False)
                print(f"  [LLM] {filename}: merge={should_merge}, conf={confidence:.2f}")
                await asyncio.sleep(0.5)

            except Exception as e:
                print(f"  [ERROR] {filename}: {e}")

    print(f"\n{'=' * 70}")
    print(f"LLM BOUNDARY REPAIR COMPLETE")
    print(f"{'=' * 70}")
    print(f"Total LLM calls: {total_calls}")
    print(f"Total cost: ${total_cost:.4f}")

    await conn.close()
    print("[DONE]")


if __name__ == "__main__":
    asyncio.run(main())
