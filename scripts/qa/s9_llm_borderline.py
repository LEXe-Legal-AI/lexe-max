"""
QA Protocol - Phase 9: LLM Trigger - Borderline Classification

For chunks with heur_confidence < 0.5 that weren't resolved by cheap heuristic,
uses LLM to classify as massima/toc/citation_list/noise.

Model: Mistral Small via OpenRouter
Cache: by content_hash (deduplicates identical chunks)

Usage (on staging server):
    cd /opt/lexe-platform/lexe-max
    export OPENROUTER_API_KEY='sk-or-...'
    uv run python scripts/qa/s9_llm_borderline.py
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

PROMPT_TEMPLATE = """Classifica il seguente testo estratto da un massimario della Corte di Cassazione.

Testo (primi 500 caratteri):
---
{text}
---

Rispondi SOLO con JSON:
{{"label": "massima"|"toc"|"citation_list"|"noise", "confidence": 0.0-1.0, "reasoning": "breve spiegazione"}}"""


async def main():
    print("=" * 70)
    print("QA PROTOCOL - PHASE 9: LLM BORDERLINE CLASSIFICATION")
    print("=" * 70)

    if not OPENROUTER_API_KEY:
        print("[ERROR] Set OPENROUTER_API_KEY")
        return

    conn = await asyncpg.connect(DB_URL)
    print("[OK] Database connected")

    qa_run_id = await conn.fetchval(
        "SELECT id FROM kb.qa_runs ORDER BY started_at DESC LIMIT 1"
    )

    # Get borderline chunks (heur_confidence < 0.5, label_method != 'llm')
    borderlines = await conn.fetch(
        """
        SELECT cl.id as label_id, cl.chunk_feature_id,
               cf.manifest_id, cf.massima_id,
               m.testo, m.content_hash
        FROM kb.chunk_labels cl
        JOIN kb.chunk_features cf ON cf.id = cl.chunk_feature_id
        JOIN kb.massime m ON m.id = cf.massima_id
        WHERE cl.qa_run_id = $1
          AND cl.heur_confidence < 0.5
          AND cl.label_method != 'llm'
        """,
        qa_run_id,
    )
    print(f"[OK] Found {len(borderlines)} borderline chunks")

    cache = {}
    total_calls = 0
    total_cached = 0
    total_cost = 0.0

    async with httpx.AsyncClient() as http_client:
        for bl in borderlines:
            manifest_id = bl["manifest_id"]
            testo = bl["testo"] or ""
            content_hash = bl["content_hash"] or hashlib.sha256(
                testo.lower().encode()
            ).hexdigest()

            # Check cache
            if content_hash in cache:
                cached = cache[content_hash]
                # Update label
                await conn.execute(
                    """
                    UPDATE kb.chunk_labels
                    SET llm_label = $1, llm_confidence = $2, llm_model = $3,
                        llm_response = $4::jsonb,
                        final_label = $1, final_confidence = $2, label_method = 'llm_cached'
                    WHERE id = $5
                    """,
                    cached["label"], cached["confidence"],
                    LLM_MODEL + "_cached", json.dumps(cached), bl["label_id"],
                )
                total_cached += 1
                continue

            prompt = PROMPT_TEMPLATE.format(text=testo[:500])

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
                    continue

                data = response.json()
                content = data["choices"][0]["message"]["content"]
                usage = data.get("usage", {})
                tokens_in = usage.get("prompt_tokens", 0)
                tokens_out = usage.get("completion_tokens", 0)
                cost = (tokens_in * 0.1 + tokens_out * 0.3) / 1_000_000

                json_match = re.search(r"\{[^}]+\}", content)
                parsed = json.loads(json_match.group()) if json_match else {}
                label = parsed.get("label", "uncertain")
                confidence = float(parsed.get("confidence", 0.5))

                if label not in ("massima", "toc", "citation_list", "noise"):
                    label = "uncertain"

                # Cache result
                cache[content_hash] = {"label": label, "confidence": confidence}

                # Update chunk label
                await conn.execute(
                    """
                    UPDATE kb.chunk_labels
                    SET llm_label = $1, llm_confidence = $2, llm_model = $3,
                        llm_response = $4::jsonb,
                        final_label = $1, final_confidence = $2, label_method = 'llm'
                    WHERE id = $5
                    """,
                    label, confidence, LLM_MODEL, json.dumps(parsed), bl["label_id"],
                )

                # Log decision
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
                    qa_run_id, manifest_id, "borderline_classification",
                    testo[:500], LLM_MODEL, "s9_borderline_v1",
                    content, json.dumps(parsed), confidence,
                    tokens_in, tokens_out, round(cost, 6), latency,
                )

                total_calls += 1
                total_cost += cost
                await asyncio.sleep(0.2)

            except Exception as e:
                print(f"  [ERROR] {e}")

    print(f"\n{'=' * 70}")
    print(f"LLM BORDERLINE COMPLETE")
    print(f"{'=' * 70}")
    print(f"Total LLM calls: {total_calls}")
    print(f"Cache hits: {total_cached}")
    print(f"Total cost: ${total_cost:.4f}")

    await conn.close()
    print("[DONE]")


if __name__ == "__main__":
    asyncio.run(main())
