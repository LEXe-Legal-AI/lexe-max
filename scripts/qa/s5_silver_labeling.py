"""
QA Protocol - Phase 5: Silver Labeling (3-step)

Step 1 - Heuristic labeling (deterministic)
Step 2 - Cheap scoring (numeric features, avoids LLM for clear cases)
Step 3 - LLM only for residual uncertain (Mistral Small, cache by content_hash)

Labels: massima, toc, citation_list, noise, uncertain

Usage (on staging server):
    cd /opt/leo-platform/lexe-api
    export OPENROUTER_API_KEY='sk-or-...'
    uv run python scripts/qa/s5_silver_labeling.py
"""

import asyncio
import hashlib
import json
import os
import re
import time

import asyncpg
import httpx

from qa_config import DB_URL
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
LLM_MODEL = "mistralai/mistral-small-latest"

# LLM prompt template
LLM_PROMPT = """Sei un esperto di diritto italiano. Classifica il seguente testo estratto da un massimario della Corte di Cassazione.

Testo (primi 500 caratteri):
---
{text}
---

Rispondi SOLO con JSON:
{{"label": "massima"|"toc"|"citation_list"|"noise", "confidence": 0.0-1.0, "reasoning": "breve spiegazione"}}

Criteri:
- "massima": principio giuridico, commento a sentenza, analisi giuridica
- "toc": indice, sommario, lista di capitoli/pagine
- "citation_list": elenco di citazioni/sentenze senza analisi
- "noise": testo non pertinente, artefatti OCR, header/footer"""


def heuristic_label(chunk: dict) -> tuple[str, float, dict]:
    """
    Step 1: Deterministic heuristic labeling.
    Returns (label, confidence, reasons).
    """
    toc_score = chunk["toc_infiltration_score"] or 0
    cit_score = chunk["citation_list_score"] or 0
    is_short = chunk["is_short"]
    char_count = chunk["char_count"]
    has_citation = chunk["has_multiple_citations"]
    starts_legal = chunk["starts_with_legal_pattern"]
    quality = chunk["quality_score"] or 0

    reasons = {}

    # Rule 1: TOC
    if toc_score > 0.6:
        reasons["toc_score"] = toc_score
        return "toc", 0.8, reasons

    # Rule 2: Citation list
    if cit_score > 0.7:
        reasons["citation_list_score"] = cit_score
        return "citation_list", 0.8, reasons

    # Rule 3: Short noise
    if is_short and not has_citation:
        reasons["is_short"] = True
        reasons["no_citation"] = True
        return "noise", 0.7, reasons

    # Rule 4: Strong massima signal
    if starts_legal and char_count > 200 and has_citation:
        reasons["starts_legal"] = True
        reasons["has_citation"] = True
        reasons["char_count"] = char_count
        return "massima", 0.9, reasons

    # Rule 5: Decent massima
    if char_count > 200 and quality > 0.7:
        reasons["char_count"] = char_count
        reasons["quality"] = quality
        return "massima", 0.8, reasons

    # Fallback: uncertain
    reasons["uncertain"] = True
    return "uncertain", 0.4, reasons


def cheap_score(chunk: dict) -> tuple[str | None, float]:
    """
    Step 2: Cheap numeric scoring for uncertain chunks.
    Returns (label_or_none, confidence).
    """
    score = 0.5  # neutral base

    toc_score = chunk["toc_infiltration_score"] or 0
    cit_score = chunk["citation_list_score"] or 0
    sentence_count = chunk["sentence_count"] or 0
    char_count = chunk["char_count"]
    has_citation = chunk["has_multiple_citations"]
    quality = chunk["quality_score"] or 0

    # Penalties (noise signals)
    if toc_score > 0.3:
        score -= 0.15
    if cit_score > 0.3:
        score -= 0.1
    if char_count < 100:
        score -= 0.2

    # Bonuses (massima signals)
    if sentence_count > 2:
        score += 0.15
    if has_citation:
        score += 0.1
    if char_count > 300:
        score += 0.1
    if quality > 0.8:
        score += 0.1

    if score > 0.7:
        return "massima", 0.65
    if score < 0.3:
        return "noise", 0.65
    return None, 0.0  # Still uncertain, needs LLM


async def llm_classify(
    http_client: httpx.AsyncClient,
    text: str,
    cache: dict,
    content_hash: str,
) -> tuple[str, float, str, dict | None]:
    """
    Step 3: LLM classification for uncertain chunks.
    Returns (label, confidence, model, raw_response).
    Cache by content_hash.
    """
    # Check cache
    if content_hash in cache:
        cached = cache[content_hash]
        return cached["label"], cached["confidence"], LLM_MODEL + "_cached", cached

    if not OPENROUTER_API_KEY:
        return "uncertain", 0.3, "none", None

    prompt = LLM_PROMPT.format(text=text[:500])

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

        if response.status_code != 200:
            return "uncertain", 0.3, LLM_MODEL, None

        data = response.json()
        content = data["choices"][0]["message"]["content"]

        # Parse JSON from response
        json_match = re.search(r"\{[^}]+\}", content)
        if json_match:
            parsed = json.loads(json_match.group())
            label = parsed.get("label", "uncertain")
            confidence = float(parsed.get("confidence", 0.5))

            # Validate label
            if label not in ("massima", "toc", "citation_list", "noise"):
                label = "uncertain"

            result = {"label": label, "confidence": confidence, "reasoning": parsed.get("reasoning")}
            cache[content_hash] = result
            return label, confidence, LLM_MODEL, parsed

    except Exception:
        pass

    return "uncertain", 0.3, LLM_MODEL, None


async def main():
    print("=" * 70)
    print("QA PROTOCOL - PHASE 5: SILVER LABELING (3-STEP)")
    print("=" * 70)

    conn = await asyncpg.connect(DB_URL)
    print("[OK] Database connected")

    qa_run_id = await conn.fetchval(
        "SELECT id FROM kb.qa_runs ORDER BY started_at DESC LIMIT 1"
    )
    print(f"[OK] qa_run_id={qa_run_id}")

    # Get all chunk features
    chunks = await conn.fetch(
        """
        SELECT cf.id, cf.manifest_id, cf.massima_id,
               cf.char_count, cf.word_count, cf.sentence_count,
               cf.is_short, cf.is_very_long,
               cf.toc_infiltration_score, cf.citation_list_score,
               cf.has_multiple_citations, cf.starts_with_legal_pattern,
               cf.quality_score,
               m.testo, m.content_hash
        FROM kb.chunk_features cf
        JOIN kb.massime m ON m.id = cf.massima_id
        WHERE cf.qa_run_id = $1
        """,
        qa_run_id,
    )
    print(f"[OK] Found {len(chunks)} chunks to label")

    # Check existing labels
    existing_ids = set()
    existing_rows = await conn.fetch(
        "SELECT chunk_feature_id FROM kb.chunk_labels WHERE qa_run_id = $1",
        qa_run_id,
    )
    existing_ids = {r["chunk_feature_id"] for r in existing_rows}

    label_counts = {}
    llm_calls = 0
    llm_cache = {}
    step_counts = {"heuristic": 0, "cheap": 0, "llm": 0}

    async with httpx.AsyncClient() as http_client:
        for chunk in chunks:
            cf_id = chunk["id"]
            if cf_id in existing_ids:
                continue

            # Step 1: Heuristic
            heur_label, heur_conf, heur_reasons = heuristic_label(dict(chunk))

            final_label = heur_label
            final_conf = heur_conf
            label_method = "heuristic"
            llm_label = None
            llm_conf = None
            llm_model = None
            llm_response = None

            if heur_label == "uncertain" or heur_conf < 0.6:
                # Step 2: Cheap scoring
                cheap_label, cheap_conf = cheap_score(dict(chunk))

                if cheap_label is not None:
                    final_label = cheap_label
                    final_conf = cheap_conf
                    label_method = "cheap_heuristic"
                    step_counts["cheap"] += 1
                else:
                    # Step 3: LLM
                    content_hash = chunk["content_hash"] or hashlib.sha256(
                        (chunk["testo"] or "").lower().encode()
                    ).hexdigest()

                    llm_label, llm_conf, llm_model, llm_response = await llm_classify(
                        http_client, chunk["testo"] or "", llm_cache, content_hash,
                    )
                    llm_calls += 1
                    step_counts["llm"] += 1

                    # Resolve: if LLM agrees with heuristic, boost confidence
                    if llm_label == heur_label:
                        final_label = llm_label
                        final_conf = max(heur_conf, llm_conf)
                    else:
                        # LLM overrides for uncertain
                        final_label = llm_label
                        final_conf = llm_conf
                    label_method = "llm"

                    # Rate limit
                    await asyncio.sleep(0.2)
            else:
                step_counts["heuristic"] += 1

            await conn.execute(
                """
                INSERT INTO kb.chunk_labels
                  (qa_run_id, chunk_feature_id,
                   heur_label, heur_confidence, heur_reasons,
                   llm_label, llm_confidence, llm_model, llm_response,
                   final_label, final_confidence, label_method)
                VALUES ($1, $2, $3, $4, $5::jsonb, $6, $7, $8, $9::jsonb, $10, $11, $12)
                """,
                qa_run_id, cf_id,
                heur_label, heur_conf, json.dumps(heur_reasons),
                llm_label, llm_conf, llm_model,
                json.dumps(llm_response) if llm_response else None,
                final_label, final_conf, label_method,
            )

            label_counts[final_label] = label_counts.get(final_label, 0) + 1

    # Summary
    print(f"\n{'=' * 70}")
    print(f"SILVER LABELING COMPLETE")
    print(f"{'=' * 70}")
    print(f"Label distribution:")
    for label, cnt in sorted(label_counts.items(), key=lambda x: -x[1]):
        total = sum(label_counts.values())
        print(f"  {label}: {cnt} ({cnt/total*100:.1f}%)")
    print(f"\nStep breakdown:")
    print(f"  Heuristic (direct): {step_counts['heuristic']}")
    print(f"  Cheap heuristic: {step_counts['cheap']}")
    print(f"  LLM calls: {step_counts['llm']}")
    print(f"  LLM cache hits: {len(llm_cache)}")

    await conn.close()
    print("[DONE]")


if __name__ == "__main__":
    asyncio.run(main())
