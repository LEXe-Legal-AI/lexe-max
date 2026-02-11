#!/usr/bin/env python3
"""Analyze content differences between Altalex and Brocardi using LLM.

Uses OpenRouter free model to classify differences:
- formatting: Only whitespace/punctuation differences
- abrogated: Article is abrogated in one source
- version_diff: Different legal versions (date of modification)
- preleggi_offset: Numbering offset for Preleggi articles
- substantive: Real content differences

Usage:
    cd lexe-max

    # Analyze all content_diff articles (with limit)
    uv run python scripts/llm_diff_analyzer.py --limit 50

    # Analyze specific articles
    uv run python scripts/llm_diff_analyzer.py --articles 26,93,105

    # Dry run (no DB updates)
    uv run python scripts/llm_diff_analyzer.py --limit 10 --dry-run
"""

import argparse
import asyncio
import os
from datetime import datetime
from pathlib import Path

import asyncpg
import httpx
import structlog

# Add parent to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Load .env file if present
try:
    from dotenv import load_dotenv
    env_file = Path(__file__).parent.parent / ".env"
    if env_file.exists():
        load_dotenv(env_file)
except ImportError:
    pass

logger = structlog.get_logger()

# Database connection
DB_URL = "postgresql://lexe_kb:lexe_kb_dev_password@localhost:5434/lexe_kb"

# LLM endpoints
LITELLM_URL = "http://localhost:4001/v1/chat/completions"  # Local LiteLLM proxy
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# API keys
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
LITELLM_KEY = os.environ.get("LITELLM_KEY", "")  # Optional for local LiteLLM

# Classification prompt
CLASSIFICATION_PROMPT = """Sei un esperto di diritto italiano. Analizza le differenze tra due versioni dello stesso articolo di legge.

ARTICOLO: {articolo} del {codice}

VERSIONE ALTALEX (fonte editoriale 2025):
---
{altalex_text}
---

VERSIONE BROCARDI (fonte web):
---
{brocardi_text}
---

Classifica la differenza in UNA delle seguenti categorie:

1. FORMATTING - Solo differenze di formattazione (spazi, punteggiatura, maiuscole/minuscole)
2. ABROGATED - L'articolo è abrogato/soppresso in una delle fonti
3. VERSION_DIFF - Versioni diverse dello stesso articolo (date di modifica diverse)
4. PRELEGGI_OFFSET - Articolo delle Disposizioni preliminari (Preleggi) con numerazione diversa
5. SUBSTANTIVE - Differenze sostanziali nel contenuto giuridico

Rispondi SOLO con un JSON nel formato:
{{"classification": "CATEGORIA", "confidence": 0.0-1.0, "reason": "breve spiegazione"}}
"""


async def get_content_diff_articles(
    conn: asyncpg.Connection,
    limit: int = 0,
    articles: list[str] | None = None,
) -> list[dict]:
    """Get articles with content_diff status."""

    if articles:
        # Specific articles
        rows = await conn.fetch("""
            SELECT
                a.id, a.codice, a.articolo, a.rubrica, a.testo as altalex_text,
                b.testo as brocardi_text, a.brocardi_similarity
            FROM kb.normativa_altalex a
            JOIN kb.normativa b ON a.brocardi_match_id = b.id
            WHERE a.brocardi_match_status = 'content_diff'
              AND a.articolo = ANY($1)
            ORDER BY a.brocardi_similarity ASC
        """, articles)
    else:
        # All content_diff with limit
        query = """
            SELECT
                a.id, a.codice, a.articolo, a.rubrica, a.testo as altalex_text,
                b.testo as brocardi_text, a.brocardi_similarity
            FROM kb.normativa_altalex a
            JOIN kb.normativa b ON a.brocardi_match_id = b.id
            WHERE a.brocardi_match_status = 'content_diff'
            ORDER BY a.brocardi_similarity ASC
        """
        if limit > 0:
            query += f" LIMIT {limit}"
        rows = await conn.fetch(query)

    return [dict(row) for row in rows]


def parse_llm_response(content: str) -> dict | None:
    """Parse LLM response with multiple fallback strategies."""
    import json
    import re

    if not content or not content.strip():
        return None

    content = content.strip()

    # Strategy 1: Handle markdown code blocks
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0].strip()
    elif "```" in content:
        parts = content.split("```")
        if len(parts) >= 2:
            content = parts[1].strip()

    # Strategy 2: Try direct JSON parse
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    # Strategy 3: Find JSON object in text
    json_match = re.search(r'\{[^{}]*"classification"[^{}]*\}', content, re.IGNORECASE)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

    # Strategy 4: Regex fallback - extract classification from text
    categories = ["FORMATTING", "ABROGATED", "VERSION_DIFF", "PRELEGGI_OFFSET", "SUBSTANTIVE"]

    for cat in categories:
        if cat.lower() in content.lower() or cat in content:
            # Try to find confidence
            conf_match = re.search(r'(\d+(?:\.\d+)?)\s*%?|confidence["\s:]+(\d+(?:\.\d+)?)', content, re.IGNORECASE)
            confidence = 0.7  # default
            if conf_match:
                val = conf_match.group(1) or conf_match.group(2)
                confidence = float(val) if float(val) <= 1 else float(val) / 100

            # Try to find reason
            reason_match = re.search(r'reason["\s:]+["\']?([^"\'}\n]+)', content, re.IGNORECASE)
            reason = reason_match.group(1).strip() if reason_match else "Extracted via regex fallback"

            return {
                "classification": cat,
                "confidence": confidence,
                "reason": reason
            }

    return None


async def call_llm_with_model(prompt: str, model: str, max_retries: int = 2) -> dict | None:
    """Call specific LLM model with retry and robust parsing."""
    if not OPENROUTER_API_KEY:
        logger.error("OPENROUTER_API_KEY not set")
        return None

    url = OPENROUTER_URL
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://lexe.pro",
        "X-Title": "LEXE Legal AI"
    }

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "Sei un assistente legale. Rispondi SOLO con JSON: {\"classification\": \"CATEGORIA\", \"confidence\": 0.9, \"reason\": \"motivo\"}"
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.1,
        "max_tokens": 300,
    }

    for attempt in range(max_retries + 1):
        try:
            async with httpx.AsyncClient(timeout=90.0) as client:
                response = await client.post(url, headers=headers, json=payload)
                response.raise_for_status()

                result = response.json()
                content = result.get("choices", [{}])[0].get("message", {}).get("content", "")

                if not content:
                    logger.warning(f"Empty response from {model}, attempt {attempt + 1}/{max_retries + 1}")
                    if attempt < max_retries:
                        await asyncio.sleep(3)
                        continue
                    return None

                parsed = parse_llm_response(content)
                if parsed:
                    return parsed

                logger.warning(f"Failed to parse {model} response, attempt {attempt + 1}")
                if attempt < max_retries:
                    await asyncio.sleep(2)

        except httpx.HTTPStatusError as e:
            logger.warning(f"HTTP error {e.response.status_code} from {model}, attempt {attempt + 1}")
            if attempt < max_retries:
                await asyncio.sleep(3)
        except Exception as e:
            logger.warning(f"Error from {model}: {e}, attempt {attempt + 1}")
            if attempt < max_retries:
                await asyncio.sleep(2)

    return None


async def call_llm_free(prompt: str) -> dict | None:
    """Call LLM with cascade: openrouter/free -> wait -> retry -> mistral-nemo fallback."""

    # Step 1: Try openrouter/free (2 attempts)
    result = await call_llm_with_model(prompt, "openrouter/free", max_retries=1)
    if result:
        return result

    # Step 2: Wait and retry openrouter/free
    logger.info("Waiting 5s before retry with free model...")
    await asyncio.sleep(5)
    result = await call_llm_with_model(prompt, "openrouter/free", max_retries=1)
    if result:
        return result

    # Step 3: Fallback to mistral-nemo (free tier)
    logger.info("Fallback to mistralai/mistral-nemo...")
    result = await call_llm_with_model(prompt, "mistralai/mistral-nemo", max_retries=2)
    if result:
        result["_model"] = "mistral-nemo"  # Track which model was used
        return result

    logger.error("All LLM attempts failed")
    return None


async def analyze_article(article: dict) -> dict:
    """Analyze a single article difference."""

    # Truncate texts if too long (save tokens)
    altalex_text = article["altalex_text"][:2000]
    brocardi_text = article["brocardi_text"][:2000]

    prompt = CLASSIFICATION_PROMPT.format(
        articolo=f"Art. {article['articolo']}",
        codice=article["codice"],
        altalex_text=altalex_text,
        brocardi_text=brocardi_text,
    )

    result = await call_llm_free(prompt)

    if result:
        return {
            "id": article["id"],
            "articolo": article["articolo"],
            "similarity": article["brocardi_similarity"],
            "classification": result.get("classification", "UNKNOWN"),
            "confidence": result.get("confidence", 0.0),
            "reason": result.get("reason", ""),
        }
    else:
        return {
            "id": article["id"],
            "articolo": article["articolo"],
            "similarity": article["brocardi_similarity"],
            "classification": "ERROR",
            "confidence": 0.0,
            "reason": "LLM call failed",
        }


async def update_classification(
    conn: asyncpg.Connection,
    article_id: str,
    classification: str,
    confidence: float,
    reason: str,
) -> None:
    """Update article with LLM classification."""

    await conn.execute("""
        UPDATE kb.normativa_altalex SET
            brocardi_match_status = $1,
            brocardi_extras = array_append(
                COALESCE(brocardi_extras, '{}'),
                $2
            ),
            updated_at = NOW()
        WHERE id = $3
    """,
        f"llm_{classification.lower()}",
        f"llm_reason: {reason} (conf: {confidence:.0%})",
        article_id,
    )


async def main():
    global OPENROUTER_API_KEY

    parser = argparse.ArgumentParser(description="LLM Diff Analyzer")
    parser.add_argument("--limit", type=int, default=50, help="Max articles to analyze")
    parser.add_argument("--articles", help="Comma-separated article numbers")
    parser.add_argument("--dry-run", action="store_true", help="Don't update DB")
    parser.add_argument("--api-key", help="OpenRouter API key (or set OPENROUTER_API_KEY)")
    parser.add_argument("--delay", type=float, default=3.0, help="Seconds between requests (default 3)")
    parser.add_argument("--batch-pause", type=int, default=50, help="Pause every N articles (default 50)")
    args = parser.parse_args()

    # API key from arg or env
    if args.api_key:
        OPENROUTER_API_KEY = args.api_key

    print(f"\n{'='*60}")
    print(f"  LLM DIFF ANALYZER - Altalex vs Brocardi")
    print(f"  {datetime.now().isoformat()}")
    print(f"  Model: openrouter/free -> mistral-nemo fallback")
    print(f"  Rate: {args.delay}s delay, pause every {args.batch_pause}")
    print(f"{'='*60}\n")

    if not OPENROUTER_API_KEY:
        print("ERROR: Set OPENROUTER_API_KEY environment variable")
        print("  export OPENROUTER_API_KEY='sk-or-v1-...'")
        return

    # Connect to DB
    conn = await asyncpg.connect(DB_URL)
    print("Connected to database")

    # Get articles to analyze
    articles_list = None
    if args.articles:
        articles_list = [a.strip() for a in args.articles.split(",")]

    articles = await get_content_diff_articles(conn, args.limit, articles_list)
    print(f"Found {len(articles)} articles to analyze")

    if not articles:
        print("No articles with content_diff status found")
        await conn.close()
        return

    # Stats
    stats = {
        "total": len(articles),
        "FORMATTING": 0,
        "ABROGATED": 0,
        "VERSION_DIFF": 0,
        "PRELEGGI_OFFSET": 0,
        "SUBSTANTIVE": 0,
        "ERROR": 0,
        "UNKNOWN": 0,
    }

    print(f"\nAnalyzing {len(articles)} articles...\n")

    for i, article in enumerate(articles, 1):
        # Rate limiting
        if i > 1:
            await asyncio.sleep(args.delay)

        # Batch pause every N articles to avoid rate limits
        if i > 1 and i % args.batch_pause == 0:
            print(f"\n  [PAUSE] Batch {i // args.batch_pause} complete, waiting 30s...\n")
            await asyncio.sleep(30)

        result = await analyze_article(article)

        classification = result["classification"].upper()
        stats[classification] = stats.get(classification, 0) + 1

        # Icon based on classification
        icons = {
            "FORMATTING": "~~",
            "ABROGATED": "XX",
            "VERSION_DIFF": "VV",
            "PRELEGGI_OFFSET": "PP",
            "SUBSTANTIVE": "!!",
            "ERROR": "??",
            "UNKNOWN": "??",
        }
        icon = icons.get(classification, "??")

        print(
            f"  [{i}/{len(articles)}] {icon} Art. {result['articolo']}: "
            f"{classification} ({result['confidence']:.0%})"
        )
        if result["reason"]:
            print(f"      -> {result['reason'][:80]}")

        # Update DB
        if not args.dry_run and classification not in ("ERROR", "UNKNOWN"):
            await update_classification(
                conn,
                result["id"],
                classification,
                result["confidence"],
                result["reason"],
            )

    await conn.close()

    # Final stats
    print(f"\n{'='*60}")
    print(f"  ANALYSIS RESULTS")
    print(f"{'='*60}")
    print(f"  Total analyzed: {stats['total']}")
    print(f"  [~~] Formatting: {stats['FORMATTING']}")
    print(f"  [XX] Abrogated: {stats['ABROGATED']}")
    print(f"  [VV] Version diff: {stats['VERSION_DIFF']}")
    print(f"  [PP] Preleggi offset: {stats['PRELEGGI_OFFSET']}")
    print(f"  [!!] Substantive: {stats['SUBSTANTIVE']}")
    print(f"  [??] Errors/Unknown: {stats['ERROR'] + stats['UNKNOWN']}")
    print(f"  Dry run: {args.dry_run}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    asyncio.run(main())
