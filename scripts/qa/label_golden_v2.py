#!/usr/bin/env python3
"""
Golden Set Labeling for Category Graph v2.4

Double-labels golden set using two models via OpenRouter:
- Labeler A: qwen/qwen3-235b-a22b-2507
- Labeler B: mistralai/mistral-large-2512
- Judge: openai/gpt-5.2 (only when A != B)

Labels:
- Materia (6 values): CIVILE, PENALE, LAVORO, TRIBUTARIO, AMMINISTRATIVO, CRISI
- Natura (2 values): SOSTANZIALE, PROCESSUALE
- Ambito (4 values, only if PROCESSUALE): GIUDIZIO, IMPUGNAZIONI, ESECUZIONE, MISURE

Usage:
    OPENROUTER_API_KEY="sk-or-..." uv run python scripts/qa/label_golden_v2.py --dry-run
    OPENROUTER_API_KEY="sk-or-..." uv run python scripts/qa/label_golden_v2.py --commit
    OPENROUTER_API_KEY="sk-or-..." uv run python scripts/qa/label_golden_v2.py --commit --resume
"""

import argparse
import asyncio
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from uuid import UUID

import asyncpg
import httpx

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.lexe_api.kb.config import KBSettings

# OpenRouter models
LABELER_A = "qwen/qwen3-235b-a22b-2507"
LABELER_B = "mistralai/mistral-large-2512"
JUDGE_MODEL = "openai/gpt-5.2"

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Valid values for each axis
MATERIE = ["CIVILE", "PENALE", "LAVORO", "TRIBUTARIO", "AMMINISTRATIVO", "CRISI"]
NATURE = ["SOSTANZIALE", "PROCESSUALE"]
AMBITI = ["GIUDIZIO", "IMPUGNAZIONI", "ESECUZIONE", "MISURE"]


@dataclass
class LabelResult:
    """Result from a single labeler."""
    model: str
    materia: str
    natura: str
    ambito: Optional[str]  # Only if natura=PROCESSUALE
    confidence: float
    rationale: str
    raw_response: str


@dataclass
class AdjudicatedResult:
    """Final adjudicated labels."""
    materia: str
    natura: str
    ambito: Optional[str]
    agreement_score: float  # 1.0 = both agree, 0.7 = partial, 0.5 = judge decided


LABELING_PROMPT = """Sei un esperto di diritto italiano. Classifica questa massima della Corte di Cassazione.

MASSIMA:
{testo}

METADATA:
- Sezione: {sezione}
- Tipo: {tipo}
- Norme citate: {norme}

TASK: Classifica secondo tre assi.

ASSE A - MATERIA (subject matter):
{materia_options}

ASSE B - NATURA (legal nature):
{natura_options}

ASSE C - AMBITO (solo se NATURA=PROCESSUALE):
{ambito_options}

REGOLE:
1. Scegli ESATTAMENTE una opzione per Materia e Natura
2. Per Ambito: scegli solo se Natura=PROCESSUALE, altrimenti lascia null
3. Non inventare etichette
4. Se incerto, scegli la migliore opzione ma abbassa confidence

OUTPUT JSON (SOLO JSON, nessun altro testo):
{{
  "materia": "UNA delle opzioni materia",
  "natura": "SOSTANZIALE o PROCESSUALE",
  "ambito": "opzione ambito o null",
  "confidence": 0.0-1.0,
  "rationale": "breve spiegazione (max 100 parole)"
}}
"""

JUDGE_PROMPT = """Sei un arbitro esperto di diritto italiano. Due classificatori sono in disaccordo.

MASSIMA:
{testo}

METADATA:
- Sezione: {sezione}
- Tipo: {tipo}
- Norme citate: {norme}

CLASSIFICATORE A ({model_a}):
Materia: {materia_a}
Natura: {natura_a}
Ambito: {ambito_a}
Rationale: {rationale_a}

CLASSIFICATORE B ({model_b}):
Materia: {materia_b}
Natura: {natura_b}
Ambito: {ambito_b}
Rationale: {rationale_b}

TASK: Decidi quale classificazione è corretta per ogni asse in disaccordo.

OUTPUT JSON (SOLO JSON, nessun altro testo):
{{
  "materia": "la materia corretta",
  "natura": "la natura corretta",
  "ambito": "ambito corretto o null",
  "winner": "A o B o MIXED",
  "rationale": "perché hai scelto così (max 100 parole)"
}}
"""


def format_options(options: List[str]) -> str:
    """Format options list for prompt."""
    return "\n".join(f"- {opt}" for opt in options)


async def call_openrouter(
    client: httpx.AsyncClient,
    api_key: str,
    model: str,
    prompt: str,
    max_retries: int = 3,
) -> Tuple[Optional[Dict], str]:
    """Call OpenRouter API and parse JSON response."""

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://lexe-api.local",
        "X-Title": "LEXE Category Labeling",
    }

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "max_tokens": 500,
    }

    for attempt in range(max_retries):
        try:
            response = await client.post(
                OPENROUTER_URL,
                headers=headers,
                json=payload,
                timeout=60.0,
            )
            response.raise_for_status()

            data = response.json()
            content = data["choices"][0]["message"]["content"]

            # Extract JSON from response
            # Handle markdown code blocks
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]

            parsed = json.loads(content.strip())

            # Handle case where model returns a list instead of dict
            if isinstance(parsed, list):
                if len(parsed) > 0 and isinstance(parsed[0], dict):
                    parsed = parsed[0]
                else:
                    return None, f"Unexpected list response: {content[:200]}"

            return parsed, content

        except json.JSONDecodeError as e:
            if attempt < max_retries - 1:
                await asyncio.sleep(1)
                continue
            return None, f"JSON parse error: {e}"

        except httpx.HTTPStatusError as e:
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)
                continue
            return None, f"HTTP error: {e}"

        except Exception as e:
            if attempt < max_retries - 1:
                await asyncio.sleep(1)
                continue
            return None, f"Error: {e}"

    return None, "Max retries exceeded"


async def label_with_model(
    client: httpx.AsyncClient,
    api_key: str,
    model: str,
    testo: str,
    sezione: Optional[str],
    tipo: Optional[str],
    norme: List[str],
) -> Optional[LabelResult]:
    """Label a massima with a single model."""

    prompt = LABELING_PROMPT.format(
        testo=testo[:2000],  # Truncate if too long
        sezione=sezione or "N/A",
        tipo=tipo or "N/A",
        norme=", ".join(norme) if norme else "nessuna",
        materia_options=format_options(MATERIE),
        natura_options=format_options(NATURE),
        ambito_options=format_options(AMBITI),
    )

    parsed, raw = await call_openrouter(client, api_key, model, prompt)

    if parsed is None:
        print(f"    [{model}] Failed: {raw}")
        return None

    # Validate and normalize response
    materia = parsed.get("materia", "").upper()
    natura = parsed.get("natura", "").upper()
    ambito = parsed.get("ambito")
    confidence = float(parsed.get("confidence", 0.5))
    rationale = parsed.get("rationale", "")

    # Validate materia
    if materia not in MATERIE:
        print(f"    [{model}] Invalid materia: {materia}")
        return None

    # Validate natura
    if natura not in NATURE:
        print(f"    [{model}] Invalid natura: {natura}")
        return None

    # Validate ambito (only if processuale)
    if natura == "PROCESSUALE":
        if ambito:
            ambito = ambito.upper()
            if ambito not in AMBITI:
                ambito = None  # Invalid, set to None for classifier
    else:
        ambito = None  # Must be None if SOSTANZIALE

    return LabelResult(
        model=model,
        materia=materia,
        natura=natura,
        ambito=ambito,
        confidence=confidence,
        rationale=rationale,
        raw_response=raw,
    )


async def adjudicate_with_judge(
    client: httpx.AsyncClient,
    api_key: str,
    testo: str,
    sezione: Optional[str],
    tipo: Optional[str],
    norme: List[str],
    result_a: LabelResult,
    result_b: LabelResult,
) -> AdjudicatedResult:
    """Call judge model to adjudicate disagreement."""

    prompt = JUDGE_PROMPT.format(
        testo=testo[:2000],
        sezione=sezione or "N/A",
        tipo=tipo or "N/A",
        norme=", ".join(norme) if norme else "nessuna",
        model_a=result_a.model,
        materia_a=result_a.materia,
        natura_a=result_a.natura,
        ambito_a=result_a.ambito or "null",
        rationale_a=result_a.rationale,
        model_b=result_b.model,
        materia_b=result_b.materia,
        natura_b=result_b.natura,
        ambito_b=result_b.ambito or "null",
        rationale_b=result_b.rationale,
    )

    parsed, _ = await call_openrouter(client, api_key, JUDGE_MODEL, prompt)

    if parsed is None:
        # Fallback to labeler A
        return AdjudicatedResult(
            materia=result_a.materia,
            natura=result_a.natura,
            ambito=result_a.ambito,
            agreement_score=0.5,
        )

    # Use judge's decision
    materia = parsed.get("materia", result_a.materia).upper()
    natura = parsed.get("natura", result_a.natura).upper()
    ambito = parsed.get("ambito")

    if materia not in MATERIE:
        materia = result_a.materia
    if natura not in NATURE:
        natura = result_a.natura
    if natura == "PROCESSUALE" and ambito:
        ambito = ambito.upper() if ambito.upper() in AMBITI else None
    else:
        ambito = None

    return AdjudicatedResult(
        materia=materia,
        natura=natura,
        ambito=ambito,
        agreement_score=0.5,  # Judge decided
    )


def compute_agreement(
    result_a: LabelResult,
    result_b: LabelResult,
) -> Tuple[bool, bool, bool, float]:
    """
    Compare two label results.
    Returns: (materia_agree, natura_agree, ambito_agree, score)
    """
    materia_agree = result_a.materia == result_b.materia
    natura_agree = result_a.natura == result_b.natura
    ambito_agree = result_a.ambito == result_b.ambito

    if materia_agree and natura_agree and ambito_agree:
        score = 1.0
    elif materia_agree and natura_agree:
        score = 0.85
    elif materia_agree or natura_agree:
        score = 0.7
    else:
        score = 0.5

    return materia_agree, natura_agree, ambito_agree, score


async def label_massima(
    client: httpx.AsyncClient,
    api_key: str,
    massima_id: UUID,
    testo: str,
    sezione: Optional[str],
    tipo: Optional[str],
    norme: List[str],
) -> Tuple[Optional[LabelResult], Optional[LabelResult], AdjudicatedResult]:
    """
    Label a single massima with both models and adjudicate if needed.
    Returns: (result_a, result_b, adjudicated)
    """

    # Label with both models
    result_a = await label_with_model(
        client, api_key, LABELER_A, testo, sezione, tipo, norme
    )
    result_b = await label_with_model(
        client, api_key, LABELER_B, testo, sezione, tipo, norme
    )

    # Handle failures
    if result_a is None and result_b is None:
        # Both failed - skip
        return None, None, None

    if result_a is None:
        # Only B succeeded
        return None, result_b, AdjudicatedResult(
            materia=result_b.materia,
            natura=result_b.natura,
            ambito=result_b.ambito,
            agreement_score=0.7,
        )

    if result_b is None:
        # Only A succeeded
        return result_a, None, AdjudicatedResult(
            materia=result_a.materia,
            natura=result_a.natura,
            ambito=result_a.ambito,
            agreement_score=0.7,
        )

    # Both succeeded - check agreement
    materia_agree, natura_agree, ambito_agree, score = compute_agreement(result_a, result_b)

    if score == 1.0:
        # Full agreement
        return result_a, result_b, AdjudicatedResult(
            materia=result_a.materia,
            natura=result_a.natura,
            ambito=result_a.ambito,
            agreement_score=1.0,
        )

    # Disagreement - call judge
    print(f"    Disagreement: A={result_a.materia}/{result_a.natura} vs B={result_b.materia}/{result_b.natura}")
    adjudicated = await adjudicate_with_judge(
        client, api_key, testo, sezione, tipo, norme, result_a, result_b
    )

    return result_a, result_b, adjudicated


async def main(dry_run: bool = True, resume: bool = False):
    """Main labeling routine."""

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: Set OPENROUTER_API_KEY environment variable")
        sys.exit(1)

    settings = KBSettings()
    conn = await asyncpg.connect(settings.kb_database_url)

    try:
        # Fetch samples to label
        if resume:
            # Only get samples not yet labeled
            rows = await conn.fetch("""
                SELECT
                    ga.massima_id,
                    ga.difficulty_bucket,
                    ga.split,
                    mf.testo_trunc,
                    mf.sezione,
                    mf.tipo,
                    mf.norms_canonical
                FROM kb.golden_category_adjudicated_v2 ga
                JOIN kb.massime_features_v2 mf ON mf.massima_id = ga.massima_id
                WHERE ga.materia_l1 = 'PENDING'
                ORDER BY ga.split, ga.massima_id
            """)
        else:
            rows = await conn.fetch("""
                SELECT
                    ga.massima_id,
                    ga.difficulty_bucket,
                    ga.split,
                    mf.testo_trunc,
                    mf.sezione,
                    mf.tipo,
                    mf.norms_canonical
                FROM kb.golden_category_adjudicated_v2 ga
                JOIN kb.massime_features_v2 mf ON mf.massima_id = ga.massima_id
                ORDER BY ga.split, ga.massima_id
            """)

        if not rows:
            print("No samples to label. Run generate_golden_v2.py first.")
            return

        print(f"Labeling {len(rows)} samples...")
        print(f"  Dry run: {dry_run}")
        print(f"  Resume: {resume}")
        print(f"  Models: {LABELER_A}, {LABELER_B}")
        print(f"  Judge: {JUDGE_MODEL}")

        stats = {
            "total": 0,
            "success": 0,
            "failed": 0,
            "full_agree": 0,
            "partial_agree": 0,
            "judge_decided": 0,
        }

        async with httpx.AsyncClient() as client:
            for i, row in enumerate(rows):
                massima_id = row["massima_id"]
                testo = row["testo_trunc"] or ""
                sezione = row["sezione"]
                tipo = row["tipo"]
                norme = row["norms_canonical"] or []
                bucket = row["difficulty_bucket"]
                split = row["split"]

                print(f"\n[{i+1}/{len(rows)}] {split}/{bucket} {str(massima_id)[:8]}...")

                result_a, result_b, adjudicated = await label_massima(
                    client, api_key, massima_id, testo, sezione, tipo, norme
                )

                stats["total"] += 1

                if adjudicated is None:
                    print("    FAILED: Both models failed")
                    stats["failed"] += 1
                    continue

                stats["success"] += 1

                if adjudicated.agreement_score == 1.0:
                    stats["full_agree"] += 1
                elif adjudicated.agreement_score >= 0.7:
                    stats["partial_agree"] += 1
                else:
                    stats["judge_decided"] += 1

                print(f"    Result: {adjudicated.materia}/{adjudicated.natura}/{adjudicated.ambito} (score={adjudicated.agreement_score})")

                if not dry_run:
                    # Store individual label results
                    if result_a:
                        await conn.execute("""
                            INSERT INTO kb.golden_category_labels_v2
                                (massima_id, labeler_model, materia_l1, natura_l1, ambito_l1,
                                 confidence, rationale)
                            VALUES ($1, $2, $3, $4, $5, $6, $7)
                        """, massima_id, LABELER_A, result_a.materia, result_a.natura,
                            result_a.ambito, result_a.confidence, result_a.rationale)

                    if result_b:
                        await conn.execute("""
                            INSERT INTO kb.golden_category_labels_v2
                                (massima_id, labeler_model, materia_l1, natura_l1, ambito_l1,
                                 confidence, rationale)
                            VALUES ($1, $2, $3, $4, $5, $6, $7)
                        """, massima_id, LABELER_B, result_b.materia, result_b.natura,
                            result_b.ambito, result_b.confidence, result_b.rationale)

                    # Update adjudicated result
                    await conn.execute("""
                        UPDATE kb.golden_category_adjudicated_v2
                        SET materia_l1 = $2,
                            natura_l1 = $3,
                            ambito_l1 = $4,
                            agreement_score = $5
                        WHERE massima_id = $1
                    """, massima_id, adjudicated.materia, adjudicated.natura,
                        adjudicated.ambito, adjudicated.agreement_score)

                # Rate limiting
                await asyncio.sleep(0.5)

        # Print stats
        print("\n" + "=" * 60)
        print("LABELING SUMMARY")
        print("=" * 60)
        print(f"  Total: {stats['total']}")
        print(f"  Success: {stats['success']}")
        print(f"  Failed: {stats['failed']}")
        print(f"  Full agreement: {stats['full_agree']}")
        print(f"  Partial agreement: {stats['partial_agree']}")
        print(f"  Judge decided: {stats['judge_decided']}")

        if stats["success"] > 0:
            agree_rate = (stats["full_agree"] + stats["partial_agree"]) / stats["success"]
            print(f"  Agreement rate: {agree_rate:.1%}")

    finally:
        await conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Label Golden Set v2")
    parser.add_argument("--dry-run", action="store_true", help="Don't commit to database")
    parser.add_argument("--commit", action="store_true", help="Commit to database")
    parser.add_argument("--resume", action="store_true", help="Resume from where left off")
    args = parser.parse_args()

    if not args.commit and not args.dry_run:
        print("Specify --dry-run or --commit")
        sys.exit(1)

    dry_run = not args.commit
    asyncio.run(main(dry_run=dry_run, resume=args.resume))
