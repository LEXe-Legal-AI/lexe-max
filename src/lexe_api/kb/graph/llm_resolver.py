"""
LLM Resolver Ensemble for Category Graph v2.5

Architettura:
- Due classifier in parallelo (Gemini Flash + Qwen)
- Un giudice (Mistral Large) per risolvere disagreement
"""
import asyncio
import json
from dataclasses import dataclass
from typing import List, Optional, Set, Tuple

import httpx

from .config import DEFAULT_THRESHOLDS, ClassificationThresholds


@dataclass
class LLMResolverResult:
    """Risultato del LLM resolver ensemble."""

    label: str
    confidence: float
    rationale: str
    accepted: bool
    # Dettagli ensemble
    model_a_label: str
    model_a_confidence: float
    model_b_label: str
    model_b_confidence: float
    agreement: bool  # True se A == B
    judge_called: bool
    judge_label: Optional[str] = None


@dataclass
class LLMResolverInput:
    """Input chirurgico per LLM resolver."""

    testo: str
    sezione: Optional[str]
    norme: List[str]
    candidate_set: Set[str]
    fallback_reason: str  # "delta_basso", "conf_bassa", etc.


LLM_CLASSIFIER_PROMPT = """Classifica questa massima della Corte di Cassazione italiana.

MASSIMA:
{testo}

METADATA:
- Sezione: {sezione}
- Norme citate: {norme}

OPZIONI (scegli UNA):
{candidate_descriptions}

Rispondi SOLO con JSON valido:
{{"label": "OPZIONE", "confidence": 0.0-1.0, "rationale": "breve spiegazione"}}
"""

LLM_JUDGE_PROMPT = """Due classificatori hanno dato risposte diverse per questa massima.

MASSIMA:
{testo}

METADATA:
- Sezione: {sezione}
- Norme: {norme}

CLASSIFICATORE A dice: {label_a} (confidence: {conf_a:.2f})
Motivazione A: {rationale_a}

CLASSIFICATORE B dice: {label_b} (confidence: {conf_b:.2f})
Motivazione B: {rationale_b}

OPZIONI VALIDE: {options}

Chi ha ragione? Rispondi SOLO con JSON valido:
{{"label": "OPZIONE_CORRETTA", "confidence": 0.0-1.0, "rationale": "breve spiegazione"}}
"""

MATERIA_DESCRIPTIONS = {
    "CIVILE": "Obbligazioni, contratti, proprietà, famiglia, successioni",
    "PENALE": "Reati, pene, circostanze, procedura penale",
    "LAVORO": "Rapporto di lavoro, licenziamento, previdenza, INPS",
    "TRIBUTARIO": "Imposte, accertamento, contenzioso tributario",
    "AMMINISTRATIVO": "PA, appalti, urbanistica, sanzioni amministrative",
    "CRISI": "Fallimento, concordato, procedure concorsuali",
}


def _parse_llm_response(content: str) -> dict:
    """Parse JSON response from LLM, handling markdown code blocks."""
    content = content.strip()

    # Handle markdown code blocks
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0]
    elif "```" in content:
        parts = content.split("```")
        if len(parts) >= 2:
            content = parts[1]

    # Try to find JSON object
    content = content.strip()
    if not content.startswith("{"):
        # Try to find JSON in the response
        start = content.find("{")
        end = content.rfind("}") + 1
        if start != -1 and end > start:
            content = content[start:end]

    return json.loads(content)


async def _call_single_llm(
    client: httpx.AsyncClient,
    api_key: str,
    model: str,
    prompt: str,
    candidate_set: Set[str],
) -> Tuple[str, float, str]:
    """
    Chiama singolo LLM e parse risposta.

    Returns:
        (label, confidence, rationale)
    """
    try:
        response = await client.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
                "max_tokens": 300,
            },
            timeout=45.0,
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]

        result = _parse_llm_response(content)
        label = result.get("label", "").upper().strip()
        confidence = float(result.get("confidence", 0.5))
        rationale = result.get("rationale", "")

        # Clamp confidence
        confidence = max(0.0, min(1.0, confidence))

        # Validate label is in candidate set
        if label not in candidate_set:
            # Try fuzzy match
            for candidate in candidate_set:
                if candidate in label or label in candidate:
                    label = candidate
                    break
            else:
                # Fallback to first candidate
                label = sorted(candidate_set)[0]
                confidence = 0.3
                rationale = f"Invalid label returned, fallback to {label}"

        return label, confidence, rationale

    except json.JSONDecodeError as e:
        return sorted(candidate_set)[0], 0.3, f"JSON parse error: {str(e)}"
    except httpx.TimeoutException:
        return sorted(candidate_set)[0], 0.3, "LLM timeout"
    except Exception as e:
        return sorted(candidate_set)[0], 0.3, f"LLM error: {str(e)}"


async def call_llm_resolver_ensemble(
    input: LLMResolverInput,
    client: httpx.AsyncClient,
    api_key: str,
    thresholds: ClassificationThresholds = DEFAULT_THRESHOLDS,
) -> LLMResolverResult:
    """
    Ensemble resolver: 2 LLM in parallelo + giudice se disagreement.

    Args:
        input: Input with text, metadata, and candidate set
        client: httpx async client
        api_key: OpenRouter API key
        thresholds: Classification thresholds

    Returns:
        LLMResolverResult with final label and ensemble details
    """
    # Build candidate descriptions
    candidate_desc = "\n".join(
        f"- {m}: {MATERIA_DESCRIPTIONS.get(m, 'N/A')}" for m in sorted(input.candidate_set)
    )

    classifier_prompt = LLM_CLASSIFIER_PROMPT.format(
        testo=input.testo[:1500],  # Truncate for cost
        sezione=input.sezione or "N/A",
        norme=", ".join(input.norme[:10]) if input.norme else "Nessuna",
        candidate_descriptions=candidate_desc,
    )

    # Step 1: Call both classifiers in parallel
    results = await asyncio.gather(
        _call_single_llm(
            client, api_key, thresholds.LLM_MODEL_A, classifier_prompt, input.candidate_set
        ),
        _call_single_llm(
            client, api_key, thresholds.LLM_MODEL_B, classifier_prompt, input.candidate_set
        ),
        return_exceptions=True,
    )

    # Handle exceptions
    if isinstance(results[0], Exception):
        label_a, conf_a, rationale_a = sorted(input.candidate_set)[0], 0.3, f"Error: {results[0]}"
    else:
        label_a, conf_a, rationale_a = results[0]

    if isinstance(results[1], Exception):
        label_b, conf_b, rationale_b = sorted(input.candidate_set)[0], 0.3, f"Error: {results[1]}"
    else:
        label_b, conf_b, rationale_b = results[1]

    # Step 2: Check agreement
    agreement = label_a == label_b

    if agreement:
        # Both agree: high confidence
        final_label = label_a
        final_conf = max(conf_a, conf_b) * 1.1  # Bonus for agreement
        final_conf = min(final_conf, 0.98)
        final_rationale = f"Agreement: {rationale_a}"
        judge_called = False
        judge_label = None
    else:
        # Disagreement: call the judge
        judge_prompt = LLM_JUDGE_PROMPT.format(
            testo=input.testo[:1000],
            sezione=input.sezione or "N/A",
            norme=", ".join(input.norme[:5]) if input.norme else "Nessuna",
            label_a=label_a,
            conf_a=conf_a,
            rationale_a=rationale_a,
            label_b=label_b,
            conf_b=conf_b,
            rationale_b=rationale_b,
            options=", ".join(sorted(input.candidate_set)),
        )

        judge_label, judge_conf, judge_rationale = await _call_single_llm(
            client, api_key, thresholds.LLM_JUDGE_MODEL, judge_prompt, input.candidate_set
        )

        final_label = judge_label
        final_conf = judge_conf * 0.9  # Penalty for disagreement
        final_rationale = f"Judge resolved: {judge_rationale}"
        judge_called = True

    accepted = final_conf >= thresholds.TH_LLM_ACCEPT

    return LLMResolverResult(
        label=final_label,
        confidence=final_conf,
        rationale=final_rationale,
        accepted=accepted,
        model_a_label=label_a,
        model_a_confidence=conf_a,
        model_b_label=label_b,
        model_b_confidence=conf_b,
        agreement=agreement,
        judge_called=judge_called,
        judge_label=judge_label,
    )
