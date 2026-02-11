"""
Category Classifier v2.5

Full classification pipeline for three-axis taxonomy:
1. Materia rules -> centroids -> LLM resolver (ensemble)
2. Natura centroids -> LLM resolver
3. Ambito rules -> centroids (only if PROCESSUALE)

v2.5 adds:
- Isotonic calibration for confidence scores
- Ensemble LLM resolver (2 classifiers + judge)
- Structured logging for all decisions
- Configurable thresholds for routing

Uses embeddings for centroid classification and OpenRouter for LLM resolver.
"""

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from uuid import UUID

import httpx
import numpy as np

from .ambito_rules import (
    derive_ambito_rule_first,
)
from .calibration import get_calibrator
from .classification_logger import ClassificationLog
from .config import DEFAULT_THRESHOLDS, ClassificationThresholds
from .llm_resolver import LLMResolverInput, call_llm_resolver_ensemble
from .materia_rules import (
    derive_materia_rule_first,
)


@dataclass
class ClassificationResult:
    """Result of classifying a single massima."""

    massima_id: UUID

    # Axis A: Materia
    materia_l1: str
    materia_confidence: float
    materia_rule: str
    materia_candidate_set: list[str]
    materia_reasons: list[str]

    # Axis B: Natura
    natura_l1: str
    natura_confidence: float
    natura_rule: str

    # Axis C: Ambito (only if PROCESSUALE)
    ambito_l1: str | None = None
    ambito_confidence: float | None = None
    ambito_rule: str | None = None

    # Topic L2
    topic_l2: str | None = None
    topic_l2_confidence: float | None = None
    topic_l2_flag: str | None = None
    abstain_reason: str | None = None

    # Composite
    composite_confidence: float = 0.0
    norms_count: int = 0


@dataclass
class CentroidCache:
    """Cache for centroid embeddings."""

    materia_centroids: dict[str, np.ndarray] = field(default_factory=dict)
    natura_centroids: dict[str, np.ndarray] = field(default_factory=dict)
    ambito_centroids: dict[str, np.ndarray] = field(default_factory=dict)
    is_loaded: bool = False


# Global centroid cache
_centroid_cache = CentroidCache()


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
LLM_RESOLVER_MODEL = "qwen/qwen3-235b-a22b-2507"

LLM_RESOLVER_PROMPT = """Classifica questa massima scegliendo SOLO tra le opzioni fornite.

MASSIMA:
{testo}

METADATA:
Sezione: {sezione}
Tipo: {tipo}
Norme: {norme}

TASK: {task_name}

OPZIONI:
{options}

OUTPUT JSON (SOLO JSON):
{{
  "choice": "UNA delle opzioni",
  "confidence": 0.0-1.0
}}

Regole:
- Non inventare etichette.
- Se sei incerto, scegli comunque la migliore opzione, ma abbassa confidence.
"""


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


async def call_llm_resolver(
    client: httpx.AsyncClient,
    api_key: str,
    testo: str,
    sezione: str | None,
    tipo: str | None,
    norme: list[str],
    task_name: str,
    options: list[str],
) -> tuple[str | None, float]:
    """
    Call LLM resolver for uncertain classifications.
    Returns: (choice, confidence)
    """
    if not api_key:
        # No API key - return None to trigger fallback
        return None, 0.0

    prompt = LLM_RESOLVER_PROMPT.format(
        testo=testo[:1500],
        sezione=sezione or "N/A",
        tipo=tipo or "N/A",
        norme=", ".join(norme[:10]) if norme else "nessuna",
        task_name=task_name,
        options="\n".join(f"- {opt}" for opt in options),
    )

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": LLM_RESOLVER_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "max_tokens": 200,
    }

    try:
        response = await client.post(
            OPENROUTER_URL,
            headers=headers,
            json=payload,
            timeout=30.0,
        )
        response.raise_for_status()

        data = response.json()
        content = data["choices"][0]["message"]["content"]

        # Extract JSON
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]

        parsed = json.loads(content.strip())
        choice = parsed.get("choice", "").upper()
        confidence = float(parsed.get("confidence", 0.5))

        # Validate choice
        if choice not in [o.upper() for o in options]:
            return None, 0.0

        return choice, confidence

    except Exception as e:
        print(f"    LLM resolver error: {e}")
        return None, 0.0


def classify_with_centroids(
    embedding: np.ndarray,
    centroids: dict[str, np.ndarray],
    candidate_set: set[str] | None = None,
) -> tuple[str, float, float]:
    """
    Classify using centroid similarity.
    Returns: (best_category, score, delta_to_second)
    """
    scores = {}
    for cat, centroid in centroids.items():
        if candidate_set and cat not in candidate_set:
            continue
        scores[cat] = cosine_similarity(embedding, centroid)

    if not scores:
        return None, 0.0, 0.0

    sorted_scores = sorted(scores.items(), key=lambda x: -x[1])
    best = sorted_scores[0]
    second = sorted_scores[1] if len(sorted_scores) > 1 else (None, 0.0)

    delta = best[1] - second[1]
    return best[0], best[1], delta


def keyword_score(text_lower: str, keywords: set[str]) -> float:
    """Compute keyword match score."""
    if not keywords:
        return 0.0
    matches = sum(1 for kw in keywords if kw.lower() in text_lower)
    return matches / len(keywords)


async def classify_materia(
    embedding: np.ndarray,
    tipo: str | None,
    sezione: str | None,
    norms: list[str],
    testo_lower: str,
    client: httpx.AsyncClient | None = None,
    api_key: str | None = None,
) -> tuple[str, float, str, list[str], list[str]]:
    """
    Classify materia using rules -> centroids -> LLM resolver.
    Returns: (materia, confidence, rule, candidate_set, reasons)
    """
    # Step 1: Try rule-based derivation
    materia, conf, rule, candidates, reasons = derive_materia_rule_first(
        tipo, sezione, norms, testo_lower
    )

    if materia is not None:
        # Rule fired - deterministic result
        return materia, conf, rule, list(candidates), reasons

    # Step 2: Centroid classification within candidate set
    if _centroid_cache.is_loaded and _centroid_cache.materia_centroids:
        best, score, delta = classify_with_centroids(
            embedding,
            _centroid_cache.materia_centroids,
            candidates,
        )

        if best and delta >= 0.15:
            # High confidence from centroids
            return best, min(0.85, score), "centroid_confident", list(candidates), reasons

        if best and delta >= 0.08:
            # Medium confidence
            return best, min(0.75, score), "centroid_medium", list(candidates), reasons

    # Step 3: LLM resolver for uncertain cases
    if client and api_key and len(candidates) <= 4:
        choice, llm_conf = await call_llm_resolver(
            client,
            api_key,
            testo_lower[:1500],
            sezione,
            tipo,
            norms,
            "Scegli la MATERIA (area giuridica)",
            list(candidates),
        )
        if choice and choice in candidates:
            # Blend centroid and LLM confidence
            final_conf = 0.6 * score + 0.4 * llm_conf if score > 0 else llm_conf * 0.8
            return choice, final_conf, "llm_resolver", list(candidates), reasons

    # Step 4: Fallback to best centroid or CIVILE
    if _centroid_cache.is_loaded and best:
        return best, min(0.60, score), "centroid_fallback", list(candidates), reasons

    # Ultimate fallback
    return "CIVILE", 0.50, "fallback_civile", list(candidates), reasons


async def classify_natura(
    embedding: np.ndarray,
    testo_lower: str,
    client: httpx.AsyncClient | None = None,
    api_key: str | None = None,
) -> tuple[str, float, str]:
    """
    Classify natura using centroids -> LLM resolver.
    Returns: (natura, confidence, rule)
    """
    # Keyword hints
    processuale_keywords = {
        "competenza",
        "ammissibilità",
        "termine",
        "decadenza",
        "notifica",
        "impugnazione",
        "ricorso",
        "appello",
        "cassazione",
        "preclusione",
        "procedimento",
        "nullità",
        "inammissibile",
        "improcedibile",
    }
    sostanziale_keywords = {
        "responsabilità",
        "danno",
        "risarcimento",
        "inadempimento",
        "contratto",
        "proprietà",
        "diritto",
        "obbligo",
        "colpa",
        "dolo",
    }

    proc_score = keyword_score(testo_lower, processuale_keywords)
    sost_score = keyword_score(testo_lower, sostanziale_keywords)

    # Centroid classification
    if _centroid_cache.is_loaded and _centroid_cache.natura_centroids:
        best, score, delta = classify_with_centroids(
            embedding,
            _centroid_cache.natura_centroids,
        )

        if best and delta >= 0.08:
            return best, min(0.90, score), "centroid_confident"

    # Keyword-based decision
    if proc_score > sost_score + 0.1:
        return "PROCESSUALE", 0.75 + proc_score * 0.2, "keywords_processuale"
    elif sost_score > proc_score + 0.1:
        return "SOSTANZIALE", 0.75 + sost_score * 0.2, "keywords_sostanziale"

    # LLM resolver for uncertain
    if client and api_key:
        choice, llm_conf = await call_llm_resolver(
            client,
            api_key,
            testo_lower[:1500],
            None,
            None,
            [],
            "Scegli la NATURA (sostanziale o processuale)",
            ["SOSTANZIALE", "PROCESSUALE"],
        )
        if choice in {"SOSTANZIALE", "PROCESSUALE"}:
            return choice, llm_conf * 0.85, "llm_resolver"

    # Fallback to centroid best or keyword
    if _centroid_cache.is_loaded and best:
        return best, min(0.65, score), "centroid_fallback"

    return "SOSTANZIALE", 0.55, "fallback_sostanziale"


async def classify_ambito(
    embedding: np.ndarray,
    norms: list[str],
    testo_lower: str,
    client: httpx.AsyncClient | None = None,
    api_key: str | None = None,
) -> tuple[str | None, float | None, str | None]:
    """
    Classify ambito using rules -> centroids.
    Only called when natura=PROCESSUALE.
    Returns: (ambito, confidence, rule) or (None, None, None)
    """
    # Step 1: Try high-precision rules
    ambito, conf, rule, candidates, reasons = derive_ambito_rule_first(norms, testo_lower)

    if ambito is not None:
        return ambito, conf, rule

    # Step 2: Centroid classification
    if _centroid_cache.is_loaded and _centroid_cache.ambito_centroids:
        # Exclude UNKNOWN from centroids
        valid_centroids = {
            k: v for k, v in _centroid_cache.ambito_centroids.items() if k != "UNKNOWN"
        }
        best, score, delta = classify_with_centroids(
            embedding,
            valid_centroids,
            candidates - {"UNKNOWN"} if candidates else None,
        )

        if best and delta >= 0.10:
            return best, min(0.85, score), "centroid_confident"

        if best and score >= 0.5:
            return best, min(0.70, score), "centroid_weak"

    # Step 3: LLM resolver if very uncertain
    if client and api_key and len(candidates) <= 4:
        choice, llm_conf = await call_llm_resolver(
            client,
            api_key,
            testo_lower[:1500],
            None,
            None,
            norms,
            "Scegli l'AMBITO processuale",
            [c for c in candidates if c != "UNKNOWN"],
        )
        if choice and choice in candidates:
            return choice, llm_conf * 0.80, "llm_resolver"

    # Fallback to UNKNOWN
    return "UNKNOWN", 0.50, "fallback_unknown"


def compute_composite_confidence(
    materia_conf: float,
    natura_conf: float,
    ambito_conf: float | None,
    materia_rule: str,
    norms_count: int,
    sezione: str | None,
) -> float:
    """
    Compute composite confidence for inference.
    """
    # Base: weighted average of axis confidences
    if ambito_conf is not None:
        base = 0.50 * materia_conf + 0.30 * natura_conf + 0.20 * ambito_conf
    else:
        base = 0.60 * materia_conf + 0.40 * natura_conf

    # Adjustments
    if "rule" in materia_rule or "tipo" in materia_rule:
        base += 0.05  # Rule-based materia is more reliable

    if norms_count == 0:
        base -= 0.03  # No norms = more uncertainty

    sez = (sezione or "").lower()
    # In DB, sezione is just "U" for Sezioni Unite
    if sez == "u" or "sez. u" in sez:
        base -= 0.05  # Sezioni Unite = cross-domain uncertainty

    return max(0.0, min(1.0, base))


async def classify_massima(
    massima_id: UUID,
    embedding: np.ndarray,
    tipo: str | None,
    sezione: str | None,
    norms: list[str],
    testo: str,
    norms_count: int,
    client: httpx.AsyncClient | None = None,
    api_key: str | None = None,
) -> ClassificationResult:
    """
    Full classification pipeline for a single massima.
    """
    testo_lower = testo.lower()

    # Classify materia
    materia, materia_conf, materia_rule, candidates, reasons = await classify_materia(
        embedding, tipo, sezione, norms, testo_lower, client, api_key
    )

    # Classify natura
    natura, natura_conf, natura_rule = await classify_natura(
        embedding, testo_lower, client, api_key
    )

    # Classify ambito (only if PROCESSUALE)
    ambito = None
    ambito_conf = None
    ambito_rule = None

    if natura == "PROCESSUALE":
        ambito, ambito_conf, ambito_rule = await classify_ambito(
            embedding, norms, testo_lower, client, api_key
        )

    # Compute composite confidence
    composite = compute_composite_confidence(
        materia_conf, natura_conf, ambito_conf, materia_rule, norms_count, sezione
    )

    return ClassificationResult(
        massima_id=massima_id,
        materia_l1=materia,
        materia_confidence=round(materia_conf, 4),
        materia_rule=materia_rule,
        materia_candidate_set=candidates,
        materia_reasons=reasons,
        natura_l1=natura,
        natura_confidence=round(natura_conf, 4),
        natura_rule=natura_rule,
        ambito_l1=ambito,
        ambito_confidence=round(ambito_conf, 4) if ambito_conf else None,
        ambito_rule=ambito_rule,
        composite_confidence=round(composite, 4),
        norms_count=norms_count,
    )


async def load_centroids_from_db(conn) -> bool:
    """
    Load centroid embeddings from database.
    Returns True if successfully loaded.
    """
    global _centroid_cache

    # TODO: Implement centroid loading from golden set embeddings
    # For now, return False to indicate centroids not available
    # Centroids will be computed from golden set after labeling

    _centroid_cache.is_loaded = False
    return False


def set_centroids(
    materia: dict[str, np.ndarray],
    natura: dict[str, np.ndarray],
    ambito: dict[str, np.ndarray],
):
    """Set centroid embeddings directly (for testing or batch processing)."""
    global _centroid_cache
    _centroid_cache.materia_centroids = materia
    _centroid_cache.natura_centroids = natura
    _centroid_cache.ambito_centroids = ambito
    _centroid_cache.is_loaded = True


# =============================================================================
# Category Classifier v2.5 - New Pipeline with Calibration + Ensemble LLM
# =============================================================================


async def classify_massima_v25(
    massima_id: UUID,
    embedding: np.ndarray,
    tipo: str | None,
    sezione: str | None,
    norms: list[str],
    testo: str,
    norms_count: int,
    client: httpx.AsyncClient | None = None,
    api_key: str | None = None,
    thresholds: ClassificationThresholds = DEFAULT_THRESHOLDS,
) -> tuple[ClassificationResult, ClassificationLog]:
    """
    Pipeline v2.5 with calibration + ensemble LLM resolver.

    Args:
        massima_id: UUID of the massima
        embedding: 1536-dim embedding vector
        tipo: Document type (civile/penale)
        sezione: Court section (L, U, 1-6)
        norms: List of norm references
        testo: Full text of the massima
        norms_count: Number of norms
        client: httpx async client for LLM calls
        api_key: OpenRouter API key
        thresholds: Classification thresholds

    Returns:
        (ClassificationResult, ClassificationLog)
    """
    start_time = time.time()
    calibrator = get_calibrator()
    testo_lower = testo.lower() if testo else ""

    # Step 1: Rule-based candidate reduction
    materia, rule_conf, rule_name, candidates, reasons = derive_materia_rule_first(
        tipo, sezione, norms, testo_lower
    )

    norm_hint_applied = "norm_hint" in rule_name

    log = ClassificationLog(
        massima_id=massima_id,
        timestamp=datetime.now().isoformat(),
        sezione=sezione,
        norm_hint_applied=norm_hint_applied,
        norm_hint_reason=reasons[0] if reasons else None,
    )

    # Step 2: If rule produced singleton, use it
    if materia is not None:
        conf_calibrated = calibrator.calibrate(rule_conf)
        log.top3_labels_raw = [materia]
        log.top3_scores_raw = [rule_conf]
        log.top3_scores_calibrated = [conf_calibrated]
        log.delta_12 = 1.0  # Singleton = max delta
        log.routing_decision = "TOP1"
        log.latency_ms_l1 = (time.time() - start_time) * 1000
        log.final_label = materia
        log.final_confidence = conf_calibrated

        # Classify natura and ambito
        natura, natura_conf, natura_rule = await classify_natura(
            embedding, testo_lower, client, api_key
        )
        ambito, ambito_conf, ambito_rule = None, None, None
        if natura == "PROCESSUALE":
            ambito, ambito_conf, ambito_rule = await classify_ambito(
                embedding, norms, testo_lower, client, api_key
            )

        composite = compute_composite_confidence(
            conf_calibrated, natura_conf, ambito_conf, rule_name, norms_count, sezione
        )

        return ClassificationResult(
            massima_id=massima_id,
            materia_l1=materia,
            materia_confidence=round(conf_calibrated, 4),
            materia_rule=rule_name,
            materia_candidate_set=list(candidates),
            materia_reasons=reasons,
            natura_l1=natura,
            natura_confidence=round(natura_conf, 4),
            natura_rule=natura_rule,
            ambito_l1=ambito,
            ambito_confidence=round(ambito_conf, 4) if ambito_conf else None,
            ambito_rule=ambito_rule,
            composite_confidence=round(composite, 4),
            norms_count=norms_count,
        ), log

    # Step 3: Centroid classification
    scores = {}
    if _centroid_cache.is_loaded and _centroid_cache.materia_centroids:
        for m in candidates:
            if m in _centroid_cache.materia_centroids:
                scores[m] = cosine_similarity(embedding, _centroid_cache.materia_centroids[m])

    if not scores:
        # Fallback: uniform scores
        scores = dict.fromkeys(candidates, 0.5)

    # Sort by score
    sorted_materie = sorted(scores.items(), key=lambda x: -x[1])
    top3 = sorted_materie[:3]

    top1_label, top1_score = top3[0]
    top2_label, top2_score = top3[1] if len(top3) > 1 else (top1_label, 0.0)
    delta_12 = top1_score - top2_score

    # Calibrate scores
    top1_calibrated = calibrator.calibrate(top1_score)

    log.top3_labels_raw = [m for m, _ in top3]
    log.top3_scores_raw = [s for _, s in top3]
    log.top3_scores_calibrated = [calibrator.calibrate(s) for _, s in top3]
    log.delta_12 = delta_12

    # Step 4: Routing decision
    needs_llm = False
    routing = "TOP1"

    if top1_calibrated < thresholds.TH_CONF_LOW:
        # Confidence too low: NEEDS_REVIEW
        routing = "NEEDS_REVIEW"
    elif delta_12 < thresholds.TH_DELTA_LLM or top1_calibrated < thresholds.TH_CONF_GO:
        # Uncertain: call LLM resolver
        needs_llm = True
        routing = "LLM_RESOLVER"
    else:
        # Confident: use Top-1
        routing = "TOP1"

    log.routing_decision = routing
    log.latency_ms_l1 = (time.time() - start_time) * 1000

    # Step 5: LLM Resolver (if needed and enabled)
    final_label = top1_label
    final_conf = top1_calibrated

    if needs_llm and thresholds.LLM_RESOLVER_ENABLED and client and api_key:
        llm_start = time.time()

        resolver_input = LLMResolverInput(
            testo=testo,
            sezione=sezione,
            norme=norms,
            candidate_set=set(log.top3_labels_raw),  # Top-3 as candidates
            fallback_reason="delta_basso" if delta_12 < thresholds.TH_DELTA_LLM else "conf_bassa",
        )

        llm_result = await call_llm_resolver_ensemble(resolver_input, client, api_key, thresholds)

        log.llm_called = True
        log.llm_candidate_set = list(resolver_input.candidate_set)
        log.llm_output_label = llm_result.label
        log.llm_output_confidence = llm_result.confidence
        log.llm_accepted = llm_result.accepted
        log.llm_agreement = llm_result.agreement
        log.llm_judge_called = llm_result.judge_called
        log.latency_ms_llm = (time.time() - llm_start) * 1000

        if llm_result.accepted:
            final_label = llm_result.label
            # Blend confidence: 60% centroid + 40% LLM
            final_conf = 0.6 * top1_calibrated + 0.4 * llm_result.confidence
            log.routing_decision = "LLM_RESOLVER"
        else:
            # LLM not accepted: use Top-1 but mark as ambiguous
            log.routing_decision = "TOP2"

    elif needs_llm and not thresholds.LLM_RESOLVER_ENABLED:
        # LLM disabled: mark as ambiguous
        log.routing_decision = "TOP2"

    log.final_label = final_label
    log.final_confidence = final_conf

    # Classify natura and ambito
    natura, natura_conf, natura_rule = await classify_natura(
        embedding, testo_lower, client, api_key
    )
    ambito, ambito_conf, ambito_rule = None, None, None
    if natura == "PROCESSUALE":
        ambito, ambito_conf, ambito_rule = await classify_ambito(
            embedding, norms, testo_lower, client, api_key
        )

    # Compute composite confidence
    composite = compute_composite_confidence(
        final_conf, natura_conf, ambito_conf, f"centroid_{routing.lower()}", norms_count, sezione
    )

    return ClassificationResult(
        massima_id=massima_id,
        materia_l1=final_label,
        materia_confidence=round(final_conf, 4),
        materia_rule=f"centroid_{routing.lower()}",
        materia_candidate_set=list(candidates),
        materia_reasons=reasons,
        natura_l1=natura,
        natura_confidence=round(natura_conf, 4),
        natura_rule=natura_rule,
        ambito_l1=ambito,
        ambito_confidence=round(ambito_conf, 4) if ambito_conf else None,
        ambito_rule=ambito_rule,
        composite_confidence=round(composite, 4),
        norms_count=norms_count,
    ), log
