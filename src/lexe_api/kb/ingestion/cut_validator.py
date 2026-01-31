"""
Cut Validator - Taglio Intelligente Chunk con LLM Fallback

Design:
1. Taglio DETERMINISTICO (default): split su confini di frase
2. LLM entra SOLO su edge cases:
   - forced_cut=true (nessun boundary trovato)
   - suspicious_end (finale troncato su citazione)
   - ambiguous_candidates (multipli candidati equivalenti)

Regole:
- LLM non inventa tagli: sceglie solo tra candidati proposti
- Output solo JSON + confidence
- Se confidence < 0.70: ignora LLM e usa deterministico
- Temperature 0.0 sempre
- Logging completo
"""

import json
import re
import time
from dataclasses import dataclass, field
from typing import Optional

import httpx
import structlog

logger = structlog.get_logger(__name__)

# ============================================================
# Configuration
# ============================================================

SOFT_CAP = 1700
HARD_CAP = 2000
MIN_CHAR = 180

# Soglie LLM
LLM_CONFIDENCE_THRESHOLD = 0.70
LLM_TEMPERATURE = 0.0
LLM_MAX_TOKENS = 256

# Pattern per end sospetti
SUSP_END_PATTERNS = re.compile(
    r"(?:sez|rv|art|artt|n|sent|ord|cass|co|c\.p\.c|c\.p\.p|c\.c|l)\.$",
    re.IGNORECASE,
)
SUSP_PUNCT = set([",", ":", ";", "(", "–", "—"])


@dataclass
class CutCandidate:
    """Candidato punto di taglio."""
    offset: int
    kind: str  # "sentence_boundary" | "near_soft_cap" | "near_hard_cap" | "forced"
    reason: str
    preview: str = ""  # 40 char intorno al punto
    score: float = 0.0  # ranking score


@dataclass
class CutDecision:
    """Decisione finale di taglio."""
    offset: int
    method: str  # "deterministic" | "llm_validated" | "llm_skipped_low_conf"
    trigger_type: Optional[str] = None  # "forced_cut" | "suspicious_end" | "ambiguous_candidates"
    forced_cut: bool = False
    candidate_index: int = 0
    candidates: list[CutCandidate] = field(default_factory=list)
    llm_confidence: Optional[float] = None
    llm_response: Optional[dict] = None
    latency_ms: int = 0


# ============================================================
# Sentence Boundary Detection
# ============================================================

# Abbreviazioni che NON terminano frase
ABBREVIATIONS = {
    "sez", "cass", "sent", "ord", "rv", "art", "artt",
    "pag", "pagg", "cfr", "nn", "vol", "ss", "segg",
    "cit", "op", "loc", "es", "ecc", "dott", "prof",
    "avv", "ing", "sig", "sigg", "co", "comma",
}


def find_sentence_boundaries(text: str, start: int, end: int) -> list[int]:
    """
    Trova tutti i confini di frase validi in una finestra.

    Un confine valido è:
    - Punto seguito da spazio + maiuscola o fine testo
    - NON dopo abbreviazione comune

    Returns:
        Lista di offset (posizioni dopo il punto)
    """
    boundaries = []
    window = text[start:end]

    # Pattern: punto + spazio + maiuscola (o fine)
    for match in re.finditer(r'\.(?:\s+[A-Z]|\s*$)', window):
        abs_pos = start + match.start() + 1  # posizione dopo il punto

        # Estrai parola prima del punto
        rel_pos = match.start()
        word_before = ""
        i = rel_pos - 1
        while i >= 0 and (window[i].isalpha() or window[i] == '.'):
            word_before = window[i] + word_before
            i -= 1

        word_clean = word_before.lower().rstrip('.')

        # Skip se abbreviazione
        if word_clean in ABBREVIATIONS:
            continue

        boundaries.append(abs_pos)

    return boundaries


def find_last_sentence_boundary(text: str, start: int, end: int) -> Optional[int]:
    """Trova l'ultimo confine di frase valido in una finestra."""
    boundaries = find_sentence_boundaries(text, start, end)
    return boundaries[-1] if boundaries else None


# ============================================================
# Cut Candidates Proposal
# ============================================================

def propose_cut_candidates(
    text: str,
    soft_cap: int = SOFT_CAP,
    hard_cap: int = HARD_CAP,
) -> list[CutCandidate]:
    """
    Genera 2-4 punti di taglio validi e spiegabili.

    Strategia:
    1. Cerca boundary vicino a hard_cap (hard_cap-250, hard_cap)
    2. Cerca boundary vicino a soft_cap (soft_cap-200, soft_cap+150)
    3. Fallback: hard_cap (forced)
    """
    text_len = len(text)
    if text_len <= hard_cap:
        # Non serve taglio
        return [CutCandidate(
            offset=text_len,
            kind="no_cut_needed",
            reason="text_under_hard_cap",
            score=1.0,
        )]

    candidates = []

    # Window 1: vicino a hard_cap
    w1_start = max(0, hard_cap - 300)
    w1_end = min(text_len, hard_cap + 50)
    boundary_hard = find_last_sentence_boundary(text, w1_start, w1_end)
    if boundary_hard and MIN_CHAR < boundary_hard <= hard_cap:
        candidates.append(CutCandidate(
            offset=boundary_hard,
            kind="near_hard_cap",
            reason="sentence_boundary_near_hard_cap",
            preview=text[max(0, boundary_hard-20):boundary_hard+20],
            score=0.8,
        ))

    # Window 2: vicino a soft_cap
    w2_start = max(0, soft_cap - 250)
    w2_end = min(text_len, soft_cap + 200)
    boundary_soft = find_last_sentence_boundary(text, w2_start, w2_end)
    if boundary_soft and MIN_CHAR < boundary_soft <= hard_cap:
        # Evita duplicati
        if not candidates or abs(boundary_soft - candidates[0].offset) > 50:
            candidates.append(CutCandidate(
                offset=boundary_soft,
                kind="near_soft_cap",
                reason="sentence_boundary_near_soft_cap",
                preview=text[max(0, boundary_soft-20):boundary_soft+20],
                score=0.9,  # Preferisci soft_cap
            ))

    # Window 3: più ampia se non abbiamo trovato nulla
    if not candidates:
        w3_start = max(0, soft_cap - 400)
        w3_end = min(text_len, hard_cap)
        boundary_wide = find_last_sentence_boundary(text, w3_start, w3_end)
        if boundary_wide and MIN_CHAR < boundary_wide:
            candidates.append(CutCandidate(
                offset=boundary_wide,
                kind="wide_search",
                reason="sentence_boundary_wide_search",
                preview=text[max(0, boundary_wide-20):boundary_wide+20],
                score=0.7,
            ))

    # Fallback: forced cut a hard_cap
    if not candidates:
        candidates.append(CutCandidate(
            offset=min(text_len, hard_cap),
            kind="forced",
            reason="no_sentence_boundary_found",
            preview=text[max(0, hard_cap-20):hard_cap+20],
            score=0.1,
        ))

    # Ordina per score (migliore prima)
    candidates.sort(key=lambda c: -c.score)

    # Limita a 4
    return candidates[:4]


# ============================================================
# Suspicious End Detection
# ============================================================

def is_suspicious_end(text: str, cut_offset: int) -> bool:
    """
    Rileva se il taglio finisce su un punto sospetto.

    Trigger per LLM validation:
    - Finisce con abbreviazione troncata (Sez., Rv., n., art.)
    - Finisce con punteggiatura sospetta (, : ; ()
    - Ultime 30 char contengono citazione iniziata ma non finita
    """
    if cut_offset <= 0:
        return True

    prefix = text[:cut_offset].rstrip()
    if not prefix:
        return True

    # Check punteggiatura finale
    if prefix[-1] in SUSP_PUNCT:
        return True

    # Check abbreviazione troncata
    tail = prefix[-25:].lower()
    if SUSP_END_PATTERNS.search(tail):
        return True

    # Check citazione iniziata (Rv. o Sez. senza numero completo)
    if re.search(r"(?:rv|sez)\.\s*\d{0,3}$", tail, re.IGNORECASE):
        return True

    return False


# ============================================================
# LLM Validator
# ============================================================

LLM_PROMPT_TEMPLATE = """Sei un esperto di testi giuridici italiani. Devi scegliere il punto di taglio migliore per un chunk di testo.

REGOLE:
1. Scegli SOLO tra i candidati forniti (non inventare nuovi punti)
2. Preferisci tagli che:
   - Finiscono su una frase completa
   - NON spezzano una citazione (Sez., Rv., n.)
   - NON lasciano periodi incompleti
3. Se nessun candidato è buono, scegli quello meno peggio con confidence bassa

TESTO (ultimi 450 char prima del taglio + primi 300 dopo):
---
{snippet_before}
[...TAGLIO QUI...]
{snippet_after}
---

CANDIDATI:
{candidates_text}

Rispondi SOLO con JSON valido:
{{"best_candidate_index": <0-based index>, "confidence": <0.0-1.0>, "reason": "<max 100 char>"}}
"""


async def llm_validate_cut(
    text: str,
    candidates: list[CutCandidate],
    api_key: str,
    model: str = "google/gemini-2.0-flash-lite-001",
    api_url: str = "https://openrouter.ai/api/v1/chat/completions",
) -> Optional[dict]:
    """
    Chiama LLM per validare/scegliere tra candidati di taglio.

    Returns:
        {
            "best_candidate_index": int,
            "confidence": float,
            "reason": str,
            "latency_ms": int,
            "cost_usd": float (stima)
        }
        oppure None se fallisce
    """
    if not api_key:
        logger.warning("llm_validate_cut: no API key, skipping")
        return None

    # Prepara snippet
    # Usa il primo candidato come riferimento per lo snippet
    ref_offset = candidates[0].offset
    snippet_before = text[max(0, ref_offset - 450):ref_offset]
    snippet_after = text[ref_offset:ref_offset + 300]

    # Formatta candidati
    candidates_text = ""
    for i, c in enumerate(candidates):
        preview = text[max(0, c.offset - 30):c.offset + 30].replace("\n", " ")
        candidates_text += f"{i}. offset={c.offset}, kind={c.kind}, preview=\"...{preview}...\"\n"

    prompt = LLM_PROMPT_TEMPLATE.format(
        snippet_before=snippet_before,
        snippet_after=snippet_after,
        candidates_text=candidates_text,
    )

    start_time = time.time()

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                api_url,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": LLM_TEMPERATURE,
                    "max_tokens": LLM_MAX_TOKENS,
                },
            )
            response.raise_for_status()
            data = response.json()

    except Exception as e:
        logger.error("llm_validate_cut failed", error=str(e))
        return None

    latency_ms = int((time.time() - start_time) * 1000)

    # Parse response
    try:
        content = data["choices"][0]["message"]["content"]
        # Estrai JSON (potrebbe essere wrapped in ```json)
        json_match = re.search(r'\{[^}]+\}', content)
        if not json_match:
            logger.warning("llm_validate_cut: no JSON in response", content=content[:200])
            return None

        result = json.loads(json_match.group())

        # Valida
        idx = result.get("best_candidate_index", 0)
        conf = result.get("confidence", 0.0)
        reason = result.get("reason", "")

        if not (0 <= idx < len(candidates)):
            idx = 0

        # Stima costo (approssimativa)
        tokens_in = len(prompt) // 4
        tokens_out = len(content) // 4
        cost_usd = (tokens_in * 0.00001 + tokens_out * 0.00004)  # ~Gemini Flash rates

        return {
            "best_candidate_index": idx,
            "confidence": float(conf),
            "reason": reason[:100],
            "latency_ms": latency_ms,
            "cost_usd": cost_usd,
            "raw_response": content[:500],
        }

    except Exception as e:
        logger.error("llm_validate_cut: parse error", error=str(e))
        return None


# ============================================================
# Main Cut Decision Logic
# ============================================================

def candidates_are_ambiguous(candidates: list[CutCandidate]) -> bool:
    """Check se ci sono 2+ candidati con score simile."""
    if len(candidates) < 2:
        return False
    # Se i top 2 hanno score entro 0.15, sono ambigui
    return abs(candidates[0].score - candidates[1].score) < 0.15


async def choose_cut(
    text: str,
    soft_cap: int = SOFT_CAP,
    hard_cap: int = HARD_CAP,
    use_llm: bool = False,
    llm_api_key: str = "",
    llm_model: str = "google/gemini-2.0-flash-lite-001",
) -> CutDecision:
    """
    Decide il punto di taglio ottimale.

    Flow:
    1. Proponi candidati deterministici
    2. Scegli il migliore
    3. Se trigger LLM (forced/suspicious/ambiguous) e use_llm=True:
       - Chiama LLM per validare
       - Se confidence >= 0.70: usa scelta LLM
       - Altrimenti: usa deterministico

    Returns:
        CutDecision con tutti i dettagli
    """
    candidates = propose_cut_candidates(text, soft_cap, hard_cap)
    chosen = candidates[0]
    forced = (chosen.kind == "forced")

    decision = CutDecision(
        offset=chosen.offset,
        method="deterministic",
        trigger_type=None,
        forced_cut=forced,
        candidate_index=0,
        candidates=candidates,
    )

    # Determina se serve LLM
    trigger = None
    if forced:
        trigger = "forced_cut"
    elif is_suspicious_end(text, chosen.offset):
        trigger = "suspicious_end"
    elif candidates_are_ambiguous(candidates):
        trigger = "ambiguous_candidates"

    decision.trigger_type = trigger

    if not trigger or not use_llm:
        return decision

    # Chiama LLM
    llm_result = await llm_validate_cut(
        text=text,
        candidates=candidates,
        api_key=llm_api_key,
        model=llm_model,
    )

    if not llm_result:
        decision.method = "llm_failed"
        return decision

    decision.llm_confidence = llm_result["confidence"]
    decision.llm_response = llm_result
    decision.latency_ms = llm_result["latency_ms"]

    if llm_result["confidence"] < LLM_CONFIDENCE_THRESHOLD:
        decision.method = "llm_skipped_low_conf"
        return decision

    # Usa scelta LLM
    llm_idx = llm_result["best_candidate_index"]
    if 0 <= llm_idx < len(candidates):
        decision.offset = candidates[llm_idx].offset
        decision.candidate_index = llm_idx
        decision.method = "llm_validated"

    return decision


def choose_cut_sync(
    text: str,
    soft_cap: int = SOFT_CAP,
    hard_cap: int = HARD_CAP,
) -> CutDecision:
    """
    Versione sincrona (solo deterministico, no LLM).
    Per uso in contesti non-async.
    """
    candidates = propose_cut_candidates(text, soft_cap, hard_cap)
    chosen = candidates[0]
    forced = (chosen.kind == "forced")

    trigger = None
    if forced:
        trigger = "forced_cut"
    elif is_suspicious_end(text, chosen.offset):
        trigger = "suspicious_end"
    elif candidates_are_ambiguous(candidates):
        trigger = "ambiguous_candidates"

    return CutDecision(
        offset=chosen.offset,
        method="deterministic",
        trigger_type=trigger,
        forced_cut=forced,
        candidate_index=0,
        candidates=candidates,
    )
