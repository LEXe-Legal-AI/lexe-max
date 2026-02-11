"""
Ambito Rules for Category Graph v2.4

High-precision rule-based derivation of Ambito (procedural scope).
Only called when natura=PROCESSUALE.

Ambito values:
- GIUDIZIO: Istruttoria, prove, competenza, notifiche
- IMPUGNAZIONI: Appello, ricorso, cassazione, termini
- ESECUZIONE: Pignoramento, espropriazione, precetto
- MISURE: Cautelare, sospensione, inibitoria

Priority: Norm-based rules first, then keyword fallback.

NOTE: Database norms are in original format (e.g., "art. 360 c.p.c.").
      This module normalizes them to canonical format (e.g., "CPC:360").
"""

import re

# Import normalization from materia_rules
from .materia_rules import normalize_norm_for_matching

# High-precision CPC norms for each Ambito
AMBITO_NORMS = {
    "MISURE": {
        "CPC:700",  # Provvedimenti d'urgenza
        "CPC:669",  # Procedimento cautelare
        "CPC:669BIS",  # Istanza cautelare
        "CPC:671",  # Sequestro giudiziario
        "CPC:672",  # Sequestro conservativo
    },
    "ESECUZIONE": {
        "CPC:474",  # Titolo esecutivo
        "CPC:475",  # Spedizione in forma esecutiva
        "CPC:479",  # Precetto
        "CPC:480",  # Forma del precetto
        "CPC:491",  # Espropriazione mobiliare
        "CPC:492",  # Forma del pignoramento
        "CPC:555",  # Espropriazione immobiliare
        "CPC:615",  # Opposizione all'esecuzione
        "CPC:617",  # Opposizione agli atti esecutivi
        "CPC:619",  # Opposizione di terzo
    },
    "IMPUGNAZIONI": {
        "CPC:323",  # Mezzi di impugnazione
        "CPC:325",  # Termine breve
        "CPC:327",  # Termine lungo
        "CPC:339",  # Appello
        "CPC:360",  # Sentenze impugnabili e motivi
        "CPC:366",  # Contenuto del ricorso
        "CPC:369",  # Deposito del ricorso
        "CPC:371",  # Ricorso incidentale
        "CPC:391",  # Rinuncia al ricorso
        "CPC:395",  # Casi di revocazione
        "CPC:404",  # Opposizione di terzo
    },
    "GIUDIZIO": {
        # Competenza
        "CPC:7",  # Competenza del giudice di pace
        "CPC:9",  # Competenza del tribunale
        "CPC:18",  # Foro generale persone fisiche
        "CPC:19",  # Foro generale persone giuridiche
        "CPC:20",  # Foro facoltativo
        "CPC:28",  # Competenza inderogabile
        "CPC:38",  # Incompetenza
        # Notifiche
        "CPC:137",  # Notificazione
        "CPC:138",  # Notificazione in mani proprie
        "CPC:139",  # Notificazione nella residenza
        "CPC:140",  # Irreperibilità
        "CPC:143",  # Notificazione a persona di residenza sconosciuta
        # Nullità
        "CPC:156",  # Rilevanza della nullità
        "CPC:157",  # Nullità formali
        "CPC:158",  # Regime della nullità
        "CPC:159",  # Estensione della nullità
        "CPC:160",  # Rinnovazione
        "CPC:161",  # Nullità della sentenza
        # Prove
        "CPC:115",  # Disponibilità delle prove
        "CPC:116",  # Valutazione delle prove
        "CPC:191",  # CTU
        "CPC:244",  # Prova testimoniale
        "CPC:2697",  # Onere della prova (CC ma spesso citato)
    },
}

# Keyword patterns for each Ambito (fallback when no norms)
AMBITO_KEYWORDS = {
    "MISURE": [
        r"\b(700|provvediment[oi]\s+d'urgenza|cautelar[ei]|"
        r"inibitor|sospension[ei]|sequestro\s+(giudiziari|conservativ)|"
        r"tutela\s+cautelare|periculum\s+in\s+mora|fumus\s+boni\s+iuris)\b"
    ],
    "ESECUZIONE": [
        r"\b(pignorament|precett[oi]|espropriazion[ei]|"
        r"opposizione\s+all'esecuzion|titolo\s+esecutivo|"
        r"esecuzione\s+forzata|vendita\s+forzata|"
        r"pignoramento\s+(mobiliare|immobiliare|presso\s+terzi))\b"
    ],
    "IMPUGNAZIONI": [
        r"\b(ricorso\s+per\s+cassazione|cassazione|impugnazion[ei]|"
        r"revocazion[ei]|opposizione\s+di\s+terzo|"
        r"termine\s+(breve|lungo)|appello|ricorso\s+incidentale|"
        r"motiv[oi]\s+di\s+ricorso|violazione\s+di\s+legge|"
        r"vizio\s+di\s+motivazione|error\s+in\s+iudicando|"
        r"error\s+in\s+procedendo)\b"
    ],
    "GIUDIZIO": [
        r"\b(competen[tz][ae]|notific[ah]|nullit[aà]|"
        r"contraddittorio|istruttor|prove|onere\s+della\s+prova|"
        r"ctu|consulente\s+tecnico|giudice\s+istruttore|"
        r"decadenz[ae]|preclusione|costituzione\s+in\s+giudizio)\b"
    ],
}


def _norm_primary(code: str) -> str:
    """Extract primary code from canonical: CPC:700 -> CPC."""
    if not code:
        return ""
    return code.split(":")[0].upper().strip()


def _extract_cpc_articles(norms: list[str]) -> set[str]:
    """
    Extract normalized CPC article references from norm list.

    Input formats (from database):
        "art. 360 c.p.c." -> "CPC:360"
        "art. 700 c.p.c." -> "CPC:700"

    Returns:
        Set of normalized CPC refs like {"CPC:360", "CPC:700"}
    """
    cpc_refs = set()
    for n in norms:
        if not n:
            continue
        code_type, number, year = normalize_norm_for_matching(n)
        if code_type == "CPC" and number:
            cpc_refs.add(f"CPC:{number}")
    return cpc_refs


def ambito_rules_high_precision(
    norms: list[str],
    testo_lower: str,
) -> tuple[str | None, float, str]:
    """
    High-precision rules for Ambito derivation.
    Only called when natura=PROCESSUALE.

    Returns:
        (ambito, confidence, rule)
        If ambito is None, classifier is needed.
    """
    cpc_refs = _extract_cpc_articles(norms)
    # Check if any norm references CPC (using normalization)
    bool(cpc_refs) or any(normalize_norm_for_matching(n)[0] == "CPC" for n in norms if n)

    # Rule 1: MISURE (cautelare) - highest priority
    misure_hits = AMBITO_NORMS["MISURE"] & cpc_refs
    if misure_hits:
        return "MISURE", 0.93, f"rule_misure_norme_{sorted(misure_hits)[0]}"

    # Keyword fallback for MISURE
    for pattern in AMBITO_KEYWORDS["MISURE"]:
        if re.search(pattern, testo_lower):
            return "MISURE", 0.88, "rule_misure_keywords"

    # Rule 2: ESECUZIONE
    esec_hits = AMBITO_NORMS["ESECUZIONE"] & cpc_refs
    if esec_hits:
        return "ESECUZIONE", 0.95, f"rule_esecuzione_norme_{sorted(esec_hits)[0]}"

    # Keyword fallback for ESECUZIONE
    for pattern in AMBITO_KEYWORDS["ESECUZIONE"]:
        if re.search(pattern, testo_lower):
            return "ESECUZIONE", 0.90, "rule_esecuzione_keywords"

    # Rule 3: IMPUGNAZIONI
    imp_hits = AMBITO_NORMS["IMPUGNAZIONI"] & cpc_refs
    if imp_hits:
        return "IMPUGNAZIONI", 0.95, f"rule_impugnazioni_norme_{sorted(imp_hits)[0]}"

    # Keyword fallback for IMPUGNAZIONI
    for pattern in AMBITO_KEYWORDS["IMPUGNAZIONI"]:
        if re.search(pattern, testo_lower):
            return "IMPUGNAZIONI", 0.90, "rule_impugnazioni_keywords"

    # Rule 4: GIUDIZIO (residual - istruttoria, competenza, notifiche)
    giudizio_hits = AMBITO_NORMS["GIUDIZIO"] & cpc_refs
    if giudizio_hits:
        return "GIUDIZIO", 0.88, f"rule_giudizio_norme_{sorted(giudizio_hits)[0]}"

    # Keyword fallback for GIUDIZIO
    for pattern in AMBITO_KEYWORDS["GIUDIZIO"]:
        if re.search(pattern, testo_lower):
            return "GIUDIZIO", 0.86, "rule_giudizio_keywords"

    # No rule hit - classifier needed
    return None, 0.0, "no_rule_hit"


def compute_ambito_candidates(
    norms: list[str],
    testo_lower: str,
) -> tuple[set[str], list[str]]:
    """
    Compute candidate set for Ambito based on weak signals.
    Used when high-precision rules don't fire.

    Returns:
        (candidate_set, reasons)
    """
    reasons: list[str] = []
    candidates: set[str] = {"GIUDIZIO", "IMPUGNAZIONI", "ESECUZIONE", "MISURE"}

    _extract_cpc_articles(norms)

    # Check for any weak signals to narrow candidates
    has_esec_signal = bool(re.search(r"\b(esecuzion|pignor|precett)\b", testo_lower))
    has_imp_signal = bool(re.search(r"\b(impugn|ricors|appell|cassaz)\b", testo_lower))
    has_misure_signal = bool(re.search(r"\b(cautelar|urgent|sospens)\b", testo_lower))

    signals = []
    if has_esec_signal:
        signals.append("ESECUZIONE")
    if has_imp_signal:
        signals.append("IMPUGNAZIONI")
    if has_misure_signal:
        signals.append("MISURE")

    if signals:
        # Narrow to detected signals + GIUDIZIO (always possible)
        candidates = set(signals) | {"GIUDIZIO"}
        reasons.append(f"weak_signals={sorted(signals)}")
    else:
        reasons.append("no_weak_signals_all_candidates")

    return candidates, reasons


def derive_ambito_rule_first(
    norms: list[str],
    testo_lower: str,
) -> tuple[str | None, float, str, set[str], list[str]]:
    """
    Attempt rule-based derivation of Ambito.
    Only call this when natura=PROCESSUALE.

    Returns:
        (ambito, confidence, rule, candidate_set, reasons)
        If ambito is None, classifier is needed.
    """
    # Try high-precision rules first
    ambito, conf, rule = ambito_rules_high_precision(norms, testo_lower)

    if ambito is not None:
        # Rule fired - singleton candidate set
        candidates = {ambito}
        reasons = [rule]
        return ambito, conf, rule, candidates, reasons

    # No rule fired - compute candidate set for classifier
    candidates, reasons = compute_ambito_candidates(norms, testo_lower)
    return None, 0.0, "needs_classifier", candidates, reasons
