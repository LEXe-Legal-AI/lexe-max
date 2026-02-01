"""
Materia Rules for Category Graph v2.4

Rule-based derivation of Materia (subject matter) using:
1. tipo field (civile/penale)
2. sezione field (Sez. L, Sez. U, etc.)
3. Norm hints from Norm Graph (DLGS:546, LEGGE:241, etc.)
4. Text keywords as fallback

Priority order for reliability:
1. tipo=penale -> PENALE (absolute)
2. CP/CPP in norms -> PENALE (strong, but check for cross-domain)
3. sezione=L -> LAVORO (strong)
4. sezione=U -> no reduction (cross-domain)
5. Norm hints for TRIBUTARIO, AMMINISTRATIVO, CRISI
6. sezione 1-6 -> exclude PENALE from candidates

NOTE: Database norms are in original format (e.g., "D.Lgs. n. 546/1992").
      This module normalizes them to canonical format (e.g., "DLGS:546:1992").
"""

import re
from typing import List, Set, Tuple, Optional

# All possible materie
MATERIE = {"CIVILE", "PENALE", "LAVORO", "TRIBUTARIO", "AMMINISTRATIVO", "CRISI"}


def normalize_norm_for_matching(norm: str) -> Tuple[str, Optional[str], Optional[str]]:
    """
    Normalize norm string to canonical components for matching.

    Input formats (from database):
        "D.Lgs. n. 546/1992" -> ("DLGS", "546", "1992")
        "L. n. 241/1990" -> ("LEGGE", "241", "1990")
        "D.P.R. n. 602/1973" -> ("DPR", "602", "1973")
        "R.D. n. 267/1942" -> ("RD", "267", "1942")
        "art. 360 c.p.c." -> ("CPC", "360", None)
        "art. 2043 c.c." -> ("CC", "2043", None)
        "art. 640 c.p." -> ("CP", "640", None)

    Returns:
        (code_type, number, year) - all uppercase, year may be None
    """
    norm = norm.strip().upper()

    # Pattern: D.LGS. N. XXX/YYYY or D.LGS. XXX/YYYY
    if match := re.search(r"D\.?LGS\.?\s*(?:N\.?\s*)?(\d+)/(\d{4})", norm):
        return ("DLGS", match.group(1), match.group(2))

    # Pattern: L. N. XXX/YYYY or LEGGE N. XXX/YYYY
    if match := re.search(r"(?:L\.?|LEGGE)\s*(?:N\.?\s*)?(\d+)/(\d{4})", norm):
        return ("LEGGE", match.group(1), match.group(2))

    # Pattern: D.P.R. N. XXX/YYYY
    if match := re.search(r"D\.?P\.?R\.?\s*(?:N\.?\s*)?(\d+)/(\d{4})", norm):
        return ("DPR", match.group(1), match.group(2))

    # Pattern: D.L. N. XXX/YYYY
    if match := re.search(r"D\.?L\.?\s*(?:N\.?\s*)?(\d+)/(\d{4})", norm):
        return ("DL", match.group(1), match.group(2))

    # Pattern: R.D. N. XXX/YYYY
    if match := re.search(r"R\.?D\.?\s*(?:N\.?\s*)?(\d+)/(\d{4})", norm):
        return ("RD", match.group(1), match.group(2))

    # Pattern: art. XXX c.p.c. (Codice Procedura Civile)
    if match := re.search(r"ART\.?\s*(\d+(?:\s*(?:BIS|TER|QUATER|QUINQUIES|SEXIES))?)\s*C\.?P\.?C\.?", norm):
        return ("CPC", match.group(1).replace(" ", ""), None)

    # Pattern: art. XXX c.c. (Codice Civile)
    if match := re.search(r"ART\.?\s*(\d+(?:\s*(?:BIS|TER|QUATER|QUINQUIES|SEXIES))?)\s*C\.?C\.?", norm):
        return ("CC", match.group(1).replace(" ", ""), None)

    # Pattern: art. XXX c.p. (Codice Penale) - NOT c.p.c.!
    if match := re.search(r"ART\.?\s*(\d+(?:\s*(?:BIS|TER|QUATER|QUINQUIES|SEXIES))?)\s*C\.?P\.?(?!C)", norm):
        return ("CP", match.group(1).replace(" ", ""), None)

    # Pattern: art. XXX c.p.p. (Codice Procedura Penale)
    if match := re.search(r"ART\.?\s*(\d+(?:\s*(?:BIS|TER|QUATER|QUINQUIES|SEXIES))?)\s*C\.?P\.?P\.?", norm):
        return ("CPP", match.group(1).replace(" ", ""), None)

    # Fallback: no match
    return ("UNKNOWN", None, None)

# Norm signatures that strongly indicate specific materie
NORM_HINTS = {
    "TRIBUTARIO": {
        "DLGS:546:1992",  # Contenzioso tributario
        "DPR:602:1973",   # Riscossione
        "DPR:633:1972",   # IVA
        "DPR:600:1973",   # Accertamento
        "DLGS:472:1997",  # Sanzioni tributarie
        "DLGS:471:1997",  # Sanzioni tributarie
        "DLGS:504:1992",  # ICI/IMU
    },
    "AMMINISTRATIVO": {
        "LEGGE:241:1990",  # Procedimento amministrativo
        "DLGS:165:2001",   # Pubblico impiego
        "DLGS:104:2010",   # Codice processo amministrativo
        "DLGS:50:2016",    # Codice appalti (nuovo)
        "DPR:445:2000",    # Documentazione amministrativa
        # NOTE: D.Lgs. 25/2008, 150/2011, 163/2006, L.689/1981 are ambiguous
        # They appear in both AMMINISTRATIVO and CIVILE contexts
        # Only add hints that are unambiguous
        "DPR:327:2001",    # TU Espropriazioni (unambiguous)
    },
    "CRISI": {
        "RD:267:1942",     # Legge fallimentare
        "DLGS:14:2019",    # Codice della crisi
    },
    "LAVORO": {
        "LEGGE:300:1970",  # Statuto lavoratori
        "DLGS:66:2003",    # Orario di lavoro
        "DLGS:81:2008",    # Sicurezza lavoro
    },
    "PENALE": {
        "CP",   # Codice penale
        "CPP",  # Codice procedura penale
    },
}


def _norm_primary(code: str) -> str:
    """Extract primary code from canonical: CC:2043 -> CC, DLGS:546:1992 -> DLGS"""
    if not code:
        return ""
    return code.split(":")[0].upper().strip()


def _normalize_norms(norms: List[str]) -> Tuple[Set[str], Set[str]]:
    """
    Normalize a list of norms from database format to canonical format.

    Returns:
        (normalized_set, primaries_set)
        normalized_set contains full canonical refs like "DLGS:546:1992"
        primaries_set contains just the code types like "DLGS", "CPC", "CP"
    """
    normalized = set()
    primaries = set()

    for norm in norms:
        if not norm:
            continue
        code_type, number, year = normalize_norm_for_matching(norm)
        if code_type and code_type != "UNKNOWN":
            primaries.add(code_type)
            if number and year:
                normalized.add(f"{code_type}:{number}:{year}")
            elif number:
                normalized.add(f"{code_type}:{number}")
            else:
                normalized.add(code_type)

    return normalized, primaries


def compute_materia_candidates(
    tipo: Optional[str],
    sezione: Optional[str],
    norms: List[str],
    testo_lower: Optional[str] = None,
) -> Tuple[Set[str], List[str]]:
    """
    Compute the candidate set for materia based on metadata and norms.

    Args:
        tipo: Document type (civile/penale)
        sezione: Court section (e.g., "L", "U", "1", "2")
        norms: List of norms in database format (e.g., "D.Lgs. n. 546/1992")
        testo_lower: Lowercase text for keyword matching

    Returns:
        (candidate_set, reasons): Narrowed candidate set and reasons for audit trail
    """
    reasons: List[str] = []
    candidates: Set[str] = set(MATERIE)

    tipo_norm = (tipo or "").strip().lower()
    sez = (sezione or "").lower()

    # Normalize norms to canonical format
    norm_set, primaries = _normalize_norms(norms)

    # Rule 1: tipo=penale -> reduce to PENALE immediately (absolute)
    if tipo_norm == "penale":
        candidates = {"PENALE"}
        reasons.append("tipo=penale")
        return candidates, reasons

    # Rule 2: CP/CPP in norms -> strong PENALE signal
    # BUT check for cross-domain if civile hints present
    has_cp_cpp = "CP" in primaries or "CPP" in primaries
    has_civile_hints = any(
        sig in norm_set or _norm_primary(sig) in primaries
        for materia, sigs in NORM_HINTS.items()
        if materia not in {"PENALE"}
        for sig in sigs
    )

    if has_cp_cpp:
        if tipo_norm == "penale":
            # tipo confirms -> absolute
            candidates = {"PENALE"}
            reasons.append("tipo_penale_confirms_cp_cpp")
            return candidates, reasons
        elif has_civile_hints:
            # Cross-domain: CP/CPP + civile norms -> resolver needed
            candidates = {"PENALE", "CIVILE"}
            reasons.append("cp_cpp_with_civile_hints_cross_domain")
            return candidates, reasons
        else:
            # Strong PENALE but not absolute
            candidates = {"PENALE"}
            reasons.append("norms_cp_cpp_strong")
            return candidates, reasons

    # Rule 3: Sezione L -> candidate set {LAVORO, CIVILE}, NOT singleton!
    # Sezione L can have CIVILE cases (privacy D.Lgs.196, processual, etc.)
    # Note: In DB, sezione is just "L", not "Sez. L"
    if sez == "l" or "sez. l" in sez:
        candidates = {"LAVORO", "CIVILE"}
        reasons.append("sezione_l_candidate_set")
        return candidates, reasons

    # Rule 3b: LAVORO keywords as reducer, NOT singleton!
    # Old keywords (contribut, inps, previdenz) are too generic:
    # - "contributi condominiali" is CIVILE
    # - "contributo di mantenimento" is CIVILE (family law)
    # Use ONLY highly specific keywords
    if testo_lower and re.search(
        r"\b(licenziament[oi]|statuto\s+dei\s+lavorator|"
        r"rapporto\s+di\s+lavoro\s+subordinato|contratto\s+di\s+lavoro\s+subordinato|"
        r"t\.?\s*f\.?\s*r\.?\s*[^\w]|trattamento\s+di\s+fine\s+rapporto)", testo_lower
    ):
        candidates = {"LAVORO", "CIVILE"}
        reasons.append("lavoro_keywords_candidate_set")
        return candidates, reasons

    # Rule 4: Sezioni Unite -> no reduction (cross-domain)
    # In DB, sezione is "U" for unite
    if sez == "u" or "sez. u" in sez:
        reasons.append("sezione_u_no_reduction")
        return candidates, reasons

    # Rule 5: Sezioni civili 1-6 -> exclude PENALE, soft CIVILE prior
    # In DB, sezione is "1", "2", "3", etc. or "6-1", "6-2", "6-3"
    # Only exclude PENALE (absolute) and LAVORO (if no hints)
    # Keep AMMINISTRATIVO, CRISI, TRIBUTARIO in candidate set for centroid
    if sez in {"1", "2", "3", "4", "5", "6"} or re.search(r"^6-[123]$", sez) or re.search(r"sez\.\s*[1-6]\b", sez):
        candidates.discard("PENALE")
        # Exclude LAVORO unless norm hints present
        lavoro_norms = NORM_HINTS.get("LAVORO", set())
        has_lavoro_hint = any(sig in norm_set for sig in lavoro_norms)
        if not has_lavoro_hint:
            candidates.discard("LAVORO")
        reasons.append("sezione_civile_soft_prior")

    # Rule 6: Norm-based hints for specialized materie
    # NOTE: Only check full signature match (not primary) because
    # primaries like "DLGS" are shared across multiple materie
    hits = []
    for materia, signatures in NORM_HINTS.items():
        if materia == "PENALE":
            continue  # Already handled above
        for sig in signatures:
            # Only check full signature match
            if sig in norm_set:
                hits.append(materia)
                break

    if hits:
        candidates = set(hits)
        reasons.append(f"norm_hints={sorted(set(hits))}")
        return candidates, reasons

    return candidates, reasons


def derive_materia_rule_first(
    tipo: Optional[str],
    sezione: Optional[str],
    norms: List[str],
    testo_lower: Optional[str] = None,
) -> Tuple[Optional[str], float, str, Set[str], List[str]]:
    """
    Attempt rule-based derivation of materia.

    Returns:
        (materia, confidence, rule, candidate_set, reasons)
        If materia is None, classifier is needed.
    """
    candidates, reasons = compute_materia_candidates(tipo, sezione, norms, testo_lower)

    # Singleton candidate set -> deterministic assignment
    if len(candidates) == 1:
        materia = next(iter(candidates))

        if materia == "PENALE":
            if (tipo or "").lower() == "penale":
                return materia, 0.92, "tipo_penale", candidates, reasons
            else:
                return materia, 0.90, "norms_cp_cpp", candidates, reasons

        if materia == "LAVORO":
            return materia, 0.95, "sezione_l_or_keywords", candidates, reasons

        if materia in {"TRIBUTARIO", "AMMINISTRATIVO", "CRISI"}:
            return materia, 0.90, f"norm_hint_{materia.lower()}", candidates, reasons

        # Fallback for singleton but unknown rule
        return materia, 0.85, "singleton_candidates", candidates, reasons

    # Multiple candidates -> needs classifier
    return None, 0.0, "needs_classifier", candidates, reasons


def get_materia_prior(
    tipo: Optional[str],
    sezione: Optional[str],
) -> Tuple[Optional[str], float]:
    """
    Get a weak prior for materia based only on tipo and sezione.
    Used when no norms are available.

    Returns:
        (materia_hint, confidence)
        Returns (None, 0.0) if no prior can be determined.
    """
    tipo_norm = (tipo or "").strip().lower()
    sez = (sezione or "").lower()

    if tipo_norm == "penale":
        return "PENALE", 0.80

    # Sezione L -> LAVORO (in DB, sezione is just "L")
    if sez == "l" or "sez. l" in sez:
        return "LAVORO", 0.85

    if tipo_norm == "civile":
        # Very weak prior - civile could be many things
        return "CIVILE", 0.40

    return None, 0.0
