#!/usr/bin/env python3
"""
Golden Set Sampling for Category Graph v2.4

Generates 600 samples (420 train + 180 test) stratified by:
1. Difficulty bucket (easy, metadata_ambiguous, procedural_heavy, cross_domain)
2. Materia (enforce minimums for minor materie)

Buckets:
- easy: sezione='Sez. L' + norm hints for TRIB/AMM/CRISI
- metadata_ambiguous: sezione IS NULL OR sezione='Sez. U'
- procedural_heavy: CPC/CPP heavy, 50% civile / 50% penale
- cross_domain: Multiple code citations, contrasting signals

Usage:
    uv run python scripts/qa/generate_golden_v2.py --dry-run
    uv run python scripts/qa/generate_golden_v2.py --commit
"""

import argparse
import asyncio
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from uuid import UUID

import asyncpg

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.lexe_api.kb.config import KBSettings
from src.lexe_api.kb.graph.materia_rules import NORM_HINTS, _norm_primary


# Bucket configuration (train + test = total)
BUCKET_CONFIG = {
    "easy": {"train": 105, "test": 30, "total": 135},
    "metadata_ambiguous": {"train": 140, "test": 60, "total": 200},
    "procedural_heavy": {"train": 105, "test": 50, "total": 155},
    "cross_domain": {"train": 70, "test": 40, "total": 110},
}
# Total: 420 train + 180 test = 600

# Materia minimums (ENFORCED, not soft)
MATERIA_MINIMUMS_TRAIN = {
    "CIVILE": 150,
    "PENALE": 80,
    "LAVORO": 50,
    "TRIBUTARIO": 25,
    "AMMINISTRATIVO": 20,
    "CRISI": 15,
}
# Total minor materie in train: 25 + 20 + 15 = 60 minimum


@dataclass
class MassimaCandidate:
    """Candidate massima for golden set."""
    massima_id: UUID
    sezione: Optional[str]
    tipo: Optional[str]
    norms: List[str]
    testo_trunc: str
    norms_count: int
    bucket: Optional[str] = None
    estimated_materia: Optional[str] = None


def normalize_norm_for_matching(norm: str) -> Tuple[str, Optional[str], Optional[str]]:
    """
    Normalize norm string for matching.
    Returns: (code_type, number, year)

    Examples:
        "D.Lgs. n. 546/1992" -> ("DLGS", "546", "1992")
        "L. n. 241/1990" -> ("LEGGE", "241", "1990")
        "art. 2043 c.c." -> ("CC", "2043", None)
        "art. 360 c.p.c." -> ("CPC", "360", None)
        "art. 640 c.p." -> ("CP", "640", None)
    """
    import re
    norm_lower = norm.lower().strip()

    # Code articles: art. XXX c.c., c.p.c., c.p., c.p.p.
    if "c.p.c." in norm_lower or "cpc" in norm_lower:
        match = re.search(r"art\.?\s*(\d+)", norm_lower)
        return ("CPC", match.group(1) if match else None, None)
    if "c.p.p." in norm_lower or "cpp" in norm_lower:
        match = re.search(r"art\.?\s*(\d+)", norm_lower)
        return ("CPP", match.group(1) if match else None, None)
    if "c.p." in norm_lower and "c.p.c." not in norm_lower and "c.p.p." not in norm_lower:
        match = re.search(r"art\.?\s*(\d+)", norm_lower)
        return ("CP", match.group(1) if match else None, None)
    if "c.c." in norm_lower:
        match = re.search(r"art\.?\s*(\d+)", norm_lower)
        return ("CC", match.group(1) if match else None, None)

    # D.Lgs., D.L., L., DPR, RD
    if "d.lgs" in norm_lower or "dlgs" in norm_lower:
        match = re.search(r"n\.?\s*(\d+)[/\s]*(\d{4})?", norm_lower)
        if match:
            return ("DLGS", match.group(1), match.group(2))
    if "d.l." in norm_lower and "d.lgs" not in norm_lower:
        match = re.search(r"n\.?\s*(\d+)[/\s]*(\d{4})?", norm_lower)
        if match:
            return ("DL", match.group(1), match.group(2))
    if norm_lower.startswith("l.") or " l. " in norm_lower or "legge" in norm_lower:
        match = re.search(r"n\.?\s*(\d+)[/\s]*(\d{4})?", norm_lower)
        if match:
            return ("LEGGE", match.group(1), match.group(2))
    if "d.p.r" in norm_lower or "dpr" in norm_lower:
        match = re.search(r"n\.?\s*(\d+)[/\s]*(\d{4})?", norm_lower)
        if match:
            return ("DPR", match.group(1), match.group(2))
    if "r.d." in norm_lower or norm_lower.startswith("rd"):
        match = re.search(r"n\.?\s*(\d+)[/\s]*(\d{4})?", norm_lower)
        if match:
            return ("RD", match.group(1), match.group(2))

    return (None, None, None)


# Norm hints in human-readable format for matching
NORM_HINTS_PATTERNS = {
    "TRIBUTARIO": [
        ("DLGS", "546", "1992"),  # Contenzioso tributario
        ("DPR", "602", "1973"),   # Riscossione
        ("DPR", "633", "1972"),   # IVA
        ("DPR", "600", "1973"),   # Accertamento
        ("DLGS", "472", "1997"),  # Sanzioni tributarie
        ("DLGS", "471", "1997"),  # Sanzioni tributarie
    ],
    "AMMINISTRATIVO": [
        ("LEGGE", "241", "1990"),  # Procedimento amministrativo
        ("DLGS", "165", "2001"),   # Pubblico impiego
        ("DLGS", "104", "2010"),   # Codice processo amministrativo
        ("DLGS", "50", "2016"),    # Codice appalti
        ("DPR", "445", "2000"),    # Documentazione amministrativa
    ],
    "CRISI": [
        ("RD", "267", "1942"),     # Legge fallimentare
        ("DLGS", "14", "2019"),    # Codice della crisi
    ],
    "LAVORO": [
        ("LEGGE", "300", "1970"),  # Statuto lavoratori
        ("DLGS", "66", "2003"),    # Orario di lavoro
        ("DLGS", "81", "2008"),    # Sicurezza lavoro
    ],
}


def estimate_materia_from_signals(
    tipo: Optional[str],
    sezione: Optional[str],
    norms: List[str],
) -> Optional[str]:
    """
    Estimate materia from strong signals for sampling purposes.
    This is NOT the final classification, just for stratification.
    """
    tipo_norm = (tipo or "").strip().lower()
    sez = (sezione or "").strip().lower()

    # Strong signals
    if tipo_norm == "penale":
        return "PENALE"

    # Sezione L -> LAVORO (sezione is just "L", not "Sez. L")
    if sez == "l":
        return "LAVORO"

    # Parse all norms
    parsed_norms = [normalize_norm_for_matching(n) for n in norms if n]
    code_types = {p[0] for p in parsed_norms if p[0]}

    # CP/CPP -> PENALE
    if "CP" in code_types or "CPP" in code_types:
        return "PENALE"

    # Norm hints for minor materie
    for materia, patterns in NORM_HINTS_PATTERNS.items():
        for code_type, num, year in patterns:
            for parsed in parsed_norms:
                p_type, p_num, p_year = parsed
                if p_type == code_type and p_num == num:
                    # Match! (year is optional)
                    if year is None or p_year is None or p_year == year:
                        return materia

    return "CIVILE"  # Default for uncertain


def classify_bucket(
    sezione: Optional[str],
    tipo: Optional[str],
    norms: List[str],
    norms_count: int,
    testo_trunc: str,
) -> str:
    """
    Classify massima into difficulty bucket.
    Sezione values are: "L", "U", "1", "2", "3", "61", "62", etc.
    """
    sez = (sezione or "").strip().lower()
    tipo_norm = (tipo or "").strip().lower()
    testo_lower = testo_trunc.lower()

    # Parse norms to get code types
    parsed_norms = [normalize_norm_for_matching(n) for n in norms if n]
    code_types = {p[0] for p in parsed_norms if p[0]}

    # Easy bucket: Clear metadata signals
    # Sezione L -> LAVORO (easy)
    if sez == "l":
        return "easy"

    # Check norm hints for minor materie (easy bucket)
    for materia, patterns in NORM_HINTS_PATTERNS.items():
        if materia in {"TRIBUTARIO", "AMMINISTRATIVO", "CRISI"}:
            for code_type, num, year in patterns:
                for parsed in parsed_norms:
                    p_type, p_num, p_year = parsed
                    if p_type == code_type and p_num == num:
                        return "easy"

    # tipo=penale with confirming CP/CPP norms -> easy
    if tipo_norm == "penale" and ("CP" in code_types or "CPP" in code_types):
        return "easy"

    # Metadata ambiguous: missing sezione or Sezioni Unite
    if not sez or sez == "u":
        return "metadata_ambiguous"

    # Procedural heavy: CPC/CPP dominated text
    has_cpc = "CPC" in code_types
    has_cpp = "CPP" in code_types
    procedural_keywords = [
        "competenza", "ammissibilità", "termine", "decadenza", "notifica",
        "impugnazione", "ricorso", "appello", "cassazione", "preclusione",
    ]
    procedural_score = sum(1 for kw in procedural_keywords if kw in testo_lower)

    if (has_cpc or has_cpp) and procedural_score >= 2:
        return "procedural_heavy"

    # Cross-domain: Multiple code types
    major_codes = {"CC", "CP", "CPP", "CPC", "DLGS", "DPR", "LEGGE", "RD"}
    codes_found = code_types & major_codes
    if len(codes_found) >= 3:
        return "cross_domain"

    # Also cross-domain if civile sezione (1, 2, 3) + penale norms (CP)
    if sez in {"1", "2", "3"} and "CP" in code_types:
        return "cross_domain"

    # Default to metadata_ambiguous
    return "metadata_ambiguous"


async def fetch_candidates(conn: asyncpg.Connection) -> List[MassimaCandidate]:
    """Fetch all active massime with features for sampling."""

    # Use the feature view
    rows = await conn.fetch("""
        SELECT
            massima_id,
            sezione,
            tipo,
            testo_trunc,
            norms_canonical,
            norms_count
        FROM kb.massime_features_v2
        ORDER BY RANDOM()
    """)

    candidates = []
    for row in rows:
        norms = row["norms_canonical"] or []
        candidate = MassimaCandidate(
            massima_id=row["massima_id"],
            sezione=row["sezione"],
            tipo=row["tipo"],
            norms=norms,
            testo_trunc=row["testo_trunc"] or "",
            norms_count=row["norms_count"] or 0,
        )
        # Classify bucket
        candidate.bucket = classify_bucket(
            candidate.sezione,
            candidate.tipo,
            candidate.norms,
            candidate.norms_count,
            candidate.testo_trunc,
        )
        # Estimate materia for stratification
        candidate.estimated_materia = estimate_materia_from_signals(
            candidate.tipo,
            candidate.sezione,
            candidate.norms,
        )
        candidates.append(candidate)

    return candidates


def stratified_sample(
    candidates: List[MassimaCandidate],
    dry_run: bool = False,
) -> Tuple[List[MassimaCandidate], List[MassimaCandidate]]:
    """
    Perform stratified sampling ensuring:
    1. Bucket quotas are met (420 train + 180 test = 600 total)
    2. Materia minimums in train are enforced
    3. procedural_heavy is 50% civile / 50% penale

    Returns: (train_samples, test_samples)
    """
    TOTAL_TRAIN = 420
    TOTAL_TEST = 180

    # Group candidates by bucket AND materia for efficient sampling
    bucket_pools: Dict[str, List[MassimaCandidate]] = defaultdict(list)
    bucket_materia_pools: Dict[Tuple[str, str], List[MassimaCandidate]] = defaultdict(list)

    for c in candidates:
        bucket_pools[c.bucket].append(c)
        bucket_materia_pools[(c.bucket, c.estimated_materia)].append(c)

    # Shuffle each pool
    for pool in bucket_pools.values():
        random.shuffle(pool)
    for pool in bucket_materia_pools.values():
        random.shuffle(pool)

    train_samples: List[MassimaCandidate] = []
    test_samples: List[MassimaCandidate] = []
    used_ids: Set[UUID] = set()

    # Track materia counts in train
    materia_counts: Dict[str, int] = defaultdict(int)
    # Track bucket counts in train
    bucket_counts: Dict[str, int] = defaultdict(int)

    def add_to_train(c: MassimaCandidate) -> bool:
        if c.massima_id not in used_ids and len(train_samples) < TOTAL_TRAIN:
            train_samples.append(c)
            used_ids.add(c.massima_id)
            materia_counts[c.estimated_materia] += 1
            bucket_counts[c.bucket] += 1
            return True
        return False

    def add_to_test(c: MassimaCandidate) -> bool:
        if c.massima_id not in used_ids and len(test_samples) < TOTAL_TEST:
            test_samples.append(c)
            used_ids.add(c.massima_id)
            return True
        return False

    # Phase 1: Ensure materia minimums in train (prioritize minor materie)
    print("\n=== Phase 1: Ensuring materia minimums ===")

    # Sort by rarity (minor materie first)
    sorted_materie = sorted(
        MATERIA_MINIMUMS_TRAIN.items(),
        key=lambda x: -x[1] if x[0] in {"CIVILE", "PENALE"} else x[1],
        reverse=True
    )

    for materia, minimum in sorted_materie:
        current = materia_counts[materia]
        needed = minimum - current

        if needed <= 0:
            continue

        print(f"  {materia}: need {needed} more (current: {current})")

        # Find candidates with this materia, prefer "easy" bucket
        for bucket in ["easy", "cross_domain", "metadata_ambiguous", "procedural_heavy"]:
            if needed <= 0 or len(train_samples) >= TOTAL_TRAIN:
                break

            # Respect bucket quota
            bucket_quota = BUCKET_CONFIG[bucket]["train"]
            bucket_current = bucket_counts[bucket]
            bucket_remaining = bucket_quota - bucket_current

            pool = bucket_materia_pools.get((bucket, materia), [])
            added = 0
            for c in pool:
                if needed <= 0 or bucket_remaining <= 0:
                    break
                if c.massima_id not in used_ids:
                    if add_to_train(c):
                        needed -= 1
                        bucket_remaining -= 1
                        added += 1

        final_count = materia_counts[materia]
        if final_count < minimum:
            print(f"    WARNING: Could only find {final_count} for {materia} (min: {minimum})")

    # Phase 2: Fill remaining bucket quotas
    print("\n=== Phase 2: Filling bucket quotas ===")
    for bucket, quotas in BUCKET_CONFIG.items():
        train_quota = quotas["train"]
        test_quota = quotas["test"]

        current_train = bucket_counts[bucket]
        need_train = max(0, train_quota - current_train)

        print(f"  {bucket}: have {current_train}, need {need_train} more train, {test_quota} test")

        pool = bucket_pools.get(bucket, [])

        # Special handling for procedural_heavy: aim for 50% civile / 50% penale
        if bucket == "procedural_heavy":
            # Refresh pools excluding used IDs
            civile_pool = [c for c in pool if c.tipo != "penale" and c.massima_id not in used_ids]
            penale_pool = [c for c in pool if c.tipo == "penale" and c.massima_id not in used_ids]

            if need_train > 0:
                # Split train quota
                civile_train_target = need_train // 2
                penale_train_target = need_train - civile_train_target

                # Add civile (up to target or available)
                civile_added = 0
                for c in civile_pool:
                    if civile_added >= civile_train_target:
                        break
                    if add_to_train(c):
                        civile_added += 1

                # Add penale (up to target or available)
                penale_added = 0
                for c in penale_pool:
                    if penale_added >= penale_train_target:
                        break
                    if add_to_train(c):
                        penale_added += 1

                # If penale was short, fill with more civile
                shortfall = need_train - civile_added - penale_added
                if shortfall > 0:
                    extra_civile = [c for c in civile_pool if c.massima_id not in used_ids]
                    for c in extra_civile[:shortfall]:
                        add_to_train(c)

            # Test split - refresh pools
            civile_test_pool = [c for c in pool if c.tipo != "penale" and c.massima_id not in used_ids]
            penale_test_pool = [c for c in pool if c.tipo == "penale" and c.massima_id not in used_ids]

            civile_test_target = test_quota // 2
            penale_test_target = test_quota - civile_test_target

            for c in civile_test_pool[:civile_test_target]:
                add_to_test(c)
            for c in penale_test_pool[:penale_test_target]:
                add_to_test(c)

            # Fill shortfall with civile
            test_shortfall = test_quota - len([s for s in test_samples if s.bucket == bucket])
            if test_shortfall > 0:
                extra_test = [c for c in civile_test_pool if c.massima_id not in used_ids]
                for c in extra_test[:test_shortfall]:
                    add_to_test(c)
        else:
            # Normal bucket filling
            available = [c for c in pool if c.massima_id not in used_ids]

            for c in available[:need_train]:
                add_to_train(c)

            # Test samples from remaining
            test_available = [c for c in pool if c.massima_id not in used_ids]
            for c in test_available[:test_quota]:
                add_to_test(c)

    # Phase 3: Report stats
    print("\n=== Sampling Results ===")
    print(f"  Train: {len(train_samples)}")
    print(f"  Test:  {len(test_samples)}")
    print(f"  Total: {len(train_samples) + len(test_samples)}")

    print("\n  Train by bucket:")
    for bucket in BUCKET_CONFIG:
        count = sum(1 for s in train_samples if s.bucket == bucket)
        print(f"    {bucket}: {count}")

    print("\n  Train by materia:")
    for materia in sorted(materia_counts.keys()):
        count = materia_counts[materia]
        minimum = MATERIA_MINIMUMS_TRAIN.get(materia, 0)
        status = "OK" if count >= minimum else "BELOW MIN"
        print(f"    {materia}: {count} (min: {minimum}) [{status}]")

    print("\n  Test by bucket:")
    for bucket in BUCKET_CONFIG:
        count = sum(1 for s in test_samples if s.bucket == bucket)
        print(f"    {bucket}: {count}")

    return train_samples, test_samples


async def save_golden_set(
    conn: asyncpg.Connection,
    train_samples: List[MassimaCandidate],
    test_samples: List[MassimaCandidate],
    dry_run: bool,
):
    """Save golden set to database."""

    if dry_run:
        print("\n[DRY RUN] Would insert golden set, but not committing.")
        return

    # Clear existing entries (if any)
    await conn.execute("DELETE FROM kb.golden_category_adjudicated_v2")
    await conn.execute("DELETE FROM kb.golden_category_labels_v2")

    # Insert train samples
    for s in train_samples:
        await conn.execute("""
            INSERT INTO kb.golden_category_adjudicated_v2
                (massima_id, materia_l1, natura_l1, difficulty_bucket, split)
            VALUES ($1, 'PENDING', 'PENDING', $2, 'train')
        """, s.massima_id, s.bucket)

    # Insert test samples
    for s in test_samples:
        await conn.execute("""
            INSERT INTO kb.golden_category_adjudicated_v2
                (massima_id, materia_l1, natura_l1, difficulty_bucket, split)
            VALUES ($1, 'PENDING', 'PENDING', $2, 'test')
        """, s.massima_id, s.bucket)

    print(f"\n[COMMITTED] Inserted {len(train_samples)} train + {len(test_samples)} test samples")


async def main(dry_run: bool = True, seed: int = 42):
    """Main sampling routine."""

    random.seed(seed)
    print(f"Random seed: {seed}")

    settings = KBSettings()
    conn = await asyncpg.connect(settings.kb_database_url)

    try:
        # Check if feature view exists
        view_exists = await conn.fetchval("""
            SELECT EXISTS (
                SELECT 1 FROM information_schema.views
                WHERE table_schema = 'kb' AND table_name = 'massime_features_v2'
            )
        """)

        if not view_exists:
            print("ERROR: View kb.massime_features_v2 does not exist.")
            print("Run migration 009_category_v2.sql first.")
            return

        print("Fetching candidates from kb.massime_features_v2...")
        candidates = await fetch_candidates(conn)
        print(f"Total candidates: {len(candidates)}")

        # Report bucket distribution
        bucket_counts = defaultdict(int)
        for c in candidates:
            bucket_counts[c.bucket] += 1

        print("\nBucket distribution in corpus:")
        for bucket, count in sorted(bucket_counts.items()):
            pct = 100 * count / len(candidates)
            print(f"  {bucket}: {count} ({pct:.1f}%)")

        # Report estimated materia distribution
        materia_counts = defaultdict(int)
        for c in candidates:
            materia_counts[c.estimated_materia] += 1

        print("\nEstimated materia distribution:")
        for materia, count in sorted(materia_counts.items(), key=lambda x: -x[1]):
            pct = 100 * count / len(candidates)
            print(f"  {materia}: {count} ({pct:.1f}%)")

        # Perform stratified sampling
        train_samples, test_samples = stratified_sample(candidates, dry_run)

        # Save to database
        await save_golden_set(conn, train_samples, test_samples, dry_run)

    finally:
        await conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Golden Set v2")
    parser.add_argument("--dry-run", action="store_true", help="Don't commit to database")
    parser.add_argument("--commit", action="store_true", help="Commit to database")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    if not args.commit and not args.dry_run:
        print("Specify --dry-run or --commit")
        sys.exit(1)

    dry_run = not args.commit
    asyncio.run(main(dry_run=dry_run, seed=args.seed))
