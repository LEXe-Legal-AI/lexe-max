"""
LEXE Knowledge Base - Deduplicator

Deduplicazione massime: hash esatto + similarity fuzzy.
"""

import hashlib
from dataclasses import dataclass
from typing import Literal

import structlog

from .cleaner import normalize_for_hash

logger = structlog.get_logger(__name__)


DuplicateType = Literal["exact", "near"]


@dataclass
class DuplicateMatch:
    """Match di duplicato trovato."""

    original_hash: str
    duplicate_hash: str
    match_type: DuplicateType
    similarity: float  # 1.0 per exact, 0-1 per near
    original_id: str | None = None
    duplicate_id: str | None = None


@dataclass
class DeduplicationResult:
    """Risultato deduplicazione batch."""

    total_input: int
    unique_count: int
    exact_duplicates: int
    near_duplicates: int
    matches: list[DuplicateMatch]


def compute_content_hash(text: str) -> str:
    """
    Calcola hash SHA256 del testo normalizzato.

    Args:
        text: Testo (verra' normalizzato)

    Returns:
        Hash hex
    """
    normalized = normalize_for_hash(text)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def compute_simhash(text: str, num_bits: int = 64) -> int:
    """
    Calcola SimHash per similarity detection.

    SimHash e' locality-sensitive: testi simili hanno hash simili.

    Args:
        text: Testo normalizzato
        num_bits: Dimensione hash (default 64)

    Returns:
        SimHash come intero
    """
    if not text:
        return 0

    # Tokenizza in shingles (n-grammi di parole)
    words = text.lower().split()
    shingles = []
    for i in range(len(words) - 2):
        shingles.append(" ".join(words[i : i + 3]))

    if not shingles:
        # Fallback per testi corti
        shingles = words

    # Vettore di pesi
    v = [0] * num_bits

    for shingle in shingles:
        # Hash dello shingle
        h = int(hashlib.md5(shingle.encode()).hexdigest(), 16)

        for i in range(num_bits):
            bit = (h >> i) & 1
            if bit:
                v[i] += 1
            else:
                v[i] -= 1

    # Converti in fingerprint
    fingerprint = 0
    for i in range(num_bits):
        if v[i] > 0:
            fingerprint |= 1 << i

    return fingerprint


def hamming_distance(a: int, b: int) -> int:
    """Calcola distanza di Hamming tra due interi."""
    return bin(a ^ b).count("1")


def simhash_similarity(a: int, b: int, num_bits: int = 64) -> float:
    """
    Calcola similarita' basata su SimHash.

    Returns:
        Similarita' 0-1 (1 = identici)
    """
    distance = hamming_distance(a, b)
    return 1.0 - (distance / num_bits)


class Deduplicator:
    """
    Deduplicatore per massime.

    Strategia a due livelli:
    1. Hash esatto per duplicati identici
    2. SimHash + soglia per near-duplicates
    """

    def __init__(
        self,
        exact_threshold: float = 1.0,
        near_threshold: float = 0.85,
        num_bits: int = 64,
    ):
        """
        Inizializza deduplicatore.

        Args:
            exact_threshold: Soglia per duplicato esatto (default 1.0)
            near_threshold: Soglia per near-duplicate (default 0.85)
            num_bits: Bit per SimHash
        """
        self.exact_threshold = exact_threshold
        self.near_threshold = near_threshold
        self.num_bits = num_bits

        # Cache di hash visti
        self._exact_hashes: dict[str, str] = {}  # hash -> id
        self._simhashes: dict[str, tuple[int, str]] = {}  # hash -> (simhash, id)

    def add_existing(self, content_hash: str, massima_id: str, text: str) -> None:
        """
        Aggiungi massima esistente al deduplicatore.

        Chiamare per popolare da database esistente.

        Args:
            content_hash: Hash SHA256 del testo normalizzato
            massima_id: ID massima nel database
            text: Testo normalizzato
        """
        self._exact_hashes[content_hash] = massima_id

        simhash = compute_simhash(text, self.num_bits)
        self._simhashes[content_hash] = (simhash, massima_id)

    def check_duplicate(
        self,
        text: str,
    ) -> tuple[bool, DuplicateMatch | None]:
        """
        Verifica se testo e' duplicato.

        Args:
            text: Testo da verificare

        Returns:
            (is_duplicate, match) - match contiene dettagli se duplicato
        """
        normalized = normalize_for_hash(text)
        content_hash = compute_content_hash(text)

        # 1. Check duplicato esatto
        if content_hash in self._exact_hashes:
            return True, DuplicateMatch(
                original_hash=content_hash,
                duplicate_hash=content_hash,
                match_type="exact",
                similarity=1.0,
                original_id=self._exact_hashes[content_hash],
            )

        # 2. Check near-duplicate con SimHash
        simhash = compute_simhash(normalized, self.num_bits)

        for existing_hash, (existing_simhash, existing_id) in self._simhashes.items():
            similarity = simhash_similarity(simhash, existing_simhash, self.num_bits)

            if similarity >= self.near_threshold:
                return True, DuplicateMatch(
                    original_hash=existing_hash,
                    duplicate_hash=content_hash,
                    match_type="near",
                    similarity=similarity,
                    original_id=existing_id,
                )

        return False, None

    def deduplicate_batch(
        self,
        items: list[tuple[str, str]],  # (id, text)
    ) -> DeduplicationResult:
        """
        Deduplica batch di testi.

        Args:
            items: Lista di (id, testo)

        Returns:
            DeduplicationResult con statistiche e match
        """
        matches: list[DuplicateMatch] = []
        unique_ids: set[str] = set()
        exact_count = 0
        near_count = 0

        for item_id, text in items:
            is_dup, match = self.check_duplicate(text)

            if is_dup and match:
                match.duplicate_id = item_id
                matches.append(match)

                if match.match_type == "exact":
                    exact_count += 1
                else:
                    near_count += 1
            else:
                # Nuovo unico - aggiungi alla cache
                unique_ids.add(item_id)
                content_hash = compute_content_hash(text)
                self._exact_hashes[content_hash] = item_id

                normalized = normalize_for_hash(text)
                simhash = compute_simhash(normalized, self.num_bits)
                self._simhashes[content_hash] = (simhash, item_id)

        return DeduplicationResult(
            total_input=len(items),
            unique_count=len(unique_ids),
            exact_duplicates=exact_count,
            near_duplicates=near_count,
            matches=matches,
        )

    def reset(self) -> None:
        """Reset cache interna."""
        self._exact_hashes.clear()
        self._simhashes.clear()


def jaccard_similarity(text1: str, text2: str, shingle_size: int = 3) -> float:
    """
    Calcola Jaccard similarity tra due testi.

    Piu' accurato di SimHash ma O(n).

    Args:
        text1, text2: Testi da confrontare
        shingle_size: Dimensione shingles (default 3 parole)

    Returns:
        Similarita' 0-1
    """
    def get_shingles(text: str) -> set[str]:
        words = text.lower().split()
        if len(words) < shingle_size:
            return set(words)
        return {
            " ".join(words[i : i + shingle_size])
            for i in range(len(words) - shingle_size + 1)
        }

    shingles1 = get_shingles(text1)
    shingles2 = get_shingles(text2)

    if not shingles1 or not shingles2:
        return 0.0

    intersection = len(shingles1 & shingles2)
    union = len(shingles1 | shingles2)

    return intersection / union if union > 0 else 0.0


def find_near_duplicates_quadratic(
    items: list[tuple[str, str]],
    threshold: float = 0.85,
) -> list[DuplicateMatch]:
    """
    Trova near-duplicates con confronto quadratico.

    Piu' preciso ma O(n^2) - usare solo per batch piccoli.

    Args:
        items: Lista di (id, text)
        threshold: Soglia similarita'

    Returns:
        Lista match
    """
    matches = []

    for i, (id1, text1) in enumerate(items):
        for id2, text2 in items[i + 1 :]:
            sim = jaccard_similarity(text1, text2)
            if sim >= threshold:
                matches.append(
                    DuplicateMatch(
                        original_hash=compute_content_hash(text1),
                        duplicate_hash=compute_content_hash(text2),
                        match_type="near",
                        similarity=sim,
                        original_id=id1,
                        duplicate_id=id2,
                    )
                )

    return matches


async def load_existing_hashes(db_pool) -> dict[str, str]:
    """
    Carica hash esistenti dal database.

    Args:
        db_pool: Connection pool database

    Returns:
        Dict content_hash -> massima_id
    """
    query = """
    SELECT id::text, content_hash, testo_normalizzato
    FROM kb.massime
    """

    async with db_pool.acquire() as conn:
        rows = await conn.fetch(query)

    result = {}
    for row in rows:
        result[row["content_hash"]] = row["id"]

    logger.info(
        "Loaded existing hashes",
        count=len(result),
    )

    return result
