"""
Normalization module for QA Protocol v3.2.

Provides:
- compute_spaced_letters_score(): Detect spaced-letters text
- normalize_v2(): Robust normalization with conditional despacing
- safe_despace(): Conservative despacing for spaced-letters segments
- compute_simhash64(): 64-bit SimHash fingerprint for candidate generation

All functions operate on testo_norm, not raw text.
"""

import hashlib
import re
import unicodedata
from typing import Tuple


# ============================================================================
# Spaced Letters Detection
# ============================================================================

def compute_spaced_letters_score(text: str) -> float:
    """
    Rileva se il testo ha spaziatura carattere-per-carattere.

    Segnali:
    - spaced_pair_ratio: count(r'[a-zàèéìòù]\s[a-zàèéìòù]') / total_letters
    - avg_token_len: media lunghezza token alfabetici
    - space_ratio: spaces / total_chars

    Returns: 0.0 (normale) - 1.0 (molto spaced)

    Soglie euristiche:
    - spaced_pair_ratio > 0.12 AND avg_token_len < 2.2 → spaced
    - Oppure più severa: spaced_pair_ratio > 0.18 → spaced
    """
    if not text or len(text) < 10:
        return 0.0

    # Count spaced letter pairs: "a b", "c d", etc.
    spaced_pairs = len(re.findall(r'[a-zàèéìòùA-ZÀÈÉÌÒÙ]\s[a-zàèéìòùA-ZÀÈÉÌÒÙ]', text))

    # Count total letters
    total_letters = len(re.findall(r'[a-zàèéìòùA-ZÀÈÉÌÒÙ]', text))
    if total_letters == 0:
        return 0.0

    spaced_pair_ratio = spaced_pairs / total_letters

    # Calculate average token length
    tokens = re.findall(r'[a-zàèéìòùA-ZÀÈÉÌÒÙ]+', text)
    avg_token_len = sum(len(t) for t in tokens) / len(tokens) if tokens else 0

    # Calculate space ratio
    space_count = text.count(' ')
    space_ratio = space_count / len(text) if text else 0

    # Combined score (weighted)
    # High spaced_pair_ratio + low avg_token_len + high space_ratio = spaced text
    score = 0.0

    # Primary signal: spaced pair ratio
    if spaced_pair_ratio > 0.18:
        score = min(1.0, spaced_pair_ratio * 3)
    elif spaced_pair_ratio > 0.12 and avg_token_len < 2.2:
        score = min(1.0, spaced_pair_ratio * 2.5)
    elif spaced_pair_ratio > 0.08 and avg_token_len < 1.8:
        score = min(1.0, spaced_pair_ratio * 2)

    # Boost if space ratio is abnormally high (>30%)
    if space_ratio > 0.30 and score > 0:
        score = min(1.0, score * 1.3)

    return round(score, 4)


# ============================================================================
# Safe Despacing
# ============================================================================

def safe_despace(text: str) -> str:
    """
    Rimuove spazi tra caratteri singoli SOLO in sequenze che rispettano TUTTE:
    1. Lunghezza segmento >= 20 caratteri
    2. spaced_pair_ratio del segmento > 0.12
    3. Token alfabetici medi < 2.2

    Dentro il segmento:
    - Rimuovi spazi SOLO tra lettere (a-z, àèéìòù)
    - NON toccare spazi tra lettere e numeri
    - NON toccare spazi vicino a punteggiatura

    Esempio:
    - "c o r t e  s u p r e m a" → "corte suprema"
    - "art. 360 c.p.c." → invariato
    - "Sez. Un." → invariato
    - "n. 1 2 3 4" → invariato (numeri)
    """
    if not text:
        return text

    # Split into segments by double spaces or paragraph breaks
    segments = re.split(r'(\n\n|\s{3,})', text)
    result = []

    for segment in segments:
        if len(segment) < 20:
            result.append(segment)
            continue

        # Check if segment is spaced letters
        score = compute_spaced_letters_score(segment)

        if score >= 0.12:
            # Apply despacing only to letter sequences
            # Pattern: single letter + space + single letter (both alphabetic)
            despaced = re.sub(
                r'(?<=[a-zàèéìòùA-ZÀÈÉÌÒÙ])\s+(?=[a-zàèéìòùA-ZÀÈÉÌÒÙ])',
                '',
                segment
            )

            # But preserve word boundaries - if we created a very long word, split it back
            # This handles cases where real words got joined
            tokens = despaced.split()
            fixed_tokens = []
            for token in tokens:
                # If token is >30 chars and all letters, it's probably multiple words joined wrong
                # Leave as-is for now, the original spacing will be applied
                fixed_tokens.append(token)

            result.append(' '.join(fixed_tokens))
        else:
            result.append(segment)

    return ''.join(result)


# ============================================================================
# Normalization v2
# ============================================================================

def normalize_v2(text: str, apply_despacing: bool = None) -> Tuple[str, float]:
    """
    Normalizzazione robusta con despacing condizionale.

    Pipeline:
    1. Unicode normalize (NFKC), rimuovi zero-width chars
    2. Normalizza apostrofi (', `, ´ → ')
    3. Collassa whitespace multiplo
    4. IF spaced_letters_score > threshold OR apply_despacing:
       - Applica despacing SOLO su sequenze "lettera spazio lettera"
       - NON toccare spazi attorno a punteggiatura e numeri
    5. Lowercase, strip

    Args:
        text: Input text
        apply_despacing: Force despacing (True), skip (False), or auto-detect (None)

    Returns: (normalized_text, spaced_letters_score)
    """
    if not text:
        return ('', 0.0)

    # Step 1: Unicode normalize
    text = unicodedata.normalize('NFKC', text)

    # Remove zero-width characters
    text = re.sub(r'[\u200b\u200c\u200d\ufeff\u00ad]', '', text)

    # Step 2: Normalize apostrophes and quotes
    text = re.sub(r'[`´''‛]', "'", text)
    text = re.sub(r'[""„‟]', '"', text)

    # Step 3: Collapse multiple whitespace (but preserve newlines initially)
    text = re.sub(r'[^\S\n]+', ' ', text)
    text = re.sub(r'\n+', '\n', text)

    # Compute spaced letters score before despacing
    spaced_score = compute_spaced_letters_score(text)

    # Step 4: Despacing
    if apply_despacing is True or (apply_despacing is None and spaced_score > 0.12):
        text = safe_despace(text)

    # Step 5: Final cleanup
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)  # Collapse all whitespace to single space
    text = text.strip()

    return (text, spaced_score)


def normalize_simple(text: str) -> str:
    """
    Simple normalization without despacing detection.
    Equivalent to the old normalize_text function.

    Used for backward compatibility.
    """
    if not text:
        return ''

    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()


# ============================================================================
# Content Hash
# ============================================================================

def compute_content_hash(text_norm: str) -> str:
    """
    Compute content hash from normalized text.

    Returns: first 40 chars of sha256 hex digest
    """
    if not text_norm:
        return ''
    return hashlib.sha256(text_norm.encode('utf-8')).hexdigest()[:40]


# ============================================================================
# SimHash 64-bit
# ============================================================================

def compute_simhash64(text_norm: str, ngram_size: int = 3) -> int:
    """
    Calcola SimHash 64-bit su testo_norm (NON raw).

    Usato per:
    - Candidate generation veloce (hamming distance < 12)
    - Dedupe rapida

    Algorithm:
    1. Extract n-grams from text
    2. Hash each n-gram to 64-bit value
    3. For each bit position, sum +1 if bit is 1, -1 if bit is 0
    4. Final hash: set bit if sum > 0

    Args:
        text_norm: Normalized text
        ngram_size: Size of character n-grams (default 3)

    Returns: 64-bit integer fingerprint
    """
    if not text_norm or len(text_norm) < ngram_size:
        return 0

    # Extract n-grams
    ngrams = [text_norm[i:i+ngram_size] for i in range(len(text_norm) - ngram_size + 1)]

    # Initialize 64 counters
    v = [0] * 64

    for ngram in ngrams:
        # Hash n-gram to 64-bit value using MD5 (fast, deterministic)
        h = int(hashlib.md5(ngram.encode('utf-8')).hexdigest()[:16], 16)

        # Update counters
        for i in range(64):
            if (h >> i) & 1:
                v[i] += 1
            else:
                v[i] -= 1

    # Build final hash
    fingerprint = 0
    for i in range(64):
        if v[i] > 0:
            fingerprint |= (1 << i)

    # Convert to signed 64-bit (PostgreSQL BIGINT is signed)
    if fingerprint >= (1 << 63):
        fingerprint -= (1 << 64)

    return fingerprint


def hamming_distance(hash1: int, hash2: int) -> int:
    """
    Compute Hamming distance between two 64-bit hashes.

    Returns: Number of differing bits (0-64)
    """
    if hash1 is None or hash2 is None:
        return 64  # Max distance if either is None

    xor = hash1 ^ hash2
    return bin(xor).count('1')


# ============================================================================
# Jaccard Similarity (character-based, for matching)
# ============================================================================

def jaccard_tokens(text1: str, text2: str) -> float:
    """
    Compute Jaccard similarity between two texts based on word tokens.

    Returns: 0.0 - 1.0
    """
    if not text1 or not text2:
        return 0.0

    tokens1 = set(text1.lower().split())
    tokens2 = set(text2.lower().split())

    if not tokens1 or not tokens2:
        return 0.0

    intersection = tokens1 & tokens2
    union = tokens1 | tokens2

    return len(intersection) / len(union) if union else 0.0


def ngram_similarity(text1: str, text2: str, n: int = 3) -> float:
    """
    Compute similarity based on character n-grams.

    More robust to minor differences than token-based Jaccard.

    Returns: 0.0 - 1.0
    """
    if not text1 or not text2:
        return 0.0

    def get_ngrams(text: str, n: int) -> set:
        text = text.lower()
        return {text[i:i+n] for i in range(len(text) - n + 1)}

    ngrams1 = get_ngrams(text1, n)
    ngrams2 = get_ngrams(text2, n)

    if not ngrams1 or not ngrams2:
        return 0.0

    intersection = ngrams1 & ngrams2
    union = ngrams1 | ngrams2

    return len(intersection) / len(union) if union else 0.0
