"""
LEXE Knowledge Base - Text Cleaner

Pulizia e normalizzazione testo giuridico per massimari.
"""

import hashlib
import re
import unicodedata
from typing import Optional


def clean_legal_text(text: str) -> str:
    """
    Pulizia editoriale per testo da massimari.

    - Unisce parole spezzate da a-capo
    - Normalizza spazi multipli
    - Normalizza virgolette e trattini
    - Rimuove caratteri di controllo

    Args:
        text: Testo grezzo da OCR

    Returns:
        Testo pulito
    """
    if not text:
        return ""

    # Unisci parole spezzate da a-capo (es. "compe-\ntenza" -> "competenza")
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)

    # Normalizza newlines
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Normalizza spazi multipli e newlines multipli
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Normalizza virgolette italiane
    text = text.replace("«", '"').replace("»", '"')
    text = text.replace(""", '"').replace(""", '"')
    text = text.replace("'", "'").replace("'", "'")

    # Normalizza trattini
    text = text.replace("–", "-").replace("—", "-")

    # Normalizza ellissi
    text = text.replace("…", "...")

    # Rimuovi caratteri di controllo (mantieni newline e tab)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

    # Normalizza Unicode (NFC)
    text = unicodedata.normalize("NFC", text)

    # Strip whitespace esterno
    text = text.strip()

    # Rimuovi spazi prima di punteggiatura
    text = re.sub(r"\s+([.,;:!?)])", r"\1", text)

    # Aggiungi spazio dopo punteggiatura se manca
    text = re.sub(r"([.,;:!?])([A-Za-zÀ-ÿ])", r"\1 \2", text)

    return text


def normalize_for_hash(text: str) -> str:
    """
    Normalizza testo per deduplicazione.

    - Lowercase
    - Rimuove punteggiatura
    - Normalizza spazi
    - Rimuove accenti (opzionale, per fuzzy match)

    Args:
        text: Testo pulito

    Returns:
        Testo normalizzato per hash/confronto
    """
    if not text:
        return ""

    # Prima pulisci
    text = clean_legal_text(text)

    # Lowercase
    text = text.lower()

    # Rimuovi punteggiatura (mantieni numeri e lettere)
    text = re.sub(r"[^\w\s]", "", text)

    # Normalizza spazi
    text = " ".join(text.split())

    return text


def compute_content_hash(text: str) -> str:
    """
    Calcola hash SHA256 del testo normalizzato.

    Args:
        text: Testo (verra' normalizzato internamente)

    Returns:
        Hash SHA256 come stringa hex
    """
    normalized = normalize_for_hash(text)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def normalize_citation_text(text: str) -> str:
    """
    Normalizza citazioni per confronto.

    Es: "Sez. U, n. 12345/2020" -> "sez u n 12345 2020"
    """
    if not text:
        return ""

    text = text.lower()
    # Rimuovi punteggiatura eccetto numeri
    text = re.sub(r"[^\w\s]", " ", text)
    text = " ".join(text.split())

    return text


def extract_first_sentence(text: str) -> str:
    """
    Estrai prima frase (per tema/titolo).

    Args:
        text: Testo massima

    Returns:
        Prima frase (max 200 chars)
    """
    if not text:
        return ""

    # Cerca "In tema di..." come pattern preferito
    match = re.search(r"[Ii]n tema di\s+([^,.]+)", text)
    if match:
        return f"In tema di {match.group(1).strip()}"

    # Altrimenti prima frase
    sentences = re.split(r"[.!?]", text)
    if sentences:
        first = sentences[0].strip()
        return first[:200] if len(first) > 200 else first

    return text[:200]


def clean_ocr_artifacts(text: str) -> str:
    """
    Rimuove artefatti comuni da OCR.

    - Caratteri ripetuti erroneamente
    - Pattern di rumore
    - Numeri di pagina isolati
    """
    if not text:
        return ""

    # Rimuovi caratteri ripetuti piu' di 3 volte (es. "|||||||")
    text = re.sub(r"(.)\1{3,}", r"\1\1", text)

    # Rimuovi linee che sono solo numeri (numeri pagina)
    lines = text.split("\n")
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        # Skip linee che sono solo numeri (1-4 cifre)
        if re.match(r"^\d{1,4}$", stripped):
            continue
        # Skip linee che sono solo punteggiatura
        if re.match(r"^[.\-_=]+$", stripped):
            continue
        cleaned_lines.append(line)

    text = "\n".join(cleaned_lines)

    # Rimuovi sequenze di punti (es. ".....")
    text = re.sub(r"\.{4,}", "...", text)

    return text


def is_likely_header_or_footer(text: str) -> bool:
    """
    Verifica se il testo e' probabilmente header/footer di pagina.
    """
    if not text or len(text) > 200:
        return False

    text_lower = text.lower().strip()

    # Pattern comuni header/footer
    patterns = [
        r"^\d+$",  # Solo numero pagina
        r"^-\s*\d+\s*-$",  # - 123 -
        r"^pagina\s+\d+",  # Pagina X
        r"^massimario\s+(civile|penale)",  # Titolo documento
        r"^corte\s+di\s+cassazione",  # Intestazione
        r"^\d{4}$",  # Anno solo
        r"^vol(ume)?\.?\s*\d+",  # Volume
    ]

    for pattern in patterns:
        if re.match(pattern, text_lower):
            return True

    return False


def merge_split_paragraphs(texts: list[str]) -> list[str]:
    """
    Unisce paragrafi che sono stati splittati erroneamente dall'OCR.

    Logica: se un blocco finisce con una parola spezzata o senza punteggiatura
    finale, uniscilo al successivo.
    """
    if not texts:
        return []

    merged = []
    current = ""

    for text in texts:
        text = text.strip()
        if not text:
            continue

        if not current:
            current = text
            continue

        # Verifica se current finisce in modo incompleto
        ends_incomplete = (
            current.endswith("-")  # Parola spezzata
            or (
                not current.endswith((".", "!", "?", ":", ";", ")"))
                and len(current) > 10
            )  # No punteggiatura finale
        )

        if ends_incomplete:
            # Unisci
            if current.endswith("-"):
                # Rimuovi trattino e unisci
                current = current[:-1] + text
            else:
                current = current + " " + text
        else:
            # Salva current e inizia nuovo
            merged.append(current)
            current = text

    if current:
        merged.append(current)

    return merged
