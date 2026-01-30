"""
Generate Embeddings for KB Massimari Benchmark
Supporta: BGE-M3, distil-ita-legal-bert, Qwen3 (via LiteLLM)

Usage:
    python scripts/generate_embeddings.py --model bge_m3
    python scripts/generate_embeddings.py --model ita_legal_bert
    python scripts/generate_embeddings.py --model qwen3 --litellm
"""
import argparse
import asyncio
import os
import sys
from dataclasses import dataclass
from typing import Optional
from uuid import UUID

import asyncpg
import numpy as np

# Config
DB_URL = "postgresql://lexe_kb:lexe_kb_dev_password@localhost:5434/lexe_kb"

# Model configs
MODEL_CONFIGS = {
    "bge_m3": {
        "hf_name": "BAAI/bge-m3",
        "dim": 1024,
        "max_tokens": 8192,
        "table": "kb.emb_bge_m3",
        "needs_chunking": False,
        "type": "huggingface",
    },
    "e5_large": {
        "hf_name": "intfloat/multilingual-e5-large",
        "dim": 1024,
        "max_tokens": 512,
        "table": "kb.emb_e5_large",
        "needs_chunking": True,
        "type": "huggingface",
    },
    "ita_legal_bert": {
        "hf_name": "dlicari/distil-ita-legal-bert",
        "dim": 768,
        "max_tokens": 512,
        "table": "kb.emb_ita_legal_bert",
        "needs_chunking": True,
        "type": "huggingface",
    },
    "qwen3": {
        "litellm_model": "text-embedding-3-large",  # placeholder for custom Qwen
        "dim": 1536,
        "max_tokens": 8191,
        "table": "kb.emb_qwen3",
        "needs_chunking": False,
        "type": "litellm",
    },
    "openai_large": {
        "litellm_model": "text-embedding-3-large",
        "dim": 2000,  # Using dimensions param (max for HNSW)
        "max_tokens": 8191,
        "table": "kb.emb_openai_large",
        "needs_chunking": False,
        "type": "openai",
    },
}

# Chunking config
CHUNK_SIZE_TOKENS = 300
CHUNK_OVERLAP_TOKENS = 60


@dataclass
class Massima:
    id: UUID
    testo: str
    anno: int
    tipo: str


@dataclass
class EmbeddingResult:
    massima_id: UUID
    chunk_idx: int
    embedding: list[float]
    text_span: str


def estimate_tokens(text: str) -> int:
    """Stima approssimativa dei token (4 char ~= 1 token per italiano)."""
    return len(text) // 4


def chunk_text(text: str, max_tokens: int, overlap_tokens: int) -> list[tuple[int, str]]:
    """
    Spezza il testo in chunk con overlap.
    Returns: list of (chunk_idx, chunk_text)
    """
    if estimate_tokens(text) <= max_tokens:
        return [(0, text)]

    # Converti tokens in caratteri (approssimativo)
    max_chars = max_tokens * 4
    overlap_chars = overlap_tokens * 4

    chunks = []
    start = 0
    chunk_idx = 0

    while start < len(text):
        end = min(start + max_chars, len(text))

        # Cerca un punto naturale di rottura (., !, ?, newline)
        if end < len(text):
            # Cerca indietro per trovare fine frase
            for i in range(end, max(start + max_chars // 2, start), -1):
                if text[i] in ".!?\n":
                    end = i + 1
                    break

        chunk_text_span = text[start:end].strip()
        if chunk_text_span:
            chunks.append((chunk_idx, chunk_text_span))
            chunk_idx += 1

        # Avanza con overlap
        start = end - overlap_chars
        if start >= len(text) - overlap_chars:
            break

    return chunks


class HuggingFaceEmbedder:
    """Embedder usando sentence-transformers."""

    def __init__(self, model_name: str):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            print("ERROR: sentence-transformers not installed")
            print("Run: uv add sentence-transformers")
            sys.exit(1)

        print(f"Loading model: {model_name}")
        self.model = SentenceTransformer(model_name, trust_remote_code=True)
        print(f"Model loaded. Dimension: {self.model.get_sentence_embedding_dimension()}")

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a batch of texts."""
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        return embeddings.tolist()


class LiteLLMEmbedder:
    """Embedder usando LiteLLM (per Qwen3 o OpenAI-compatible)."""

    def __init__(self, model_name: str, dimensions: Optional[int] = None):
        try:
            import litellm
            self.litellm = litellm
        except ImportError:
            print("ERROR: litellm not installed")
            print("Run: uv add litellm")
            sys.exit(1)

        self.model_name = model_name
        self.dimensions = dimensions
        print(f"Using LiteLLM model: {model_name}")
        if dimensions:
            print(f"  Dimensions: {dimensions}")

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings via LiteLLM."""
        kwargs = {
            "model": self.model_name,
            "input": texts,
        }
        if self.dimensions:
            kwargs["dimensions"] = self.dimensions

        response = self.litellm.embedding(**kwargs)
        return [item["embedding"] for item in response.data]


class OpenAIEmbedder:
    """Embedder usando OpenAI API direttamente (per text-embedding-3-large)."""

    def __init__(self, model_name: str = "text-embedding-3-large", dimensions: int = 2000):
        try:
            from openai import OpenAI
            self.client = OpenAI()  # Uses OPENAI_API_KEY env var
        except ImportError:
            print("ERROR: openai not installed")
            print("Run: uv add openai")
            sys.exit(1)

        self.model_name = model_name
        self.dimensions = dimensions
        print(f"Using OpenAI model: {model_name}")
        print(f"  Dimensions: {dimensions}")

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings via OpenAI API."""
        response = self.client.embeddings.create(
            model=self.model_name,
            input=texts,
            dimensions=self.dimensions,
        )
        return [item.embedding for item in response.data]


async def load_massime(conn: asyncpg.Connection) -> list[Massima]:
    """Load all massime from database."""
    rows = await conn.fetch("""
        SELECT m.id, m.testo, d.anno, d.tipo
        FROM kb.massime m
        JOIN kb.documents d ON d.id = m.document_id
        ORDER BY d.anno, m.id
    """)

    return [
        Massima(
            id=row["id"],
            testo=row["testo"],
            anno=row["anno"],
            tipo=row["tipo"],
        )
        for row in rows
    ]


async def clear_embeddings(conn: asyncpg.Connection, table: str):
    """Clear existing embeddings for fresh run."""
    await conn.execute(f"DELETE FROM {table}")
    print(f"Cleared {table}")


async def insert_embeddings(
    conn: asyncpg.Connection,
    table: str,
    results: list[EmbeddingResult],
    batch_size: int = 100,
):
    """Insert embeddings in batches."""
    for i in range(0, len(results), batch_size):
        batch = results[i : i + batch_size]

        # Prepare values
        values = [
            (r.massima_id, r.chunk_idx, r.embedding)
            for r in batch
        ]

        await conn.executemany(
            f"""
            INSERT INTO {table} (massima_id, chunk_idx, embedding)
            VALUES ($1, $2, $3)
            ON CONFLICT (massima_id, chunk_idx) DO UPDATE
            SET embedding = EXCLUDED.embedding
            """,
            values,
        )

    print(f"Inserted {len(results)} embeddings into {table}")


async def generate_for_model(
    model_key: str,
    use_litellm: bool = False,
    batch_size: int = 32,
    clear_first: bool = True,
):
    """Generate embeddings for a specific model."""
    config = MODEL_CONFIGS.get(model_key)
    if not config:
        print(f"ERROR: Unknown model '{model_key}'")
        print(f"Available: {list(MODEL_CONFIGS.keys())}")
        return

    print("=" * 70)
    print(f"GENERATE EMBEDDINGS: {model_key}")
    print("=" * 70)
    print(f"Dimension: {config['dim']}")
    print(f"Max tokens: {config['max_tokens']}")
    print(f"Table: {config['table']}")
    print(f"Needs chunking: {config['needs_chunking']}")

    # Initialize embedder based on type
    model_type = config.get("type", "huggingface")

    if model_type == "openai":
        embedder = OpenAIEmbedder(
            model_name=config.get("litellm_model", "text-embedding-3-large"),
            dimensions=config["dim"],
        )
    elif model_type == "litellm" or use_litellm:
        embedder = LiteLLMEmbedder(
            model_name=config.get("litellm_model", "text-embedding-3-large"),
            dimensions=config.get("dim") if config.get("dim", 0) != 1536 else None,
        )
    else:
        embedder = HuggingFaceEmbedder(config["hf_name"])

    # Connect to DB
    conn = await asyncpg.connect(DB_URL)
    print("[OK] Database connected")

    # Clear existing
    if clear_first:
        await clear_embeddings(conn, config["table"])

    # Load massime
    massime = await load_massime(conn)
    print(f"Loaded {len(massime)} massime")

    # Prepare texts and track chunks
    texts_to_embed = []
    text_metadata = []  # (massima_id, chunk_idx)

    for m in massime:
        if config["needs_chunking"]:
            chunks = chunk_text(m.testo, config["max_tokens"], CHUNK_OVERLAP_TOKENS)
        else:
            chunks = [(0, m.testo)]

        for chunk_idx, chunk_text_span in chunks:
            texts_to_embed.append(chunk_text_span)
            text_metadata.append((m.id, chunk_idx))

    print(f"Total texts to embed: {len(texts_to_embed)}")
    if len(texts_to_embed) > len(massime):
        print(f"  (includes {len(texts_to_embed) - len(massime)} sub-chunks)")

    # Generate embeddings in batches
    all_embeddings = []
    for i in range(0, len(texts_to_embed), batch_size):
        batch_texts = texts_to_embed[i : i + batch_size]
        print(f"Embedding batch {i // batch_size + 1}/{(len(texts_to_embed) + batch_size - 1) // batch_size}")

        batch_embeddings = embedder.embed(batch_texts)
        all_embeddings.extend(batch_embeddings)

    # Create results
    results = [
        EmbeddingResult(
            massima_id=text_metadata[i][0],
            chunk_idx=text_metadata[i][1],
            embedding=all_embeddings[i],
            text_span=texts_to_embed[i][:100],
        )
        for i in range(len(all_embeddings))
    ]

    # Insert
    await insert_embeddings(conn, config["table"], results)

    # Stats
    stats = await conn.fetchrow(f"""
        SELECT COUNT(*) as total,
               COUNT(DISTINCT massima_id) as unique_massime,
               COUNT(*) FILTER (WHERE chunk_idx > 0) as sub_chunks
        FROM {config['table']}
    """)

    print(f"\nFinal stats for {config['table']}:")
    print(f"  Total embeddings: {stats['total']}")
    print(f"  Unique massime: {stats['unique_massime']}")
    print(f"  Sub-chunks: {stats['sub_chunks']}")

    await conn.close()
    print("\n[DONE]")


def main():
    parser = argparse.ArgumentParser(description="Generate embeddings for KB benchmark")
    parser.add_argument(
        "--model",
        required=True,
        choices=list(MODEL_CONFIGS.keys()),
        help="Model to use for embedding",
    )
    parser.add_argument(
        "--litellm",
        action="store_true",
        help="Use LiteLLM instead of local model",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for embedding (default: 32)",
    )
    parser.add_argument(
        "--no-clear",
        action="store_true",
        help="Don't clear existing embeddings",
    )

    args = parser.parse_args()

    asyncio.run(
        generate_for_model(
            model_key=args.model,
            use_litellm=args.litellm,
            batch_size=args.batch_size,
            clear_first=not args.no_clear,
        )
    )


if __name__ == "__main__":
    main()
