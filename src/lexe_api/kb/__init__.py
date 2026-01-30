"""
LEXE Knowledge Base - Massimari Giurisprudenziali

Modulo per la gestione della Knowledge Base con:
- Ingestion: estrazione PDF, parsing, embedding
- Retrieval: hybrid search (dense + BM25 + trgm + RRF)
- Graph: Apache AGE per relazioni tra massime/norme
"""

__version__ = "0.1.0"

from .config import (
    EMBEDDING_DIMS,
    EmbeddingChannel,
    EmbeddingModel,
    GraphExpansionConfig,
    HybridSearchConfig,
    KBConfig,
    KBSettings,
    SystemConfig,
)

# Lazy imports per submodules (evita import circolari)
def get_ingestion_module():
    """Get ingestion submodule."""
    from . import ingestion
    return ingestion


def get_retrieval_module():
    """Get retrieval submodule."""
    from . import retrieval
    return retrieval


__all__ = [
    # Version
    "__version__",
    # Config
    "KBConfig",
    "KBSettings",
    "EmbeddingModel",
    "EmbeddingChannel",
    "SystemConfig",
    "EMBEDDING_DIMS",
    "HybridSearchConfig",
    "GraphExpansionConfig",
    # Lazy loaders
    "get_ingestion_module",
    "get_retrieval_module",
]
