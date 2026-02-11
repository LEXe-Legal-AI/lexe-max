"""
LEXE Knowledge Base - Configuration
"""

from dataclasses import dataclass, field
from enum import Enum

from pydantic import Field
from pydantic_settings import BaseSettings


class EmbeddingModel(str, Enum):
    """Modelli embedding supportati."""

    QWEN3 = "qwen3"
    E5_LARGE = "e5-large"
    BGE_M3 = "bge-m3"
    LEGAL_BERT_IT = "legal-bert-it"


class EmbeddingChannel(str, Enum):
    """Canali embedding per ogni massima."""

    TESTO = "testo"  # Testo completo massima
    TEMA = "tema"  # Tema/titolo breve
    CONTESTO = "contesto"  # Contesto sezione editoriale


class SystemConfig(str, Enum):
    """Configurazioni sistema retrieval (S1-S5)."""

    S1_ATOMICO = "S1"  # Atomico base + evidenza
    S2_HYBRID = "S2"  # Dense + BM25 + trgm + RRF
    S3_RERANK = "S3"  # S2 + reranker
    S4_GRAPH = "S4"  # S3 + graph expansion pesato
    S5_FULL = "S5"  # S4 + temporal + query rewrite


# Dimensioni embedding per modello
EMBEDDING_DIMS: dict[EmbeddingModel, int] = {
    EmbeddingModel.QWEN3: 1536,
    EmbeddingModel.E5_LARGE: 1024,
    EmbeddingModel.BGE_M3: 1024,
    EmbeddingModel.LEGAL_BERT_IT: 768,
}


class KBSettings(BaseSettings):
    """Settings per Knowledge Base da environment variables."""

    # Database
    kb_database_url: str = Field(
        default="postgresql://lexe_kb:lexe_kb_dev_password@localhost:5434/lexe_kb",
        description="Connection string per KB database",
    )

    # Embedding
    kb_default_model: EmbeddingModel = Field(
        default=EmbeddingModel.QWEN3,
        description="Modello embedding di default",
    )
    kb_litellm_url: str = Field(
        default="http://localhost:4000/v1",
        description="URL LiteLLM per embedding",
    )

    # Retrieval
    kb_default_system: SystemConfig = Field(
        default=SystemConfig.S2_HYBRID,
        description="Configurazione sistema di default",
    )
    kb_rrf_k: int = Field(default=60, description="K per RRF fusion")
    kb_dense_limit: int = Field(default=50, description="Limite risultati dense search")
    kb_sparse_limit: int = Field(default=50, description="Limite risultati sparse search")
    kb_trgm_limit: int = Field(default=20, description="Limite risultati trgm")
    kb_final_limit: int = Field(default=20, description="Limite risultati finali")
    kb_min_similarity: float = Field(default=0.5, description="Similarita' minima")

    # Graph
    kb_graph_expansion_depth: int = Field(default=2, description="Profondita' espansione grafo")
    kb_graph_min_weight: float = Field(default=0.3, description="Peso minimo archi")
    kb_graph_top_n: int = Field(default=5, description="Top N nodi per seed")

    # Reranker
    kb_reranker_model: str = Field(
        default="BAAI/bge-reranker-v2-m3",
        description="Modello reranker",
    )
    kb_reranker_top_k: int = Field(default=10, description="Top K per reranker")

    # OCR/Ingestion
    kb_ocr_strategy: str = Field(default="hi_res", description="Strategia unstructured")
    kb_ocr_languages: list[str] = Field(default=["ita"], description="Lingue OCR")
    kb_max_retries: int = Field(default=3, description="Max retry ingestion job")

    # Deduplication
    kb_dedup_threshold: float = Field(default=0.95, description="Soglia dedup esatto")
    kb_near_dedup_threshold: float = Field(default=0.85, description="Soglia near-dedup")

    model_config = {"env_prefix": "LEXE_", "env_file": ".env", "extra": "ignore"}


@dataclass
class HybridSearchConfig:
    """Configurazione per hybrid search."""

    dense_limit: int = 50
    bm25_limit: int = 50
    trgm_limit: int = 20
    rrf_k: int = 60
    final_limit: int = 20
    min_similarity: float = 0.5
    model: EmbeddingModel = EmbeddingModel.QWEN3
    channel: EmbeddingChannel = EmbeddingChannel.TESTO


@dataclass
class GraphExpansionConfig:
    """Configurazione per graph expansion."""

    expansion_depth: int = 2
    top_n_per_seed: int = 5
    min_weight: float = 0.3
    edge_types: list[str] = field(default_factory=lambda: ["SAME_PRINCIPLE", "CITES", "APPLIES"])


@dataclass
class KBConfig:
    """Configurazione completa Knowledge Base."""

    settings: KBSettings = field(default_factory=KBSettings)
    hybrid: HybridSearchConfig = field(default_factory=HybridSearchConfig)
    graph: GraphExpansionConfig = field(default_factory=GraphExpansionConfig)

    @classmethod
    def from_env(cls) -> "KBConfig":
        """Crea config da environment variables."""
        settings = KBSettings()
        return cls(
            settings=settings,
            hybrid=HybridSearchConfig(
                dense_limit=settings.kb_dense_limit,
                bm25_limit=settings.kb_sparse_limit,
                trgm_limit=settings.kb_trgm_limit,
                rrf_k=settings.kb_rrf_k,
                final_limit=settings.kb_final_limit,
                min_similarity=settings.kb_min_similarity,
                model=settings.kb_default_model,
            ),
            graph=GraphExpansionConfig(
                expansion_depth=settings.kb_graph_expansion_depth,
                top_n_per_seed=settings.kb_graph_top_n,
                min_weight=settings.kb_graph_min_weight,
            ),
        )
