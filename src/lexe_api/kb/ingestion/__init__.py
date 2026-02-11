"""
LEXE Knowledge Base - Ingestion Pipeline

Moduli per l'estrazione e processing di PDF massimari:
- extractor: Estrazione PDF con unstructured (hi_res OCR)
- cleaner: Pulizia testo giuridico
- parser: Parsing gerarchico deterministico
- massima_extractor: Estrazione massime atomiche
- citation_parser: Estrazione citazioni e norme
- deduplicator: Deduplicazione hash + similarity
- embedder: Generazione multi-embedding
- pipeline: Orchestrazione pipeline completa
"""

from .citation_parser import (
    CitationType,
    ParsedCitation,
    extract_all_citations,
    extract_codice_citations,
    extract_eu_citations,
    extract_legge_citations,
    extract_pronuncia_citations,
    get_cited_norms,
    get_cited_pronounce,
)
from .cleaner import (
    clean_legal_text,
    clean_ocr_artifacts,
    compute_content_hash,
    extract_first_sentence,
    is_likely_header_or_footer,
    merge_split_paragraphs,
    normalize_citation_text,
    normalize_for_hash,
)
from .deduplicator import (
    DeduplicationResult,
    Deduplicator,
    DuplicateMatch,
    compute_simhash,
    find_near_duplicates_quadratic,
    jaccard_similarity,
    simhash_similarity,
)
from .embedder import (
    EmbeddingClient,
    EmbeddingRequest,
    EmbeddingResult,
    MultiEmbedder,
    create_embedding_client_from_env,
    validate_embedding_dims,
)
from .extractor import (
    ExtractedElement,
    ExtractionResult,
    OCRQualityMetrics,
    calculate_ocr_metrics,
    extract_pdf_sync,
    extract_pdf_with_quality,
    filter_elements_by_category,
    group_elements_by_page,
)
from .massima_extractor import (
    ExtractedCitation,
    ExtractedMassima,
    calculate_massima_importance,
    extract_citation,
    extract_massime_from_elements,
    extract_tema,
    is_massima_text,
)
from .parser import (
    ParsedDocument,
    SectionNode,
    detect_section_type,
    extract_toc_from_first_pages,
    flatten_sections_with_elements,
    get_section_context,
    is_likely_section_header,
    parse_document_structure,
)
from .pipeline import (
    IngestionJob,
    IngestionPipeline,
    JobStatus,
    PipelineResult,
    create_job,
    run_single_document,
)

__all__ = [
    # extractor
    "ExtractionResult",
    "ExtractedElement",
    "OCRQualityMetrics",
    "calculate_ocr_metrics",
    "extract_pdf_sync",
    "extract_pdf_with_quality",
    "filter_elements_by_category",
    "group_elements_by_page",
    # cleaner
    "clean_legal_text",
    "clean_ocr_artifacts",
    "compute_content_hash",
    "extract_first_sentence",
    "is_likely_header_or_footer",
    "merge_split_paragraphs",
    "normalize_citation_text",
    "normalize_for_hash",
    # parser
    "ParsedDocument",
    "SectionNode",
    "detect_section_type",
    "extract_toc_from_first_pages",
    "flatten_sections_with_elements",
    "get_section_context",
    "is_likely_section_header",
    "parse_document_structure",
    # massima_extractor
    "ExtractedCitation",
    "ExtractedMassima",
    "calculate_massima_importance",
    "extract_citation",
    "extract_massime_from_elements",
    "extract_tema",
    "is_massima_text",
    # citation_parser
    "CitationType",
    "ParsedCitation",
    "extract_all_citations",
    "extract_codice_citations",
    "extract_eu_citations",
    "extract_legge_citations",
    "extract_pronuncia_citations",
    "get_cited_norms",
    "get_cited_pronounce",
    # deduplicator
    "Deduplicator",
    "DeduplicationResult",
    "DuplicateMatch",
    "compute_simhash",
    "find_near_duplicates_quadratic",
    "jaccard_similarity",
    "simhash_similarity",
    # embedder
    "EmbeddingClient",
    "EmbeddingRequest",
    "EmbeddingResult",
    "MultiEmbedder",
    "create_embedding_client_from_env",
    "validate_embedding_dims",
    # pipeline
    "IngestionJob",
    "IngestionPipeline",
    "JobStatus",
    "PipelineResult",
    "create_job",
    "run_single_document",
]
