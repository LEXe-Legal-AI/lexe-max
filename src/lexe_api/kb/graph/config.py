"""
Category Graph v2.5 Configuration

Soglie operative, modelli LLM e feature flags per il sistema di classificazione.
"""

from dataclasses import dataclass


@dataclass
class ClassificationThresholds:
    """Soglie operative per routing L1."""

    # Confidence calibrata minima per accettare Top-1
    TH_CONF_GO: float = 0.80

    # Confidence sotto cui forzare Top-2 + NEEDS_REVIEW
    TH_CONF_LOW: float = 0.65

    # Delta minimo tra Top-1 e Top-2 per evitare LLM
    TH_DELTA_LLM: float = 0.12

    # Confidence LLM minima per accettare sua scelta
    TH_LLM_ACCEPT: float = 0.70

    # Feature flags
    LLM_RESOLVER_ENABLED: bool = True
    GRAPH_MODE_EXPERIMENTAL: bool = False

    # Budget
    MAX_LLM_CALLS_PER_RUN: int = 8000

    # Ensemble LLM (due classifier + un giudice)
    # Model IDs da OpenRouter
    LLM_MODEL_A: str = "google/gemini-2.5-flash-lite-preview-09-2025"
    LLM_MODEL_B: str = "qwen/qwen3-235b-a22b-2507"
    LLM_JUDGE_MODEL: str = "mistralai/mistral-large-2512"


DEFAULT_THRESHOLDS = ClassificationThresholds()
