"""SciSense: Gaussian Process uncertainty gating for LLM-generated insights."""

from .gp_engine import GPEngine, GPResult
from .llm_layer import LLMReasoningLayer, SciInsight, get_certainty_level
from .pipeline import PipelineOutput, run_scisense

__all__ = [
    "GPEngine",
    "GPResult",
    "LLMReasoningLayer",
    "SciInsight",
    "get_certainty_level",
    "PipelineOutput",
    "run_scisense",
]
__version__ = "0.2.0"
