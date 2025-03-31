"""Initialization for the whisper_lm_transformers package."""

from transformers.pipelines import PIPELINE_REGISTRY

from .logits_processor import (
    WhisperLLMLogitsProcessor,
    WhisperLMLogitsProcessor,
)
from .whisper_with_lm import WhisperWithLM
from .whisper_with_lm_pipeline import WhisperWithLMPipeline

PIPELINE_REGISTRY.register_pipeline(
    task="whisper-with-lm",
    pipeline_class=WhisperWithLMPipeline,
    pt_model=WhisperWithLM,
    type="audio",
)

__all__ = [
    "WhisperLMLogitsProcessor",
    "WhisperLLMLogitsProcessor",
    "WhisperWithLM",
]
