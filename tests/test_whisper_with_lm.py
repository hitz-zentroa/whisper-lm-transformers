#!/usr/bin/env pytest
"""Tests for the WhisperWithLM class functionality.

This file tests the WhisperWithLM class to ensure that the integration of the
KenLM language model with the Whisper generation process works correctly,
focusing on output correctness.
"""

import pytest
from transformers import WhisperProcessor

from whisper_lm_transformers import WhisperWithLM

from .utils import decode_and_normalize


@pytest.mark.usefixtures(
    "whisper_config", "lm_config", "audio_array", "text_ref", "normalizer"
)
def test_whisper_with_lm(whisper_config, lm_config, audio_array, text_ref, normalizer):
    """Test the WhisperWithLM model for correct text generation with LMs.

    Args:
        whisper_config: Configuration fixture for Whisper model.
        lm_config: Language model configuration fixture.
        audio_array: Audio data for testing.
        text_ref: Reference text for comparison.
        normalizer: Text normalizing function.
    """
    processor = WhisperProcessor.from_pretrained(whisper_config["model"])

    # Load the model
    model = WhisperWithLM.from_pretrained(
        whisper_config["model"],
    )

    # Prepare audio
    inputs = processor(audio=audio_array, sampling_rate=16000, return_tensors="pt")

    generated = model.generate(
        input_features=inputs["input_features"],
        tokenizer=processor.tokenizer,  # pylint: disable=no-member
        num_beams=5,
        lm_model=lm_config["path"],
        lm_alpha=lm_config["alpha"],
        lm_beta=lm_config["beta"],
        language=whisper_config["lang"],
    )

    # Decode and normalize the text
    text_ref, text_out = decode_and_normalize(
        processor, normalizer, text_ref, generated
    )

    assert text_out == text_ref
