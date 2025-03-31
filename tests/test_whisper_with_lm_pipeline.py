#!/usr/bin/env python
"""Tests for the custom `WhisperWithLMPipeline` pipeline using a language model."""

import os

import pytest
from transformers import pipeline

import whisper_lm_transformers  # pylint: disable=unused-import # noqa: F401


@pytest.mark.usefixtures(
    "whisper_config", "lm_config", "audio_array", "text_ref", "normalizer"
)
def test_whisper_with_lm_pipeline(
    whisper_config, lm_config, audio_array, text_ref, normalizer
):
    """
    Validate the custom "whisper-with-lm" pipeline for transcription.

    Args:
        whisper_config (dict):
            Configuration for the Whisper model (including 'model' key).
        lm_config (dict):
            Configuration for the KenLM model (including 'path', 'alpha',
            'beta').
        audio_array (np.ndarray):
            The audio data array for transcription.
        text_ref (str):
            The reference transcription to compare against.
        normalizer (function):
            A function to normalize both the model output and reference text.
    """
    # Create an instance for your custom "whisper-with-lm" task pipeline
    asr_pipeline = pipeline(
        "whisper-with-lm",
        model=whisper_config["model"],
        lm_model=lm_config["path"],
        lm_alpha=lm_config["alpha"],
        lm_beta=lm_config["beta"],
        language=whisper_config["lang"],
    )

    # Run the pipeline on the audio data
    result = asr_pipeline(audio_array)

    # Normalize the text
    text_out = normalizer(result["text"]).strip()
    text_ref = normalizer(text_ref).strip()

    # Assert they match
    assert text_out == text_ref


@pytest.mark.skipif(not os.getenv("TEST_LLM"), reason="Skipping LLM tests.")
@pytest.mark.usefixtures(
    "whisper_config", "llm_config", "audio_array", "text_ref", "normalizer"
)
def test_whisper_with_llm_pipeline(
    whisper_config, llm_config, audio_array, text_ref, normalizer
):
    """
    Validate the custom "whisper-with-llm" pipeline for transcription.

    Args:
        whisper_config (dict):
            Configuration for the Whisper model (including 'model' key).
        llm_config (dict):
            Configuration for the LLM model (including 'path', 'alpha',
            'beta').
        audio_array (np.ndarray):
            The audio data array for transcription.
        text_ref (str):
            The reference transcription to compare against.
        normalizer (function):
            A function to normalize both the model output and reference text.
    """
    # Create an instance for your custom "whisper-with-lm" task pipeline
    asr_pipeline = pipeline(
        "whisper-with-lm",
        model=whisper_config["model"],
        llm_model=llm_config["path"],
        lm_alpha=llm_config["alpha"],
        lm_beta=llm_config["beta"],
        language=whisper_config["lang"],
    )

    # Run the pipeline on the audio data
    result = asr_pipeline(audio_array)

    # Normalize the text
    text_out = normalizer(result["text"]).strip()
    text_ref = normalizer(text_ref).strip()

    # Assert they match
    assert text_out == text_ref
