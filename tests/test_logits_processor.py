#!/usr/bin/env pytest
"""Test suite for ensuring Whisper model integration with KenLM works as expected.

This file contains tests that validate the functionality of Whisper model with
and without KenLM logits processing, checking aspects like transcription
accuracy and Word Error Rate (WER).
"""

import os

import pytest
from evaluate import load
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from transformers.generation import LogitsProcessorList

from whisper_lm_transformers import (
    WhisperLLMLogitsProcessor,
    WhisperLMLogitsProcessor,
)

from .utils import decode_and_normalize


@pytest.mark.usefixtures("whisper_config", "audio_array", "text_ref", "normalizer")
def test_nolm(whisper_config, audio_array, text_ref, normalizer):
    """Test the Whisper model without language model integration.

    Args:
        whisper_config: Configuration fixture for Whisper model.
        audio_array: Audio data for testing.
        text_ref: Reference text for comparison.
        normalizer: Text normalizing function.
    """
    processor = WhisperProcessor.from_pretrained(whisper_config["model"])

    # Load the whisper acoustic model
    model = WhisperForConditionalGeneration.from_pretrained(whisper_config["model"])

    # Prepare audio
    inputs = processor(audio=audio_array, sampling_rate=16000, return_tensors="pt")

    # Generate the transcription
    generated = model.generate(
        input_features=inputs["input_features"],
        num_beams=5,
        do_sample=False,
        temperature=0,
        language=whisper_config["lang"],
    )

    # Decode the tokens
    text_out = processor.decode(generated[0], skip_special_tokens=True)

    # Normalize the sentences
    text_ref = normalizer(text_ref).strip()
    text_out = normalizer(text_out).strip()

    assert text_out != text_ref

    # Calculate the WER
    wer_metric = load("wer")
    wer = wer_metric.compute(references=[text_ref], predictions=[text_out])
    assert 1.0 > wer > 0.1


@pytest.mark.usefixtures(
    "whisper_config", "lm_config", "audio_array", "text_ref", "normalizer"
)
def test_lm_logits_processor(
    whisper_config, lm_config, audio_array, text_ref, normalizer
):
    """
    Test the logits processor for the integration of LMs with Whisper.

    Args:
        whisper_config (dict):
            Configuration for the Whisper model specific to the test language.
        lm_config (dict):
            Contains the path to the KenLM model and parameters such as alpha
            and beta which influence the logits adjustment.
        audio_array (np.ndarray):
            The audio data array for transcription.
        text_ref (str):
            The expected correct transcription to compare the model output
            against.
        normalizer (function):
            A function to normalize both the output and reference
            transcriptions for accurate comparison.
    """
    processor = WhisperProcessor.from_pretrained(whisper_config["model"])

    # Load the whisper acoustic model
    model = WhisperForConditionalGeneration.from_pretrained(whisper_config["model"])

    # Create the logits processor for that integrates the LM
    lm_processor = WhisperLMLogitsProcessor(
        tokenizer=processor.tokenizer,  # pylint: disable=no-member
        lm_model=lm_config["path"],
        lm_alpha=lm_config["alpha"],
        lm_beta=lm_config["beta"],
        top_k=6,
    )

    # Run generate with a LogitsProcessorList
    logits_processors = LogitsProcessorList([lm_processor])

    # Prepare audio
    inputs = processor(audio=audio_array, sampling_rate=16000, return_tensors="pt")

    # Generate the transcription
    generated = model.generate(
        input_features=inputs["input_features"],
        num_beams=5,
        logits_processor=logits_processors,
        language=whisper_config["lang"],
    )

    # Decode and normalize the text
    text_ref, text_out = decode_and_normalize(
        processor, normalizer, text_ref, generated
    )

    assert text_out == text_ref


@pytest.mark.skipif(not os.getenv("TEST_LLM"), reason="Skipping LLM tests.")
@pytest.mark.usefixtures(
    "whisper_config", "llm_config", "audio_array", "text_ref", "normalizer"
)
def test_llm_logits_processor(
    whisper_config, llm_config, audio_array, text_ref, normalizer
):
    """
    Test the logits processor for the integration of LMs with Whisper.

    Args:
        whisper_config (dict):
            Configuration for the Whisper model specific to the test language.
        llm_config (dict):
            Contains the path to the LLM model and parameters such as alpha
            and beta which influence the logits adjustment.
        audio_array (np.ndarray):
            The audio data array for transcription.
        text_ref (str):
            The expected correct transcription to compare the model output
            against.
        normalizer (function):
            A function to normalize both the output and reference
            transcriptions for accurate comparison.
    """
    processor = WhisperProcessor.from_pretrained(whisper_config["model"])

    # Load the whisper acoustic model
    model = WhisperForConditionalGeneration.from_pretrained(whisper_config["model"])

    # Create the logits processor for that integrates the LM
    lm_processor = WhisperLLMLogitsProcessor(
        tokenizer=processor.tokenizer,  # pylint: disable=no-member
        lm_model=llm_config["path"],
        lm_alpha=llm_config["alpha"],
        lm_beta=llm_config["beta"],
        top_k=6,
    )

    # Run generate with a LogitsProcessorList
    logits_processors = LogitsProcessorList([lm_processor])

    # Prepare audio
    inputs = processor(audio=audio_array, sampling_rate=16000, return_tensors="pt")

    # Generate the transcription
    generated = model.generate(
        input_features=inputs["input_features"],
        num_beams=5,
        logits_processor=logits_processors,
        language=whisper_config["lang"],
    )

    # Decode and normalize the text
    text_ref, text_out = decode_and_normalize(
        processor, normalizer, text_ref, generated
    )

    assert text_out == text_ref
