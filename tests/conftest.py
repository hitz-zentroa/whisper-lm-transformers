"""Shared pytest fixtures for configuring the whisper_lm_transformers tests.

This module defines fixtures for various configurations and components such as
language settings, Whisper model configurations, and language model
configurations used in testing the Whisper-LM integration.
"""

import logging
import os

import pytest
from datasets import Audio, load_dataset
from whisper.audio import load_audio
from whisper.normalizers import BasicTextNormalizer


@pytest.fixture(name="lang")
def fixture_lang():
    """
    Fixture to provide the language code used in the tests.

    Returns:
        str:
            A language code representing the specific language model to be
            tested, e.g., 'eu' for Basque.
    """
    return "eu"


@pytest.fixture(name="whisper_config")
def fixture_whisper_config(lang):  # pylint: disable=
    """
    Provide a configuration dictionary for the Whisper model tests.

    Args:
        lang (str):
            Language code provided by the `lang` fixture.

    Returns:
        dict:
            Configuration for the Whisper model including keys for 'lang' and
            'model' with the path to the pretrained model.
    """
    return {
        "key": "value",
        "lang": lang,
        "model": f"zuazo/whisper-tiny-{lang}",
    }


@pytest.fixture(name="lm_config")
def fixture_lm_config(lang):
    """
    Provide the configuration for the KenLM model integration in the tests.

    Args:
        lang (str): Language code used to locate the language-specific KenLM
        model file.

    Returns:
        dict:
            Dictionary containing the path to the KenLM model and tuning
            parameters like alpha and beta.
    """
    path = os.path.dirname(__file__)
    return {
        "path": os.path.join(path, f"5gram-{lang}.bin"),
        "alpha": 0.33582368603855817,
        "beta": 0.6882556478819416,
    }


@pytest.fixture(name="llm_config")
def fixture_llm_config():
    """
    Provide the configuration for the LLM model integration in the tests.

    Returns:
        dict:
            Dictionary containing the path to the LLM model and tuning
            parameters like alpha and beta.
    """
    return {
        "path": "HiTZ/latxa-7b-v1.2",
        "alpha": 2.733293955541733,
        "beta": 0.0017859540619529915,
    }


@pytest.fixture(name="normalizer")
def fixture_normalizer():
    """
    Provide a text normalization function for sentence comparisons.

    Returns:
        function:
            A function from the `BasicTextNormalizer` class configured to
            remove diacritics.
    """
    return BasicTextNormalizer(remove_diacritics=True)


@pytest.fixture(name="ds", scope="session")
def fixture_ds():
    """
    Load the audio dataset, ensuring the correct sample rate.

    Returns:
        Dataset:
            A `datasets.Dataset` object with all audio samples resampled to
            16 kHz for consistency.
    """
    ds = load_dataset("openslr", "SLR76", split="train", trust_remote_code=True)
    first_example = ds[0]["audio"]
    if first_example["sampling_rate"] != 16_000:
        logging.info(
            "Resampling audio: %d -> %d", first_example["sampling_rate"], 16_000
        )
        ds = ds.cast_column("audio", Audio(sampling_rate=16_000))
    return ds


@pytest.fixture(name="example_index", scope="session")
def fixture_example_index():
    """
    Provide an index for selecting a specific audio example from the dataset.

    Returns:
        int:
            Index of the audio example used to demonstrate or test model
            performance.
    """
    return 28


@pytest.fixture(name="audio_array", scope="session")
def fixture_audio_array(ds, example_index):
    """
    Load and provide the audio array from the dataset at the specified index.

    Args:
        ds (Dataset):
            The dataset loaded by the `ds` fixture.
        example_index (int):
            The index provided by the `example_index` fixture.

    Returns:
        np.ndarray: The audio array for the specified example.
    """
    audio_path = ds[example_index]["audio"]["path"]
    if os.path.exists(audio_path):
        return load_audio(audio_path)
    return ds[example_index]["audio"]["array"]


@pytest.fixture(name="text_ref", scope="session")
def fixture_text_ref(ds, example_index):
    """
    Provide the reference transcription for the specified audio example.

    Args:
        ds (Dataset):
            The dataset loaded by the `ds` fixture.
        example_index (int):
            The index provided by the `example_index` fixture.

    Returns:
        str:
            The reference transcription for the audio example.
    """
    return ds[example_index]["sentence"]
