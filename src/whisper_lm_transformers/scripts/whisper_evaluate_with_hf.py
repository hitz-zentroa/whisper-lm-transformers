#!/usr/bin/env python
"""Evaluates a Whisper model in OpenAI format in a dataset (CV by default).

Example:
Evaluate the Basque Tiny model in OpenSLR-76:

```shell
whisper_evaluate_with_hf \
    --dataset openslr \
    --dataset_name SLR76 \
    --dataset_split train \
    --language eu \
    --temperature 0 \
    --beam_size 5 \
    --lm_path 5gram-eu.bin \
    --lm_alpha 0.33582368603855817 \
    --lm_beta 0.6882556478819416 \
    zuazo/whisper-tiny-eu
```
"""

import argparse
import json
import logging
import os
import sys
from collections import defaultdict

import jiwer
import numpy as np
from datasets import concatenate_datasets, load_dataset
from tabulate import tabulate
from tqdm import tqdm
from transformers import pipeline, set_seed
from whisper.normalizers import BasicTextNormalizer


def parse_none(value):
    """Convert `"None"` string to `None` value in Python.

    Used to parse command line values.

    Parameters
    ----------
    value : str
        The input value.

    Returns
    -------
    str or None
        The output value with None parsed.
    """
    return None if value == "None" else value


def int_or_none(value):
    """Try to convert the value to an integer, unless the value is "None".

    Parameters
    ----------
    value : int or None
        An input of an integer number of `None`.

    Returns
    -------
    int or None
        The final value as integer or `None`.
    """
    if value.lower() == "none":
        return None
    return int(value)


def str_or_none(value):
    """Try to convert the value to an str, unless the value is "None".

    Parameters
    ----------
    value : int or None
        An input of an integer number of `None`.

    Returns
    -------
    int or None
        The final value as integer or `None`.
    """
    if value.lower() == "none":
        return None
    return str(value)


def tuple_type(strings):
    """Parse a string representation of numbers separated by commas.

    This function is specifically designed to process command-line argument
    strings.

    Parameters
    ----------
    strings : str
        A string input representing a tuple, typically formatted like
        "(1.0, 2.0, 3.0)".

    Returns
    -------
    tuple
        A tuple of floats created from the input string.

    Example
    -------
    ```python
    >>> tuple_type("(0.0, 0.2, 0.4, 0.6, 0.8, 1.0)")
    (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
    ```
    """
    if strings == "None":
        return None
    if isinstance(strings, (float, int)):
        return (float(strings),)
    if isinstance(strings, (list, tuple)):
        return tuple(float(x) for x in strings)
    strings = strings.replace("(", "").replace(")", "")
    mapped_int = map(float, strings.split(","))
    return tuple(mapped_int)


def parse_args():
    """Parse command line arguments.

    Returns
    -------
    namespace
        The namespace populated with the command line argument values.
    """
    parser = argparse.ArgumentParser(
        description="Evaluates a Whipser model in OpenAI format."
    )
    parser.add_argument(
        "model",
        help="Path or name of the OpenAI model to load.",
    )
    parser.add_argument(
        "--audios",
        "-a",
        default=[],
        nargs="+",
        help="Transcribe a list of audios instead of using a dataset.",
    )
    parser.add_argument(
        "--language",
        "--lang",
        type=str_or_none,
        default=None,
        help="The language in ISO-639-1 (two-letter code).",
    )
    parser.add_argument(
        "--dataset",
        "-d",
        default="mozilla-foundation/common_voice_13_0",
        help="Path or name of the Hugging Face dataset. Defaults to CV 13.",
    )
    parser.add_argument(
        "--dataset_name",
        "-dn",
        default="eu",
        help=(
            "Defining the name of the dataset configuration for Hugging Face. "
            "For Common Voice datasets, this represents the language. "
            "Defaults to `eu`."
        ),
    )
    parser.add_argument(
        "--dataset_split",
        "-ds",
        nargs="+",
        default=["test"],
        help=(
            "Which splits of the data to load, separated by spaces. "
            "Defaults to `test`."
        ),
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--use_audio_array",
        action="store_true",
        help="Whether to use the array subkey instead of the audio path.",
    )
    parser.add_argument(
        "--skip_normalize",
        "-n",
        action="store_true",
        help="Whether to normalize the text (enabled by default)",
    )
    parser.add_argument(
        "--with_diacritics",
        action="store_true",
        help="Leave the diacritics when normalizing.",
    )
    parser.add_argument(
        "--temperature",
        type=tuple_type,
        default=(0.0),
        help=(
            "Temperature is a form of controlled randomness. "
            "A list of numbers can be provided separated by commas. "
            "Defaults to 0, which means disabled. The logits will be divided "
            "by this number. "
            "`> 1.0` leads to a more random sampling behaviour. "
            "`< 1.0` makes model more confident in its predictions and "
            "reducing randomness."
        ),
    )
    parser.add_argument(
        "--beam_size",
        type=int_or_none,
        default=5,
        help="Number of beams in beam search, enables Beam Search.",
    )
    parser.add_argument(
        "--with_timestamps",
        action="store_true",
        help="Enable timestamps prediction.",
    )
    parser.add_argument(
        "--lm_path",
        type=str,
        default=None,
        help="A KenLM n-gram language model path.",
    )
    parser.add_argument(
        "--llm_path",
        type=str,
        default=None,
        help="A Hugging Face language model path or URI.",
    )
    parser.add_argument(
        "--lm_alpha",
        type=float,
        default=None,
        help="KenLM Language Model weight.",
    )
    parser.add_argument(
        "--lm_beta",
        type=float,
        default=None,
        help="KenLM word insertion weight.",
    )
    parser.add_argument(
        "--lm_eos",
        type=str,
        default=None,
        help="KenLM End-of-String characters.",
    )
    parser.add_argument(
        "--lm_normalize",
        type=bool,
        default=True,
        help="Whether to normalize the text for the KenLM.",
    )
    parser.add_argument(
        "--lm_token_threshold",
        type=int,
        default=None,
        help=(
            "Minimum number of tokens in a sequence required before applying "
            "language model scoring. This prevents premature evaluation on "
            "short sequences."
        ),
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help=(
            "Directory to save the evaluation outputs. "
            "If not provided, no files will be saved."
        ),
    )

    levels = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
    parser.add_argument("--log-level", "-l", default="INFO", choices=levels)
    args = parser.parse_args()
    return args


def compute_measures(label_texts, predicted_texts):
    """Compute the metrics: WER, CER, etc.

    Parameters
    ----------
    label_texts : list
        Original groundtruth values.
    predicted_texts : list
        Predicted values.

    Returns
    -------
    dict
        The measures with the following keys: cer, wer, mer, wil, wip, hits,
        substitutions, deletions, insertions.
    """
    measures = jiwer.compute_measures(label_texts, predicted_texts)
    measures["cer"] = jiwer.cer(label_texts, predicted_texts)
    measures.pop("hypothesis", None)
    measures.pop("ops", None)
    measures.pop("truth", None)
    return measures


def evaluate(
    model,
    dataset,
    normalizer=None,
    text_column="text",
    audio_subkey="path",
):
    """Evaluate a Whisper ASR model on a given dataset using `transcribe()`.

    Parameters
    ----------
    model : whisper.Whisper
        The Whisper ASR model to be used for transcription.
    dataset : datasets.Dataset
        A dataset containing audio data and corresponding ground truth text.
    normalizer : function, optional
        A function used to normalize text data. If None, normalization is
        skipped.
    text_column : str, optional
        The name of the column in the dataset containing the ground truth text.
    audio_subkey : str, optional
        The key in the dataset that points to the audio data, which could be a
        path or an array.

    Returns
    -------
    tuple :
    - entence_measures : dict
        A dictionary containing lists of computed sentence-level measures
        (e.g., CER, WER) across all examples in the dataset.
    - label_texts : list
        A list of all ground truth texts used for evaluation.
    - predicted_texts : list
        A list of all predicted texts generated by the model.

    Example
    -------
    ```python
    sentence_measures, label_texts, predicted_texts = evaluate(
        model, dataset, transcribe_options
    )
    ```
    """
    # Aggregating all reference and hypothesis texts for dataset-level metrics
    label_texts = []
    predicted_texts = []

    # Iterate through the dataset
    logging.info("Evaluating the dataset:")
    sentence_measures = defaultdict(list)
    for example in tqdm(dataset):
        # Transcribe the example
        label_text = example[text_column]
        audio = example["audio"][audio_subkey]
        if isinstance(audio, list):
            audio = np.array(audio)
        if isinstance(audio, np.ndarray):
            audio = audio.astype(np.float32)
        predicted_text = model(audio)["text"]

        # Normalize the text
        if normalizer is not None:
            label_text = normalizer(label_text).strip()
            predicted_text = normalizer(predicted_text).strip()

        # Append for dataset-level calculation
        label_texts.append(label_text)
        predicted_texts.append(predicted_text)

        # Compute the sentence-level scores:
        measures = compute_measures(label_text, predicted_text)
        for name, score in measures.items():
            if isinstance(score, (float, int)):
                sentence_measures[name].append(score)

    return sentence_measures, label_texts, predicted_texts


def pretty_print_scores(sentence_scores, dataset_scores):
    """Print a summary table of sentence-level and dataset-level scores.

    Parameters
    ----------
    sentence_scores : dict
        A dictionary containing arrays of scores for CER and WER at the
        sentence level. Each key ('cer', 'wer') should map to a list or array
        of float values representing the respective metric.
    dataset_scores : dict
        A dictionary containing single values for CER and WER at the dataset
        level. Each key ('cer', 'wer') should map to a float value representing
        the respective metric.

    Returns
    -------
    None. The function directly prints the formatted table to the console.

    Example
    -------
    ```python
    >>> sentence_scores = {'cer': [0.1, 0.2, 0.15], 'wer': [0.3, 0.25, 0.35]}
    >>> dataset_scores = {'cer': 0.15, 'wer': 0.3}
    >>> pretty_print_scores(sentence_scores, dataset_scores)
    ```
    """
    table_headers = ["Metric Level", "CER", "WER"]
    table_data = [
        [
            "Sentence-level",
            (
                f"{np.mean(sentence_scores['cer']) * 100:.2f} "
                f"± {np.std(sentence_scores['cer']) * 100:.2f}"
            ),
            (
                f"{np.mean(sentence_scores['wer']) * 100:.2f} "
                f"± {np.std(sentence_scores['wer']) * 100:.2f}"
            ),
        ],
        [
            "Dataset-level",
            f"{dataset_scores['cer'] * 100:.2f}",
            f"{dataset_scores['wer'] * 100:.2f}",
        ],
    ]

    print(tabulate(table_data, headers=table_headers, tablefmt="grid"))


def save_json(data, file_path):
    """Save data to a JSON file.

    Parameters
    ----------
    data : dict or list
        The data to save.
    file_path : str
        The path of output file.
    """
    with open(file_path, "w", encoding="utf-8") as fhandle:
        json.dump(data, fhandle, indent=4, ensure_ascii=False)


def main():  # pylint: disable=too-many-locals,too-many-branches,too-many-statements
    """Start the program."""
    args = parse_args()
    logging.basicConfig(level=args.log_level)

    # Print the command line run:
    logging.info("Command: %s", " ".join(sys.argv))

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    logging.info("Loading model: %s", args.model)
    model = pipeline(
        "whisper-with-lm",
        model=args.model,
        lm_model=args.lm_path,
        llm_model=args.llm_path,
        lm_alpha=args.lm_alpha,
        lm_beta=args.lm_beta,
        num_beams=args.beam_size,
        language=args.language,
        return_timestamps=args.with_timestamps,
    )

    logging.info("Loading dataset: %s", args.dataset)
    logging.info("- name: %s", args.dataset_name)

    # Load and concatenate the specified dataset splits
    datasets_splits = []
    for split_name in args.dataset_split:
        logging.info("- split: %s", split_name)
        dataset_split = load_dataset(
            args.dataset,
            parse_none(args.dataset_name),
            split=split_name,
            cache_dir=None,  # Specify your cache directory here if needed
        )
        datasets_splits.append(dataset_split)

    # Concatenate all specified splits into a single dataset
    if len(datasets_splits) > 1:
        dataset = concatenate_datasets(datasets_splits).shuffle(seed=args.seed)
    else:
        dataset = datasets_splits[0]

    # Which subkey to use to extract the audio
    audio_subkey = "array" if args.use_audio_array else "path"

    # Select the transcription field
    text_columns = [
        "sentence",
        "transcript",
        "transcription",
        "text",
        "normalized_text",
    ]
    text_column = [c for c in text_columns if c in dataset.features][0]

    # Clean up unused columns
    remove_columns = [
        "accent",
        "age",
        "client_id",
        "down_votes",
        "gender",
        "locale",
        "path",
        "segment",
        "up_votes",
    ]
    remove_columns = [c for c in remove_columns if c in dataset.features]
    dataset = dataset.remove_columns(remove_columns)

    # Transcribe a list of audios:
    if len(args.audios) > 0:
        logging.info("Transcriptions:")
        for audio in args.audios:
            logging.debug("Transcribing audio: %s", audio)
            result = model(
                audio,
                temperature=tuple_type(args.temperature),
            )
            print("- " + audio + ":", result["text"])
        sys.exit(0)

    # Text normalizing function:
    if not args.skip_normalize:
        normalizer = BasicTextNormalizer(remove_diacritics=not args.with_diacritics)
    else:
        normalizer = None

    logging.info("Using Hugging Face to transcribe")
    sentence_measures, label_texts, predicted_texts = evaluate(
        model,
        dataset,
        normalizer,
        text_column,
        audio_subkey,
    )

    # Print sentence-level scores
    print()
    print("Sentence-level scores:")
    for name, score in sentence_measures.items():
        score = np.array(score)
        print(f"Average {name}: {score.mean()} ± {score.std()}")

    # Compute dataset-level scores
    dataset_measures = compute_measures(label_texts, predicted_texts)

    # Print dataset-level scores
    print()
    print("Dataset-level scores:")
    for name, score in dataset_measures.items():
        print(f"Final {name}: {score}")

    # Print a summary table including only the CER/WER scores
    print()
    print("Summary:")
    pretty_print_scores(sentence_measures, dataset_measures)

    # Saving the output files with the results
    if args.output_dir:
        # Create the output directory
        os.makedirs(args.output_dir, exist_ok=True)

        # Save command-line arguments
        save_json(
            vars(args),
            os.path.join(args.output_dir, "args.json"),
        )

        # Save dataset-level metrics
        save_json(
            dataset_measures,
            os.path.join(args.output_dir, "dataset_level_results.json"),
        )

        # Save sentence-level metrics
        measures = []
        for i in range(len(label_texts)):  # pylint: disable=consider-using-enumerate
            result_data = {
                "label_text": label_texts[i],
                "predicted_text": predicted_texts[i],
            }
            for name, scores in sentence_measures.items():
                result_data[name] = scores[i]
            measures.append(result_data)
        save_json(
            measures,
            os.path.join(args.output_dir, "sentence_level_results.json"),
        )
    logging.info("Output directory: %s", args.output_dir)

    logging.info("Finished.")


if __name__ == "__main__":
    main()
