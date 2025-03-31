#!/usr/bin/env python
"""Optimize the language model based on Deep Speech work.

Based on the following code with many changes:
https://github.com/mozilla/DeepSpeech/blob/master/lm_optimizer.py

Example:
Optimize the LM for the Basque fine-tuned Tiny model:

```shell
whisper_lm_optimizer_with_hf zuazo/whisper-tiny-eu \
  --dataset_split "train+validation" \
  --dataset_shuffle True \
  --dataset_name "eu" \
  --language "eu" \
  --beam_size 5 \
  --lm_path "5gram-eu.bin" \
  --n_trials 100 \
  --journal_storage --n_jobs 28
```
"""

import argparse
import logging
import math
import os
import random
import sys
import tempfile
from pathlib import Path

import jiwer
import joblib
import optuna
import torch
import torch.multiprocessing as mp
from datasets import load_dataset
from optuna.storages import JournalFileStorage, JournalStorage
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


def objective_with_hf(
    trial,
    model,
    dataset,
    skip_normalize,
    lm_path,
    llm_path,
    lm_alpha_min,
    lm_beta_min,
    lm_alpha_max,
    lm_beta_max,
    beam_size,
    language,
    with_timestamps,
    use_cer,
    device,
):  # pylint: disable=too-many-locals,too-many-arguments,too-many-positional-arguments
    """Objective function used by the Optuna framework to run a trial.

    It uses Whisper.transcribe() function with no batches support.

    Args:
        trial (int):
            The Optuna study trial number.
        model (nn.Module):
            The model to test preloaded into the correct device.
        dataset (datasets.Dataset):
            The dataset split to test.
        skip_normalize (bool):
            Whether to normalize the text.
        lm_path (str):
            A KenLM n-gram language model path
        llm_path (str):
            A Hugging Face language model path or URI.
        lm_alpha_min (float):
            The minimum of the alpha hyperparameter of the CTC decoder explored
            during hyperparameter optimization. Language Model weight.
        lm_beta_min (float):
            The minimum beta hyperparameter of the CTC decoder explored during
            hyperparameter optimization. Word insertion weight.
        lm_alpha_max (float):
            The maximum of the alpha hyperparameter of the CTC decoder explored
            during hyperparameter optimization. Language Model weight.
        lm_beta_max (float):
            The maximum beta hyperparameter of the CTC decoder explored during
            hyperparameter optimization. Word insertion weight.
        beam_size (int):
            Number of beams in beam search, enables Beam Search.
        language (str):
            The language in ISO-639-1 (two-letter code).
        with_timestamps (bool):
            Enable timestamps prediction.
        use_cer (bool):
            Whether to use the CER as metric instead of the WER.
        device (str, optional):
            The device to which tensors will be moved ('cuda' or 'cpu').

    Returns:
        float: The CER if the model is character based, the WER elsewhere.
    """
    logging.debug("Using transcribe()")

    lm_alpha = trial.suggest_float("lm_alpha", lm_alpha_min, lm_alpha_max)
    lm_beta = trial.suggest_float("lm_beta", lm_beta_min, lm_beta_max)

    if isinstance(model, str):
        logging.debug("Loading the model: %s", model)
        model = pipeline(
            "whisper-with-lm",
            model=model,
            lm_model=lm_path,
            llm_model=llm_path,
            lm_alpha=lm_alpha,
            lm_beta=lm_beta,
            num_beams=beam_size,
            language=language,
            return_timestamps=with_timestamps,
            device=device,
        )

    # Text normalizing function:
    if not skip_normalize:
        normalizer = BasicTextNormalizer(remove_diacritics=True)
    else:
        normalizer = None

    score_func = jiwer.cer if use_cer else jiwer.wer

    references = []
    predictions = []
    logging.debug("Starting main optimization loop.")
    for step, example in enumerate(dataset):
        # Transcribe the example:
        label_text = example["sentence"]
        audio = str(example["audio"]["path"])
        predicted_text = model(audio)["text"]
        # Compute the score:
        if normalizer:
            label_text = normalizer(label_text).strip().lower()
            predicted_text = normalizer(predicted_text).strip().lower()

        references.append(label_text)
        predictions.append(predicted_text)

        # Report intermediate incremental objective value every "batch-size":
        measure = score_func(references, predictions)
        logging.debug("Intermediate score: %f", measure)
        trial.report(measure, step)

        # Handle pruning based on the intermediate value:
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    measure = score_func(references, predictions)
    logging.debug("Final score: %f", measure)
    return measure


def optimize_study(study_name, storage, n_trials, *args, gpu_id=None, **kwargs):
    """Optimizes the study for a given number of trials.

    This function is used with multiprocessing (`n_jobs > 1`).

    Parameters
    ----------
    study_name : str
        The name of the study to be created or loaded.
    storage : str
        The storage location (like a database) where the study's results
        should be stored or from where it should be loaded.
    objective : Callable
        The objective function to be minimized during optimization.
    n_trials : int
        The number of trials to be performed.
    gpu_id : int, str, None
        The number of GPU to use.

    Returns
    -------
    None
        This function returns None. It updates the study database in place.
    """
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    study = optuna.create_study(
        study_name=study_name, storage=storage, load_if_exists=True
    )
    study.optimize(lambda x: objective_with_hf(x, *args, **kwargs), n_trials=n_trials)


def detect_available_gpus(n_gpus):
    """Detect and lists the IDs of GPUs available.

    This function checks the environment variable `CUDA_VISIBLE_DEVICES` to
    determine which GPUs have been made available there.

    Args:
        n_gpus (int or None):
            The number of GPUs to detect. If None, the function attempts to use
            all available GPUs.

    Returns:
        list of str:
            A list containing the string identifiers of the GPUs to be used.

    Notes:
        The GPU IDs are strings because they are often used in environments and
        configurations where string types are required.

    Examples:
    To use all available GPUs:

    >>> detect_available_gpus(None)
    ['0', '1', '2', '3']

    To specify a certain number of GPUs:

    >>> detect_available_gpus(2)
    ['0', '1']

    When `CUDA_VISIBLE_DEVICES` is set to '0,2':

    >>> os.environ['CUDA_VISIBLE_DEVICES'] = '0,2'
    >>> detect_available_gpus(None)
    ['0', '2']
    """
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        # Use the GPUs specified in the environment variable
        gpu_ids = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
        n_gpus = len(gpu_ids)
        logging.info("Detected GPUs from CUDA_VISIBLE_DEVICES: %s", gpu_ids)
    elif n_gpus is None:
        # Use all available GPUs
        n_gpus = torch.cuda.device_count()
        gpu_ids = list(map(str, range(n_gpus)))
        logging.info("Using available GPUs: %s", gpu_ids)
    else:
        # Use the first `n_gpus` as specified
        gpu_ids = list(map(str, range(n_gpus)))
        logging.info("Using the first %s GPUs: %s", n_gpus, gpu_ids)
    return gpu_ids


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
        default=None,
        help="Transcribe a list of audios instead of using a dataset.",
    )
    parser.add_argument(
        "--language",
        "--lang",
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
        default="validation",
        help="Which split of the data to load. Defaults to `test`.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--dataset_n",
        type=int,
        default=None,
        help=(
            "The number of examples to sample from the dataset. "
            "It takes all by default."
        ),
    )
    parser.add_argument(
        "--dataset_shuffle",
        type=bool,
        default=True,
        help="Whether to shuffle the dataset examples.",
    )
    parser.add_argument(
        "--skip_normalize",
        "-n",
        action="store_true",
        help="Whether to normalize the text (enabled by default)",
    )
    parser.add_argument(
        "--beam_size",
        type=int,
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
        "--use_cer",
        action="store_true",
        help="Whether to use the CER as metric instead of the WER.",
    )
    parser.add_argument(
        "--study_name",
        default="lm_optimizer",
        help="Name of the optuna study.",
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=25,
        help=(
            "The number of parallel jobs per GPU. If this argument is set to "
            "-1, the number is set to CPU count."
        ),
    )
    parser.add_argument(
        "--journal_storage",
        action="store_true",
        help=(
            "Use Journal storage backend in Optuna instead of SQLite. "
            "Recommended with big n_jobs values but still experimental."
        ),
    )
    parser.add_argument("--storage", default=None, help="Optuna storage URL.")
    parser.add_argument(
        "--joblib_backend",
        default="loky",
        help=(
            "Joblib parallelization backend implementation: "
            "loky, multiprocessing, threading."
        ),
    )
    parser.add_argument(
        "--use_tmp",
        action="store_true",
        help=(
            "Use /tmp for storage files of the backend. "
            "Recommended for network mounted directories with locking issues."
        ),
    )
    parser.add_argument(
        "--n_trials",
        type=int,
        default=100,
        help="The number of trials to run during hyperparameter optimization.",
    )
    parser.add_argument(
        "--n_gpus",
        type=int,
        default=None,
        help="Number of GPUs to use. If None, all available GPUs are used.",
    )
    parser.add_argument(
        "--lm_alpha_min",
        type=float,
        default=0,
        help=(
            "The minimum of the alpha hyperparameter of the CTC decoder "
            "explored during hyperparameter optimization. Language Model "
            "weight."
        ),
    )
    parser.add_argument(
        "--lm_alpha_max",
        type=float,
        default=5,
        help=(
            "The maximum of the alpha hyperparameter of the CTC decoder "
            "explored during hyperparameter optimization. Language Model "
            "weight."
        ),
    )
    parser.add_argument(
        "--lm_beta_min",
        type=float,
        default=0,
        help=(
            "The maximum beta hyperparameter of the CTC decoder explored "
            "during hyperparameter optimization. Word insertion weight."
        ),
    )
    parser.add_argument(
        "--lm_beta_max",
        type=float,
        default=5,
        help=(
            "The maximum beta hyperparameter of the CTC decoder explored "
            "during hyperparameter optimization. Word insertion weight."
        ),
    )
    parser.add_argument(
        "--device",
        default=None,
        help="The device to which tensors will be moved ('cuda' or 'cpu').",
    )
    levels = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
    parser.add_argument("--log-level", "-l", default="INFO", choices=levels)
    args = parser.parse_args()
    return args


def main():  # pylint: disable=too-many-locals,too-many-branches,too-many-statements
    """Start the program."""

    def objective_fn(trial):
        """Pass the preloaded model and dataset to the objective function.

        This function is used without multiprocessing (`n_jobs == 1`).

        Parameters
        ----------
        trial : int
            The Optuna study trial number.

        Returns
        -------
        float
            The score.
        """
        logging.info("Using HF to optimize")
        return objective_with_hf(
            trial,
            model,
            dataset,
            args.skip_normalize,
            args.lm_path,
            args.llm_path,
            args.lm_alpha_min,
            args.lm_beta_min,
            args.lm_alpha_max,
            args.lm_beta_max,
            args.beam_size,
            args.language,
            args.with_timestamps,
            args.use_cer,
            args.device,
        )

    args = parse_args()
    logging.basicConfig(level=args.log_level)

    # Print the command line run:
    logging.info("Command: %s", " ".join(sys.argv))
    optuna.logging.enable_propagation()
    optuna.logging.disable_default_handler()

    # Decide the device to use if not set
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Detect available GPUs
    gpu_ids = detect_available_gpus(args.n_gpus)
    n_gpus = len(gpu_ids)
    logging.info("Available GPUs: %s", n_gpus)

    # Load only one model if required (only for threading)
    if args.joblib_backend == "threading":
        logging.info("Loading model: %s", args.model)
        model = pipeline(
            "whisper-with-lm",
            model=args.model,
            lm_model=args.lm_path,
            llm_model=args.llm_path,
            num_beams=args.beam_size,
            language=args.language,
            return_timestamps=args.with_timestamps,
        )
    else:
        model = args.model

        # Force the 'spawn' start method once at the top level
        mp.set_start_method("spawn", force=True)

    logging.info("Loading dataset: %s", args.dataset)
    logging.info("- name: %s", args.dataset_name)
    logging.info("- split: %s", args.dataset_split)
    dataset = load_dataset(
        args.dataset,
        parse_none(args.dataset_name),
        split=args.dataset_split,
        token=True,
    )
    dataset = dataset.remove_columns(
        [
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
    )

    # Shuffle the dataset in a reproducible way
    if args.dataset_shuffle:
        dataset = dataset.shuffle(seed=args.seed)

    # Limit the number of examples
    if args.dataset_n is not None:
        dataset_len = len(dataset)
        logging.info(
            "Subsample the dataset to: %d (from %d)",
            args.dataset_n,
            dataset_len,
        )
        indices = random.sample(range(dataset_len), args.dataset_n)
        dataset = dataset.select(indices)

    logging.info("Metric: %s", "CER" if args.use_cer else "WER")
    if args.n_jobs == 1 and n_gpus < 1:
        logging.info("Creating study:")
        study = optuna.create_study()
        logging.info("Optimizing the LM:")
        study.optimize(objective_fn, n_jobs=1, n_trials=args.n_trials)
    else:  # parallel processing
        # n_trials = math.ceil(args.n_trials / args.n_jobs)
        n_trials = math.ceil(args.n_trials / (args.n_jobs * n_gpus))
        logging.info(
            "Distributing %d trials across %d jobs.",
            args.n_trials,
            n_gpus * args.n_jobs,
        )

        logging.info("Number of trials: %d", n_trials)
        # It is recommended to use /tmp to store the logs with network disks
        tmp_dir = tempfile.gettempdir() if args.use_tmp else str(Path.home())
        logging.info("Initializing storage:")
        if args.journal_storage:
            storage_path = os.path.join(tmp_dir, f"{args.study_name}-journal.log")
            storage = JournalStorage(JournalFileStorage(storage_path))
        elif args.storage is not None:
            storage = args.storage
        else:
            storage_path = os.path.join(tmp_dir, f"{args.study_name}.db")
            storage = f"sqlite:///{storage_path}"
        study = optuna.create_study(
            study_name=args.study_name, storage=storage, load_if_exists=True
        )

        logging.info("Creating job arguments.")
        job_args = []
        for gpu_id in gpu_ids:
            for _ in range(args.n_jobs):
                job_args.append(
                    joblib.delayed(optimize_study)(
                        args.study_name,
                        storage,
                        n_trials,
                        model,
                        dataset,
                        args.skip_normalize,
                        args.lm_path,
                        args.llm_path,
                        args.lm_alpha_min,
                        args.lm_beta_min,
                        args.lm_alpha_max,
                        args.lm_beta_max,
                        args.beam_size,
                        args.language,
                        args.with_timestamps,
                        args.use_cer,
                        args.device,
                        gpu_id=gpu_id,
                    )
                )

        logging.info("Optimizing the LM in parallel: %s:", args.joblib_backend)
        joblib.Parallel(n_jobs=args.n_jobs * n_gpus, backend=args.joblib_backend)(
            job_args
        )
    metric = "CER" if args.use_cer else "WER"
    print(
        f"Best params: lm_alpha={study.best_params['lm_alpha']} "
        f"and lm_beta={study.best_params['lm_beta']} "
        f"with {metric}={study.best_value}"
    )


if __name__ == "__main__":
    main()
