# Whisper-LM-Transformers

[![arXiv - Paper](https://img.shields.io/badge/cs.CL-2503.23542-b31b1b?&logo=arxiv&logoColor=red)](https://arxiv.org/abs/2503.23542)
[![PyPI - Version](https://img.shields.io/pypi/v/whisper-lm-transformers)](https://pypi.org/project/whisper-lm-transformers/)
[![Build](https://github.com/hitz-zentroa/whisper-lm-transformers/actions/workflows/python-app.yml/badge.svg)](https://github.com/hitz-zentroa/whisper-lm-transformers/actions)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![HiTZ](https://img.shields.io/badge/HiTZ-Basque%20Center%20for%20Language%20Technology-blueviolet)](http://www.hitz.eus/)

KenLM and Large language model integration with Whisper ASR models implemented
in Hugging Face library.

## Installation

Install the package from PyPI:

```shell
pip install whisper-lm-transformers
```

Or clone and install locally:

```shell
git clone https://github.com/hitz-zentroa/whisper-lm-transformers.git
cd whisper-lm-transformers
pip install .
```

Besides, a recent version of
[KenLM](https://github.com/kpu/kenlm)
is required to use n-gram language models:

```shell
pip install https://github.com/kpu/kenlm/archive/master.zip
```

## Usage Examples

### 1) Using Hugging Face Pipeline

There is a new
[pipeline](https://huggingface.co/docs/transformers/en/main_classes/pipelines)
task called "whisper-with-lm". Once imported, you can do:

```python
>>> from transformers import pipeline
>>> from huggingface_hub import hf_hub_download
>>> import whisper_lm_transformers  # Required to register the new pipeline

>>> # Download the n-gram model
>>> lm_model = hf_hub_download(
...     repo_id="HiTZ/whisper-lm-ngrams", filename="5gram-eu.bin"
... )

>>> # Example: KenLM-based decoding
>>> pipe = pipeline(
...     "whisper-with-lm",
...     model="zuazo/whisper-tiny-eu",
...     lm_model=lm_model, # Provide a kenlm model path
...     lm_alpha=0.33582369,
...     lm_beta=0.68825565,
...     language="eu",
... )

>>> # Transcribe an audio file or array
>>> pipe("tests/fixtures/audio.mp3")["text"]
'Talka diskoetxearekin grabatzen ditut beti abestien maketak.'

```

**Note:** In the example above, we use our [Basque KenLM
model](https://huggingface.co/HiTZ/whisper-lm-ngrams). Optimize
the `lm_alpha`, `lm_beta`, etc., for best results with your own models.

#### Integrating a Large Language Model

If you prefer to use a Large LM:

```python
>>> # Load the pipeline
>>> pipe = pipeline(
...     "whisper-with-lm",
...     model="zuazo/whisper-tiny-eu",
...     llm_model="HiTZ/latxa-7b-v1.2", # Hugging Face LLM name or path
...     lm_alpha=2.73329396,
...     lm_beta=0.00178595,
...     language="eu",
... )

>>> # Transcribe an audio file or array
>>> pipe("tests/fixtures/audio.mp3")["text"]
'Talka diskoetxearekin grabatzen ditut beti abestien maketak.'

```

**Caution:** Running large LMs side-by-side with Whisper requires sufficient
GPU memory.

## 2) Using the `WhisperWithLM` Class Directly

If you prefer manual control, you can use the `WhisperWithLM` class:

```python
>>> from datasets import Audio, load_dataset
>>> from transformers import WhisperProcessor
>>> from huggingface_hub import hf_hub_download
>>> from whisper.audio import load_audio

>>> from whisper_lm_transformers import WhisperWithLM

>>> # Download the n-gram model
>>> lm_model = hf_hub_download(
...     repo_id="HiTZ/whisper-lm-ngrams", filename="5gram-eu.bin"
... )

>>> # Load the model
>>> model_name = "zuazo/whisper-tiny-eu"
>>> processor = WhisperProcessor.from_pretrained(model_name)
>>> model = WhisperWithLM.from_pretrained(model_name)

>>> # Load an audio example
>>> ds = load_dataset("openslr", "SLR76", split="train", trust_remote_code=True)
>>> audio = load_audio(ds[28]["audio"]["path"])

>>> # Process the audio and generate the output
>>> inputs = processor(audio=audio, sampling_rate=16000, return_tensors="pt")
>>> generated = model.generate(
...     input_features=inputs["input_features"],
...     tokenizer=processor.tokenizer,
...     lm_model=lm_model, # Provide a kenlm model path
...     lm_alpha=0.33582369,
...     lm_beta=0.68825565,
...     num_beams=5,
...     language="eu",
... )
>>> processor.decode(generated[0], skip_special_tokens=True)
'Talka diskoetxearekin grabatzen ditut beti abestien maketak.'

```

### Audio Processing Note

In the last example, we used OpenAI’s `load_audio()` function for reproduction.
You can also use
[standard HF audio processing methods](https://huggingface.co/docs/datasets/en/audio_process)
, e.g.  `ds.cast_column("audio", Audio(sampling_rate=16000))`. However, keep
consistent sample rates and methods, as different audio preprocessing can yield
different internal logits, thus altering the final LM integration results. For
example, if you have optimized the language model using our
[whisper-lm repository](https://github.com/hitz-zentroa/whisper-lm) based on
OpenAI's Whisper implementation, we recommend re-running the optimization with
the scripts provided here for the best results.

## Included Scripts

The package includes the following scripts:

* `whisper_evaluate_with_hf`: Evaluates a Whisper model in a dataset.
* `whisper_lm_optimizer_with_hf`: Optimize the n-gram or large language model.

Run them with `--help` to see how to use them.

### Optimizing Language Model Parameters

Optimize the alpha and beta parameters for the LM:

```shell
whisper_lm_optimizer_with_hf "zuazo/whisper-tiny-eu" \
    --study_name "lm_optimizer-lm-whisper-tiny-eu" \
    --dataset_split "train+validation" \
    --dataset_shuffle "True" \
    --dataset_name "eu" \
    --language "eu" \
    --beam_size 5 \
    --lm_path "5gram-eu.bin" \
    --n_trials 100
    --journal_storage \
    --n_jobs 28
```

We can also optimize for a large language models using the `--llm_path`
argument:

```shell
whisper_lm_optimizer_with_hf "zuazo/whisper-tiny-eu" \
    --study_name "lm_optimizer-llm-whisper-tiny-eu" \
    --dataset_split "train+validation" \
    --dataset_name "eu" \
    --dataset_shuffle "True" \
    --dataset_n 4000 \
    --language "eu" \
    --beam_size 5 \
    --llm_path "HiTZ/latxa-7b-v1.2" \
    --lm_alpha_min 0 --lm_beta_min 0 \
    --lm_alpha_max 3 --lm_beta_max 3 \
    --n_trials 100 \
    --journal_storage \
    --n_jobs 1
```

In this case, we will limit the jobs to 1 per GPU, because we are loading both
the Whisper and the LLM model in the GPU memory. This was run in 7 NVIDIA
A100-SXM4-80GB GPU.

### Evaluating Models

Evaluate the performance on standard datasets or your own data:

```shell
whisper_evaluate_with_hf "zuazo/whisper-tiny-eu" \
    --dataset "mozilla-foundation/common_voice_13_0" \
    --dataset_name "eu" \
    --dataset_split "test" \
    --language "eu" \
    --temperature 0 \
    --beam_size 5 \
    --lm_path "5gram-eu.bin" \
    --lm_alpha 0.33582369 --lm_beta 0.68825565
```

## Contributing

Contributions, bug reports, and feature requests are welcome! Please check out
[CONTRIBUTING.md](CONTRIBUTING.md) for details on how to set up your
environment and run tests before submitting changes.

## Citation

If you find this helpful in your research, please cite:

```bibtex
@misc{dezuazo2025whisperlmimprovingasrmodels,
      title={Whisper-LM: Improving ASR Models with Language Models for Low-Resource Languages},
      author={Xabier de Zuazo and Eva Navas and Ibon Saratxaga and Inma Hernáez Rioja},
      year={2025},
      eprint={2503.23542},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2503.23542},
}
```

Please, check the related paper preprint in
[arXiv:2503.23542](https://arxiv.org/abs/2503.23542)
for more details.
