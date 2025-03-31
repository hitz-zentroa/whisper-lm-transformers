"""Module providing a custom logits processor for Whisper models.

This module contains the WhisperLMLogitsProcessor class which integrates
KenLM language models with Whisper's generation capabilities, enabling
the use of shallow fusion techniques for improved speech recognition
performance.
"""

import logging
import string
from threading import Lock

import kenlm
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import LogitsProcessor
from whisper.normalizers import BasicTextNormalizer


class WhisperLMLogitsProcessor(
    LogitsProcessor
):  # pylint: disable=too-many-instance-attributes,too-few-public-methods
    """Custom logits processor to integrate LMs with HF Whisper implementation.

    Each decoding step retains only the top-K tokens based on acoustic
    probabilities (based on the official Whisper implementation), then adjusts
    their scores by incorporating language model scores and optionally a word
    count penalty, promoting more linguistically plausible sequences.

    Args:
        tokenizer (`WhisperTokenizer`):
            A tokenizer instance capable of decoding token IDs to text, used
            for language model scoring.
        lm_model (`kenlm.Model` or `str`):
            Either a KenLM model instance or the path to a KenLM model file.
        lm_alpha (`float`, optional, defaults to 0.5):
            Weight for the language model score in the logit adjustment.
        lm_beta (`float`, optional, defaults to 0.0):
            Weight for the word count penalty in the logit adjustment.
        lm_eos (`str`, optional, defaults to "!?."):
            Characters considered as end of sentence for LM scoring purposes.
        lm_eow (`str`, optional, defaults to string.punctuation):
            Characters considered as end of word boundaries for LM scoring purposes.
        lm_normalize (`bool`, optional, defaults to True):
            Whether to normalize text using `BasicTextNormalizer` before LM scoring.
        lm_token_threshold (`int`, optional, defaults to 4):
            Minimum number of tokens a sequence must have before LM scoring is applied.
        top_k (`int`, optional):
            The number of top tokens to consider during each decoding step.
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        tokenizer,
        lm_model,
        lm_alpha=0.5,
        lm_beta=0.0,
        lm_eos="!?.",
        lm_eow=string.punctuation,
        lm_normalize=True,
        lm_token_threshold=4,
        top_k=None,
    ):  # pylint: disable=too-many-positional-arguments
        """Construct the WhisperLMLogitsProcessor object.

        Args:
            tokenizer (`transformers.WhisperTokenizer`):
                Whisper tokenizer (or any tokenizer that can decode IDs).
            lm_model (`str`):
                The KenLM model path or loaded instance.
            lm_alpha (`float`):
                Weight for LM score.
            lm_beta (`float`):
                Weight for word count.
            lm_eos (`str`):
                Set of characters to treat as end-of-sentence in LM scoring.
            lm_eow (`Set[str]`):
                Set of punctuation chars that might denote word boundary.
            lm_normalize (`bool`):
                Whether to apply `BasicTextNormalizer` prior to LM scoring.
            lm_token_threshold (`int`):
                Minimal # tokens in sequence before doing LM scoring.
            top_k (`int`):
                How many tokens to keep from the acoustic top-k per beam step.
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.special_token_ids = self.tokenizer.all_special_ids
        if isinstance(lm_model, str):
            self.lm_model = kenlm.Model(lm_model)
        else:
            self.lm_model = lm_model
        self.top_k = top_k
        self.lm_alpha = lm_alpha
        self.lm_beta = lm_beta

        self.lm_eos = lm_eos or ""
        self.lm_eow = lm_eow if lm_eow is not None else set(string.punctuation)
        self.lm_normalize = lm_normalize
        self.lm_normalizer = BasicTextNormalizer() if lm_normalize else None

        # Minimum number of tokens before we consider LM scoring
        self.lm_token_threshold = lm_token_threshold

    def __call__(  # pylint: disable=too-many-locals
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:  # pylint: disable=too-many-locals
        """Process logits with LM scoring during Whisper model generation.

        This method modifies the logits based on KenLM scores and is called at
        each decoding step. It applies top-k filtering followed by language
        model scoring to refine the predictions.

        Args:
            input_ids (torch.LongTensor):
                Tensor of token ids being processed.
            scores (torch.FloatTensor):
                Raw logits from the model that are to be processed.

        Returns:
            torch.FloatTensor:
                The adjusted logits after applying language model scoring.
        """
        beam_size, _ = scores.shape

        # Softmax to normalize the scores comming from the pipeline
        scores = F.log_softmax(scores.float(), dim=-1)

        for j in range(beam_size):
            row = scores[j]

            # Step 1: find top_k from acoustic distribution
            top_vals, top_inds = torch.topk(row, self.top_k)
            # fill new row with -inf
            new_row = torch.full_like(row, float("-inf"))

            prefix_ids = input_ids[j].tolist()

            for i in range(self.top_k):
                token_id = top_inds[i].item()
                acoustic_logprob = top_vals[i].item()

                # Build the new sequence
                sequence = prefix_ids + [token_id]

                # Compute KenLM-based fused score
                lm_score, word_count = self.lm_score_and_word_count(sequence)

                if lm_score is None:
                    # No LM score if text is empty
                    fused_score = acoustic_logprob
                else:
                    fused_score = (
                        acoustic_logprob
                        + self.lm_alpha * lm_score
                        + self.lm_beta * word_count
                    )

                new_row[token_id] = fused_score

            # Overwrite the row
            scores[j] = new_row

        return scores

    def lm_score_and_word_count(self, sequence_ids) -> tuple:
        """Get language model score and word count for a sequence.

        Args:
            sequence (tuple of int): A sequence of token IDs.

        Returns:
            float: The language model score for the decoded text of the sequence.
            int: The number of words in the decoded text of the sequence.
        """
        if self.lm_model is None:
            return None, 0

        # Convert sequence of tokens to text
        sequence = [t for t in sequence_ids if t not in self.special_token_ids]
        if len(sequence) < self.lm_token_threshold:
            return None, 0
        text = self.tokenizer.decode(sequence, skip_special_tokens=True)

        # Early return for empty text
        if not text:
            return None, 0
        logging.debug('LM text: "%s"', text)

        # Normalize the text
        if self.lm_normalize:
            normalized_text = self.lm_normalizer(text)
        else:
            normalized_text = text
        logging.debug('LM text normalized: "%s"', normalized_text)

        # Check for end of sentence and end of word:
        eos = text[-1] in self.lm_eos

        word_count = len(normalized_text.split())
        logging.debug("Word count: %d", word_count)

        # In KenLM, the most probable sequences have a higher score:
        score = self.lm_model.score(normalized_text, bos=True, eos=eos)
        logging.debug("LM score: %f", score)

        return score, word_count


class LLMSingleton:
    """A singleton class to manage the loading and accessing of LLMs.

    This ensures that each model and tokenizer is loaded only once in a
    thread-safe manner, making them reusable across different parts of an
    application.
    """

    _models = {}
    _tokenizers = {}
    _models_lock = Lock()
    _tokenizers_lock = Lock()

    @classmethod
    def get_model(cls, model_name):
        """Retrieve or load a model singleton based on the given model name.

        Args:
            model_name (str): The name of the model to load or retrieve.

        Returns:
            An instance of the model corresponding to `model_name`.
        """
        with cls._models_lock:
            if model_name not in cls._models:
                logging.debug("Loading model: %s", model_name)
                model = AutoModelForCausalLM.from_pretrained(model_name)
                cls._models[model_name] = model
            return cls._models[model_name]

    @classmethod
    def get_tokenizer(cls, tokenizer_name):
        """Retrieve or load a tokenizer singleton based on the given name.

        Args:
            tokenizer_name (str): The name of the tokenizer to load or retrieve.

        Returns:
            An instance of the tokenizer corresponding to `tokenizer_name`.
        """
        with cls._tokenizers_lock:
            if tokenizer_name not in cls._tokenizers:
                logging.debug("Loading tokenizer: %s", tokenizer_name)
                tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
                cls._tokenizers[tokenizer_name] = tokenizer
            return cls._tokenizers[tokenizer_name]


class WhisperLLMLogitsProcessor(
    WhisperLMLogitsProcessor
):  # pylint: disable=too-many-instance-attributes,too-few-public-methods
    """Logits processor to integrate HF LLMs with HF Whisper implementation.

    Args:
        tokenizer (`WhisperTokenizer`):
            A tokenizer instance capable of decoding token IDs to text, used
            for language model scoring.
        lm_model (`kenlm.Model` or `str`):
            Either a KenLM model instance or the path to a KenLM model file.
        lm_model (`str`):
            The name or path path of an LLM model.
        lm_alpha (`float`, optional, defaults to 0.5):
            Weight for the language model score in the logit adjustment.
        lm_beta (`float`, optional, defaults to 0.0):
            Weight for the word count penalty in the logit adjustment.
        lm_eos (`str`, optional, defaults to "!?."):
            Characters considered as end of sentence for LM scoring purposes.
        lm_eow (`str`, optional, defaults to string.punctuation):
            Characters considered as end of word boundaries for LM scoring purposes.
        lm_normalize (`bool`, optional, defaults to True):
            Whether to normalize text using `BasicTextNormalizer` before LM scoring.
        lm_token_threshold (`int`, optional, defaults to 4):
            Minimum number of tokens a sequence must have before LM scoring is applied.
        top_k (`int`, optional):
            The number of top tokens to consider during each decoding step.
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        tokenizer,
        lm_model=None,
        lm_alpha=0.5,
        lm_beta=0.0,
        lm_eos="!?.",
        lm_eow=string.punctuation,
        lm_normalize=True,
        lm_token_threshold=4,
        top_k=None,
    ):  # pylint: disable=too-many-positional-arguments
        """Construct the WhisperLMLogitsProcessor object.

        Args:
            tokenizer (`transformers.WhisperTokenizer`):
                Whisper tokenizer (or any tokenizer that can decode IDs).
            lm_model (`str`):
                The name or path path of an LLM model.
            lm_alpha (`float`):
                Weight for LM score.
            lm_beta (`float`):
                Weight for word count.
            lm_eos (`str`):
                Set of characters to treat as end-of-sentence in LM scoring.
            lm_eow (`Set[str]`):
                Set of punctuation chars that might denote word boundary.
            lm_normalize (`bool`):
                Whether to apply `BasicTextNormalizer` prior to LM scoring.
            lm_token_threshold (`int`):
                Minimal # tokens in sequence before doing LM scoring.
            top_k (`int`):
                How many tokens to keep from the acoustic top-k per beam step.
        """
        super().__init__(
            tokenizer,
            None,
            lm_alpha,
            lm_beta,
            lm_eos,
            lm_eow,
            lm_normalize,
            lm_token_threshold,
            top_k,
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.lm_model = LLMSingleton.get_model(lm_model).to(self.device)
        self.lm_tokenizer = LLMSingleton.get_tokenizer(lm_model)

    def lm_score_and_word_count(self, sequence_ids) -> tuple:
        """Get language model score and word count for a sequence.

        Args:
            sequence (tuple of int): A sequence of token IDs.

        Returns:
            float: The language model score for the decoded text of the sequence.
            int: The number of words in the decoded text of the sequence.
        """
        if self.lm_model is None:
            return None, 0

        # Convert sequence of tokens to text
        sequence = [t for t in sequence_ids if t not in self.special_token_ids]
        if len(sequence) < self.lm_token_threshold:
            return None, 0
        text = self.tokenizer.decode(sequence, skip_special_tokens=True)

        # Early return for empty text
        if not text:
            return None, 0
        logging.debug('LLM text: "%s"', text)

        # Normalize the text
        if self.lm_normalize:
            normalized_text = self.lm_normalizer(text)
        else:
            normalized_text = text
        logging.debug('LLM text normalized: "%s"', normalized_text)

        word_count = len(normalized_text.split())
        logging.debug("Word count: %d", word_count)

        # Tokenize the input
        tokens = self.lm_tokenizer(normalized_text, return_tensors="pt").to(self.device)

        # Get input IDs and attention mask
        input_ids = tokens["input_ids"]
        attention_mask = tokens["attention_mask"]

        # outputs = self.lm_model(**tokens)
        # Calculate output from the model
        outputs = self.lm_model(
            input_ids, attention_mask=attention_mask, labels=input_ids
        )

        # Get the log probabilities of the last token
        log_probs = outputs.logits[:, -1, :].softmax(dim=-1)
        # Use the highest log probability as the score
        score = log_probs.max().item()
        logging.debug("LLM score: %f", score)

        return score, word_count
