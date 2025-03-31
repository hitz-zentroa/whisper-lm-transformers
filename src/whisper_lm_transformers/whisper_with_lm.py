"""Extends WhisperForConditionalGeneration for language model integration.

This module defines the WhisperWithLM class, a subclass of the
WhisperForConditionalGeneration class from Hugging Face's transformers library.

It improves Whisper models with language model support, using shallow fusion
to combine token logits from both Whisper's acoustic model and an external
KenLM model during the generation process.
"""

from transformers import WhisperForConditionalGeneration
from transformers.generation import LogitsProcessorList

from .logits_processor import (
    WhisperLLMLogitsProcessor,
    WhisperLMLogitsProcessor,
)


class WhisperWithLM(
    WhisperForConditionalGeneration
):  # pylint: disable=too-many-ancestors,too-many-instance-attributes
    """A subclass of `WhisperForConditionalGeneration` with LM integration.

    This class modifies the generation method to use a custom
    `WhisperLMLogitsProcessor`, applying shallow fusion to improve the
    linguistic accuracy of the generated text based on language model
    probabilities.

    Methods:
        generate:
            Extends the Whisper model's generate method by setting up the
            `WhisperLMLogitsProcessor` with provided KenLM configuration.
    """

    def __init__(self, *args, **kwargs):
        """Initialize the required attributes in the constructor."""
        super().__init__(*args, **kwargs)
        # Initialize attributes
        self.tokenizer = None
        self.lm_model = None
        self.llm_model = None
        self.lm_alpha = 0.5
        self.lm_beta = 0.0
        self.lm_top_k = None
        self.lm_token_threshold = 4
        self.lm_eos_chars = ".?!"
        self.lm_normalize = True

    def _retrieve_logit_processors(
        self, generation_config, logits_processor, begin_index, num_beams, device
    ):  # pylint: disable=too-many-arguments,too-many-positional-arguments
        """Retrieve and possibly initialize custom logit processors for Whisper.

        This includes the KenLM based logits processor to integrate the LMs.

        Args:
            generation_config (`GenerationConfig`):
                Configuration settings for generation.
            logits_processor (`LogitsProcessorList`):
                Existing list of logits processors to which the custom
                processor will be added.
            begin_index (`int`):
                The index at which to start the logit processing in the input
                sequence.
            num_beams (`int`):
                The number of beams used in beam search.
            device (`torch.device`):
                The device on which the tensors are located.

        Returns:
            `LogitsProcessorList`: An updated list of logits processors that
            includes the KenLM logits processor.
        """
        # Build the custom LM processor if user provided an LM
        if logits_processor is None:
            logits_processor = LogitsProcessorList()

        top_k = num_beams + 1 if self.lm_top_k is None else self.lm_top_k

        if self.lm_model is not None:
            lm_logits_processor_klass = WhisperLMLogitsProcessor
            lm_model = self.lm_model
        elif self.llm_model is not None:
            lm_logits_processor_klass = WhisperLLMLogitsProcessor
            lm_model = self.llm_model
        else:
            lm_logits_processor_klass = None
            lm_model = None

        # Insert the custom top-k + LM fusion in front or at the end
        if lm_logits_processor_klass is not None:
            lm_proc = lm_logits_processor_klass(
                tokenizer=self.tokenizer,  # or pass in from outside
                lm_model=lm_model,
                top_k=top_k,
                lm_alpha=self.lm_alpha,
                lm_beta=self.lm_beta,
                lm_eos=self.lm_eos_chars,
                lm_normalize=self.lm_normalize,
                lm_token_threshold=self.lm_token_threshold,
            )
            # For convenience, just prepend it
            logits_processor.insert(0, lm_proc)

        # Call super() to add normal Whisper processors
        logits_processor = (
            super()._retrieve_logit_processors(  # pylint: disable=no-member
                generation_config=generation_config,
                logits_processor=logits_processor,
                begin_index=begin_index,
                num_beams=num_beams,
                device=device,
            )
        )

        return logits_processor

    def generate(
        self,
        tokenizer,
        lm_model=None,
        llm_model=None,
        lm_alpha: float = 0.5,
        lm_beta: float = 0.0,
        lm_top_k=None,
        lm_token_threshold: int = 4,
        lm_eos_chars: str = ".?!",
        lm_normalize: bool = True,
        **kwargs,
    ):  # pylint: disable=too-many-arguments,too-many-positional-arguments
        """Generate method that integrates LM scoring with Whisper generation.

        Extends WhisperForConditionalGeneration's generate method by setting up
        a custom logits processor that uses KenLM for language model scoring.

        Args:
            tokenizer:
                Tokenizer associated with the Whisper model.
            lm_model (`str` or kenlm.Model`, optional):
                Path to the KenLM model file or an already loaded KenLM model.
            llm_model (`str`, optional):
                Name or path of the LLM model.
            lm_alpha (`float`, optional):
                The weight for the language model score.
            lm_alpha (`float`, optional):
                The weight for the language model score.
            lm_beta (`float`, optional):
                The weight for the token count penalty.
            lm_top_k (`int`, optional):
                Number of top tokens to consider in the beam search.
            lm_token_threshold (`int`, optional):
                Minimum number of tokens for language model scoring.
            lm_eos_chars (str, optional):
                Characters considered as end of sentence for LM scoring.
            lm_normalize (bool, optional):
                Whether to normalize the text before LM scoring.
            **kwargs:
                Additional keyword arguments passed to the parent generate
                method.

        Returns:
            torch.LongTensor: The generated token ids.
        """
        self.tokenizer = tokenizer
        self.lm_model = lm_model
        self.llm_model = llm_model
        self.lm_alpha = lm_alpha
        self.lm_beta = lm_beta
        self.lm_top_k = lm_top_k
        self.lm_token_threshold = lm_token_threshold
        self.lm_eos_chars = lm_eos_chars
        self.lm_normalize = lm_normalize

        return super().generate(**kwargs)  # pylint: disable=no-member
