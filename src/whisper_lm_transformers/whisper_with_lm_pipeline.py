"""Custom pipeline for Whisper + LM integration.

This module defines `WhisperWithLMPipeline`, a pipeline that inherits from
`AutomaticSpeechRecognitionPipeline` and leverages `WhisperWithLM` to allow
KenLM shallow-fusion decoding in an easy-to-use pipeline format.
"""

from typing import Any, Dict, Optional

from transformers import AutomaticSpeechRecognitionPipeline

from .whisper_with_lm import WhisperWithLM


class WhisperWithLMPipeline(
    AutomaticSpeechRecognitionPipeline
):  # pylint: disable=too-many-instance-attributes
    """A pipeline for automatic speech recognition using Whisper and LMs."""

    def __init__(
        self,
        model=None,
        lm_model: Optional[str] = None,
        llm_model: Optional[str] = None,
        lm_alpha: float = 0.5,
        lm_beta: float = 0.0,
        lm_top_k: Optional[int] = None,
        lm_token_threshold: int = 4,
        lm_eos_chars: str = ".?!",
        lm_normalize: bool = True,
        num_beams: Optional[int] = 5,
        language: Optional[str] = None,
        **kwargs,
    ):  # pylint: disable=too-many-arguments,too-many-positional-arguments
        """Initialize the "whisper-lm" pipeline, with the passed arguments.

        Args:
            model (`str` or `WhisperWithLM`, *optional*):
                The model to use. If a string, it will be loaded from that
                checkpoint. If `None`, must pass a `model` kwarg to the parent
                pipeline or rely on other arguments.
            lm_model (`str` or *optional*):
                Path to the KenLM `.bin` file or None.
            llm_model (`str` or *optional*):
                Path or name to the LLM model or None.
            lm_alpha (`float`, *optional*, defaults to 0.5):
                Weight for LM log-prob.
            lm_beta (`float`, *optional*, defaults to 0.0):
                Weight for word count penalty.
            lm_top_k (`int`, *optional*):
                Number of top tokens for beam expansions.
            lm_token_threshold (`int`, *optional*, defaults to 4):
                Minimum number of tokens for LM scoring.
            lm_eos_chars (`str`, *optional*, defaults to ".?!"):
                Characters considered as end-of-sentence for LM scoring.
            lm_normalize (`bool`, *optional*, defaults to True):
                Whether to apply `BasicTextNormalizer` or not.
            language (`str` or *optional*):
                The language of the audio.
            num_beams (`int` or *optionals*`):
                Switching from greedy search to beam search.
            **kwargs:
                Additional arguments passed to the parent pipeline's
                constructor.
        """
        # If user passes a string for model, might or might not be a
        # WhisperWithLM checkpoint
        if isinstance(model, str):
            model = WhisperWithLM.from_pretrained(model)

        # If user didn't pass a model, fallback to parent constructor's logic
        super().__init__(model=model, **kwargs)

        # These default attributes are used later in `_sanitize_parameters`.
        # We'll store them in the pipeline instance so the user can do:
        #   pipeline("whisper-with-lm", model="...", lm_model="...")
        self.lm_model = lm_model
        self.llm_model = llm_model
        self.lm_alpha = lm_alpha
        self.lm_beta = lm_beta
        self.lm_top_k = lm_top_k
        self.lm_token_threshold = lm_token_threshold
        self.lm_eos_chars = lm_eos_chars
        self.lm_normalize = lm_normalize
        self.num_beams = num_beams
        self.language = language

    def _sanitize_parameters(self, **kwargs):  # pylint: disable=arguments-differ
        """Override `_sanitize_parameters` to handle LM-specific arguments.

        Returns:
            A tuple of `(preprocess_params, forward_params, postprocess_params)`.
        """
        if "lm_model" in kwargs:
            self.lm_model = kwargs.pop("lm_model")
        if "llm_model" in kwargs:
            self.llm_model = kwargs.pop("llm_model")
        if "lm_alpha" in kwargs:
            self.lm_alpha = kwargs.pop("lm_alpha")
        if "lm_beta" in kwargs:
            self.lm_beta = kwargs.pop("lm_beta")
        if "lm_top_k" in kwargs:
            self.lm_top_k = kwargs.pop("lm_top_k")
        if "lm_token_threshold" in kwargs:
            self.lm_token_threshold = kwargs.pop("lm_token_threshold")
        if "lm_eos_chars" in kwargs:
            self.lm_eos_chars = kwargs.pop("lm_eos_chars")
        if "lm_normalize" in kwargs:
            self.lm_normalize = kwargs.pop("lm_normalize")
        if "num_beams" in kwargs:
            self.num_beams = kwargs.pop("num_beams")
        if "language" in kwargs:
            self.language = kwargs.pop("language")

        return super()._sanitize_parameters(**kwargs)

    def _forward(
        self, model_inputs: Dict[str, Any], **generate_kwargs
    ):  # pylint: disable=arguments-differ
        """Pass KenLM arguments to the `WhisperWithLM.generate(...)` call."""
        # Insert the LM integration arguments
        generate_kwargs.update(
            {
                "tokenizer": self.tokenizer,
                "lm_model": self.lm_model,
                "llm_model": self.llm_model,
                "lm_alpha": self.lm_alpha,
                "lm_beta": self.lm_beta,
                "lm_top_k": self.lm_top_k,
                "lm_token_threshold": self.lm_token_threshold,
                "lm_eos_chars": self.lm_eos_chars,
                "lm_normalize": self.lm_normalize,
                "num_beams": self.num_beams,
                "language": self.language,
            }
        )
        return super()._forward(model_inputs, **generate_kwargs)
