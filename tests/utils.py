"""Utility functions used in tests."""


def decode_and_normalize(processor, normalizer, text_ref, generated):
    """
    Decode the tokens and normalize the reference and output text.

    Args:
        processor (`transformers.WhisperProcessor`):
            The WhisperProcessor for decoding tokens.
        normalizer (`function`):
            A function to normalize the text (fixture from conftest.py).
        text_ref (`str`):
            The reference transcription to compare against.
        generated (`torch.LongTensor`):
            The model output tokens.

    Returns:
        tuple (str, str): The normalized reference text and normalized model output.
    """
    text_out = processor.decode(generated[0], skip_special_tokens=True)
    text_ref_norm = normalizer(text_ref).strip()
    text_out_norm = normalizer(text_out).strip()
    return text_ref_norm, text_out_norm
