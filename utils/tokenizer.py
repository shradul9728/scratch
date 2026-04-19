"""
Tokenizers for the GPT project.
  - CharTokenizer: character-level encoding
  - BPETokenizer: sub-word encoding via tiktoken
"""
from typing import List


class CharTokenizer:
    """
    Character-level tokenizer.
    Builds a vocabulary from the input text and maps each unique character to an integer.
    """

    def __init__(self, text: str):
        """
        Build vocabulary from the given text.

        Args:
            text: The full training corpus as a string.
        """
        chars = sorted(set(text))
        self._stoi = {ch: i for i, ch in enumerate(chars)}
        self._itos = {i: ch for ch, i in self._stoi.items()}

    @property
    def vocab_size(self) -> int:
        """Number of unique tokens in the vocabulary."""
        return len(self._stoi)

    @property
    def stoi(self) -> dict:
        """String-to-integer mapping."""
        return dict(self._stoi)

    @property
    def itos(self) -> dict:
        """Integer-to-string mapping."""
        return dict(self._itos)

    def encode(self, text: str) -> List[int]:
        """
        Encode a string into a list of integer token IDs.

        Args:
            text: Input string to encode.

        Returns:
            List of integer token IDs.
        """
        return [self._stoi[ch] for ch in text]

    def decode(self, ids: List[int]) -> str:
        """
        Decode a list of integer token IDs back into a string.

        Args:
            ids: List of integer token IDs.

        Returns:
            Decoded string.
        """
        return ''.join(self._itos[i] for i in ids)


class BPETokenizer:
    """
    Sub-word tokenizer backed by OpenAI's tiktoken library.
    Uses the cl100k_base encoding (same as GPT-4).
    """

    def __init__(self, encoding_name: str = "cl100k_base"):
        """
        Initialize the BPE tokenizer.

        Args:
            encoding_name: Name of the tiktoken encoding to use.
        """
        import tiktoken
        self._enc = tiktoken.get_encoding(encoding_name)

    @property
    def vocab_size(self) -> int:
        """Number of tokens in the BPE vocabulary."""
        return self._enc.n_vocab

    def encode(self, text: str) -> List[int]:
        """
        Encode a string into a list of sub-word token IDs.

        Args:
            text: Input string to encode.

        Returns:
            List of integer token IDs.
        """
        return self._enc.encode(text, allowed_special="all", disallowed_special=())

    def decode(self, ids: List[int]) -> str:
        """
        Decode a list of sub-word token IDs back into a string.

        Args:
            ids: List of integer token IDs.

        Returns:
            Decoded string.
        """
        return self._enc.decode(ids)
