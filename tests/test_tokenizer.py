"""Tests for utils/tokenizer.py"""
import pytest

from utils.tokenizer import CharTokenizer, BPETokenizer


# ──────────────── CharTokenizer ────────────────

class TestCharTokenizer:
    SAMPLE = "hello world"

    def test_vocab_size(self):
        tok = CharTokenizer(self.SAMPLE)
        unique_chars = set(self.SAMPLE)
        assert tok.vocab_size == len(unique_chars)

    def test_stoi_itos_consistency(self):
        tok = CharTokenizer(self.SAMPLE)
        for ch, idx in tok.stoi.items():
            assert tok.itos[idx] == ch

    def test_encode_returns_list_of_ints(self):
        tok = CharTokenizer(self.SAMPLE)
        encoded = tok.encode(self.SAMPLE)
        assert isinstance(encoded, list)
        assert all(isinstance(i, int) for i in encoded)
        assert len(encoded) == len(self.SAMPLE)

    def test_roundtrip(self):
        tok = CharTokenizer(self.SAMPLE)
        assert tok.decode(tok.encode(self.SAMPLE)) == self.SAMPLE

    def test_roundtrip_multiline(self):
        text = "First line.\nSecond line.\n"
        tok = CharTokenizer(text)
        assert tok.decode(tok.encode(text)) == text


# ──────────────── BPETokenizer ────────────────

class TestBPETokenizer:
    def test_vocab_size(self):
        tok = BPETokenizer()
        assert tok.vocab_size > 50000  # cl100k_base has ~100k tokens

    def test_roundtrip(self):
        tok = BPETokenizer()
        text = "Hello, world! This is a BPE test."
        assert tok.decode(tok.encode(text)) == text

    def test_encode_returns_list_of_ints(self):
        tok = BPETokenizer()
        encoded = tok.encode("Testing 123")
        assert isinstance(encoded, list)
        assert all(isinstance(i, int) for i in encoded)
