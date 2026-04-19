"""Tests for utils/data_loader.py"""
import os
import tempfile
import pytest

from utils.data_loader import load_text


def test_load_text_returns_string(tmp_path):
    """load_text should return a non-empty string."""
    p = tmp_path / "sample.txt"
    p.write_text("Hello, world!", encoding='utf-8')
    text = load_text(str(p))
    assert isinstance(text, str)
    assert len(text) > 0
    assert text == "Hello, world!"


def test_load_text_file_not_found():
    """load_text should raise FileNotFoundError for missing files."""
    with pytest.raises(FileNotFoundError):
        load_text("nonexistent_file_xyz.txt")


def test_load_text_empty_file(tmp_path):
    """load_text should raise ValueError for empty files."""
    p = tmp_path / "empty.txt"
    p.write_text("", encoding='utf-8')
    with pytest.raises(ValueError):
        load_text(str(p))
