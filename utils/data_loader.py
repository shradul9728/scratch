"""
Data loader utility.
Reads raw text corpora from disk.
"""
import os


def load_text(path: str) -> str:
    """
    Read a raw text file and return its contents as a single string.

    Args:
        path: Path to the text file.

    Returns:
        The full text content of the file.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is empty.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Data file not found: {path}")

    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()

    if not text:
        raise ValueError(f"Data file is empty: {path}")

    return text
