"""Byte-level BPE tokenizer (CS336 Assignment 1)."""

from __future__ import annotations

from collections import defaultdict

import regex as re

# GPT-2 pre-tokenization pattern (contractions, words, numbers, punctuation, whitespace).
GPT2_SPLIT_PATTERN = (
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)
_PRETOKEN_RE = re.compile(GPT2_SPLIT_PATTERN)


def pretokenize(text: str) -> list[str]:
    """Split ``text`` into pre-tokens using the GPT-2 regex."""
    return _PRETOKEN_RE.findall(text)


def count_pairs(sequences: list[tuple[list[int], int]]) -> dict[tuple[int, int], int]:
    """Count adjacent pairs across ``(sequence, weight)`` items, weighting each
    pair occurrence by its sequence's count."""
    counts: dict[tuple[int, int], int] = defaultdict(int)
    for seq, weight in sequences:
        for a, b in zip(seq, seq[1:]):
            counts[(a, b)] += weight
    return dict(counts)


def merge_pair(seq: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """Replace every non-overlapping left-to-right occurrence of ``pair`` in
    ``seq`` with ``new_id``."""
    merged: list[int] = []
    i = 0
    n = len(seq)
    while i < n:
        if i < n - 1 and seq[i] == pair[0] and seq[i + 1] == pair[1]:
            merged.append(new_id)
            i += 2
        else:
            merged.append(seq[i])
            i += 1
    return merged
