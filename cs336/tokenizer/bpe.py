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


def _word_counts(text: str, special_tokens: list[str]) -> dict[tuple[int, ...], int]:
    """Pre-tokenize ``text`` into byte-id tuples with frequencies, dropping any
    special-token spans so merges never cross them."""
    if special_tokens:
        split_re = re.compile("|".join(re.escape(t) for t in special_tokens))
        segments = split_re.split(text)
    else:
        segments = [text]

    counts: dict[tuple[int, ...], int] = defaultdict(int)
    for segment in segments:
        for token in pretokenize(segment):
            counts[tuple(token.encode("utf-8"))] += 1
    return dict(counts)


def train_bpe(
    text: str, vocab_size: int, special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Train a byte-level BPE tokenizer.

    Returns ``(vocab, merges)`` where ``vocab`` maps id -> bytes and ``merges``
    is the ordered list of merged byte pairs. Ties in pair frequency are broken
    by choosing the lexicographically greatest pair.
    """
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    next_id = 256
    for token in special_tokens:
        vocab[next_id] = token.encode("utf-8")
        next_id += 1

    merges: list[tuple[bytes, bytes]] = []
    words = [[list(seq), count] for seq, count in _word_counts(text, special_tokens).items()]

    while len(vocab) < vocab_size:
        counts = count_pairs([(seq, count) for seq, count in words])
        if not counts:
            break
        best = max(counts, key=lambda p: (counts[p], (vocab[p[0]], vocab[p[1]])))
        merges.append((vocab[best[0]], vocab[best[1]]))
        vocab[next_id] = vocab[best[0]] + vocab[best[1]]
        for word in words:
            word[0] = merge_pair(word[0], best, next_id)
        next_id += 1

    return vocab, merges


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
