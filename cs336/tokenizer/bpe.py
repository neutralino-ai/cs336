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
        for a, b in zip(seq, seq[1:], strict=False):
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


class BPETokenizer:
    """Encode/decode text with a trained byte-level BPE vocabulary."""

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []
        self._token_to_id = {token: i for i, token in vocab.items()}
        self._merge_ranks = {pair: rank for rank, pair in enumerate(merges)}
        if self.special_tokens:
            # match longest special tokens first so overlapping ones resolve greedily
            ordered = sorted(self.special_tokens, key=len, reverse=True)
            self._special_re = re.compile("(" + "|".join(re.escape(t) for t in ordered) + ")")
        else:
            self._special_re = None

    def _bpe(self, token: str) -> list[int]:
        pieces = [bytes([b]) for b in token.encode("utf-8")]
        while len(pieces) >= 2:
            # find the adjacent pair with the lowest (earliest) merge rank
            best_rank = None
            best_i = None
            for i in range(len(pieces) - 1):
                rank = self._merge_ranks.get((pieces[i], pieces[i + 1]))
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank = rank
                    best_i = i
            if best_i is None:
                break
            pieces[best_i : best_i + 2] = [pieces[best_i] + pieces[best_i + 1]]
        return [self._token_to_id[p] for p in pieces]

    def encode(self, text: str) -> list[int]:
        if self._special_re is None:
            segments = [text]
        else:
            segments = self._special_re.split(text)

        ids: list[int] = []
        special_set = set(self.special_tokens)
        for segment in segments:
            if not segment:
                continue
            if segment in special_set:
                ids.append(self._token_to_id[segment.encode("utf-8")])
            else:
                for token in pretokenize(segment):
                    ids.extend(self._bpe(token))
        return ids

    def decode(self, ids: list[int]) -> str:
        data = b"".join(self.vocab[i] for i in ids)
        return data.decode("utf-8", errors="replace")


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
