import pytest

from cs336.tokenizer.bpe import (
    BPETokenizer,
    count_pairs,
    merge_pair,
    pretokenize,
    train_bpe,
    train_bpe_fast,
)


def test_merge_pair_replaces_all_occurrences():
    # sequence with the pair (1, 2) appearing twice
    seq = [1, 2, 3, 1, 2]
    assert merge_pair(seq, (1, 2), 4) == [4, 3, 4]


def test_merge_pair_handles_overlap_left_to_right():
    # (1, 1) appears overlapping; left-to-right scan merges the first pair,
    # leaving the trailing 1 unmerged
    seq = [1, 1, 1]
    assert merge_pair(seq, (1, 1), 9) == [9, 1]


def test_merge_pair_no_match_returns_equal_sequence():
    seq = [1, 2, 3]
    assert merge_pair(seq, (4, 5), 6) == [1, 2, 3]


def test_count_pairs_counts_adjacent_pairs_with_weights():
    # two pre-token sequences with multiplicities (word, count)
    sequences = [([1, 2, 1, 2], 3), ([1, 2, 3], 1)]
    counts = count_pairs(sequences)
    assert counts[(1, 2)] == 3 + 3 + 1  # two (1,2) in first*3, one in second
    assert counts[(2, 1)] == 3
    assert counts[(2, 3)] == 1


def test_count_pairs_ignores_length_one_sequences():
    counts = count_pairs([([5], 10)])
    assert counts == {}


def test_pretokenize_keeps_leading_space_with_word():
    # GPT-2 style: a space attaches to the following word
    assert pretokenize("Hello world") == ["Hello", " world"]


def test_pretokenize_splits_contraction_and_punctuation():
    assert pretokenize("don't!") == ["don", "'t", "!"]


def test_pretokenize_separates_digits_from_letters():
    assert pretokenize("abc123") == ["abc", "123"]


def test_train_bpe_base_vocab_has_256_bytes():
    vocab, merges = train_bpe("ab", vocab_size=256, special_tokens=[])
    assert len(vocab) == 256
    assert vocab[0] == b"\x00"
    assert vocab[97] == b"a"
    assert merges == []


def test_train_bpe_merge_order_and_tiebreak():
    # "aaab" -> bytes [97,97,97,98].
    # round 1: (a,a) most frequent (count 2) -> merge to b"aa"
    # round 2: tie at count 1 between (b"aa",b"a") and (b"a",b"b");
    #          lexicographically greater pair (b"aa",b"a") wins -> b"aaa"
    vocab, merges = train_bpe("aaab", vocab_size=258, special_tokens=[])
    assert merges == [(b"a", b"a"), (b"aa", b"a")]
    assert vocab[256] == b"aa"
    assert vocab[257] == b"aaa"
    assert len(vocab) == 258


def test_train_bpe_adds_special_tokens_and_never_merges_them():
    vocab, merges = train_bpe(
        "aaab", vocab_size=300, special_tokens=["<|endoftext|>"]
    )
    # special token occupies an id and appears verbatim in the vocab
    assert b"<|endoftext|>" in vocab.values()
    # no merge ever contains bytes spanning the special token
    for a, b in merges:
        assert b"<|endoftext|>" not in a + b


def test_tokenizer_roundtrip_plain_text():
    vocab, merges = train_bpe("the cat sat on the mat", vocab_size=300, special_tokens=[])
    tok = BPETokenizer(vocab, merges)
    ids = tok.encode("the cat sat")
    assert all(isinstance(i, int) for i in ids)
    assert tok.decode(ids) == "the cat sat"


def test_tokenizer_applies_merges_to_shrink_sequence():
    # after training, "the" should encode to fewer ids than its 3 raw bytes
    vocab, merges = train_bpe("the the the the", vocab_size=300, special_tokens=[])
    tok = BPETokenizer(vocab, merges)
    assert len(tok.encode("the")) < len(b"the")


def test_tokenizer_handles_special_token_as_single_id():
    vocab, merges = train_bpe("hello world", vocab_size=300, special_tokens=["<|endoftext|>"])
    tok = BPETokenizer(vocab, merges, special_tokens=["<|endoftext|>"])
    special_id = next(i for i, b in vocab.items() if b == b"<|endoftext|>")
    ids = tok.encode("hello<|endoftext|>world")
    assert special_id in ids
    assert tok.decode(ids) == "hello<|endoftext|>world"


def test_tokenizer_roundtrip_multibyte_unicode():
    text = "héllo 世界"
    vocab, merges = train_bpe(text, vocab_size=320, special_tokens=[])
    tok = BPETokenizer(vocab, merges)
    assert tok.decode(tok.encode(text)) == text


# --- optimized trainer: must be byte-identical to the naive reference ---

_EQUIV_CASES = [
    ("aaab", 258, []),
    ("the cat sat on the mat. " * 10, 320, []),
    ("aaaaa bbbbb aaaaa ababab", 280, []),  # overlap-heavy
    ("hello world hello there world", 300, ["<|endoftext|>"]),
    ("héllo 世界 héllo 世界 abc", 330, []),
]


@pytest.mark.parametrize("text,vocab_size,special", _EQUIV_CASES)
def test_train_bpe_fast_matches_naive(text, vocab_size, special):
    v_slow, m_slow = train_bpe(text, vocab_size, special)
    v_fast, m_fast = train_bpe_fast(text, vocab_size, special)
    assert m_fast == m_slow
    assert v_fast == v_slow
