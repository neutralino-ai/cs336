from cs336.tokenizer.bpe import count_pairs, merge_pair, pretokenize


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
