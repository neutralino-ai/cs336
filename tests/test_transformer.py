import torch
import torch.nn.functional as F

from cs336.models.transformer import (
    Embedding,
    Linear,
    MultiHeadSelfAttention,
    RMSNorm,
    RotaryPositionalEmbedding,
    SwiGLU,
    TransformerBlock,
    TransformerLM,
    cross_entropy,
    scaled_dot_product_attention,
    softmax,
)


def test_softmax_sums_to_one_and_matches_torch():
    x = torch.randn(4, 7)
    out = softmax(x, dim=-1)
    assert torch.allclose(out.sum(dim=-1), torch.ones(4))
    assert torch.allclose(out, torch.softmax(x, dim=-1), atol=1e-6)


def test_softmax_is_numerically_stable_for_large_values():
    x = torch.tensor([1000.0, 1001.0, 1002.0])
    out = softmax(x, dim=-1)
    assert torch.isfinite(out).all()
    assert torch.allclose(out.sum(), torch.tensor(1.0))


def test_linear_applies_weight_without_bias():
    lin = Linear(3, 2)
    with torch.no_grad():
        lin.weight.copy_(torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 1.0]]))
    x = torch.tensor([[2.0, 3.0, 5.0]])
    out = lin(x)
    assert torch.allclose(out, torch.tensor([[2.0, 8.0]]))
    assert out.shape == (1, 2)


def test_embedding_looks_up_rows():
    emb = Embedding(5, 4)
    with torch.no_grad():
        emb.weight.copy_(torch.arange(20, dtype=torch.float32).reshape(5, 4))
    ids = torch.tensor([0, 2])
    out = emb(ids)
    assert torch.allclose(out[0], torch.tensor([0.0, 1.0, 2.0, 3.0]))
    assert torch.allclose(out[1], torch.tensor([8.0, 9.0, 10.0, 11.0]))


def test_rmsnorm_matches_manual_formula():
    torch.manual_seed(0)
    d = 8
    norm = RMSNorm(d, eps=1e-5)
    with torch.no_grad():
        norm.weight.copy_(torch.linspace(0.5, 2.0, d))
    x = torch.randn(2, 3, d)
    out = norm(x)
    rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-5)
    expected = x / rms * norm.weight
    assert torch.allclose(out, expected, atol=1e-5)


def test_rmsnorm_is_scale_invariant_up_to_weight():
    d = 16
    norm = RMSNorm(d)
    with torch.no_grad():
        norm.weight.fill_(1.0)
    x = torch.randn(1, d)
    assert torch.allclose(norm(x), norm(x * 10.0), atol=1e-4)


def test_rope_position_zero_is_identity():
    d = 8
    rope = RotaryPositionalEmbedding(d, max_seq_len=64)
    x = torch.randn(1, 1, d)
    assert torch.allclose(rope(x, torch.tensor([0])), x, atol=1e-5)


def test_rope_preserves_vector_norm():
    d = 8
    rope = RotaryPositionalEmbedding(d, max_seq_len=64)
    x = torch.randn(2, 4, d)
    out = rope(x, torch.arange(4))
    assert torch.allclose(out.norm(dim=-1), x.norm(dim=-1), atol=1e-4)


def test_rope_dot_product_depends_only_on_relative_position():
    d = 8
    rope = RotaryPositionalEmbedding(d, max_seq_len=64)
    q = torch.randn(1, 1, d)
    k = torch.randn(1, 1, d)

    def at(x, pos):
        return rope(x, torch.tensor([pos]))[0, 0]

    # both pairs have relative offset (m - n) = 2 -> dot products must match
    dp1 = at(q, 5) @ at(k, 3)
    dp2 = at(q, 2) @ at(k, 0)
    assert torch.allclose(dp1, dp2, atol=1e-4)


def test_sdpa_matches_manual_softmax_attention():
    torch.manual_seed(0)
    q = torch.randn(2, 3, 4)
    k = torch.randn(2, 5, 4)
    v = torch.randn(2, 5, 6)
    out = scaled_dot_product_attention(q, k, v)
    scores = q @ k.transpose(-1, -2) / (4**0.5)
    expected = torch.softmax(scores, dim=-1) @ v
    assert torch.allclose(out, expected, atol=1e-5)


def test_sdpa_causal_mask_blocks_future_tokens():
    torch.manual_seed(0)
    x = torch.randn(1, 3, 4)
    mask = torch.tril(torch.ones(3, 3)).bool()  # True = allowed
    out = scaled_dot_product_attention(x, x, x, mask=mask)
    # position 0 may only attend to key 0, so its output equals value 0 exactly
    assert torch.allclose(out[0, 0], x[0, 0], atol=1e-5)


def test_mha_preserves_shape():
    d_model, heads, t = 16, 4, 5
    mha = MultiHeadSelfAttention(d_model, heads)
    x = torch.randn(2, t, d_model)
    assert mha(x).shape == (2, t, d_model)


def test_mha_is_causal():
    torch.manual_seed(0)
    d_model, heads, t = 16, 4, 6
    rope = RotaryPositionalEmbedding(d_model // heads, max_seq_len=32)
    mha = MultiHeadSelfAttention(d_model, heads, rope=rope)
    x = torch.randn(1, t, d_model)
    out1 = mha(x)
    x2 = x.clone()
    x2[0, -1] += 100.0  # perturb the last (future) token
    out2 = mha(x2)
    # every earlier position must be unchanged
    assert torch.allclose(out1[0, :-1], out2[0, :-1], atol=1e-5)


def test_swiglu_matches_formula():
    torch.manual_seed(0)
    ff = SwiGLU(8, 16)
    x = torch.randn(2, 3, 8)
    out = ff(x)
    a = ff.w1(x)
    silu = a * torch.sigmoid(a)
    expected = ff.w2(silu * ff.w3(x))
    assert torch.allclose(out, expected, atol=1e-5)
    assert out.shape == (2, 3, 8)


def test_transformer_block_preserves_shape():
    torch.manual_seed(0)
    block = TransformerBlock(d_model=16, num_heads=4, d_ff=32)
    x = torch.randn(2, 5, 16)
    assert block(x).shape == (2, 5, 16)


def test_transformer_lm_output_shape():
    model = TransformerLM(
        vocab_size=50, context_length=16, d_model=16, num_layers=2, num_heads=4, d_ff=32
    )
    ids = torch.randint(0, 50, (2, 8))
    assert model(ids).shape == (2, 8, 50)


def test_transformer_lm_is_causal_end_to_end():
    torch.manual_seed(0)
    model = TransformerLM(
        vocab_size=50, context_length=16, d_model=16, num_layers=2, num_heads=4, d_ff=32
    )
    ids = torch.randint(0, 50, (1, 8))
    out1 = model(ids)
    ids2 = ids.clone()
    ids2[0, -1] = (ids[0, -1] + 1) % 50  # change only the last token
    out2 = model(ids2)
    # logits for every earlier position must be unaffected by a future token
    assert torch.allclose(out1[0, :-1], out2[0, :-1], atol=1e-5)


def test_cross_entropy_matches_torch():
    torch.manual_seed(0)
    logits = torch.randn(7, 10)
    targets = torch.randint(0, 10, (7,))
    ours = cross_entropy(logits, targets)
    ref = F.cross_entropy(logits, targets)
    assert torch.allclose(ours, ref, atol=1e-5)


def test_cross_entropy_is_stable_for_large_logits():
    logits = torch.tensor([[0.0, 1000.0], [1000.0, 0.0]])
    targets = torch.tensor([1, 0])
    loss = cross_entropy(logits, targets)
    assert torch.isfinite(loss)
    assert loss < 1e-3  # confident + correct -> near-zero loss
