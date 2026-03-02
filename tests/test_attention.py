import torch
from translator.models.attention import Attention


def test_attention_shapes():
    """Verify attention produces correct output shapes."""
    batch, src_len, encoder_dim, decoder_dim = 8, 12, 1024, 512

    attn = Attention(encoder_dim, decoder_dim)

    decoder_hidden = torch.randn(batch, decoder_dim)
    encoder_outputs = torch.randn(batch, src_len, encoder_dim)
    mask = torch.ones(batch, src_len, dtype=torch.bool)

    context, weights = attn(decoder_hidden, encoder_outputs, mask)

    assert context.shape == (batch, encoder_dim), f"context shape: {context.shape}"
    assert weights.shape == (batch, src_len), f"weights shape: {weights.shape}"
    print("✓ Attention shapes correct")


def test_attention_weights_sum_to_one():
    """Verify attention weights form a valid probability distribution."""
    batch, src_len, encoder_dim, decoder_dim = 8, 12, 1024, 512

    attn = Attention(encoder_dim, decoder_dim)

    decoder_hidden = torch.randn(batch, decoder_dim)
    encoder_outputs = torch.randn(batch, src_len, encoder_dim)
    mask = torch.ones(batch, src_len, dtype=torch.bool)

    _, weights = attn(decoder_hidden, encoder_outputs, mask)

    # Weights should sum to 1 along src_len dimension
    sums = weights.sum(dim=1)
    assert torch.allclose(sums, torch.ones(batch), atol=1e-5), f"weights sum: {sums}"

    # Weights should be non-negative
    assert (weights >= 0).all(), "Negative attention weights found"
    print("✓ Attention weights sum to 1 and are non-negative")


def test_attention_mask():
    """Verify padding positions receive zero attention weight."""
    batch, src_len, encoder_dim, decoder_dim = 2, 6, 1024, 512

    attn = Attention(encoder_dim, decoder_dim)

    decoder_hidden = torch.randn(batch, decoder_dim)
    encoder_outputs = torch.randn(batch, src_len, encoder_dim)

    # First example: 4 real tokens, 2 padding
    # Second example: 3 real tokens, 3 padding
    mask = torch.tensor([
        [True, True, True, True, False, False],
        [True, True, True, False, False, False],
    ])

    _, weights = attn(decoder_hidden, encoder_outputs, mask)

    # Padding positions should have zero weight
    assert weights[0, 4].item() < 1e-6, f"Padding weight not zero: {weights[0, 4]}"
    assert weights[0, 5].item() < 1e-6, f"Padding weight not zero: {weights[0, 5]}"
    assert weights[1, 3].item() < 1e-6, f"Padding weight not zero: {weights[1, 3]}"
    assert weights[1, 4].item() < 1e-6, f"Padding weight not zero: {weights[1, 4]}"
    assert weights[1, 5].item() < 1e-6, f"Padding weight not zero: {weights[1, 5]}"

    # Real positions should still sum to 1
    assert torch.allclose(weights[0, :4].sum(), torch.tensor(1.0), atol=1e-5)
    assert torch.allclose(weights[1, :3].sum(), torch.tensor(1.0), atol=1e-5)
    print("✓ Attention mask works — padding gets zero weight")


if __name__ == "__main__":
    test_attention_shapes()
    test_attention_weights_sum_to_one()
    test_attention_mask()
