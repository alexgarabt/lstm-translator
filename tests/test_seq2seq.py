import torch
from translator.models.encoder import Encoder
from translator.models.decoder import Decoder
from translator.models.seq2seq import Seq2Seq

def _build_model(vocab_size=20, embed_dim=32, hidden_dim=64, num_layers=1):
    encoder = Encoder(vocab_size, embed_dim, hidden_dim, num_layers, dropout=0.0)
    decoder = Decoder(vocab_size, embed_dim, hidden_dim, 2 * hidden_dim, num_layers, dropout=0.0)
    model = Seq2Seq(encoder, decoder, pad_token_id=0, bos_token_id=1, eos_token_id=2)
    return model

def test_forward_shapes():
    """Verify Seq2Seq produces correct output shapes."""
    vocab_size, batch, src_len, trg_len = 20, 4, 8, 6

    model = _build_model(vocab_size)

    src = torch.randint(1, vocab_size, (batch, src_len))
    trg = torch.randint(1, vocab_size, (batch, trg_len))
    src_lengths = torch.full((batch,), src_len)

    logits, attention = model(src, src_lengths, trg, teacher_forcing_ratio=1.0)

    # trg_len - 1 predictions (no prediction for <START>)
    assert logits.shape == (batch, trg_len - 1, vocab_size), f"logits shape: {logits.shape}"
    assert len(attention) == trg_len - 1, f"attention steps: {len(attention)}"
    assert attention[0].shape == (batch, src_len), f"attention shape: {attention[0].shape}"
    print("✓ Seq2Seq forward shapes correct")


def test_forward_with_padding():
    """Verify Seq2Seq handles padded batches correctly."""
    vocab_size, batch, embed_dim, hidden_dim = 20, 2, 32, 64
    pad_id = 0

    model = _build_model(vocab_size, embed_dim, hidden_dim)

    # Different source lengths: first has 5 real tokens, second has 3
    src = torch.tensor([
        [1, 5, 3, 7, 2, 0, 0],   # 5 real + 2 padding
        [1, 4, 2, 0, 0, 0, 0],   # 3 real + 4 padding
    ])
    trg = torch.tensor([
        [1, 8, 6, 2, 0],         # 4 real + 1 padding
        [1, 9, 2, 0, 0],         # 3 real + 2 padding
    ])
    src_lengths = torch.tensor([5, 3])

    # Should run without errors
    logits, attention = model(src, src_lengths, trg, teacher_forcing_ratio=1.0)

    assert logits.shape == (batch, trg.shape[1] - 1, vocab_size)
    print("✓ Seq2Seq handles padding correctly")


def test_backward_runs():
    """Verify gradients flow through the entire model."""
    vocab_size = 20
    model = _build_model(vocab_size)

    src = torch.randint(1, vocab_size, (4, 8))
    trg = torch.randint(1, vocab_size, (4, 6))
    src_lengths = torch.full((4,), 8)

    logits, _ = model(src, src_lengths, trg, teacher_forcing_ratio=1.0)

    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, vocab_size),
        trg[:, 1:].reshape(-1),
        ignore_index=0,
    )
    loss.backward()

    # Check that all parameters received gradients
    params_with_grad = 0
    params_total = 0
    for name, param in model.named_parameters():
        params_total += 1
        if param.grad is not None and param.grad.abs().sum() > 0:
            params_with_grad += 1

    assert params_with_grad == params_total, (
        f"Only {params_with_grad}/{params_total} parameters received gradients"
    )
    print(f"✓ Gradients flow to all {params_total} parameters")


if __name__ == "__main__":
    test_forward_shapes()
    test_forward_with_padding()
    test_backward_runs()
