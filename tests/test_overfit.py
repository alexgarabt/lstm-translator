import torch
import torch.nn.functional as F
from tqdm import trange

from translator.models.encoder import Encoder
from translator.models.decoder import Decoder
from translator.models.seq2seq import Seq2Seq


def test_overfit_copy():
    """
    Overfit test: train the model to COPY input sequences.
    If it can't memorize 10 sequences in 200 epochs, there's a bug.
    
    Uses a tiny vocabulary with raw token ids (no tokenizer needed).
    Vocab: 0=<pad>, 1=<bos>, 2=<eos>, 3-12=data tokens
    """
    torch.manual_seed(42)

    # Config
    vocab_size = 13
    pad_id, bos_id, eos_id = 0, 1, 2
    embed_dim, hidden_dim, num_layers = 32, 64, 1
    lr = 1e-3
    num_epochs = 200

    # Build model
    encoder = Encoder(vocab_size, embed_dim, hidden_dim, num_layers, dropout=0.0)
    decoder = Decoder(vocab_size, embed_dim, hidden_dim, 2 * hidden_dim, num_layers, dropout=0.0)
    model = Seq2Seq(encoder, decoder, pad_token_id=pad_id)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Create 10 copy pairs: src = [bos, tokens..., eos], trg = same
    # Model should learn: output = input
    sequences = [
        [bos_id, 3, 4, 5, eos_id],
        [bos_id, 6, 7, 8, 9, eos_id],
        [bos_id, 3, 3, 3, eos_id],
        [bos_id, 10, 11, 12, eos_id],
        [bos_id, 5, 4, 3, eos_id],
        [bos_id, 7, 7, eos_id],
        [bos_id, 8, 9, 10, 11, eos_id],
        [bos_id, 3, 12, eos_id],
        [bos_id, 6, 6, 6, 6, eos_id],
        [bos_id, 4, 5, 6, 7, 8, eos_id],
    ]

    # Pad to same length
    max_len = max(len(s) for s in sequences)
    padded = [s + [pad_id] * (max_len - len(s)) for s in sequences]

    # src and trg are identical (copy task)
    src = torch.tensor(padded)
    trg = torch.tensor(padded)
    src_lengths = torch.tensor([len(s) for s in sequences])

    # Train
    model.train()
    for epoch in trange(num_epochs, desc="Overfit copy"):
        logits, _ = model(src, src_lengths, trg, teacher_forcing_ratio=1.0)

        loss = F.cross_entropy(
            logits.reshape(-1, vocab_size),
            trg[:, 1:].reshape(-1),
            ignore_index=pad_id,
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    final_loss = loss.item()
    print(f"Final loss: {final_loss:.6f}")

    # Verify: model should reproduce inputs exactly
    model.eval()
    with torch.no_grad():
        logits, _ = model(src, src_lengths, trg, teacher_forcing_ratio=0.0)
        predictions = logits.argmax(dim=-1)  # (batch, trg_len-1)

    correct = 0
    total = 0
    for i in range(len(sequences)):
        seq_len = len(sequences[i]) - 1  # exclude bos from target comparison
        pred = predictions[i, :seq_len].tolist()
        expected = sequences[i][1:]  # target without bos

        if pred == expected:
            correct += 1
        else:
            print(f"  MISMATCH seq {i}: expected {expected}, got {pred}")
        total += 1

    accuracy = correct / total
    print(f"Copy accuracy: {correct}/{total} = {accuracy:.0%}")

    assert final_loss < 0.1, f"Loss too high: {final_loss}"
    assert accuracy >= 0.8, f"Accuracy too low: {accuracy:.0%}"
    print("✓ Overfit copy test passed")


def test_overfit_reverse():
    """
    Overfit test: train the model to REVERSE input sequences.
    This verifies attention learns non-trivial alignment patterns.
    
    src: [bos, 3, 4, 5, eos] → trg: [bos, 5, 4, 3, eos]
    """
    torch.manual_seed(42)

    vocab_size = 13
    pad_id, bos_id, eos_id = 0, 1, 2
    embed_dim, hidden_dim, num_layers = 32, 64, 1
    lr = 1e-3
    num_epochs = 300  # reverse is harder than copy

    encoder = Encoder(vocab_size, embed_dim, hidden_dim, num_layers, dropout=0.0)
    decoder = Decoder(vocab_size, embed_dim, hidden_dim, 2 * hidden_dim, num_layers, dropout=0.0)
    model = Seq2Seq(encoder, decoder, pad_token_id=pad_id)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Create pairs: src = [bos, a, b, c, eos], trg = [bos, c, b, a, eos]
    src_sequences = [
        [bos_id, 3, 4, 5, eos_id],
        [bos_id, 6, 7, 8, 9, eos_id],
        [bos_id, 10, 11, 12, eos_id],
        [bos_id, 5, 4, 3, eos_id],
        [bos_id, 7, 8, eos_id],
        [bos_id, 3, 6, 9, 12, eos_id],
        [bos_id, 4, 4, 4, eos_id],
        [bos_id, 8, 3, eos_id],
    ]

    trg_sequences = []
    for seq in src_sequences:
        # Reverse the data tokens (between bos and eos)
        data = seq[1:-1]
        trg_sequences.append([bos_id] + data[::-1] + [eos_id])

    # Pad
    max_src = max(len(s) for s in src_sequences)
    max_trg = max(len(s) for s in trg_sequences)

    src_padded = [s + [pad_id] * (max_src - len(s)) for s in src_sequences]
    trg_padded = [s + [pad_id] * (max_trg - len(s)) for s in trg_sequences]

    src = torch.tensor(src_padded)
    trg = torch.tensor(trg_padded)
    src_lengths = torch.tensor([len(s) for s in src_sequences])

    # Train
    model.train()
    for epoch in trange(num_epochs, desc="Overfit reverse"):
        logits, _ = model(src, src_lengths, trg, teacher_forcing_ratio=1.0)

        loss = F.cross_entropy(
            logits.reshape(-1, vocab_size),
            trg[:, 1:].reshape(-1),
            ignore_index=pad_id,
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    final_loss = loss.item()
    print(f"Final loss: {final_loss:.6f}")

    # Verify
    model.eval()
    with torch.no_grad():
        logits, attention = model(src, src_lengths, trg, teacher_forcing_ratio=0.0)
        predictions = logits.argmax(dim=-1)

    correct = 0
    total = 0
    for i in range(len(src_sequences)):
        expected = trg_sequences[i][1:]  # without bos
        seq_len = len(expected)
        pred = predictions[i, :seq_len].tolist()

        if pred == expected:
            correct += 1
        else:
            print(f"  MISMATCH seq {i}: src={src_sequences[i]}, expected={expected}, got={pred}")
        total += 1

    accuracy = correct / total
    print(f"Reverse accuracy: {correct}/{total} = {accuracy:.0%}")

    assert final_loss < 0.1, f"Loss too high: {final_loss}"
    assert accuracy >= 0.7, f"Accuracy too low: {accuracy:.0%}"
    print("✓ Overfit reverse test passed")


if __name__ == "__main__":
    test_overfit_copy()
    print()
    test_overfit_reverse()
