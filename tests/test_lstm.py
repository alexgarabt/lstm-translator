import torch
import torch.nn as nn
from translator.models.lstm import LSTMCell, LSTM


def test_lstm_cell_matches_pytorch():
    """Verify our LSTMCell produces identical results to PyTorch's."""
    torch.manual_seed(42)
    input_size, hidden_size, batch_size = 32, 64, 8

    our_cell = LSTMCell(input_size, hidden_size)
    torch_cell = nn.LSTMCell(input_size, hidden_size)

    # Copy weights: our W is (4H, H+X) = [W_hh | W_ih]
    with torch.no_grad():
        torch_cell.weight_hh.copy_(our_cell.W[:, :hidden_size])
        torch_cell.weight_ih.copy_(our_cell.W[:, hidden_size:])
        torch_cell.bias_ih.copy_(our_cell.b)
        torch_cell.bias_hh.zero_()

    x = torch.randn(batch_size, input_size)
    h = torch.randn(batch_size, hidden_size)
    c = torch.randn(batch_size, hidden_size)

    h_ours, c_ours = our_cell(x, h, c)
    h_torch, c_torch = torch_cell(x, (h, c))

    h_diff = (h_ours - h_torch).abs().max().item()
    c_diff = (c_ours - c_torch).abs().max().item()

    assert h_diff < 1e-5, f"h_t differs by {h_diff}"
    assert c_diff < 1e-5, f"c_t differs by {c_diff}"
    print("✓ LSTMCell matches PyTorch")


def test_lstm_sequence():
    """Verify LSTM processes a full sequence with correct output shapes."""
    torch.manual_seed(42)
    batch, seq_len, input_size, hidden_size, num_layers = 4, 10, 32, 64, 2

    lstm = LSTM(input_size, hidden_size, num_layers)
    x = torch.randn(batch, seq_len, input_size)

    outputs, (h_n, c_n) = lstm(x)

    assert outputs.shape == (batch, seq_len, hidden_size), f"outputs shape: {outputs.shape}"
    assert h_n.shape == (num_layers, batch, hidden_size), f"h_n shape: {h_n.shape}"
    assert c_n.shape == (num_layers, batch, hidden_size), f"c_n shape: {c_n.shape}"
    print("✓ LSTM sequence shapes correct")


def test_lstm_initial_states():
    """Verify LSTM accepts custom initial states."""
    torch.manual_seed(42)
    batch, seq_len, input_size, hidden_size, num_layers = 4, 10, 32, 64, 2

    lstm = LSTM(input_size, hidden_size, num_layers)
    x = torch.randn(batch, seq_len, input_size)

    h_0 = torch.randn(num_layers, batch, hidden_size)
    c_0 = torch.randn(num_layers, batch, hidden_size)

    outputs, (h_n, c_n) = lstm(x, initial_states=(h_0, c_0))

    # Results should differ from zero-initialized
    outputs_zero, _ = lstm(x, initial_states=None)
    assert not torch.allclose(outputs, outputs_zero), "Custom init states had no effect"
    print("✓ LSTM initial states work")


if __name__ == "__main__":
    test_lstm_cell_matches_pytorch()
    test_lstm_sequence()
    test_lstm_initial_states()
