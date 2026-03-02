import torch
import torch.nn as nn

# Process on temporal step (t)
class LSTMCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        """
        Parameters
        ----------
        input_size : input dimension (e.g. 256 for embeddings)
        hidden_size : dimension of the hidden state h_t (e.g. 512)
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # W is fusion matrix of all gates parameters
        # W = {Wi,Wf,Wc,Wo} -> dim = [4 * hidden_size, (hidden_size + input_size)]
        # Each row -> W -> [h_{t-1}, x_t] -> gate dimension
        self.W = nn.Parameter(torch.randn(4 * hidden_size, input_size + hidden_size))

        # Bias -> for all the gates -> {bi, bf, bc, bo}
        self.b = nn.Parameter(torch.zeros(4 * hidden_size))

        # Initialize our paramters
        self._initialize_weights()

    def _initialize_weights(self):
        # Xavier uniform initialization for 
        nn.init.xavier_uniform_(self.W)
        nn.init.zeros_(self.b)

        # make the first bias (bias forget gate) = 1.0
        nn.init.ones_(self.b[self.hidden_size:2 * self.hidden_size])

    def forward(self, x_t: torch.Tensor, h_prev: torch.Tensor, c_prev: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Concatenation [h_prev, x_t]
        combined = torch.cat([h_prev, x_t], dim=1)

        # W @ x + b -> linear regression
        gates = torch.nn.functional.linear(combined, self.W, self.b)

        # recover the paramters
        i_gate, f_gate, c_candidate, o_gate = gates.chunk(4, dim=1)

        # apply activation
        # values from [0,1] (mask for information forget,add,ouput)
        f_gate = torch.sigmoid(f_gate)
        i_gate = torch.sigmoid(i_gate)
        o_gate = torch.sigmoid(o_gate)
        # values form [-1,1] positive and negative content
        c_candidate = torch.tanh(c_candidate)

        # update cell state
        c_t = f_gate * c_prev + i_gate * c_candidate

        # Expose the hidden state (normalize cell state to avoid bid numbers)
        h_t = o_gate * torch.tanh(c_t)

        return h_t,c_t


# Process (n) temporal steps, t -> {0,1,...,n}
class LSTM(nn.Module):
    hidden_size: int
    input_size: int
    num_layers: int
    cells: nn.ModuleList

    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1):
        super().__init__()

        self.hidden_size = hidden_size
        self.input_size = input_size
        self.num_layers = num_layers

        # Stack of LSTM cells
        # LSTM_0 -> input, input_size
        # LSTM_... -> hidden_state, hidden_size
        self.cells = nn.ModuleList()
        for layer in range(num_layers):
            layer_input_size = input_size if layer == 0 else hidden_size
            self.cells.append(LSTMCell(layer_input_size, hidden_size))

    def forward(self, x: torch.Tensor, initial_states: tuple | None = None) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Porcess a whole secuence
        
        Returns:
            outputs: (batch, seq_len, hidden_size) — h_t (last layer last step)
            (h_n, c_n): final states from all the layers
                h_n: (num_layers, batch, hidden_size)
                c_n: (num_layers, batch, hidden_size)
        """

        batch_size, seq_len, _ = x.shape

        if initial_states is None:
            h = [torch.zeros(batch_size, self.hidden_size, device=x.device) for _ in range(self.num_layers)]
            c = [torch.zeros(batch_size, self.hidden_size, device=x.device) for _ in range(self.num_layers)]
        else:
            h_0, c_0 = initial_states
            # h_0 -> (num_layers, batch, hidden) -> list(batch, hidden)
            h = [h_0[layer] for layer in range(self.num_layers)]
            c = [c_0[layer] for layer in range(self.num_layers)]

        # outputs (last layer)
        outputs = []

        for t in range(seq_len):
            # input -> step (t)
            layer_input = x[:, t, :]

            # propagate through all the stack
            for layer in range (self.num_layers):
                h[layer], c[layer] = self.cells[layer](layer_input, h[layer], c[layer])
                layer_input = h[layer]

            outputs.append(h[-1])

        # (batch, hidden) -> (batch, seq_len, hidden)
        outputs = torch.stack(outputs, dim=1)

        # list of (batch, hidden) → (num_layers, batch, hidden)
        h_n = torch.stack(h, dim=0)
        c_n = torch.stack(c, dim=0)

        return outputs, (h_n, c_n)

def test_lstm_cell():
    """
    Test our implementation vs pytorch implementation
    """
    torch.manual_seed(42)
    input_size, hidden_size, batch_size = 32, 64, 8

    our_cell = LSTMCell(input_size, hidden_size)
    torch_cell = nn.LSTMCell(input_size, hidden_size)

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

    print(f"Max diff h_t: {h_diff:.2e}")
    print(f"Max diff c_t: {c_diff:.2e}")

    assert h_diff < 1e-5, f"h_t differs by {h_diff}"
    assert c_diff < 1e-5, f"c_t differs by {c_diff}"
    print("✓ LSTMCell matches PyTorch!")
