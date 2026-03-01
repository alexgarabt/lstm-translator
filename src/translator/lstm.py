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
        # W = {Wf,Wi,Wc,Wo} -> dim = [4 * hidden_size, (hidden_size + input_size)]
        # Each row -> W -> [h_{t-1}, x_t] -> gate dimension
        self.W = nn.Parameter(torch.randn(4 * hidden_size, input_size + hidden_size))

        # Bias -> for all the gates -> {bf, bi, bc, bo}
        self.b = nn.Parameter(torch.zeros(4 * input_size))

        # Initialize our paramters
        self._initialize_weights()

    def _initialize_weights(self):
        # Xavier uniform initialization for 
        nn.init.xavier_uniform_(self.W)
        nn.init.zeros_(self.b)

        # make the first bias (bias forget gate) = 1.0
        nn.init.ones_(self.b[:self.hidden_size])

    def forward(self, x_t: torch.Tensor, h_prev: torch.Tensor, c_prev: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Concatenation [h_prev, x_t]
        combined = torch.cat([h_prev, x_t], dim=1)

        # W @ x + b -> linear regression
        gates = torch.nn.functional.linear(combined, self.W, self.b)

        # recover the paramters
        f_gate, i_gate, c_candidate, o_gate = gates.chunk(4, dim=1)

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
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1):
        super().__init__()

        self.hidden_size = hidden_size
        self.input_size = input_size

        # Stack of LSTM cells
        # LSTM_0 -> input, input_size
        # LSTM_... -> hidden_state, hidden_size
        self.cells = nn.ModuleList()
        for layer in range(num_layers):
            layer_input_size = input_size if layer == 0 else hidden_size
            self.cells.append(LSTMCell(layer_input_size, hidden_size))

    def forward(self, x: torch.Tensor, initial_states: tuple | None = None) -> tuple[torch.Tensor, torch.Tensor]
        """
        Porcess a whole secuence
        
        Returns:
            outputs: (batch, seq_len, hidden_size) — h_t (last layer last step)
            (h_n, c_n): final states from all the layers
                h_n: (num_layers, batch, hidden_size)
                c_n: (num_layers, batch, hidden_size)
        """

        batch_size, seq_len, _ = x.shape
