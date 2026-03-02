import torch
import torch.nn as nn
from .lstm import LSTM

# BiLSTM encoder architecture
class Encoder(nn.Module):

    def __init__(self, vocab_size: int, embedded_dim: int, hidden_dim: int, num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # token_id -> embedding
        self.embedding = nn.Embedding(vocab_size, embedded_dim)

        # right to left & left to right LSTM
        self.lstm_forward = LSTM(embedded_dim, hidden_dim, num_layers)
        self.lstm_backward = LSTM(embedded_dim, hidden_dim, num_layers)

        # concatenate forward + backward
        self.h_projection = nn.Linear(2* hidden_dim, hidden_dim)
        self.c_projection = nn.Linear(2* hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)

   def forward(self, src: torch.Tensor, src_lengths: torch.Tensor) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Parameters:
            src: (batch, src_len) tokens id
            src_lengths: (batch,) real length of each sequence
        Returns:
            encoder_outputs: (batch, src_len, 2*hidden_dim) state in each position
            (h_dec, c_dec): initial state for the decoder
                h_dec: (num_layers, batch, hidden_dim)
                c_dec: (num_layers, batch, hidden_dim)
        """
        # FIX: CHECK FOR SRC_LENGTHS why is not used

        # embedding + dropout
        embedded = self.dropout(self.embedding(src))
        outputs_fwd, (h_fwd, c_fwd) = self.lstm_forward(embedded)

        # reverse embedding
        embedded_reverse = torch.flip(embedded, dims=[1])
        outputs_bwd, (h_bwd, c_bwd) = self.lstm_backward(embedded_reverse)
        # order for states
        outputs_bwd = torch.flip(outputs_bwd, dims=[1])

        # concatenate forward + backward -> batch, src_len, 2*hidden_dim
        encoder_outputs = torch.cat([outputs_fwd, outputs_bwd], dim=2)

        # initial states
        h_dec_layer = []
        c_dec_layer = []
        for layer in range(self.num_layers):
            # concatenate each layer
            h_cat = torch.cat([h_fwd[layer], h_bwd[layer]], dim=1)
            c_cat = torch.cat([c_fwd[layer], c_bwd[layer]], dim=1)

            # proyect & normalize tanh(x)
            h_dec_layer.append(torch.tanh(self.h_projection(h_cat)))
            c_dec_layer.append(torch.tanh(self.c_projection(c_cat)))

        # (num_layers, batch, hidden_dim)
        h_dec = torch.stack(h_dec_layer, dim=0)
        c_dec = torch.stack(c_dec_layer, dim=0)

        return encoder_outputs, (h_dec, c_dec)
