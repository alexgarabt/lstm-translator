import torch
import torch.nn as nn
from .lstm import LSTM, LSTMCell
from .attention import Attention

class Decoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, encoder_dim: int, num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # embed_dim + encoder_dim
        # decoder {embedded(y_{t-1}) + context_attn}
        self.lstm  = LSTM(embed_dim + encoder_dim, hidden_dim, num_layers)

        # Attention
        self.attention = Attention(encoder_dim, hidden_dim)

        # combinate decoder state + context (encoder)
        self.combination = nn.Linear(hidden_dim + encoder_dim, hidden_dim)

        # final projection h_tilde -> logits
        self.output_projection = nn.Linear(hidden_dim, vocab_size)

        self.dropout = nn.Dropout(dropout)

    def forward_step(self,
        token: torch.Tensor,            
        h_prev: list[torch.Tensor],     
        c_prev: list[torch.Tensor],     
        encoder_outputs: torch.Tensor,  
        context_prev: torch.Tensor,     
        mask: torch.Tensor,             
    ) -> tuple[torch.Tensor, list, list, torch.Tensor, torch.Tensor]:
        """
        One step of decodification

        Parameters:
            (batch,) — token id del paso anterior
            lista de (batch, hidden_dim) por capa
            lista de (batch, hidden_dim) por capa
            (batch, src_len, encoder_dim)
            (batch, encoder_dim) — contexto del paso anterior
            (batch, src_len)

        Returns:
            logits: (batch, vocab_size) — distribución sobre el vocabulario
            h_new: lista de hidden states actualizados
            c_new: lista de cell states actualizados
            context: (batch, encoder_dim) — nuevo vector de contexto
            attn_weights: (batch, src_len) — pesos de atención
        """

        # (batch,) → (batch, embed_dim)
        embedded = self.dropout(self.embedding(token))

        # concatenate the embedding from the last step
        lstm_input = torch.cat([embedded, context_prev], dim=1)

        # one step (propagated through all the layers) -> recurrence (we need the previous context + new generated y)
        h_new = []
        c_new = []
        layer_input = lstm_input
        for layer in range(self.num_layers):
            h_layer, c_layer = self.lstm.cells[layer](layer_input, h_prev[layer], c_prev[layer])
            h_new.append(h_layer)
            c_new.append(c_layer)
            layer_input = h_layer  

        # h_new[-1]: (batch, hidden_dim)
        context, attn_weights = self.attention(h_new[-1], encoder_outputs, mask)

        # combinate hidden state + context
        combined = torch.cat([h_new[-1], context], dim=1)
        # normalize
        h_tilde = torch.tanh(self.combination(combined))

        # proyect the vocab logits
        logits = self.output_projection(h_tilde)

        return logits, h_new, c_new, context, attn_weights
