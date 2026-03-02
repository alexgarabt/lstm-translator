import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, encoder_dim: int, decoder_dim: int):
        """
        Parameters
        ----------
        encoder_dim : dimension of the encoder outputs (2 * hidden_dim BiLSTM)
        decoder_dim : dimension of the encoder hidden state (hidden_dim)
        """
        super().__init__()

        # encoder_dim (2*hidden_dim) -> decoder_dim (hidden_dim)
        self.encoder_projection = nn.Linear(encoder_dim, decoder_dim, bias=False)

    def forward(self, decoder_hidden: torch.Tensor, encoder_outputs: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
            decoder_hidden: torch.Tensor,    # (batch, decoder_dim)
            encoder_outputs: torch.Tensor,   # (batch, src_len, encoder_dim)
            mask: torch.Tensor,              # (batch, src_len) — True real positions, False is for padding
        Returns:
            context: (batch, encoder_dim) — vector de contexto
            attn_weights: (batch, src_len) — pesos de atención (para visualización)
        """

        encoder_projected = self.encoder_projection(encoder_outputs)
        # dot product attention
        # result -> (batch, src_len, hidden) @ (batch, hidden, 1) -> (batch, src_len, 1)
        scores = torch.bmm(encoder_projected, decoder_hidden.unsqueeze(2)).squeeze(2)
        # mask -infinite in padding positions
        scores = scores.masked_fill(~mask, float('-inf'))

        attn_weights = F.softmax(scores, dim=1)
        # attn_weights @ encoder_outputs -> (batch, 2*hidden)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)

        return context, attn_weights
