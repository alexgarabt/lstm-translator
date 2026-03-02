import torch
import random
import torch.nn as nn

from .encoder import Encoder
from .decoder import Decoder

class Seq2Seq(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, pad_token_id: int):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.pad_token_id = pad_token_id

    def create_mask(self, src: torch.Tensor) -> torch.Tensor:
        """
        Create the mask for the padding in attention
        True = real position, False = padding
        (batch, src_len)
        """
        return src != self.pad_token_id

    def forward(self, src: torch.Tensor, src_lengths: torch.Tensor, trg: torch.Tensor, teach_forcing_ratio: float = 0.5) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Parameters
        ----------
        src : input tokens in first secuence
        src_lengths : true length of each sentence (without padding)  
        trg : target tokens in second sequence. trg[:, 0] is <START>, trg[:, -1] is <END>  
        teacher_forcing_ratio : probability of using the correct token vs. the model’s prediction  
        
        Returns
        -------
        all_logits : (batch, trg_len - 1, vocab_size) — predictions for each step  
                     trg_len - 1 because we do not predict anything for <START>  
        all_attention : list of (batch, src_len) — attention weights for each step  
        """
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        mask = self.create_mask(src)

        # encoder
        encoder_outputs, (h_dec, c_dec) = self.encoder(src, src_lengths)

        # initial states -> decoder
        h = [h_dec[layer] for layer in range(self.decoder.num_layers)]
        c = [c_dec[layer] for layer in range(self.decoder.num_layers)]

        # initial context -> zeros
        encoder_dim = encoder_outputs.shape[2]
        context = torch.zeros(batch_size, encoder_dim, device=src.device)

        # first token <START>
        current_token = trg[:, 0]

        all_logits = []
        all_attention = []

        for t in range(1, trg_len):
            logits, h, c, context, attn_weights = self.decoder.forward_step(current_token, h, c, encoder_outputs, context, mask)

            all_logits.append(logits)
            all_attention.append(attn_weights)

            # teacher forcing and autoregressive sampling
            if random.random() < teach_forcing_ratio:
                current_token = trg[:, t]
            else:
                current_token = logits.argmax(dim=1)

        all_logits = torch.stack(all_logits, dim=1)
        return all_logits, all_attention


