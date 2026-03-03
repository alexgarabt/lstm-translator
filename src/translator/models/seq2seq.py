import torch
import random
import torch.nn as nn

from .encoder import Encoder
from .decoder import Decoder

class Seq2Seq(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, pad_token_id: int, bos_token_id: int, eos_token_id: int):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

    def create_mask(self, src: torch.Tensor) -> torch.Tensor:
        """
        Mask out padding, BOS and EOS from attention.
        Only content tokens are visible to the decoder.
        """
        mask = src != self.pad_token_id
        mask = mask & (src != self.bos_token_id)
        mask = mask & (src != self.eos_token_id)
        return mask

    def forward(self, src: torch.Tensor, src_lengths: torch.Tensor, trg: torch.Tensor, teacher_forcing_ratio: float = 0.5) -> tuple[torch.Tensor, list[torch.Tensor]]:
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
            if random.random() < teacher_forcing_ratio:
                current_token = trg[:, t]
            else:
                current_token = logits.argmax(dim=1)

        all_logits = torch.stack(all_logits, dim=1)
        return all_logits, all_attention

    @torch.no_grad()
    def beam_search(
        self,
        src: torch.Tensor,              # (1, src_len) — single example, no batch
        src_lengths: torch.Tensor,      # (1,)
        bos_id: int,
        eos_id: int,
        beam_width: int = 5,
        max_len: int = 50,
        length_penalty: float = 0.6,
    ) -> list[int]:
        """
        Beam search decoding for a single sentence.
    
        Returns the best token sequence (list of ids, without bos/eos).
        """
        device = src.device
    
        # 1. Encode — only once, shared by all beams
        encoder_outputs, (h_dec, c_dec) = self.encoder(src, src_lengths)
        mask = self.create_mask(src)
    
        # 2. Initialize decoder states
        h = [h_dec[layer] for layer in range(self.decoder.num_layers)]
        c = [c_dec[layer] for layer in range(self.decoder.num_layers)]
        context = torch.zeros(1, encoder_outputs.shape[2], device=device)
    
        # 3. Each beam is: (log_prob, token_sequence, h, c, context, finished)
        initial_beam = {
            'log_prob': 0.0,
            'tokens': [],
            'h': h,
            'c': c,
            'context': context,
            'finished': False,
        }
        beams = [initial_beam]
    
        # Current input token for all beams: <bos>
        completed = []
    
        for step in range(max_len):
            all_candidates = []
    
            for beam in beams:
                if beam['finished']:
                    completed.append(beam)
                    continue
    
                # Determine input token
                if len(beam['tokens']) == 0:
                    token = torch.tensor([bos_id], device=device)
                else:
                    token = torch.tensor([beam['tokens'][-1]], device=device)
    
                # One decoder step
                logits, h_new, c_new, context_new, _ = self.decoder.forward_step(
                    token, beam['h'], beam['c'],
                    encoder_outputs, beam['context'], mask,
                )
    
                # Log probabilities over vocabulary
                log_probs = torch.nn.functional.log_softmax(logits, dim=1)  # (1, vocab_size)
    
                # Top-k candidates from this beam
                topk_log_probs, topk_ids = log_probs.topk(beam_width, dim=1)  # (1, beam_width)
    
                for i in range(beam_width):
                    token_id = topk_ids[0, i].item()
                    token_log_prob = topk_log_probs[0, i].item()
    
                    new_beam = {
                        'log_prob': beam['log_prob'] + token_log_prob,
                        'tokens': beam['tokens'] + [token_id],
                        'h': h_new,
                        'c': c_new,
                        'context': context_new,
                        'finished': token_id == eos_id,
                    }
                    all_candidates.append(new_beam)
    
            if not all_candidates:
                break
    
            # Sort by score with length penalty and keep top beam_width
            def score(beam):
                length = len(beam['tokens'])
                # Length penalty: ((5 + length) / 6) ^ alpha
                # Prevents beam search from preferring shorter sequences
                lp = ((5 + length) / 6) ** length_penalty
                return beam['log_prob'] / lp
    
            all_candidates.sort(key=score, reverse=True)
            beams = all_candidates[:beam_width]
    
            # If all active beams are finished, stop
            if all(b['finished'] for b in beams):
                break
    
        # Collect all finished beams + any unfinished ones
        completed.extend([b for b in beams if b['finished']])
        if not completed:
            completed = beams  # fallback: use best unfinished
    
        # Return the best sequence
        best = max(completed, key=score)
        # Remove eos if present
        tokens = best['tokens']
        if tokens and tokens[-1] == eos_id:
            tokens = tokens[:-1]
    
        return tokens

