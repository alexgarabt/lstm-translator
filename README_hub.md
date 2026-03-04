---
license: mit
language:
  - en
  - es
tags:
  - translation
  - lstm
  - seq2seq
  - attention
  - pytorch
  - from-scratch
pipeline_tag: translation
---

# LSTM English-to-Spanish Translator

A sequence-to-sequence neural machine translation model built **entirely from scratch** — custom LSTM cells, encoder, decoder, attention mechanism, and beam search — as a deep learning educational project.

**Code**: [github.com/alexgarabt/lstm-translator](https://github.com/alexgarabt/lstm-translator)

<video src="https://huggingface.co/alexgara/lstm-en-es-translator/resolve/main/img/lstm.mp4" controls width="600"></video>

## Architecture

| Component | Details |
|---|---|
| **Architecture** | BiLSTM Encoder → Dot-Product Attention → LSTM Decoder |
| **Parameters** | ~31.9M |
| **Encoder** | 2-layer Bidirectional LSTM (custom), 512 hidden units per direction |
| **Decoder** | 2-layer LSTM (custom) with Luong dot-product attention |
| **Tokenizer** | SentencePiece BPE, 16K vocab per language |
| **Training data** | Tatoeba (~222K pairs) + Europarl filtered (~400K pairs) = ~622K pairs |

```
Source (EN) → [Embedding] → [BiLSTM Encoder] → encoder_outputs
                                                      ↓
                                              [Dot-Product Attention]
                                                      ↓
Target (ES) → [Embedding] → [LSTM Decoder] → [Output Projection] → predicted tokens
```

## Quick Start

```bash
pip install torch sentencepiece huggingface_hub
```

```python
import torch
from huggingface_hub import hf_hub_download

REPO_ID = "alexgara/lstm-en-es-translator"

# Download
checkpoint_path = hf_hub_download(REPO_ID, "model.pt")
en_tok_path = hf_hub_download(REPO_ID, "spm_en.model")
es_tok_path = hf_hub_download(REPO_ID, "spm_es.model")

# For full usage with the translator package, see the GitHub repo
```

For full inference with greedy/beam decoding, clone the [GitHub repo](https://github.com/alexgarabt/lstm-translator) and run:

```bash
uv run python scripts/inference.py --interactive
```

## Example Translations

| English | Greedy | Beam (k=5) |
|---|---|---|
| Hello | hola. | hola. |
| How are you? | ¿cómo estás? | ¿cómo estás? |
| I love you | te quiero. | te amo. |
| The cat is black | el gato es negro. | el gato es negro. |
| Where is the hospital? | ¿dónde está el hospital? | ¿dónde está el hospital? |
| I want to eat | quiero quiero. | quiero comer. |
| I don't understand | no no lo entiendo. | no entiendo. |

Beam search eliminates the repetition artifacts visible in greedy decoding.

## Training

### Hyperparameters

| Parameter | Value |
|---|---|
| Embedding dimension | 256 |
| Hidden dimension | 512 |
| Encoder dimension | 1024 (bidirectional) |
| Layers | 2 |
| Dropout | 0.35 |
| Batch size | 128 |
| Learning rate | 3e-4 (AdamW) |
| Gradient clipping | 1.0 |
| Label smoothing | 0.1 |
| Teacher forcing | Linear decay 1.0 → 0.3 |
| Max sequence length | 35 tokens |
| Epochs | 40 |

### Training Curves

<table>
<tr>
<td><img src="img/epoch_train_loss.svg" width="400" alt="Train Loss"/></td>
<td><img src="img/epoch_val_loss.svg" width="400" alt="Val Loss"/></td>
</tr>
<tr>
<td align="center">Train Loss (epoch)</td>
<td align="center">Validation Loss (epoch)</td>
</tr>
</table>

<table>
<tr>
<td><img src="img/train_loss.svg" width="270" alt="Train Loss (step)"/></td>
<td><img src="img/train_grad_norm.svg" width="270" alt="Gradient Norm"/></td>
<td><img src="img/train_attention_entropy.svg" width="270" alt="Attention Entropy"/></td>
</tr>
<tr>
<td align="center">Train Loss (step)</td>
<td align="center">Gradient Norm</td>
<td align="center">Attention Entropy</td>
</tr>
</table>

The apparent uptick in train loss after epoch ~16 is caused by teacher forcing decay (the training task gets harder as the model relies more on its own predictions). The validation loss — always evaluated fully autoregressively — decreases monotonically.

### Attention Visualization

The model learns interpretable word alignments:

<table>
<tr>
<td><img src="img/context1.png" width="270"/></td>
<td><img src="img/context2.png" width="270"/></td>
<td><img src="img/context3.png" width="270"/></td>
</tr>
</table>

### Dataset

| Source | Pairs | Description |
|---|---|---|
| [Tatoeba](https://opus.nlpl.eu/Tatoeba.php) | ~222K | Short conversational sentences |
| [Europarl](https://opus.nlpl.eu/Europarl.php) | ~400K | Parliamentary proceedings (filtered ≤30 words) |
| **Total** | **~622K** | Mixed register |

## What's Built From Scratch

Every neural network component is implemented from first principles — no `torch.nn.LSTM` or pre-built seq2seq modules:

- **LSTMCell** — fused gates, Xavier init, forget bias = 1.0
- **BiLSTM Encoder** — forward + backward with learned projection
- **Dot-Product Attention** — score, mask, softmax, context
- **LSTM Decoder** — step-by-step with attention and teacher forcing
- **Beam Search** — k-best decoding with length normalization

## Files in This Repo

| File | Description |
|---|---|
| `model.pt` | Model checkpoint (weights + optimizer state) |
| `hparams.json` | Training hyperparameters |
| `spm_en.model` | English SentencePiece tokenizer |
| `spm_es.model` | Spanish SentencePiece tokenizer |
| `data/combined.en` | English training sentences |
| `data/combined.es` | Spanish training sentences |
| `config.py` | Model configuration dataclass |
| `train.py` | Training script |

## Limitations

- Best for short-to-medium sentences (under 30 words)
- English → Spanish only
- LSTM architecture is inherently sequential (slower inference than Transformers)
- Trained on conversational + parliamentary text — may struggle with specialized domains

## License

MIT

## Citation

```bibtex
@misc{lstm-en-es-translator,
  author = {Alex Gara},
  title = {LSTM English-to-Spanish Translator},
  year = {2025},
  url = {https://github.com/alexgarabt/lstm-translator}
}
```
