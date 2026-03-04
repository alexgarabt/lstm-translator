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

A sequence-to-sequence neural machine translation model built **entirely from scratch** — including the LSTM cells, encoder, decoder, and attention mechanism — as a deep learning educational project.

## Model Description

This is a BiLSTM encoder-decoder with Luong (dot-product) attention, trained on ~620K parallel sentence pairs from Tatoeba and Europarl. Every component is implemented from first principles in PyTorch, with no use of `torch.nn.LSTM` or pre-built seq2seq modules.

| Component | Details |
|---|---|
| **Architecture** | BiLSTM Encoder → Dot-Product Attention → LSTM Decoder |
| **Parameters** | ~31.9M |
| **Encoder** | 2-layer Bidirectional LSTM (custom), 512 hidden units per direction |
| **Decoder** | 2-layer LSTM (custom) with attention and teacher forcing |
| **Attention** | Luong dot-product with learned projection |
| **Tokenizer** | SentencePiece BPE, 16K vocab per language |
| **Training data** | Tatoeba (~222K pairs) + Europarl filtered (~400K pairs) |
| **Framework** | PyTorch |

## Architecture

```
Source (EN) → [Embedding] → [BiLSTM Encoder] → encoder_outputs
                                                      ↓
                                              [Dot-Product Attention]
                                                      ↓
Target (ES) → [Embedding] → [LSTM Decoder] → [Output Projection] → predicted tokens
```

**Key implementation details:**
- Custom `LSTMCell` with fused weight matrices for efficient GPU computation
- Forget gate bias initialized to 1.0 for stable gradient flow
- Bidirectional encoder with learned projection for decoder initialization
- Attention masking for proper handling of padded sequences
- Beam search decoding for higher quality translations

## Usage

### Installation

```bash
git clone https://github.com/alexgarabt/lstm-translator.git
cd lstm-translator
uv sync
```

### Quick Translation

```python
import torch
from huggingface_hub import hf_hub_download
from translator.data.tokenizer import Tokenizer
from translator.models.encoder import Encoder
from translator.models.decoder import Decoder
from translator.models.seq2seq import Seq2Seq

REPO_ID = "alexgara/lstm-en-es-translator"

# Download model and tokenizers
checkpoint_path = hf_hub_download(REPO_ID, "model.pt")
en_tok_path = hf_hub_download(REPO_ID, "spm_en.model")
es_tok_path = hf_hub_download(REPO_ID, "spm_es.model")

# Load tokenizers
src_tok = Tokenizer(en_tok_path)
trg_tok = Tokenizer(es_tok_path)

# Build and load model
encoder = Encoder(src_tok.vocab_size, 256, 512, 2, dropout=0.0)
decoder = Decoder(trg_tok.vocab_size, 256, 512, 1024, 2, dropout=0.0)
model = Seq2Seq(encoder, decoder,
                pad_token_id=src_tok.pad_id,
                bos_token_id=src_tok.bos_id,
                eos_token_id=src_tok.eos_id)

checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Translate
def translate(text, max_len=50):
    src_ids = src_tok.encode(text)
    src = torch.tensor([src_ids])
    mask = src != src_tok.pad_id

    with torch.no_grad():
        enc_out, (h, c) = model.encoder(src, torch.tensor([len(src_ids)]))
        h = [h[l] for l in range(2)]
        c = [c[l] for l in range(2)]
        context = torch.zeros(1, enc_out.shape[2])
        token = torch.tensor([trg_tok.bos_id])
        result = []

        for _ in range(max_len):
            logits, h, c, context, _ = model.decoder.forward_step(
                token, h, c, enc_out, context, mask
            )
            token = logits.argmax(dim=1)
            if token.item() == trg_tok.eos_id:
                break
            result.append(token.item())

    return trg_tok.decode(result)

print(translate("How are you?"))
# Output: ¿cómo estás?
```

## Example Translations

| English | Spanish (Greedy) | Spanish (Beam) |
|---|---|---|
| Hello | hola. | hola. |
| How are you? | ¿cómo estás? | ¿cómo estás? |
| I love you | te quiero. | te amo. |
| The cat is black | el gato es negro. | el gato es negro. |
| Where is the hospital? | ¿dónde está el hospital? | ¿dónde está el hospital? |
| I want to eat | quiero quiero. | quiero comer. |
| She is my sister | ella es mi hermana. | ella es mi hermana. |
| I don't understand | no no lo entiendo. | no entiendo. |

Beam search eliminates the repetition artifacts present in greedy decoding.

## Training

### Hyperparameters

| Parameter | Value |
|---|---|
| Embedding dimension | 256 |
| Hidden dimension | 512 |
| Encoder dimension | 1024 (2 × 512) |
| Layers | 2 |
| Dropout | 0.35 |
| Batch size | 128 |
| Learning rate | 3e-4 |
| Optimizer | Adam |
| Gradient clipping | 1.0 (max norm) |
| Label smoothing | 0.1 |
| Teacher forcing | Linear decay 1.0 → 0.3 |
| Max sequence length | 35 tokens |
| BPE vocabulary | 16,000 per language |

### Dataset

| Source | Pairs | Description |
|---|---|---|
| [Tatoeba](https://opus.nlpl.eu/Tatoeba.php) | ~222K | Short conversational sentences (median 6 words) |
| [Europarl](https://opus.nlpl.eu/Europarl.php) | ~400K (filtered) | European Parliament proceedings, filtered to ≤30 words |
| **Total** | **~622K** | Mixed conversational + formal register |

### Training from scratch

```bash
git clone https://github.com/alexgarabt/lstm-translator.git
cd lstm-translator
uv sync

# Train (downloads data, trains tokenizers, trains model)
uv run python scripts/train_v2.py

# Monitor with TensorBoard
uv run tensorboard --logdir training/runs_v2/
```

### Parameter Breakdown

| Component | Parameters | % of Total |
|---|---|---|
| Encoder embedding | 4.1M | 12.8% |
| Encoder BiLSTM (2 layers × 2 directions) | 7.3M | 23.0% |
| Encoder projections | 1.0M | 3.3% |
| Decoder embedding | 4.1M | 12.8% |
| Decoder LSTM (2 layers) | 5.8M | 18.1% |
| Attention + combination | 1.3M | 4.1% |
| Output projection | 8.2M | 25.8% |
| **Total** | **~31.9M** | **100%** |

## What Was Built From Scratch

This project implements every neural network component from the ground up:

- **LSTMCell**: Four gates (input, forget, candidate, output) with fused weight matrices, Xavier initialization, and forget gate bias = 1.0
- **LSTM**: Multi-layer wrapper that iterates the cell over timesteps
- **BiLSTM Encoder**: Forward + backward LSTMs with state concatenation and learned projection
- **Dot-Product Attention**: Score computation, padding mask with -∞, softmax normalization, weighted context
- **LSTM Decoder**: Token embedding + context concatenation, step-by-step generation with attention
- **Seq2Seq**: Orchestration with configurable teacher forcing schedule
- **Beam Search**: K-best sequence decoding with length normalization
- **SentencePiece BPE Tokenizer**: Wrapper with special token handling
- **Training Loop**: Gradient clipping, label smoothing, TensorBoard logging, checkpoint management

## Attention Visualization

The model learns interpretable word alignments. Below are attention heatmaps from validation examples showing the model correctly attending to source words when generating each target token:

- "He lives in a huge house" → "él vive en una casa enorme" (clean diagonal alignment)
- "Someone opened the door" → "alguien abrió la puerta" (correct cross-lingual mapping)
- "I have nothing particular to say" → "no tengo nada particular que decir" (learned Spanish double negation)

## Limitations

- Best suited for short-to-medium sentences (under 30 words)
- Trained primarily on conversational and parliamentary text — may struggle with specialized domains
- LSTM architecture is inherently sequential, making inference slower than Transformer-based models
- Single direction only (English → Spanish)

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

## Acknowledgments

- Training data from [OPUS](https://opus.nlpl.eu/) (Tatoeba and Europarl corpora)
- Tokenization with [SentencePiece](https://github.com/google/sentencepiece)
- Attention mechanism based on [Luong et al. (2015)](https://arxiv.org/abs/1508.04025)
