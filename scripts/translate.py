# scripts/translate.py

import torch
from pathlib import Path
from translator.config import Config
from translator.data.tokenizer import Tokenizer
from translator.models.encoder import Encoder
from translator.models.decoder import Decoder
from translator.models.seq2seq import Seq2Seq


def load_model(config: Config, src_tokenizer: Tokenizer, trg_tokenizer: Tokenizer, checkpoint_path: str):
    encoder = Encoder(
        vocab_size=src_tokenizer.vocab_size,
        embedded_dim=config.embed_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        dropout=0.0,  # no dropout at inference
    )
    decoder = Decoder(
        vocab_size=trg_tokenizer.vocab_size,
        embed_dim=config.embed_dim,
        hidden_dim=config.hidden_dim,
        encoder_dim=config.encoder_dim,
        num_layers=config.num_layers,
        dropout=0.0,
    )
    model = Seq2Seq(
        encoder, decoder,
        pad_token_id=src_tokenizer.pad_id,
        bos_token_id=src_tokenizer.bos_id,
        eos_token_id=src_tokenizer.eos_id,
    )

    checkpoint = torch.load(checkpoint_path, map_location=config.device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.to(config.device)

    return model


def translate(model: Seq2Seq, src_tokenizer: Tokenizer, trg_tokenizer: Tokenizer, text: str, device: str, max_len: int = 30) -> str:
    src_ids = src_tokenizer.encode(text)
    src = torch.tensor([src_ids], device=device)
    src_lengths = torch.tensor([len(src_ids)], device=device)
    mask = src != src_tokenizer.pad_id

    with torch.no_grad():
        enc_out, (h, c) = model.encoder(src, src_lengths)

        h = [h[l] for l in range(model.decoder.num_layers)]
        c = [c[l] for l in range(model.decoder.num_layers)]
        context = torch.zeros(1, enc_out.shape[2], device=device)

        token = torch.tensor([trg_tokenizer.bos_id], device=device)
        result = []

        for _ in range(max_len):
            logits, h, c, context, attn = model.decoder.forward_step(
                token, h, c, enc_out, context, mask
            )
            token = logits.argmax(dim=1)
            if token.item() == trg_tokenizer.eos_id:
                break
            result.append(token.item())

    return trg_tokenizer.decode(result)


def main():
    config = Config(
        data_dir=Path("training/data"),
        embed_dim=256,
        hidden_dim=512,
        num_layers=2,
        device="cuda",
    )

    # Load tokenizers
    src_tokenizer = Tokenizer(config.data_dir / "spm_en.model")
    trg_tokenizer = Tokenizer(config.data_dir / "spm_es.model")

    # Load model from best checkpoint
    checkpoint_path = "training/checkpoint_dir/model_epoch_33_loss_4.0347.pt"
    model = load_model(config, src_tokenizer, trg_tokenizer, checkpoint_path)

    # Test sentences
    tests = [
        "Hello",
        "How are you?",
        "I love you",
        "The cat is black",
        "Where is the hospital?",
        "I want to eat",
        "She is my sister",
        "It is very cold today",
        "I don't understand",
        "Thank you very much",
    ]

    for text in tests:
        result = translate(model, src_tokenizer, trg_tokenizer, text, config.device)
        print(f"EN: {text}")
        print(f"ES: {result}")
        print()


if __name__ == "__main__":
    main()
