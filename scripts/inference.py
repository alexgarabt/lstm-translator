"""
Translate English → Spanish using the pre-trained model from HuggingFace Hub.

Usage:
    uv run python scripts/inference.py
    uv run python scripts/inference.py --text "How are you?"
    uv run python scripts/inference.py --beam-width 5
    uv run python scripts/inference.py --interactive
"""

import argparse
import torch
from pathlib import Path

from translator.config import Config
from translator.data.tokenizer import Tokenizer
from translator.models.encoder import Encoder
from translator.models.decoder import Decoder
from translator.models.seq2seq import Seq2Seq

from hub import download_checkpoint, download_tokenizers, download_hparams


def config_from_hparams(hparams: dict, device: str) -> Config:
    """Build a Config from the downloaded hparams.json."""
    return Config(
        data_dir=Path("."),            # not used for inference
        embed_dim=hparams["embed_dim"],
        hidden_dim=hparams["hidden_dim"],
        num_layers=hparams["num_layers"],
        device=device,
    )


def load_model(
    config: Config,
    src_tokenizer: Tokenizer,
    trg_tokenizer: Tokenizer,
    checkpoint_path: Path,
) -> Seq2Seq:
    """Build model from config and load checkpoint weights."""
    encoder = Encoder(
        vocab_size=src_tokenizer.vocab_size,
        embedded_dim=config.embed_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        dropout=0.0,
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
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    model.to(config.device)
    return model


def translate_greedy(
    model: Seq2Seq,
    src_tokenizer: Tokenizer,
    trg_tokenizer: Tokenizer,
    text: str,
    device: str,
    max_len: int = 50,
) -> str:
    """Greedy decoding — pick the most probable token at each step."""
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
            logits, h, c, context, _ = model.decoder.forward_step(
                token, h, c, enc_out, context, mask
            )
            token = logits.argmax(dim=1)
            if token.item() == trg_tokenizer.eos_id:
                break
            result.append(token.item())

    return trg_tokenizer.decode(result)


def translate_beam(
    model: Seq2Seq,
    src_tokenizer: Tokenizer,
    trg_tokenizer: Tokenizer,
    text: str,
    device: str,
    beam_width: int = 5,
) -> str:
    """Beam search decoding — keep B best hypotheses at each step."""
    src_ids = src_tokenizer.encode(text)
    src = torch.tensor([src_ids], device=device)
    src_lengths = torch.tensor([len(src_ids)], device=device)

    tokens = model.beam_search(
        src, src_lengths,
        bos_id=trg_tokenizer.bos_id,
        eos_id=trg_tokenizer.eos_id,
        beam_width=beam_width,
    )
    return trg_tokenizer.decode(tokens)


def main():
    parser = argparse.ArgumentParser(description="Translate EN → ES")
    parser.add_argument("--text", type=str, default=None, help="Text to translate")
    parser.add_argument("--beam-width", type=int, default=5, help="Beam width (0 = greedy only)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--interactive", action="store_true", help="Interactive translation loop")
    parser.add_argument("--cache-dir", type=str, default="hub_cache", help="HF cache directory")
    args = parser.parse_args()

    print("Downloading model from HuggingFace Hub...")
    hparams = download_hparams(args.cache_dir)
    en_tok_path, es_tok_path = download_tokenizers(args.cache_dir)
    checkpoint_path = download_checkpoint(args.cache_dir)

    config = config_from_hparams(hparams, args.device)
    src_tokenizer = Tokenizer(str(en_tok_path))
    trg_tokenizer = Tokenizer(str(es_tok_path))
    model = load_model(config, src_tokenizer, trg_tokenizer, checkpoint_path)
    print(f"Model loaded ({hparams.get('total_parameters', '?')} params) on {args.device}\n")

    def translate(text: str) -> None:
        greedy = translate_greedy(model, src_tokenizer, trg_tokenizer, text, args.device)
        print(f"  EN:     {text}")
        print(f"  GREEDY: {greedy}")
        if args.beam_width > 0:
            beam = translate_beam(
                model, src_tokenizer, trg_tokenizer, text, args.device, args.beam_width
            )
            print(f"  BEAM:   {beam}")
        print()

    if args.text:
        translate(args.text)
        return

    if args.interactive:
        print("Interactive mode (Ctrl+C to exit)\n")
        try:
            while True:
                text = input("EN> ").strip()
                if text:
                    translate(text)
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            return

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
        "If i showed you my house, my neighborhood back then, would you understand where I am from?",
        "In business today, too many executives spend money they haven't earned, to buy things they don't need, to impress people they don't even like."

    ]
    for text in tests:
        translate(text)


if __name__ == "__main__":
    main()
