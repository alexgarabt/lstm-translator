import json
import dataclasses
from pathlib import Path
from translator.config import Config
from translator.data.preprocessing import load_pairs, train_val_test_split, save_texts
from translator.data.tokenizer import Tokenizer
from translator.data.dataset import TranslationDataset
from translator.models.encoder import Encoder
from translator.models.decoder import Decoder
from translator.models.seq2seq import Seq2Seq
from translator.training.trainer import Trainer


def save_config(config: Config, model_params: int, dataset_size: int):
    """Save hyperparameters + run metadata as JSON in the checkpoint dir."""
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = config.checkpoint_dir / "hparams.json"

    # Convert Config dataclass to dict, handling Path objects
    hparams = {}
    for field in dataclasses.fields(config):
        value = getattr(config, field.name)
        hparams[field.name] = str(value) if isinstance(value, Path) else value

    # Add derived / extra info
    hparams["encoder_dim"] = config.encoder_dim
    hparams["total_parameters"] = model_params
    hparams["training_pairs"] = dataset_size

    with open(path, "w") as f:
        json.dump(hparams, f, indent=2)

    print(f"Hyperparameters saved: {path}")


def main():
    config = Config(
        data_dir=Path("training/data"),
        max_length=35,
        vocab_size=16000,
        val_size=3000,
        test_size=3000,
        embed_dim=256,
        hidden_dim=512,
        num_layers=2,
        dropout=0.35,
        batch_size=128,
        learning_rate=3e-4,
        max_epochs=40,
        gradient_clip=1.0,
        teacher_forcing_start=1.0,
        teacher_forcing_end=0.3,
        label_smoothing=0.1,
        log_every=100,
        eval_every_epoch=1,
        checkpoint_dir=Path("training/checkpoint_v4"),
        tensorboard_dir=Path("training/runs_v4"),
        device="cuda",
    )

    # Load combined data directly
    en_path = config.data_dir / "combined.en"
    es_path = config.data_dir / "combined.es"

    pairs = load_pairs(en_path, es_path, max_length=config.max_length)
    train_pairs, val_pairs, test_pairs = train_val_test_split(
        pairs, val_size=config.val_size, test_size=config.test_size
    )

    # Load existing v2 tokenizers (same data, same vocab_size — no need to retrain)
    en_model = str(config.data_dir / "spm_en_v2")
    es_model = str(config.data_dir / "spm_es_v2")

    src_tokenizer = Tokenizer(f"{en_model}.model")
    trg_tokenizer = Tokenizer(f"{es_model}.model")

    # Create datasets
    train_dataset = TranslationDataset(train_pairs, src_tokenizer, trg_tokenizer)
    val_dataset = TranslationDataset(val_pairs, src_tokenizer, trg_tokenizer)

    # Build model
    encoder = Encoder(
        vocab_size=src_tokenizer.vocab_size,
        embedded_dim=config.embed_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        dropout=config.dropout,
    )
    decoder = Decoder(
        vocab_size=trg_tokenizer.vocab_size,
        embed_dim=config.embed_dim,
        hidden_dim=config.hidden_dim,
        encoder_dim=config.encoder_dim,
        num_layers=config.num_layers,
        dropout=config.dropout,
    )
    model = Seq2Seq(
        encoder, decoder,
        pad_token_id=src_tokenizer.pad_id,
        bos_token_id=src_tokenizer.bos_id,
        eos_token_id=src_tokenizer.eos_id,
    )

    # Save hyperparameters
    model_params = sum(p.numel() for p in model.parameters())
    save_config(config, model_params, len(train_pairs))

    # Train
    trainer = Trainer(model, train_dataset, val_dataset, src_tokenizer, trg_tokenizer, config)
    trainer.fit()


if __name__ == "__main__":
    main()
