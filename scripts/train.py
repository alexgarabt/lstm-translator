from pathlib import Path
from translator.config import Config
from translator.data.preprocessing import load_pairs, train_val_test_split, save_texts
from translator.data.tokenizer import Tokenizer
from translator.data.dataset import TranslationDataset
from translator.models.encoder import Encoder
from translator.models.decoder import Decoder
from translator.models.seq2seq import Seq2Seq
from translator.training.trainer import Trainer


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
        batch_size=96,
        learning_rate=3e-4,
        max_epochs=20,
        gradient_clip=1.0,
        teacher_forcing_start=1.0,
        teacher_forcing_end=0.3,
        label_smoothing=0.1,
        log_every=100,
        eval_every_epoch=1,
        checkpoint_dir=Path("training/checkpoint_v2"),
        tensorboard_dir=Path("training/runs_v2"),
        device="cuda",
    )

    # Load combined data directly
    en_path = config.data_dir / "combined.en"
    es_path = config.data_dir / "combined.es"

    pairs = load_pairs(en_path, es_path, max_length=config.max_length)
    train_pairs, val_pairs, test_pairs = train_val_test_split(
        pairs, val_size=config.val_size, test_size=config.test_size
    )

    # Save text files for tokenizer training
    train_prefix = config.data_dir / "train_v2"
    save_texts(train_pairs, train_prefix)

    # Train tokenizers on combined corpus
    en_model = str(config.data_dir / "spm_en_v2")
    es_model = str(config.data_dir / "spm_es_v2")

    Tokenizer.train_model(f"{train_prefix}.en", en_model, vocab_size=config.vocab_size)
    Tokenizer.train_model(f"{train_prefix}.es", es_model, vocab_size=config.vocab_size)

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

    # Train
    trainer = Trainer(model, train_dataset, val_dataset, src_tokenizer, trg_tokenizer, config)
    trainer.fit()


if __name__ == "__main__":
    main()
