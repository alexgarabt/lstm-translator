from translator.config import Config
from translator.data.download import download_tatoeba
from translator.data.preprocessing import load_pairs, train_val_test_split, save_texts
from translator.data.tokenizer import Tokenizer
from translator.data.dataset import TranslationDataset
from translator.models.encoder import Encoder
from translator.models.decoder import Decoder
from translator.models.seq2seq import Seq2Seq
from translator.training.trainer import Trainer


def main():
    config = Config()

    # 1. Download data
    en_path, es_path = download_tatoeba(config.data_dir)

    # 2. Load and split
    pairs = load_pairs(en_path, es_path, max_length=config.max_length)
    train_pairs, val_pairs, test_pairs = train_val_test_split(
        pairs, val_size=config.val_size, test_size=config.test_size
    )

    # 3. Save separate text files for tokenizer training
    train_prefix = config.data_dir / "train"
    save_texts(train_pairs, train_prefix)

    # 4. Train tokenizers (one for EN, one for ES)
    en_model = str(config.data_dir / "spm_en")
    es_model = str(config.data_dir / "spm_es")

    Tokenizer.train_model(f"{train_prefix}.en", en_model, vocab_size=config.vocab_size)
    Tokenizer.train_model(f"{train_prefix}.es", es_model, vocab_size=config.vocab_size)

    src_tokenizer = Tokenizer(f"{en_model}.model")
    trg_tokenizer = Tokenizer(f"{es_model}.model")

    # 5. Create datasets
    train_dataset = TranslationDataset(train_pairs, src_tokenizer, trg_tokenizer)
    val_dataset = TranslationDataset(val_pairs, src_tokenizer, trg_tokenizer)

    # 6. Build model
    encoder = Encoder(
        vocab_size=src_tokenizer.vocab_size,
        embed_dim=config.embed_dim,
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
    model = Seq2Seq(encoder, decoder, pad_token_id=src_tokenizer.pad_id)

    # 7. Train
    trainer = Trainer(model, train_dataset, val_dataset, src_tokenizer, trg_tokenizer, config)
    trainer.fit()


if __name__ == "__main__":
    main()
