"""
Continue training from a checkpoint with cosine annealing LR decay.

Usage:
    uv run python scripts/continue_training.py

"""

import torch
from pathlib import Path
from translator.config import Config
from translator.data.preprocessing import load_pairs, train_val_test_split
from translator.data.tokenizer import Tokenizer
from translator.data.dataset import TranslationDataset
from translator.models.encoder import Encoder
from translator.models.decoder import Decoder
from translator.models.seq2seq import Seq2Seq
from translator.training.trainer import Trainer


RESUME_CHECKPOINT = "training/checkpoint_v4/model_epoch_39_loss_X.XXXX.pt" 
EXTRA_EPOCHS = 20
LR_START = 3e-4
LR_END = 3e-5


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
        learning_rate=LR_START,
        max_epochs=EXTRA_EPOCHS,
        gradient_clip=1.0,
        teacher_forcing_start=0.3,   # fixed at v4's final value
        teacher_forcing_end=0.3,
        label_smoothing=0.1,
        log_every=100,
        eval_every_epoch=1,
        checkpoint_dir=Path("training/checkpoint_v5"),
        tensorboard_dir=Path("training/runs_v5"),
        device="cuda",
    )

    # ── Data ────────────────────────────────────────────────────────────
    pairs = load_pairs(
        config.data_dir / "combined.en",
        config.data_dir / "combined.es",
        max_length=config.max_length,
    )
    train_pairs, val_pairs, _ = train_val_test_split(
        pairs, val_size=config.val_size, test_size=config.test_size
    )

    src_tokenizer = Tokenizer(str(config.data_dir / "spm_en_v2.model"))
    trg_tokenizer = Tokenizer(str(config.data_dir / "spm_es_v2.model"))

    train_dataset = TranslationDataset(train_pairs, src_tokenizer, trg_tokenizer)
    val_dataset = TranslationDataset(val_pairs, src_tokenizer, trg_tokenizer)

    # ── Model ───────────────────────────────────────────────────────────
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

    # ── Scheduler (created before Trainer so it can be restored) ────────
    steps_per_epoch = len(train_dataset) // config.batch_size + 1
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        torch.optim.AdamW(model.parameters(), lr=LR_START),  # dummy, replaced by Trainer
        T_max=EXTRA_EPOCHS * steps_per_epoch,
        eta_min=LR_END,
    )

    # ── Train (resume handled internally) ───────────────────────────────
    trainer = Trainer(
        model, train_dataset, val_dataset,
        src_tokenizer, trg_tokenizer, config,
        resume_from=RESUME_CHECKPOINT,
        scheduler=scheduler,
    )

    # Reassign scheduler to use Trainer's actual optimizer
    trainer.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        trainer.optimizer,
        T_max=EXTRA_EPOCHS * len(trainer.train_loader),
        eta_min=LR_END,
    )

    trainer.fit()


if __name__ == "__main__":
    main()
