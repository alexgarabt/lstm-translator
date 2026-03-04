"""
Continue training from a checkpoint with cosine annealing LR decay.

Usage:
    uv run python scripts/continue_training.py

Resumes from the best checkpoint in checkpoint_v2/, adds a cosine LR scheduler,
and trains for additional epochs. Logs to a new TensorBoard directory (runs_v3/)
so you can compare with the original run.
"""

import torch
from pathlib import Path
from translator.config import Config
from translator.data.preprocessing import load_pairs, train_val_test_split, save_texts
from translator.data.tokenizer import Tokenizer
from translator.data.dataset import TranslationDataset
from translator.models.encoder import Encoder
from translator.models.decoder import Decoder
from translator.models.seq2seq import Seq2Seq
from translator.training.trainer import Trainer


def find_best_checkpoint(checkpoint_dir: Path) -> Path:
    """Find the checkpoint with the lowest validation loss."""
    checkpoints = list(checkpoint_dir.glob("*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")

    best = None
    best_loss = float("inf")
    for ckpt in checkpoints:
        checkpoint = torch.load(ckpt, map_location="cpu", weights_only=False)
        val_loss = checkpoint.get("val_loss", float("inf"))
        if val_loss < best_loss:
            best_loss = val_loss
            best = ckpt

    print(f"Best checkpoint: {best} (val_loss={best_loss:.4f})")
    return best


def main():
    # ── Config ──────────────────────────────────────────────────────────
    # Same as v2 but with more epochs and new output dirs
    EXTRA_EPOCHS = 20          # how many MORE epochs to train
    LR_START = 3e-4            # initial lr (same as v2)
    LR_END = 3e-5              # final lr after cosine decay (10x reduction)
    CHECKPOINT_DIR = Path("training/checkpoint_v2")  # where v2 checkpoints are

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
        max_epochs=EXTRA_EPOCHS,       # only the NEW epochs
        gradient_clip=1.0,
        teacher_forcing_start=1.0,     # will be overridden below
        teacher_forcing_end=0.3,
        label_smoothing=0.1,
        log_every=100,
        eval_every_epoch=1,
        checkpoint_dir=Path("training/checkpoint_v3"),
        tensorboard_dir=Path("training/runs_v3"),
        device="cuda",
    )

    # ── Load data (same as v2) ──────────────────────────────────────────
    en_path = config.data_dir / "combined.en"
    es_path = config.data_dir / "combined.es"

    pairs = load_pairs(en_path, es_path, max_length=config.max_length)
    train_pairs, val_pairs, test_pairs = train_val_test_split(
        pairs, val_size=config.val_size, test_size=config.test_size
    )

    # ── Load existing tokenizers (DO NOT retrain) ───────────────────────
    src_tokenizer = Tokenizer(str(config.data_dir / "spm_en_v2.model"))
    trg_tokenizer = Tokenizer(str(config.data_dir / "spm_es_v2.model"))

    # ── Create datasets ─────────────────────────────────────────────────
    train_dataset = TranslationDataset(train_pairs, src_tokenizer, trg_tokenizer)
    val_dataset = TranslationDataset(val_pairs, src_tokenizer, trg_tokenizer)

    # ── Build model ─────────────────────────────────────────────────────
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

    # ── Load checkpoint ─────────────────────────────────────────────────
    best_ckpt = find_best_checkpoint(CHECKPOINT_DIR)
    checkpoint = torch.load(best_ckpt, map_location=config.device, weights_only=False)

    model.load_state_dict(checkpoint["model_state_dict"])
    resumed_epoch = checkpoint["epoch"]
    print(f"Resumed from epoch {resumed_epoch + 1}, val_loss={checkpoint['val_loss']:.4f}")

    # ── Create trainer ──────────────────────────────────────────────────
    # Override teacher forcing: continue from where v2 left off
    # v2 had 40 epochs, linear from 1.0 to 0.3
    # At epoch 39: tf = 1.0 + (39/39) * (0.3 - 1.0) = 0.3
    # So we start at 0.3 and keep it at 0.3 (model should be autoregressive)
    config.teacher_forcing_start = 0.3
    config.teacher_forcing_end = 0.3

    trainer = Trainer(
        model, train_dataset, val_dataset,
        src_tokenizer, trg_tokenizer, config
    )

    # ── Restore optimizer state ─────────────────────────────────────────
    trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # ── Adjust global step for TensorBoard continuity ───────────────────
    # v2 had ~40 epochs × ~4813 steps/epoch ≈ 192K steps
    steps_per_epoch = len(trainer.train_loader)
    trainer.global_step = (resumed_epoch + 1) * steps_per_epoch
    print(f"Resuming from global_step={trainer.global_step}")

    # ── Add cosine annealing LR scheduler ───────────────────────────────
    # Decays from LR_START to LR_END over EXTRA_EPOCHS
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        trainer.optimizer,
        T_max=EXTRA_EPOCHS * steps_per_epoch,  # total steps for cosine cycle
        eta_min=LR_END,
    )

    # ── Monkey-patch trainer to use scheduler ───────────────────────────
    # Store original train_epoch to wrap it
    _original_train_epoch = trainer.train_epoch

    def train_epoch_with_scheduler(epoch: int) -> float:
        """Wraps train_epoch to step the LR scheduler after each batch."""
        import torch.nn.utils
        from functools import partial
        from tqdm import tqdm

        trainer.model.train()
        total_loss = 0
        tf_ratio = trainer._get_teacher_forcing_ratio(epoch)

        progress_bar = tqdm(trainer.train_loader, desc=f"Epoch {epoch+1}", leave=False)

        for batch in progress_bar:
            src = batch["src"].to(trainer.config.device)
            trg = batch["trg"].to(trainer.config.device)
            src_lengths = batch["src_lengths"].to(trainer.config.device)

            # Forward
            logits, attention = trainer.model(src, src_lengths, trg, tf_ratio)
            loss = trainer.criterion(
                logits.reshape(-1, logits.shape[-1]),
                trg[:, 1:].reshape(-1),
            )

            # Backward
            trainer.optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                trainer.model.parameters(),
                trainer.config.gradient_clip,
            )
            trainer.optimizer.step()

            # Step scheduler every batch (cosine annealing works per-step)
            scheduler.step()

            # Logging
            total_loss += loss.item()
            trainer.global_step += 1

            current_lr = scheduler.get_last_lr()[0]
            progress_bar.set_postfix(
                loss=f"{loss.item():.4f}",
                grad=f"{grad_norm:.2f}",
                lr=f"{current_lr:.2e}",
            )

            if trainer.global_step % trainer.config.log_every == 0:
                trainer._log_step(loss.item(), grad_norm, attention, tf_ratio)
                # Also log the learning rate
                trainer.writer.add_scalar(
                    "train/learning_rate", current_lr, trainer.global_step
                )

        avg_loss = total_loss / len(trainer.train_loader)
        return avg_loss

    trainer.train_epoch = train_epoch_with_scheduler

    # ── Also patch _log_epoch to log the resumed epoch number ───────────
    _original_log_epoch = trainer._log_epoch

    def log_epoch_with_offset(epoch: int, train_loss: float, val_loss: float):
        """Log with the real epoch number (offset by v2 epochs)."""
        real_epoch = resumed_epoch + 1 + epoch
        # Log under the real epoch number for comparison
        trainer.writer.add_scalar("epoch/train_loss", train_loss, real_epoch)
        trainer.writer.add_scalar("epoch/val_loss", val_loss, real_epoch)

        # Weight and gradient histograms
        for name, param in trainer.model.named_parameters():
            if param.grad is not None:
                trainer.writer.add_histogram(f"weights/{name}", param, real_epoch)
                trainer.writer.add_histogram(f"gradients/{name}", param.grad, real_epoch)

        # Attention heatmaps
        trainer._log_attention_examples(real_epoch)

    trainer._log_epoch = log_epoch_with_offset

    # ── Also patch save_checkpoint to include real epoch ─────────────────
    _original_save = trainer.save_checkpoint

    def save_checkpoint_with_offset(epoch: int, val_loss: float):
        """Save with the real epoch number."""
        real_epoch = resumed_epoch + 1 + epoch
        config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = config.checkpoint_dir / f"model_epoch_{real_epoch}_loss_{val_loss:.4f}.pt"
        torch.save(
            {
                "epoch": real_epoch,
                "model_state_dict": trainer.model.state_dict(),
                "optimizer_state_dict": trainer.optimizer.state_dict(),
                "val_loss": val_loss,
                "config": config,
            },
            path,
        )
        print(f"Checkpoint saved: {path}")

    trainer.save_checkpoint = save_checkpoint_with_offset

    # ── Also patch the print in fit() ───────────────────────────────────
    _original_fit = trainer.fit

    def fit_with_info():
        """Main training loop with resume info."""
        best_val_loss = float("inf")

        print(f"\n{'='*60}")
        print(f"CONTINUING TRAINING")
        print(f"{'='*60}")
        print(f"Resumed from epoch: {resumed_epoch + 1}")
        print(f"Extra epochs: {EXTRA_EPOCHS}")
        print(f"Total epochs after this: {resumed_epoch + 1 + EXTRA_EPOCHS}")
        print(f"LR schedule: {LR_START:.1e} → {LR_END:.1e} (cosine)")
        print(f"Teacher forcing: fixed at {config.teacher_forcing_start}")
        print(f"Device: {config.device}")
        print(f"Parameters: {sum(p.numel() for p in trainer.model.parameters()):,}")
        print(f"TensorBoard: tensorboard --logdir training/runs_v3/")
        print(f"{'='*60}\n")

        for epoch in range(config.max_epochs):
            train_loss = trainer.train_epoch(epoch)
            val_loss = trainer.evaluate()

            trainer._log_epoch(epoch, train_loss, val_loss)

            real_epoch = resumed_epoch + 1 + epoch
            current_lr = scheduler.get_last_lr()[0]
            print(
                f"Epoch {real_epoch + 1} | "
                f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
                f"LR: {current_lr:.2e} | "
                f"TF: {config.teacher_forcing_start:.2f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                trainer.save_checkpoint(epoch, val_loss)

                # Also save as best_model.pt for easy access
                best_path = config.checkpoint_dir / "best_model.pt"
                torch.save(
                    {
                        "epoch": real_epoch,
                        "model_state_dict": trainer.model.state_dict(),
                        "optimizer_state_dict": trainer.optimizer.state_dict(),
                        "val_loss": val_loss,
                        "config": config,
                    },
                    best_path,
                )

        trainer.writer.close()
        print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")

    fit_with_info()


if __name__ == "__main__":
    main()
