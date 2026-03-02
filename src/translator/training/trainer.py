import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from functools import partial
from tqdm import tqdm
import matplotlib.pyplot as plt

from ..config import Config
from ..models.seq2seq import Seq2Seq
from ..data.dataset import TranslationDataset, collate_fn
from ..data.tokenizer import Tokenizer
from .metrics import (
    compute_total_gradient_norm,
    compute_attention_entropy,
    plot_attention,
)


class Trainer:
    def __init__(
        self,
        model: Seq2Seq,
        train_dataset: TranslationDataset,
        val_dataset: TranslationDataset,
        src_tokenizer: Tokenizer,
        trg_tokenizer: Tokenizer,
        config: Config,
    ):
        self.model = model.to(config.device)
        self.config = config
        self.src_tokenizer = src_tokenizer
        self.trg_tokenizer = trg_tokenizer

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=1e-5,
        )

        # Loss: CrossEntropy ignoring padding tokens
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=trg_tokenizer.pad_id,
            label_smoothing=config.label_smoothing,
        )

        # DataLoaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=partial(collate_fn, pad_id=src_tokenizer.pad_id),
            num_workers=2,
            pin_memory=True,  # speeds up CPU -> GPU transfer
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            collate_fn=partial(collate_fn, pad_id=src_tokenizer.pad_id),
        )

        # TensorBoard
        self.writer = SummaryWriter(config.tensorboard_dir)
        self.global_step = 0

        # Fixed validation examples for attention visualization across epochs
        self.viz_examples = self._get_viz_examples(val_dataset, n=5)

    def _get_viz_examples(self, dataset: TranslationDataset, n: int) -> list[dict]:
        """Select N fixed examples to visualize attention evolution."""
        examples = []
        for i in range(min(n, len(dataset))):
            examples.append(dataset[i])
        return examples

    def _get_teacher_forcing_ratio(self, epoch: int) -> float:
        """Linear decay of teacher forcing ratio across epochs."""
        cfg = self.config
        progress = epoch / max(cfg.max_epochs - 1, 1)
        return cfg.teacher_forcing_start + progress * (cfg.teacher_forcing_end - cfg.teacher_forcing_start)

    def train_epoch(self, epoch: int) -> float:
        """Train one full epoch. Returns average loss."""
        self.model.train()
        total_loss = 0
        tf_ratio = self._get_teacher_forcing_ratio(epoch)

        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}", leave=False)

        for batch in progress_bar:
            src = batch['src'].to(self.config.device)
            trg = batch['trg'].to(self.config.device)
            src_lengths = batch['src_lengths'].to(self.config.device)

            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                # Forward pass
                logits, attention = self.model(src, src_lengths, trg, tf_ratio)
                # Loss: compare predictions with target (without <START>)
                # logits: (batch, trg_len-1, vocab_size)
                # trg[:, 1:]: (batch, trg_len-1) — correct tokens without <START>
                loss = self.criterion(
                    logits.reshape(-1, logits.shape[-1]),
                    trg[:, 1:].reshape(-1),
                )

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping — returns norm before clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.gradient_clip,
            ).item()

            # Parameter update
            self.optimizer.step()

            # Logging
            total_loss += loss.item()
            self.global_step += 1

            progress_bar.set_postfix(loss=f"{loss.item():.4f}", grad=f"{grad_norm:.2f}")

            if self.global_step % self.config.log_every == 0:
                self._log_step(loss.item(), grad_norm, attention, tf_ratio)

        avg_loss = total_loss / len(self.train_loader)
        return avg_loss

    @torch.no_grad()
    def evaluate(self) -> float:
        """Evaluate on validation set. Returns average loss."""
        self.model.eval()
        total_loss = 0

        for batch in self.val_loader:
            src = batch['src'].to(self.config.device)
            trg = batch['trg'].to(self.config.device)
            src_lengths = batch['src_lengths'].to(self.config.device)

            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                # Evaluation: teacher forcing = 0 (fully autoregressive)
                logits, _ = self.model(src, src_lengths, trg, teacher_forcing_ratio=0.0)

                loss = self.criterion(
                    logits.reshape(-1, logits.shape[-1]),
                    trg[:, 1:].reshape(-1),
                )
            total_loss += loss.item()

        return total_loss / len(self.val_loader)

    def _log_step(self, loss: float, grad_norm: float, attention: list, tf_ratio: float):
        """Log metrics to TensorBoard every N steps."""
        step = self.global_step

        # Basic scalars
        self.writer.add_scalar('train/loss', loss, step)
        self.writer.add_scalar('train/grad_norm', grad_norm, step)
        self.writer.add_scalar('train/teacher_forcing', tf_ratio, step)

        # Attention entropy from last decoder step
        if attention:
            entropy = compute_attention_entropy(attention[-1])
            self.writer.add_scalar('train/attention_entropy', entropy, step)

    def _log_epoch(self, epoch: int, train_loss: float, val_loss: float):
        """Log per-epoch metrics to TensorBoard."""
        self.writer.add_scalar('epoch/train_loss', train_loss, epoch)
        self.writer.add_scalar('epoch/val_loss', val_loss, epoch)

        # Weight and gradient histograms
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                self.writer.add_histogram(f'weights/{name}', param, epoch)
                self.writer.add_histogram(f'gradients/{name}', param.grad, epoch)

        # Attention heatmaps for fixed validation examples
        self._log_attention_examples(epoch)

    @torch.no_grad()
    def _log_attention_examples(self, epoch: int):
        """Generate attention heatmaps for fixed validation examples."""
        self.model.eval()

        for i, example in enumerate(self.viz_examples):
            src = example['src'].unsqueeze(0).to(self.config.device)    # (1, src_len)
            trg = example['trg'].unsqueeze(0).to(self.config.device)    # (1, trg_len)
            src_lengths = torch.tensor([len(example['src'])]).to(self.config.device)

            logits, attention = self.model(src, src_lengths, trg, teacher_forcing_ratio=0.0)

            # Source tokens for heatmap labels
            src_tokens = self.src_tokenizer.decode_with_tokens(example['src'].tolist())

            # Predicted tokens (not reference)
            predicted_ids = logits.argmax(dim=-1).squeeze(0).tolist()
            trg_tokens = self.trg_tokenizer.decode_with_tokens(predicted_ids)

            # Attention: list of (1, src_len) -> list of (src_len,)
            attn_for_plot = [a.squeeze(0) for a in attention]

            fig = plot_attention(src_tokens, trg_tokens, attn_for_plot)
            self.writer.add_figure(f'attention/example_{i}', fig, epoch)
            plt.close(fig)

        self.model.train()

    def save_checkpoint(self, epoch: int, val_loss: float):
        """Save model checkpoint."""
        self.config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = self.config.checkpoint_dir / f"model_epoch_{epoch}_loss_{val_loss:.4f}.pt"
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'config': self.config,
        }, path)
        print(f"Checkpoint saved: {path}")

    def fit(self):
        """Main training loop."""
        best_val_loss = float('inf')

        print(f"Training on {self.config.device}")
        print(f"Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"TensorBoard: tensorboard --logdir {self.config.tensorboard_dir}")
        print()

        for epoch in range(self.config.max_epochs):
            train_loss = self.train_epoch(epoch)
            val_loss = self.evaluate()

            self._log_epoch(epoch, train_loss, val_loss)

            tf_ratio = self._get_teacher_forcing_ratio(epoch)
            print(
                f"Epoch {epoch+1}/{self.config.max_epochs} | "
                f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
                f"TF: {tf_ratio:.2f}"
            )

            # Save if best so far
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint(epoch, val_loss)

        self.writer.close()
        print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
