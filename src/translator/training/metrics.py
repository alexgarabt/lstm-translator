import torch
import torch.nn as nn
import matplotlib
# non-GUI backend
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.figure import Figure


def compute_gradient_norm(model: nn.Module) -> dict[str, float]:
    """
    Compute the L2 norm of gradients for each parameter group.
    Useful for diagnosing vanishing/exploding gradients.
    """
    norms = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            norms[name] = param.grad.norm(2).item()
    return norms


def compute_total_gradient_norm(model: nn.Module) -> float:
    """Global gradient norm (same metric used by gradient clipping)."""
    total_norm = 0.0
    for param in model.parameters():
        if param.grad is not None:
            total_norm += param.grad.norm(2).item() ** 2
    return total_norm ** 0.5


def compute_attention_entropy(attn_weights: torch.Tensor) -> float:
    """
    Mean entropy of attention weights.
    High entropy = diffuse attention (doesn't know where to look).
    Low entropy = focused attention (knows exactly where to look).

    attn_weights: (batch, src_len)
    """
    eps = 1e-8
    entropy = -(attn_weights * torch.log(attn_weights + eps)).sum(dim=1)  # (batch,)
    return entropy.mean().item()


def plot_attention(src_tokens: list[str], trg_tokens: list[str], attention_weights: list[torch.Tensor]) -> Figure:
    """
    Generate an attention heatmap for TensorBoard.

    src_tokens: ["<s>", "The", "black", "cat", "</s>"]
    trg_tokens: ["El", "gato", "negro"]  (generated tokens, without <s>)
    attention_weights: list of tensors, one per generated token
    """
    # Stack: list of (src_len,) -> (trg_len, src_len)
    attn_matrix = torch.stack(attention_weights).detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=(max(6, len(src_tokens) * 0.6), max(4, len(trg_tokens) * 0.5)))

    cax = ax.matshow(attn_matrix, cmap='viridis')
    fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)

    # Labels
    ax.set_xticklabels([''] + src_tokens, rotation=45, ha='left')
    ax.set_yticklabels([''] + trg_tokens)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    ax.set_xlabel('Source (EN)')
    ax.set_ylabel('Target (ES)')

    plt.tight_layout()
    return fig
