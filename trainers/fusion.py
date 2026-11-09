"""Contextual fusion in route order: 00, V0, 0T, VT."""

import math

import numpy as np
import torch
from scipy.special import softmax
from torch import nn


class ContextualFusion(nn.Module):
    def __init__(self, num_classes, alpha=16.0, beta=10.849894523620605,
                 ema_rate=0.01):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha, dtype=torch.float32))
        self.beta = nn.Parameter(torch.tensor(beta, dtype=torch.float32))
        self.register_buffer(
            "shared_ema",
            torch.full((num_classes,), 1.0 / num_classes, dtype=torch.float64),
        )
        self.register_buffer("batches_seen", torch.tensor(0, dtype=torch.long))
        self.ema_rate = ema_rate

    def forward(self, logits):
        # Only alpha and beta receive gradients from the fusion objective.
        logits = logits.detach().float()
        probabilities = logits.double().softmax(dim=-1)
        entropy = -(
            probabilities * probabilities.clamp_min(1e-300).log()
        ).sum(dim=-1) / math.log(logits.shape[-1])
        routes = logits.argmax(dim=-1)
        disagreement = (routes != routes[:, :1]).any(dim=1)
        table = -(logits.shape[-1] * self.shared_ema.clamp_min(1e-12)).log()
        context = table[routes].float() * disagreement[:, None]
        weights = (-self.alpha * entropy.float() + self.beta * context).softmax(dim=1)
        output = (weights.unsqueeze(-1) * logits).sum(dim=1)
        return output, weights, probabilities.mean(dim=(0, 1))

    @torch.no_grad()
    def after_step(self, batch_mean):
        self.alpha.clamp_(min=0)
        self.beta.clamp_(min=0)
        self.shared_ema.mul_(1 - self.ema_rate).add_(
            batch_mean, alpha=self.ema_rate
        )
        self.batches_seen.add_(1)


def estimate_test_statistics(logits, batch_size=256):
    """Estimate a shared frequency from the complete unlabeled collection.

    No labels or training EMA enter this function. Logits may be a read-only
    memory map of shape (images, 4, classes).
    """
    n, routes, classes = logits.shape
    if n == 0 or routes != 4 or classes < 2:
        raise ValueError("Expected nonempty logits with shape (N, 4, C), C >= 2")
    entropy = np.empty((n, 4), dtype=np.float32)
    top = np.empty((n, 4), dtype=np.int64)
    probability_sum = np.zeros((4, classes), dtype=np.float64)
    for start in range(0, n, batch_size):
        z = np.asarray(logits[start:start + batch_size], dtype=np.float64)
        if not np.isfinite(z).all():
            raise ValueError("Non-finite branch logits")
        p = softmax(z, axis=-1)
        entropy[start:start + len(z)] = -(
            p * np.log(np.maximum(p, 1e-300))
        ).sum(axis=-1) / math.log(classes)
        top[start:start + len(z)] = z.argmax(axis=-1)
        probability_sum += p.sum(axis=0)
    mean_probability = probability_sum.mean(axis=0) / n
    disagreement = (top != top[:, :1]).any(axis=1)
    context = -np.log(classes * np.maximum(mean_probability[top], 1e-12))
    context = context.astype(np.float32)
    context *= disagreement[:, None]
    return entropy, context, mean_probability


def fuse_test_logits(logits, entropy, context, alpha, beta, device="cpu"):
    """Apply the learned scalars and return FP32 logits and route weights."""
    z = torch.as_tensor(np.asarray(logits).copy(), dtype=torch.float32, device=device)
    h = torch.as_tensor(np.asarray(entropy).copy(), dtype=torch.float32, device=device)
    q = torch.as_tensor(np.asarray(context).copy(), dtype=torch.float32, device=device)
    a = torch.as_tensor(alpha, dtype=torch.float32, device=device)
    b = torch.as_tensor(beta, dtype=torch.float32, device=device)
    weights = (-a * h + b * q).softmax(dim=1)
    return (weights.unsqueeze(-1) * z).sum(dim=1), weights

