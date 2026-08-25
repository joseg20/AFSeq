#!/usr/bin/env python3
"""The four loss functions of the WSTS+ grid search, plus our own (bce_dice).

Ported verbatim from with_wstsplus/losses_wsts.py on 2026-08-25; train.py
imports it as `losses`. seq_loss is the entry point.

Lahrichi et al. search over BCE, Focal, Dice and Jaccard. Their Focal uses
alpha = inverse frequency of positive pixels and gamma = 2 (their default).

"Inverse frequency" is computed PER BATCH here, from the actual target, rather
than as a fixed constant: the positive rate swings enormously between fire days
(a fire covering 3% of the patch vs one covering 0.05%), so a single global
constant would over-weight the quiet days and under-weight the active ones.

All of them take logits, so the model's head stays unchanged.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _pos_weight_from(y: torch.Tensor, cap: float = 1000.0) -> torch.Tensor:
    """Inverse frequency of the positive class, capped to stay finite.

    A patch with no fire at all would give an infinite weight; the cap keeps the
    gradient usable in that case.
    """
    pos = y.sum()
    total = y.numel()
    neg = total - pos
    if pos < 1:
        return torch.tensor(cap, device=y.device)
    return torch.clamp(neg / pos, max=cap)


def bce(logits: torch.Tensor, y: torch.Tensor, pos_weight: float | None = None):
    pw = (torch.tensor(pos_weight, device=y.device) if pos_weight
          else _pos_weight_from(y))
    return F.binary_cross_entropy_with_logits(logits, y, pos_weight=pw)


def focal(logits: torch.Tensor, y: torch.Tensor, gamma: float = 2.0):
    """Focal loss with alpha = inverse positive frequency, gamma = 2.

    This follows the PAPER, which states alpha is "set as the inverse frequency
    of positive class pixels" and gamma is 2.

    Their CODE does something else, and the two disagree. BaseModel.py calls
    torchvision's sigmoid_focal_loss with `alpha = 1 - pos_class_weight` (the
    configurable `alpha_focal` line is commented out), and every UTAE config
    ships `pos_class_weight: 10`, so alpha = -9. Current torchvision rejects that
    outright ("alpha must be in the range [0,1] or -1 for ignore"); older
    versions guarded with `if alpha >= 0`, which means a negative alpha silently
    SKIPPED class balancing altogether. So their effective loss was focal with
    gamma=2 and no alpha at all.

    We implement the paper's description. To reproduce their code path instead,
    drop the alpha term entirely.
    """
    p = torch.sigmoid(logits)
    pt = torch.where(y > 0.5, p, 1 - p).clamp(1e-6, 1 - 1e-6)
    inv = _pos_weight_from(y)
    alpha_pos = inv / (1 + inv)            # normalised to (0, 1)
    alpha = torch.where(y > 0.5, alpha_pos, 1 - alpha_pos)
    return (-alpha * (1 - pt) ** gamma * torch.log(pt)).mean()


def dice(logits: torch.Tensor, y: torch.Tensor, eps: float = 1.0):
    p = torch.sigmoid(logits)
    inter = (p * y).sum()
    return 1 - (2 * inter + eps) / (p.sum() + y.sum() + eps)


def jaccard(logits: torch.Tensor, y: torch.Tensor, eps: float = 1.0):
    p = torch.sigmoid(logits)
    inter = (p * y).sum()
    union = p.sum() + y.sum() - inter
    return 1 - (inter + eps) / (union + eps)


def bce_dice(logits: torch.Tensor, y: torch.Tensor, pos_weight: float = 20.0):
    """Our own recipe, kept so the new runs can be compared with the old ones."""
    return bce(logits, y, pos_weight=pos_weight) + dice(logits, y)


REGISTRY = {"bce": bce, "focal": focal, "dice": dice, "jaccard": jaccard,
            "bce_dice": bce_dice}


def seq_loss(name: str, preds: torch.Tensor, Y: torch.Tensor, V: torch.Tensor,
             horizon: int, pos_weight: float = 20.0) -> torch.Tensor:
    """Apply a loss per forecast day, skipping days the event does not have.

    WHY THE MASK: sequences are padded to a common horizon, but a fire observed
    for 3 more days has no ground truth for days 4 and 5. V[:, h] flags which
    (sample, horizon) pairs are real. Without the mask those absent days would
    enter the loss as all-zero targets and teach the model that fire vanishes.
    """
    fn = REGISTRY[name]
    if isinstance(preds, (list, tuple)):
        preds = torch.stack(preds, 1)
    if preds.dim() == 4:
        preds = preds.unsqueeze(2)
    tot, n = 0.0, 0
    for h in range(horizon):
        vm = V[:, h] > 0.5
        if vm.sum() == 0:
            continue
        p, y = preds[vm, h], Y[vm, h]
        # BCE and our BCE+Dice take an explicit positive weight; Focal derives
        # its alpha from the batch, and Dice/Jaccard are unweighted by design.
        if name in ("bce", "bce_dice"):
            tot = tot + fn(p, y, pos_weight=pos_weight)
        else:
            tot = tot + fn(p, y)
        n += 1
    return tot / max(n, 1)
