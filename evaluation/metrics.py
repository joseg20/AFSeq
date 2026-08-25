"""Pooled average precision on the GPU, verified against sklearn.

Why: accumulating every prediction and calling sklearn's average_precision_score
once at the end sorts ~80 million values on the CPU. That is what pinned 16
cores while the GPU sat at 4%. Sorting is exactly what a GPU is for.

The subtlety is TIES. sklearn groups equal scores into a single threshold, so
all predictions with the same value share one precision/recall point. With
float16 probabilities there are a great many ties, and a naive cumsum that
treats each element as its own threshold gives a different (slightly optimistic)
number. This implementation collapses ties the same way, and the test below
checks it on the real distribution rather than on random data.
"""
import sys

import numpy as np
import torch
from sklearn.metrics import average_precision_score


def pooled_ap_gpu(y, s, device="cuda"):
    """Average precision, same definition as sklearn, computed on device.

    y: int/bool labels, s: scores. Accepts numpy or torch, any float width.
    """
    y = torch.as_tensor(y, device=device).to(torch.float32).ravel()
    s = torch.as_tensor(s, device=device).to(torch.float32).ravel()
    n_pos = y.sum()
    if n_pos == 0:
        return float("nan")

    s, order = torch.sort(s, descending=True)
    y = y[order]

    tp = torch.cumsum(y, 0)
    fp = torch.cumsum(1.0 - y, 0)

    # collapse ties: keep only the LAST index of each run of equal scores, so a
    # group of tied predictions contributes one threshold, as sklearn does
    keep = torch.ones_like(s, dtype=torch.bool)
    keep[:-1] = s[1:] != s[:-1]

    tp = tp[keep]
    fp = fp[keep]
    precision = tp / (tp + fp)
    recall = tp / n_pos

    prev_recall = torch.cat([torch.zeros(1, device=recall.device), recall[:-1]])
    return float(((recall - prev_recall) * precision).sum().item())


def _test():
    """Compare against sklearn on data shaped like ours: float16, very sparse."""
    rng = np.random.default_rng(23)
    print("%-34s %10s %10s %12s" % ("caso", "sklearn", "gpu", "diferencia"))
    print("-" * 70)
    worst = 0.0
    for name, n, pos_rate, dtype in (
            ("pequeno float32", 100_000, 0.02, np.float32),
            ("float16 (muchos empates)", 1_000_000, 0.01, np.float16),
            ("muy escaso, como WSTS+", 5_000_000, 0.003, np.float16),
            ("saturado en 0 y 1", 1_000_000, 0.01, np.float16)):
        y = (rng.random(n) < pos_rate).astype(np.int8)
        s = rng.random(n).astype(dtype)
        if name.startswith("saturado"):
            s = np.clip(s * 3 - 1, 0, 1).astype(dtype)   # lots of exact 0 and 1
        s = np.where(y == 1, np.minimum(
            s.astype(np.float32) + 0.3, 1.0).astype(dtype), s)
        a = average_precision_score(y, s.astype(np.float32))
        b = pooled_ap_gpu(y, s)
        worst = max(worst, abs(a - b))
        print("%-34s %10.6f %10.6f %12.2e" % (name, a, b, abs(a - b)))
    print()
    print("  peor diferencia: %.2e" % worst)
    print("  veredicto:", "EQUIVALENTE" if worst < 1e-6 else
          "DIFIERE -- no usar sin revisar")
    return worst < 1e-6


if __name__ == "__main__":
    sys.exit(0 if _test() else 1)
