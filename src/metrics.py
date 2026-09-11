"""Evaluation metrics for multi-label Charades classification.

Two flavors of mAP (both computed from per-class AP via sklearn):

- **Macro mAP** — unweighted mean of per-class APs (Charades / AVA standard).
  Each class counts equally; only classes with ≥1 positive in the eval set are included.

- **Weighted mAP** — mean of per-class APs weighted by positive count in the eval set.
  Frequent classes contribute more; useful when class imbalance is severe.
"""
import numpy as np
from sklearn.metrics import average_precision_score


def charades_map(y_true, y_score):
    """Compute macro and weighted mAP for multi-label predictions.

    Args:
        y_true, y_score: np arrays [N, C] (binary labels, predicted probabilities).

    Returns:
        dict with keys:
            macro    — unweighted mean AP over classes with positives
            weighted — positive-count-weighted mean AP
            per_class_ap — array [C], NaN where class has no positives
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=np.float64)
    # Guard against occasional NaN/Inf logits (sklearn AP requires finite scores).
    if not np.isfinite(y_score).all():
        n_bad = int((~np.isfinite(y_score)).sum())
        print(f"[metrics] warning: {n_bad} non-finite scores → clipped to [0, 1]", flush=True)
        y_score = np.nan_to_num(y_score, nan=0.0, posinf=1.0, neginf=0.0)
    y_score = np.clip(y_score, 0.0, 1.0)
    n_classes = y_true.shape[1]
    aps = np.full(n_classes, np.nan, dtype=np.float64)
    n_pos = np.zeros(n_classes, dtype=np.float64)
    for c in range(n_classes):
        n_pos[c] = y_true[:, c].sum()
        if n_pos[c] > 0:
            aps[c] = average_precision_score(y_true[:, c], y_score[:, c])

    valid = ~np.isnan(aps)
    macro = float(np.nanmean(aps)) if valid.any() else float("nan")
    weighted = float(np.average(aps[valid], weights=n_pos[valid])) if valid.any() else float("nan")

    return {"macro": macro, "weighted": weighted, "per_class_ap": aps}
