"""Parse Charades annotations into multi-hot labels over the 157 action classes.

Two split protocols (select with --protocol):

  train_val_test  (recommended)
      train — 80% of official Charades train
      val   — 20% holdout from official train (checkpoint selection)
      test  — official Charades test (final evaluation only)

  train_test  (simple baseline — test used for both checkpoint + final eval)
      train — full official Charades train
      val   — same videos as test (official test, used during training)
      test  — official Charades test (same as val → val mAP == test mAP)
"""
import json
import os

import numpy as np
import pandas as pd

NUM_CLASSES = 157
VAL_FRAC = 0.2
SPLIT_SEED = 42
SPLIT_FILENAME = "train_val_split.json"
PROTOCOLS = ("train_test", "train_val_test")


def load_classes(annot_dir):
    """Return (id2idx, names) from Charades_v1_classes.txt."""
    path = os.path.join(annot_dir, "Charades_v1_classes.txt")
    id2idx, names = {}, []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            cid, name = line.split(" ", 1)
            id2idx[cid] = i
            names.append(name)
    assert len(names) == NUM_CLASSES, f"expected {NUM_CLASSES} classes, got {len(names)}"
    return id2idx, names


def parse_actions(cell):
    """'c092 11.9 21.2;c147 0 12.6' -> ['c092', 'c147']  (deduplicated)."""
    if not isinstance(cell, str) or not cell.strip():
        return []
    out = []
    for entry in cell.split(";"):
        entry = entry.strip()
        if entry:
            out.append(entry.split()[0])
    return list(dict.fromkeys(out))


def normalize_protocol(protocol):
    if protocol not in PROTOCOLS:
        raise ValueError(f"protocol must be one of {PROTOCOLS}, got {protocol!r}")
    return protocol


def describe_protocol(protocol):
    protocol = normalize_protocol(protocol)
    if protocol == "train_val_test":
        return "train/val/test (val=20% holdout from train; test=final eval only)"
    return "train/test (official test used for checkpoint selection and final eval)"


def checkpoint_split(protocol):
    """Which split name is used during training to pick best.pth."""
    return "val" if normalize_protocol(protocol) == "train_val_test" else "test"


def split_path(annot_dir):
    return os.path.join(annot_dir, SPLIT_FILENAME)


def create_train_val_split(annot_dir, val_frac=VAL_FRAC, seed=SPLIT_SEED):
    """Hold out val_frac of official train videos for validation."""
    ids, _ = _labels_from_csv(annot_dir, "train")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(ids))
    n_val = max(1, int(round(len(ids) * val_frac)))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    return {
        "seed": seed,
        "val_frac": val_frac,
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "train_ids": [ids[i] for i in train_idx],
        "val_ids": [ids[i] for i in val_idx],
    }


def load_train_val_split(annot_dir, val_frac=VAL_FRAC, seed=SPLIT_SEED):
    """Load or create train_val_split.json (only needed for train_val_test)."""
    path = split_path(annot_dir)
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    split = create_train_val_split(annot_dir, val_frac, seed)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(split, f, indent=2)
    print(f"Created {path} ({split['n_train']} train / {split['n_val']} val)", flush=True)
    return split


def _labels_from_csv(annot_dir, csv_split):
    """Labels from Charades_v1_{train,test}.csv without sub-splitting."""
    id2idx, _ = load_classes(annot_dir)
    csv = os.path.join(annot_dir, f"Charades_v1_{csv_split}.csv")
    df = pd.read_csv(csv)

    ids, rows = [], []
    for _, r in df.iterrows():
        classes = parse_actions(r.get("actions", ""))
        if not classes:
            continue
        vec = np.zeros(NUM_CLASSES, dtype=np.float32)
        for c in classes:
            if c in id2idx:
                vec[id2idx[c]] = 1.0
        ids.append(str(r["id"]))
        rows.append(vec)
    return ids, np.stack(rows)


def build_labels(annot_dir, split, protocol="train_test"):
    """Return (ids, labels) for split in {'train','val','test'}."""
    protocol = normalize_protocol(protocol)

    if split == "test":
        return _labels_from_csv(annot_dir, "test")

    if protocol == "train_test":
        if split == "train":
            return _labels_from_csv(annot_dir, "train")
        if split == "val":
            # During training, validate on official test (same videos as final eval).
            return _labels_from_csv(annot_dir, "test")
        raise ValueError(f"unknown split: {split!r}")

    # train_val_test
    if split in ("train", "val"):
        ids, labels = _labels_from_csv(annot_dir, "train")
        tvs = load_train_val_split(annot_dir)
        keep = set(tvs["train_ids"] if split == "train" else tvs["val_ids"])
        idx = [i for i, vid in enumerate(ids) if vid in keep]
        return [ids[i] for i in idx], labels[idx]

    raise ValueError(f"unknown split: {split!r} (use train, val, or test)")


def positive_counts(annot_dir, protocol="train_test"):
    """Per-class positive count on the training fold (for BCE pos_weight)."""
    _, labels = build_labels(annot_dir, "train", protocol=protocol)
    return labels.sum(axis=0)
