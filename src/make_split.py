"""Create train_val_split.json (required only for --protocol train_val_test).

    python src/make_split.py --annot-dir $SCRATCH/charades/annotations
"""
import argparse
import os

import annotations


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--annot-dir", default=os.environ.get("ANNOT_DIR", ""))
    args = p.parse_args()
    if not args.annot_dir:
        p.error("--annot-dir is required (or set ANNOT_DIR)")
    split = annotations.load_train_val_split(args.annot_dir)
    path = annotations.split_path(args.annot_dir)
    print(f"Split file: {path}")
    print(f"  protocol:     train_val_test")
    print(f"  train videos: {split['n_train']}")
    print(f"  val videos:   {split['n_val']}")
    print(f"  test videos:  official Charades test CSV (evaluate.py only)")
    print("\nFor --protocol train_test, this file is not used.")


if __name__ == "__main__":
    main()
