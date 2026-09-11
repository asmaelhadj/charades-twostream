"""Collect mAP from every runs/<protocol>/<name>/ and print comparison tables.

    python src/summarize.py
    python src/summarize.py --protocol train_val_test
"""
import argparse
import glob
import json
import os

import annotations


def _load_test_maps(run_dir, protocol, best, best_weighted):
    """Prefer multi-view eval json when present, else single-view, else best val."""
    preferred = sorted(glob.glob(os.path.join(run_dir, "per_class_ap_s*c*.json")))
    candidates = preferred + [os.path.join(run_dir, "per_class_ap.json")]
    for pc in candidates:
        if not os.path.exists(pc):
            continue
        with open(pc, encoding="utf-8") as f:
            pc_data = json.load(f)
        return (
            pc_data.get("macro_mAP", pc_data.get("mAP")),
            pc_data.get("weighted_mAP"),
            os.path.basename(pc),
        )
    if protocol == "train_test":
        return best, best_weighted, "history(best)"
    return None, None, None


def collect_rows(protocol_filter=None):
    rows = []
    for hist in sorted(glob.glob("runs/*/*/history.json")):
        run_dir = os.path.dirname(hist)
        name = os.path.basename(run_dir)
        protocol = os.path.basename(os.path.dirname(run_dir))
        with open(hist, encoding="utf-8") as f:
            h = json.load(f)
        protocol = h.get("protocol", protocol)
        if protocol_filter and protocol != protocol_filter:
            continue
        best = h.get("best_map")
        best_weighted = h.get("best_weighted_map")
        test_macro, test_weighted, src = _load_test_maps(
            run_dir, protocol, best, best_weighted,
        )
        rows.append((protocol, name, best, best_weighted, test_macro, test_weighted, src))
    return rows


def print_table(rows, title):
    if not rows:
        return
    print(f"\n{title}")
    print(f"{'run':20s} {'val macro':>10s} {'val wtd':>10s} "
          f"{'test macro':>11s} {'test wtd':>10s}  {'eval':s}")
    print("-" * 78)
    for _, name, best, best_w, test_macro, test_w, src in rows:
        b = f"{best:.4f}" if best is not None else "  -  "
        bw = f"{best_w:.4f}" if best_w is not None else "  -  "
        tm = f"{test_macro:.4f}" if test_macro is not None else "  -  "
        tw = f"{test_w:.4f}" if test_w is not None else "  -  "
        tag = src or "-"
        print(f"{name:20s} {b:>10s} {bw:>10s} {tm:>11s} {tw:>10s}  {tag}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--protocol", choices=annotations.PROTOCOLS, default=None)
    args = p.parse_args()

    rows = collect_rows(args.protocol)
    if not rows:
        print("No runs found under runs/<protocol>/<name>/. Train something first.")
        return

    by_protocol = {}
    for row in rows:
        by_protocol.setdefault(row[0], []).append(row)

    for protocol in sorted(by_protocol):
        if args.protocol and protocol != args.protocol:
            continue
        print_table(by_protocol[protocol], f"Protocol: {protocol}")
        print(f"  → {annotations.describe_protocol(protocol)}")


if __name__ == "__main__":
    main()
