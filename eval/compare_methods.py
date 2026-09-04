"""Paired comparison between methods on the per-plant metrics `run_comparison.py` writes.

Every method is scored on the identical held-out plants, so differences are tested by resampling
plants with the pairing intact. An unpaired test on this data is dominated by between-plant
variance -- plants differ enormously in how predictable their partners are -- and will call real
differences insignificant.
"""
import argparse
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from antheia.metrics import paired_bootstrap


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--per-plant", default=ROOT / "results/per_plant_val_modelled.parquet")
    ap.add_argument("--metric", default="recall@10")
    ap.add_argument("--against", default=None, help="reference method; default is the best")
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    d = pd.read_parquet(args.per_plant)
    wide = d.pivot(index="plant", columns="method", values=args.metric)
    ref = args.against or wide.mean().idxmax()
    print(f"[paired] {args.metric}, {len(wide)} plants, reference = {ref}\n")
    rows = []
    for m in wide.columns:
        if m == ref:
            continue
        diff, lo, hi, p = paired_bootstrap(wide[ref].to_numpy(), wide[m].to_numpy(), args.n, args.seed)
        rows.append({"method": m, f"{args.metric}": wide[m].mean(), "delta_vs_ref": diff,
                     "lo": lo, "hi": hi, "p": p, "significant": bool(lo > 0 or hi < 0)})
    out = pd.DataFrame(rows).sort_values("delta_vs_ref")
    out.insert(0, "reference", ref)
    print(out.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    dest = Path(args.per_plant).with_name(f"paired_{args.metric.replace('@', '')}.csv")
    out.to_csv(dest, index=False)
    print(f"\n[wrote] {dest}")


if __name__ == "__main__":
    main()
