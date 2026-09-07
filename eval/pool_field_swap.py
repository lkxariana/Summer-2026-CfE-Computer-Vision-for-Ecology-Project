"""Pool the field-swap arms over seeds: seed-averaged per-plant nR@10, paired bootstrap, strata.

Reads results/field_swap_perplant_s{seed}.npz for every seed present, averages each plant's
nrecall@10 across seeds within an arm, then compares arms with a paired bootstrap over plants --
the same protocol as the encoding ladder, so the noise floor is the one already established
(single-seed differences of +-0.02 are routine).
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from antheia.metrics import paired_bootstrap

CONTRASTS = [
    ("field replaces", "reference"), ("field added", "reference"), ("field, no impute", "reference"),
    ("no-text surface", "reference"), ("no-text field", "no-text surface"),
]


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preset", default="swap", choices=["swap", "joint"])
    args = ap.parse_args()
    stem = "field_swap" if args.preset == "swap" else "joint_field_swap"
    files = sorted((ROOT / "results").glob(f"{stem}_perplant_s*.npz"))
    runs = [np.load(f, allow_pickle=True) for f in files]
    if args.preset == "joint":
        # the taxonomy-free control lives in the swap files; same seeds, same reference (verified
        # bit-identical), so it pairs across files
        for i, f in enumerate(files):
            sw = ROOT / "results" / f.name.replace("joint_field_swap", "field_swap")
            if sw.exists():
                d = dict(runs[i]); d["no-text surface"] = np.load(sw)["no-text surface"]; runs[i] = d
    seeds = [f.stem.split("_s")[-1] for f in files]
    keys = lambda r: list(r.files) if hasattr(r, "files") else list(r.keys())
    arms = [k for k in keys(runs[0]) if k not in ("plants", "seen")]
    seen = runs[0]["seen"]
    for r in runs:
        assert (r["plants"] == runs[0]["plants"]).all()
    print(f"{len(runs)} seeds ({', '.join(seeds)}), {len(seen)} plants, genus seen {seen.sum()} / unseen {(~seen).sum()}\n")

    avg = {a: np.mean([r[a] for r in runs if a in keys(r)], 0) for a in arms}
    rows = []
    print(f"{'arm':<18} {'nR@10':>7} {'seen':>7} {'unseen':>7}   per-seed")
    for a in arms:
        per = [f"{r[a].mean():.4f}" for r in runs if a in keys(r)]
        print(f"{a:<18} {avg[a].mean():.4f} {avg[a][seen].mean():.4f} {avg[a][~seen].mean():.4f}   {' '.join(per)}")
        rows.append(dict(arm=a, nrecall10=avg[a].mean(), seen=avg[a][seen].mean(), unseen=avg[a][~seen].mean(),
                         n_seeds=len(per)))
    contrasts = CONTRASTS if args.preset == "swap" else (
        [(a, "reference") for a in arms if a.startswith("joint")]
        + [(a, "no-text surface") for a in arms if a.startswith("no-text joint")])
    print("\npaired bootstrap on seed-averaged per-plant nR@10 (10k resamples):")
    for a, b in contrasts:
        if a not in avg or b not in avg:
            continue
        d, lo, hi, p = paired_bootstrap(avg[a], avg[b], 10000, 42)
        du, lou, hiu, pu = paired_bootstrap(avg[a][~seen], avg[b][~seen], 10000, 42)
        print(f"  {a:<18} - {b:<16} {d:+.4f} [{lo:+.4f},{hi:+.4f}] p={p:.4f}   "
              f"unseen genus {du:+.4f} [{lou:+.4f},{hiu:+.4f}] p={pu:.3f}")
        rows.append(dict(arm=f"{a} - {b}", nrecall10=d, lo=lo, hi=hi, p=p, unseen=du, p_unseen=pu))
    pd.DataFrame(rows).to_csv(ROOT / f"results/{stem}_pooled.csv", index=False)
    print(f"\n[wrote] results/{stem}_pooled.csv")


if __name__ == "__main__":
    main()
