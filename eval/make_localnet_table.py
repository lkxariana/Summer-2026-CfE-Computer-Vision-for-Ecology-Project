"""Table 3 -- local-network completion -- from bundles under runs/*/*/localnet/s*/.

  python eval/make_localnet_table.py [--out results/tables_localnet.md]

Mean per-network AUPR with a bootstrap CI over networks, lift over connectance, pooled AUPR / AUROC over
all blocks, and structure fidelity of the predicted network at matched connectance: Spearman degree
correlations (plants, pollinators), NODF predicted vs observed, and the share of predicted links that are
observed. Seeds averaged where several exist. Row order and citations follow make_ladder_table.GROUPS.
"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT))
from antheia.bundle import RUNS
from eval.make_ladder_table import GROUPS


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=ROOT / "results/tables_localnet.md")
    args = ap.parse_args()
    rows = []
    for f in RUNS.glob("*/*/localnet/s*/metrics.json"):
        c = json.load(open(f.with_name("config.json"))); m = json.load(open(f))
        rows.append({"model": c["model"], "seed": c["seed"], **m})
    df = pd.DataFrame(rows)
    if df.empty:
        print("no localnet bundles"); return
    num = df.select_dtypes("number").columns.difference(["seed"])
    g = df.groupby("model"); pooled = g[num].mean(); pooled["n_seeds"] = g.seed.nunique()
    n_net = int(df.n_networks.iloc[0]); conn = df.mean_connectance.iloc[0]
    cols = [("mean_aupr", "mean AUPR"), ("mean_aupr_lift", "lift"), ("pooled_aupr", "pooled AUPR"), ("pooled_auroc", "pooled AUROC"),
            ("mean_deg_spearman_plants", "deg rho plants"), ("mean_deg_spearman_polls", "deg rho polls"),
            ("mean_nodf_pred", "NODF pred"), ("mean_link_precision_at_L", "precision@L")]
    nodf_obs = df.mean_nodf_obs.iloc[0]
    lines = [f"## Table 3 — local-network completion ({n_net} surveyed networks; mean connectance {conn:.3f} = chance AUPR; "
             f"observed NODF {nodf_obs:.1f})", "",
             "| Method | Reference | seeds | " + " | ".join(c for _, c in cols) + " |", "|---|---|:---:|" + "---:|" * len(cols)]
    best = {k: pooled[k].max() for k, _ in cols}
    for group, items in GROUPS:
        present = [(n, l, r) for n, l, r in items if n in pooled.index]
        if not present:
            continue
        lines.append(f"| *{group}* | | | " + " | " * (len(cols) - 1) + " |")
        for name, label, ref in present:
            r = pooled.loc[name]; cells = []
            for k, _ in cols:
                v = r[k]; s = f"{v:.3f}" if k != "mean_nodf_pred" else f"{v:.1f}"
                # for NODF the best is the one closest to observed
                is_best = (abs(v - nodf_obs) == (pooled["mean_nodf_pred"] - nodf_obs).abs().min()) if k == "mean_nodf_pred" else abs(v - best[k]) < 1e-9
                cells.append(f"**{s}**" if is_best else s)
            ci = f" [{r['mean_aupr_lo']:.3f},{r['mean_aupr_hi']:.3f}]" if r["n_seeds"] == 1 else ""
            cells[0] = cells[0] + ci
            lines.append(f"| {label} | {ref} | {int(r['n_seeds'])} | " + " | ".join(cells) + " |")
    lines += ["", "*Each network's plants x pollinators block is scored with every local-network pair removed from training. "
              "An absent pair among surveyed species is an observed non-interaction. Predicted networks for the structure "
              "columns take the top-L pairs, L = observed links; NODF closest to observed is bold. Bootstrap CI over networks.*"]
    Path(args.out).write_text("\n".join(lines) + "\n"); print("\n".join(lines))


if __name__ == "__main__":
    main()
