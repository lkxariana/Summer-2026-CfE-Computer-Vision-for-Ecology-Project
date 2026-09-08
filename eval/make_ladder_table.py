"""Table 2 of the connectivity ladder from bundles (plan §6, §8.1): grouped, cited, seed-pooled.

  python eval/make_ladder_table.py --split cold_plant/val [--out results/tables_ladder.md]

Rows are the comparative set only -- nulls, ecological baselines, published architectures we
re-implemented, and our frozen models. Intermediate arms (sweeps, ablations) are excluded by
construction: a row appears only if its bundle name is in GROUPS.
"""
import argparse
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from antheia.bundle import load_bundles, seed_pooled

GROUPS = [
    ("Nulls", [
        ("baseline_popularity", "Pollinator popularity", "Aiyappa et al. 2025, ICML"),
        ("baseline_cooccurrence", "Co-occurrence count N", "Blanchet, Cazelles & Gravel 2020, Ecol Lett"),
        ("baseline_abundance", "Abundance neutral model", "Vázquez, Chacoff & Cagnolo 2009, Ecology"),
    ]),
    ("Ecological baselines", [
        ("baseline_phenology_abundance", "Phenology x abundance", "Vizentin-Bugoni et al. 2014, Proc R Soc B"),
        ("baseline_congeneric", "Congeneric transfer", "cf. Strydom et al. 2022, MEE"),
        ("baseline_svd_taxonomic", "Truncated SVD + taxonomic imputation", "Strydom et al. 2022, MEE"),
        ("baseline_antheia_spatial", "Co-occurrence PCA-15 + N (roadmap proof-of-concept; ANTHEIA v1)", "Strydom et al. 2021, Phil Trans R Soc B; Li, Cher & Jacobs 2026"),
        ("baseline_antheia_scalar", "Co-occurrence PCA-15 + N + Delta (ANTHEIA v1 scalar)", "Li, Cher & Jacobs 2026"),
        ("baseline_pair_gbm", "Gradient boosting on pair features", "Pichler et al. 2020, MEE"),
        ("baseline_nectar_ungated", "Spatial x phenological overlap product", "Baiotto et al. 2026 (bioRxiv), Eq. 1"),
        ("baseline_nectar_like", "NECTAR-style plausibility (genus constraint x overlap product)", "Baiotto et al. 2026 (bioRxiv)"),
    ]),
    ("Published architectures, re-implemented", [
        ("baseline_two_tower", "Two-tower retrieval (sampled softmax, logQ)", "Yi et al. 2019, RecSys"),
        ("baseline_pairnet", "Pair MLP (NCF-style)", "He et al. 2017, WWW"),
        ("baseline_widedeep", "Wide & Deep", "Cheng et al. 2016"),
        ("baseline_dcnv2", "DCN-V2 (2 cross layers)", "Wang et al. 2021, WWW"),
        ("baseline_tabicl", "TabICL (in-context tabular)", "Qu et al. 2025"),
    ]),
    ("Ours", [
        ("baseline_ours_gbm", "Boosted ranker: taxonomy + spatial + per-cell", "this work"),
        ("baseline_routed", "Genus-routed experts", "this work"),
        ("M1.0_reference", "Embedding model (identity + field blocks)", "this work"),
        ("M2.1_fusion_identity", "Fusion re-ranker on the embedding model (identity tokens)", "this work"),
        ("M2.2_fusion_joint", "Fusion re-ranker + joint field tokens", "this work"),
        ("M2.8_fusion_routed", "Fusion re-ranker on the routed tree retriever", "this work"),
        ("M2.8b_fusion_routed_affine", "Fusion re-ranker on the routed tree retriever (affine base)", "this work"),
        ("M2.9_fusion_coocneg", "Fusion re-ranker + co-occurrence negatives", "this work"),
        ("M2.10_fusion_genus", "Fusion re-ranker + genus-profile tokens", "this work"),
        ("M2.11_fusion_genus_coocneg", "Fusion re-ranker + genus tokens + co-occurrence negatives", "this work"),
        ("M3.1_rgcn", "R-GCN over species, taxa, cell x month", "this work"),
        ("M2.12_fusion_on_rgcn", "Fusion re-ranker on the R-GCN retriever", "this work"),
    ]),
]
COLS = [("aupr", "AUPR"), ("aupr_at_0.25", "AUPR 1:3"), ("aupr_at_0.5", "AUPR 1:1"), ("auroc", "AUROC"),
        ("nrecall_at_10", "nR@10"), ("nrecall_at_50", "nR@50"), ("nrecall_at_10__genus_unseen", "unseen-genus nR@10")]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="cold_plant/val")
    ap.add_argument("--out", default=ROOT / "results/tables_ladder.md")
    args = ap.parse_args()
    df = load_bundles(); df = df[df["split"] == args.split]
    pooled = seed_pooled(df, group=("model", "split")).set_index("model")
    prev = df["prevalence"].iloc[0] if len(df) else float("nan")
    lines = [f"## Table 2 — cold-plant validation ({args.split}), {int(df['n_queries'].iloc[0])} plants x "
             f"{int(df['n_candidates'].iloc[0])} candidates, chance AUPR {prev:.5f}", "",
             "| Method | Reference | seeds | " + " | ".join(c for _, c in COLS) + " |",
             "|---|---|:---:|" + "---:|" * len(COLS)]
    best = {k: pooled[k].max() for k, _ in COLS if k in pooled}
    for group, rows in GROUPS:
        present = [(n, l, r) for n, l, r in rows if n in pooled.index]
        if not present:
            continue
        lines.append(f"| *{group}* | | | " + " | " * (len(COLS) - 1) + " |")
        for name, label, ref in present:
            r = pooled.loc[name]
            cells = []
            for k, _ in COLS:
                v = r.get(k, float("nan"))
                s = f"{v:.3f}" if pd.notna(v) else "—"
                cells.append(f"**{s}**" if pd.notna(v) and abs(v - best[k]) < 1e-9 else s)
            ns = int(r["n_seeds"]); flag = "" if (ns >= 3 or name.startswith("baseline_") and ns >= 1) else " (incomplete)"
            lines.append(f"| {label} | {ref} | {ns}{flag} | " + " | ".join(cells) + " |")
    lines += ["", "*AUPR at network prevalence is primary; AUPR 1:3 and 1:1 re-weight negatives from the full ranking "
              "(the population version of sampled-negative evaluation). Deterministic baselines are single runs; learned "
              "models are averaged over seeds {42, 0, 1}. Bold = column best.*"]
    Path(args.out).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
