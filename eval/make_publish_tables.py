"""Publishable tables (compact): Table 1 = four regimes x {AUPR, AUROC, nR@10}; Table 2 = within-site completion.

Rows: nulls, ecological baselines, ANTHEIA v1 (prior work from this group), published architectures, ours (R-GCN,
+re-ranker, symmetric R-GCN, +re-ranker). Everything else (ablations, hand-engineered trees, re-expressed AUPRs, nR@50,
strata, degree correlations) goes to the appendix tables produced by make_ladder_table.py / make_localnet_table.py.

  python eval/make_publish_tables.py [--out results/tables_publish.md]
"""
import argparse
import subprocess
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from antheia.bundle import RUNS as RUNS_DIR, load_bundles, seed_pooled

ROWS = [
    ("Nulls", [("baseline_popularity", "Pollinator popularity"), ("baseline_cooccurrence", "Co-occurrence N")]),
    ("Ecological baselines", [
        ("baseline_phenology_abundance", "Phenology x abundance"),
        ("baseline_congeneric", "Congeneric transfer"),
        ("baseline_svd_taxonomic", "SVD + taxonomic imputation"),
        ("baseline_pair_gbm", "Pair-feature GBM"),
        ("baseline_nectar_like", "NECTAR-style plausibility")]),
    ("ANTHEIA v1 (prior work, this group)", [
        ("baseline_antheia_spatial", "Co-occurrence PCA + N"),
        ("baseline_antheia_scalar", "Co-occurrence PCA + N + phenology overlap")]),
    ("Published architectures", [
        ("baseline_two_tower", "Two-tower retrieval"),
        ("baseline_widedeep", "Wide & Deep"),
        ("baseline_dcnv2", "DCN-V2")]),
    ("Ours", [
        ("M3.1_rgcn", "R-GCN, taxon + cell nodes (plant-side leave-own-edges-out)"),
        ("M2.12_fusion_on_rgcn", "  + identity re-ranker"),
        ("M3.7_rgcn_sym", "R-GCN, taxon + cell nodes (symmetric leave-own-edges-out)"),
        ("M4.0_system_sym", "  + identity re-ranker"),
        ("A2_rgcn_sym_notaxon", "R-GCN, species + cell nodes (symmetric leave-own-edges-out)"),
        ("M5.0_system_notaxon", "  + identity re-ranker (**final**)")]),
]
SPLITS = [("cold_plant/val", "cold plant"), ("cold_poll/val", "cold pollinator"), ("cold_both/val", "cold both"), ("warm/val", "warm")]
METS = [("aupr", "AUPR"), ("auroc", "AUROC"), ("nrecall_at_10", "nR@10")]
DET = {"baseline_popularity", "baseline_cooccurrence", "baseline_abundance", "baseline_phenology_abundance", "baseline_congeneric",
       "baseline_svd_taxonomic", "baseline_nectar_ungated", "baseline_nectar_like", "baseline_tabicl"}


def stale_models(df):
    fix = subprocess.run(["git", "log", "--format=%H", "-1", "--grep=negative-sampling pool"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    out = set()
    for _, r in df.iterrows():
        if r["model"] in DET or r["split"].startswith("cold_plant") or not fix:
            continue
        if subprocess.run(["git", "merge-base", "--is-ancestor", fix, r["git"]], cwd=ROOT).returncode != 0:
            out.add((r["model"], r["split"]))
    return out


def fmt(v, best, dagger=""):
    if pd.isna(v):
        return "—"
    s = f"{v:.3f}"
    return (f"**{s}**" if abs(v - best) < 1e-9 else s) + dagger


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=ROOT / "results/tables_publish.md")
    args = ap.parse_args()
    df = load_bundles(); stale = stale_models(df)
    pooled = seed_pooled(df, group=("model", "split")).set_index(["model", "split"])
    names = [n for _, rows in ROWS for n, _ in rows]
    # ---- Table 1
    head = "| Method | " + " | ".join(f"{lab}: {m}" for _, lab in SPLITS for _, m in METS) + " |"
    lines = ["## Table 1 — link prediction across regimes (validation). AUPR at network prevalence, AUROC, normalised recall@10.", "",
             head, "|---|" + "---:|" * (len(SPLITS) * len(METS))]
    best = {}
    for sp, _ in SPLITS:
        for k, _ in METS:
            vals = [pooled.loc[(n, sp), k] for n in names if (n, sp) in pooled.index]
            best[(sp, k)] = max(vals) if vals else float("nan")
    chance = {sp: pooled.xs(sp, level="split")["prevalence"].iloc[0] for sp, _ in SPLITS if sp in pooled.index.get_level_values("split")}
    for group, rows in ROWS:
        lines.append(f"| *{group}* |" + " |" * (len(SPLITS) * len(METS)))
        for n, label in rows:
            cells = []
            for sp, _ in SPLITS:
                for k, _ in METS:
                    v = pooled.loc[(n, sp), k] if (n, sp) in pooled.index else float("nan")
                    cells.append(fmt(v, best[(sp, k)], "†" if (n, sp) in stale else ""))
            lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines += ["", "Chance AUPR: " + ", ".join(f"{lab} {chance[sp]:.4f}" for sp, lab in SPLITS if sp in chance) + ". Learned models: mean of seeds "
              "{42, 0, 1} where available; deterministic baselines single run. † = run predates the pollinator-side protocol fix (being re-run). "
              "Bold = column best. Re-expressed AUPR (1:3, 1:1), nR@50 and strata: appendix.", ""]
    # ---- Table 2 (within-site)
    loc = df[df.split == "localnet"]
    lp = seed_pooled(loc, group=("model", "split")).set_index("model")
    cols = [("mean_aupr", "mean AUPR"), ("mean_auroc", "mean AUROC"), ("mean_link_precision_at_L", "precision@L"), ("mean_nodf_pred", "NODF (obs 36)"),
            ("mean_aupr_warm", "AUPR warm plants"), ("mean_aupr_cold", "AUPR cold plants")]
    lines += ["## Table 2 — within-site network completion (91 surveyed networks; every local pair removed from training; chance AUPR = connectance 0.135)", "",
              "| Method | " + " | ".join(c for _, c in cols) + " |", "|---|" + "---:|" * len(cols)]
    lbest = {k: lp.loc[[n for n in names if n in lp.index], k].max() for k, _ in cols}
    for group, rows in ROWS:
        present = [(n, l) for n, l in rows if n in lp.index]
        if not present:
            continue
        lines.append(f"| *{group}* |" + " |" * len(cols))
        for n, label in present:
            r = lp.loc[n]; cells = []
            for k, _ in cols:
                v = r.get(k, float("nan"))
                if k == "mean_nodf_pred":
                    cells.append("—" if pd.isna(v) else (f"**{v:.0f}**" if abs(v - 36.0) == min(abs(lp.loc[[m for m in names if m in lp.index], k] - 36.0)) else f"{v:.0f}"))
                elif k == "mean_aupr" and "mean_aupr_lo" in r and r.get("n_seeds", 1) == 1:
                    cells.append(fmt(v, lbest[k]) + f" [{r['mean_aupr_lo']:.3f}, {r['mean_aupr_hi']:.3f}]")
                else:
                    cells.append(fmt(v, lbest[k]))
            lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines += ["", "precision@L and NODF from the top-L pairs per network (L = observed links). Warm = plant keeps at least one edge outside the site; "
              "cold = none. Seeds averaged where several exist; single-seed rows show the bootstrap CI over networks."]
    Path(args.out).write_text("\n".join(lines) + "\n"); print("\n".join(lines))
    appendix(Path(args.out).with_name("tables_publish_appendix.md"))


def appendix(out_path):
    """Appendix tables: A1 cold-plant strata (low-degree, unseen-genus, zero-shot), A2 within-site by dataset."""
    import glob, json
    import numpy as np
    df = load_bundles()
    names = [n for _, rows in ROWS for n, _ in rows]; labels = {n: l for _, rows in ROWS for n, l in rows}
    lines = ["## Table A1 — cold-plant validation by stratum (nR@10 unless noted; seeds averaged)", "",
             "| Method | low-degree plants (<= 2 partners): nR@10 / AUPR | other plants: nR@10 | unseen-genus: nR@10 / AUPR | zero-shot (text-imputed features): nR@10 |",
             "|---|---:|---:|---:|---:|"]
    cp = df[df.split == "cold_plant/val"]
    for n in names:
        g = cp[cp.model == n]
        if g.empty: continue
        m = lambda k: g[k].mean() if k in g else float("nan")
        lines.append(f"| {labels[n].strip()} | {m('nrecall_at_10__low_degree'):.3f} / {m('aupr__low_degree'):.3f} | {m('nrecall_at_10__not_low_degree'):.3f} | "
                     f"{m('nrecall_at_10__genus_unseen'):.3f} / {m('aupr__genus_unseen'):.3f} | {m('nrecall_at_10__zeroshot'):.3f} |")
    lines += ["", "## Table A2 — within-site mean AUPR by survey source (seeds averaged)", ""]
    nets = pd.read_parquet(ROOT / "data/network/local_networks.parquet").drop_duplicates("network").set_index("network")["dataset"]
    per = {}
    for n in names:
        fs = glob.glob(str(RUNS_DIR / n / "*" / "localnet" / "s*" / "per_network.csv"))
        if not fs: continue
        d = pd.concat([pd.read_csv(f) for f in fs]).groupby("network")["aupr"].mean()
        d = d.to_frame("aupr").join(nets, how="left")
        per[n] = d.groupby("dataset")["aupr"].mean()
    if per:
        tab = pd.DataFrame(per).T; tab.columns = [str(c) for c in tab.columns]
        counts = nets.value_counts()
        lines.append("| Method | " + " | ".join(f"{c} (n={counts.get(c, 0)})" for c in tab.columns) + " |")
        lines.append("|---|" + "---:|" * len(tab.columns))
        for n in tab.index:
            lines.append(f"| {labels[n].strip()} | " + " | ".join(f"{tab.loc[n, c]:.3f}" for c in tab.columns) + " |")
    Path(out_path).write_text("\n".join(lines) + "\n"); print("\n".join(lines))


if __name__ == "__main__":
    main()
