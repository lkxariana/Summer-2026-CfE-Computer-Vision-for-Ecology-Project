"""Emit the paper tables from the built artifacts, as markdown matching docs/paper/tables.md.

Every number is computed here rather than transcribed, so a table cannot drift from the data it
describes. Run after the network and universe are built.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def counts(nodes_side, labels):
    sub = nodes_side[nodes_side["label"].isin(labels)]
    return len(sub), int((sub["rank"] == "species").sum()), int((sub["rank"] == "genus").sum())


def table1(net, universe):
    e_all = pd.read_parquet(net / "edges.parquet")
    pl = pd.read_parquet(net / "nodes_plants.parquet")
    po = pd.read_parquet(net / "nodes_pollinators.parquet")
    u = json.load(open(universe))
    up = {p["label"] for p in u["plants"]}
    uq = {p["label"] for p in u["pollinators"]}

    rows, cols = {}, {}
    for tier, e in [("Tier A", e_all[e_all["tier"] == "A"]), ("Tier A+B", e_all)]:
        plants, pl_sp, pl_gen = counts(pl, set(e["plant"]))
        polls, po_sp, po_gen = counts(po, set(e["pollinator"]))
        pdeg = e["plant"].value_counts()
        qdeg = e["pollinator"].value_counts()
        srcs = {s for v in e["sources"] for s in v.split(",")}
        fy = e["first_year"].dropna()
        modelled = int((e["plant"].isin(up) & e["pollinator"].isin(uq)).sum())
        cols[tier] = {
            "Interactions": f"{len(e):,}",
            "Plant taxa (species / genus)": f"{pl_sp:,} / {pl_gen:,}",
            "Pollinator taxa (species / genus)": f"{po_sp:,} / {po_gen:,}",
            "Connectance": f"{100 * len(e) / (plants * polls):.2f}%",
            "Plant degree — median / max": f"{int(pdeg.median())} / {int(pdeg.max()):,}",
            "Pollinator degree — median / max": f"{int(qdeg.median())} / {int(qdeg.max()):,}",
            "Interactions with ≥2 independent records": f"{100 * (e['n_observations'] >= 2).mean():.1f}%",
            "Interactions with ≥2 source datasets": f"{100 * (e['n_sources'] >= 2).mean():.1f}%",
            "Source datasets": f"{len(srcs):,}",
            "Interactions carrying a year": f"{100 * e['first_year'].notna().mean():.1f}%",
            "First record — median (5–95%)":
                f"{int(fy.median())} ({int(np.percentile(fy, 5))}–{int(np.percentile(fy, 95))})",
            "First record — full extent": f"{int(fy.min())}–{int(fy.max())}",
            "Modelled subgraph (both sides feature-covered)": f"{modelled:,}",
        }
    rows = list(cols["Tier A"])
    out = ["| | Tier A | Tier A+B |", "|---|---:|---:|"]
    out += [f"| {r} | {cols['Tier A'][r]} | {cols['Tier A+B'][r]} |" for r in rows]
    return "\n".join(out)


GROUPS = [
    ("*Nulls*", ["popularity", "cooccurrence", "abundance"]),
    ("*Structured ecological baselines*", ["congeneric", "phenology_abundance", "trait_matching"]),
    ("*Learned representations*", ["svd_taxonomic", "lightfm"]),
    ("*Feature-based*", ["pair_gbm", "two_tower", "two_tower_percell"]),
]
LABELS = {
    "popularity": "Pollinator popularity", "cooccurrence": "Co-occurrence count",
    "abundance": "Abundance neutral model", "congeneric": "Congeneric transfer",
    "phenology_abundance": "Phenology x abundance likelihood",
    "trait_matching": "Trait matching (reduced coverage)",
    "svd_taxonomic": "Truncated SVD + taxonomic imputation", "lightfm": "LightFM (WARP)",
    "pair_gbm": "Gradient boosting on pair features",
    "two_tower": "**Two-tower retrieval (ours)**",
    "two_tower_percell": "**+ per-cell phenology encoder (ours)**",
}
SETS = [("All held-out plants", "val"), ("Expert field networks", "web-of-life"),
        ("Specimen records", "gbif-us-bees")]


def table2(results, curves="modelled"):
    frames, prev = {}, {}
    for title, tag in SETS:
        f = Path(results) / f"comparison_{tag}_{curves}.csv"
        if f.exists():
            d = pd.read_csv(f).set_index("method")
            frames[title] = d
            prev[title] = d["connectance"].iloc[0]
    head = ["| | | | " + " | ".join(f"{t} | " for t in frames) + "|",
            "|---|---|:---:|" + "---:|---:|" * len(frames),
            "| **Method** | **Reference** | | " +
            " | ".join("R@10 | PR-AUC" for _ in frames) + " |"]
    sub = []
    body = []
    for group, methods in GROUPS:
        body.append(f"| {group} | | | " + " | ".join("|" for _ in frames) + " |")
        for m in methods:
            cells, ref, cs = [], "", ""
            for title in frames:
                d = frames[title]
                if m in d.index:
                    cells += [f"{d.loc[m, 'recall@10']:.3f}", f"{d.loc[m, 'pr_auc']:.4f}"]
                    ref = d.loc[m, "reference"] or "—"
                    cs = "\u2713" if d.loc[m, "cold_start"] else ""
                else:
                    cells += ["—", "—"]
            body.append(f"| {LABELS[m]} | {ref} | {cs} | " + " | ".join(cells) + " |")
    n = {t: len(pd.read_parquet(Path(results) / f"per_plant_{tag}_modelled.parquet")
                .query("method == 'popularity'")) for t, tag in SETS
         if (Path(results) / f"per_plant_{tag}_modelled.parquet").exists()}
    cap = ("\n*Cold-start capable methods only: each scores a plant with no training interactions. "
           "Held-out plants per set: " + ", ".join(f"{t} {v}" for t, v in n.items()) +
           ". Prevalence baseline for PR-AUC: " +
           ", ".join(f"{t} {prev[t]:.5f}" for t in prev) + ".*")
    return "\n".join(head + sub + body) + "\n" + cap


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--network", default=ROOT / "data/network")
    ap.add_argument("--universe", default=ROOT / "data/network/modelled_universe.json")
    ap.add_argument("--out", default=ROOT / "results/tables.md")
    ap.add_argument("--results", default=ROOT / "results")
    ap.add_argument("--tables", nargs="*", default=["1", "2"])
    args = ap.parse_args()
    parts = []
    if "1" in args.tables:
        parts.append("## Table 1 — The interaction network\n\n" + table1(Path(args.network), args.universe))
    if "2" in args.tables:
        parts.append("## Table 2 — Model comparison\n\n" + table2(args.results))
    text = "\n\n".join(parts)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(text + "\n")
    print(text)
    print(f"\n[wrote] {args.out}")


if __name__ == "__main__":
    main()
