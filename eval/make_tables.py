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


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--network", default=ROOT / "data/network")
    ap.add_argument("--universe", default=ROOT / "data/network/modelled_universe.json")
    ap.add_argument("--out", default=ROOT / "results/tables.md")
    ap.add_argument("--tables", nargs="*", default=["1"])
    args = ap.parse_args()
    parts = []
    if "1" in args.tables:
        parts.append("## Table 1 — The interaction network\n\n" + table1(Path(args.network), args.universe))
    text = "\n\n".join(parts)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(text + "\n")
    print(text)
    print(f"\n[wrote] {args.out}")


if __name__ == "__main__":
    main()
