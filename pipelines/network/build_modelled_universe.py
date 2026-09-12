"""Freeze the modelled universe: the taxa every downstream feature cache is indexed to.

A taxon is in the universe when it can be given a spatially explicit surface — a per-cell,
per-week flowering or activity curve. Two routes exist on each side:

  plants       direct         the e98 opportunity surface already holds the species
               zeroshot_text  a BioCLIP-2 text-conditioned head predicts the curve
  pollinators  direct         GBIF occurrences support a supervised per-species SDM head
               zeroshot_lesinr  the LE-SINR head predicts it from the Linnaean text embedding

Pollinators outside the core flower-visiting orders are dropped: they are predators and
incidental visitors (Araneae, Psocodea, Mesostigmata, ...) rather than pollinators. So are taxa
whose order the checklists disagree on, since that disagreement is exactly the membership question.

Genus aggregation -- defining a genus node from its feature-covered congeners -- is not used.
It underperformed the text zero-shot head on held-out covered plants (0.493 vs 0.566 curve
overlap). `has_congener` records where it would have been available so the ablation stays
possible without rebuilding the universe.

Ordering is frozen here (by label, within side). Nothing downstream may reindex.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from antheia.paths import DATA_ROOT, HF_CACHE

CORE_ORDERS = {"Hymenoptera", "Lepidoptera", "Diptera", "Coleoptera", "Hemiptera", "Apodiformes"}
from antheia.paths import REPO_ROOT as ROOT


def genera(names):
    return {n.split()[0] for n in names if " " in n}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--network", default=ROOT / "data/network")
    ap.add_argument("--surface-species",
                    default=str(DATA_ROOT) + "/opportunity_surface_e98_species.json")
    ap.add_argument("--occ", default=str(DATA_ROOT) + "/pollinator_sdm/pollinator_occ_gbifv3.npz")
    ap.add_argument("--out", default=ROOT / "data/network/modelled_universe.json")
    args = ap.parse_args()

    net = Path(args.network)
    surf = set(json.load(open(args.surface_species))) - {"unknown"}
    occ = set(map(str, np.load(args.occ, allow_pickle=True)["names"]))
    surf_gen, occ_gen = genera(surf), genera(occ)

    pl = pd.read_parquet(net / "nodes_plants.parquet").sort_values("label").reset_index(drop=True)
    pl["feature_source"] = np.where(pl["label"].isin(surf), "direct", "zeroshot_text")
    pl["has_congener"] = pl["label"].str.split(" ").str[0].isin(surf_gen)

    pol = pd.read_parquet(net / "nodes_pollinators.parquet")
    # An order conflict means the checklists disagree on whether the taxon is a flower visitor at
    # all (GBIF has the gall-mite genus Eriophyes in both Trombidiformes and Diptera); drop rather
    # than resolve by majority, which picks the larger and wrong group.
    ambiguous = pol.get("order_conflict", pd.Series(False, index=pol.index)).fillna(False)
    keep_order = pol["order"].isin(CORE_ORDERS) & ~ambiguous
    dropped = pol[~keep_order]
    pol = pol[keep_order].sort_values("label").reset_index(drop=True)
    pol["feature_source"] = np.where(pol["label"].isin(occ), "direct", "zeroshot_lesinr")
    pol["has_congener"] = pol["label"].str.split(" ").str[0].isin(occ_gen)

    edges = pd.read_parquet(net / "edges.parquet", columns=["plant", "pollinator"])
    keep = edges["plant"].isin(set(pl["label"])) & edges["pollinator"].isin(set(pol["label"]))

    universe = {
        "n_plants": len(pl), "n_pollinators": len(pol),
        "n_interactions_modellable": int(keep.sum()),
        "n_interactions_total": len(edges),
        "excluded_pollinator_taxa": len(dropped),
        "excluded_order_conflict": int(ambiguous.sum()),
        "excluded_pollinator_orders": sorted(dropped["order"].dropna().unique().tolist()),
        "plants": [{"id": r.plant_id, "label": r.label, "rank": r["rank"],
                    "feature_source": r.feature_source, "has_congener": bool(r.has_congener)}
                   for _, r in pl.iterrows()],
        "pollinators": [{"id": r.pollinator_id, "label": r.label, "rank": r["rank"],
                         "feature_source": r.feature_source, "has_congener": bool(r.has_congener)}
                        for _, r in pol.iterrows()],
    }
    json.dump(universe, open(args.out, "w"), indent=1)

    for name, df in [("plants", pl), ("pollinators", pol)]:
        c = df["feature_source"].value_counts().to_dict()
        print(f"[{name:11s}] {len(df):6d}  " + "  ".join(f"{k}={v}" for k, v in sorted(c.items())))
    print(f"[excluded   ] {len(dropped):6d} pollinator taxa outside the core orders")
    print(f"[edges      ] {keep.sum():6d} / {len(edges)} modellable ({100 * keep.mean():.1f}%)")
    print(f"[wrote      ] {args.out}")


if __name__ == "__main__":
    main()
