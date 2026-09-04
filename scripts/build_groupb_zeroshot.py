"""Group B species list for the pollinator SDM: network taxa with no occurrence records.

The SDM deliverable models a pollinator one of two ways. Group A taxa have GBIF occurrences and
get a supervised per-species head. Group B taxa are in the interaction network but absent from
the occurrence extract, so their activity curve comes from the LE-SINR zero-shot head conditioned
on a BioCLIP-2 text embedding of the Linnaean name.

Emits the CSV that `pipelines/sdm/text_embeddings.py --set zeroshot` consumes. Restricted to the
core flower-visiting orders; the remaining orders (Araneae, Psocodea, Mesostigmata, ...) are
predators and incidental visitors rather than pollinators and are excluded from the modelled universe.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

CORE_ORDERS = {"Hymenoptera", "Lepidoptera", "Diptera", "Coleoptera", "Hemiptera", "Apodiformes"}
ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nodes", default=ROOT / "data/network/nodes_pollinators.parquet")
    ap.add_argument("--occ", default="/scratch/cher/antheia-data/pollinator_sdm/pollinator_occ_gbifv3.npz")
    ap.add_argument("--out", default="/scratch/cher/antheia-data/pollinator_sdm/groupB_zeroshot_species.csv")
    args = ap.parse_args()

    occ = set(map(str, np.load(args.occ, allow_pickle=True)["names"]))
    pol = pd.read_parquet(args.nodes)
    # same membership rule as the modelled universe: core orders, and not a taxon whose order
    # the checklists disagree on
    ambiguous = pol.get("order_conflict", pd.Series(False, index=pol.index)).fillna(False)
    gb = pol[pol["order"].isin(CORE_ORDERS) & ~ambiguous & ~pol["label"].isin(occ)]
    out = pd.DataFrame({
        "sourceTaxonName": gb["label"].values,
        "sourceTaxonOrderName": gb["order"].fillna("").values,
        "sourceTaxonFamilyName": gb["family"].fillna("").values,
        "rank": gb["rank"].values,
    }).sort_values("sourceTaxonName")
    out.to_csv(args.out, index=False)
    print(f"[groupB] {len(pol)} network pollinators -> {len(out)} zero-shot taxa "
          f"({(~pol['order'].isin(CORE_ORDERS)).sum()} dropped as non-core order, "
          f"{pol['label'].isin(occ).sum()} have occurrences)")
    print(f"[groupB] wrote {args.out}")


if __name__ == "__main__":
    main()
