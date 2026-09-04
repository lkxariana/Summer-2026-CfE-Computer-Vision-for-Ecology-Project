"""Collapse nodes that are the same taxon under different checklist identifiers.

`build_edges.py` keys a node on the identifier it resolved from the GloBI record, preferring GBIF.
One name routinely carries several such identifiers -- synonyms, superseded keys, or the same genus
published in more than one checklist -- so the node tables hold 919 plant and 625 pollinator labels
that appear two or more times (`Aceria` under six distinct GBIF keys). Every feature source
downstream is keyed on the name: the opportunity surface, the occurrence extract, and the text
embeddings all look a taxon up by its binomial. Left uncollapsed, those rows become separate
candidates sharing one identical feature vector, splitting a taxon's true partners across nodes
and diluting any retrieval metric computed over them.

Rank never disagrees within a duplicated label, so (label, rank) is the merge key. Family and order
are filled from whichever member of the group carries them. Where members disagree on a non-null
value the field is flagged `<field>_conflict` rather than resolved by majority -- GBIF places
the gall-mite genus Eriophyes in both Trombidiformes/Eriophyidae and Diptera/Cecidomyiidae, and the
larger group is the wrong one. A family conflict is usually a revision that changed nothing else
(Hydrophyllaceae sunk into Boraginaceae) and is only recorded; an order conflict decides whether a
pollinator counts as a flower visitor at all, so the universe drops those taxa.

Edges are remapped onto canonical identifiers and deduplicated: counts sum, source and type sets
union, tier keeps the stronger evidence, and degree is recomputed from the collapsed edge list.
"""
import argparse
import shutil
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ID_PREF = {"GBIF": 0, "COL": 1, "ITIS": 2, "WFO": 3, "NAME": 4}
# edges_no_coordinates.parquet is name-keyed and holds dropped records; it carries no
# identifiers to remap.
EDGE_FILES = ["edges.parquet"]


def canonical_id(ids):
    return sorted(ids, key=lambda i: (ID_PREF.get(i.split(":")[0], 9), i))[0]


def merge_nodes(nodes, idcol, class_cols):
    """One row per (label, rank); returns the merged table and an old->new id map."""
    out, mapping = [], {}
    for (label, rank), g in nodes.groupby(["label", "rank"], dropna=False):
        cid = canonical_id(g[idcol].tolist())
        for old in g[idcol]:
            mapping[old] = cid
        row = {idcol: cid, "label": label, "rank": rank,
               "n_merged": len(g), "merged_ids": ",".join(sorted(g[idcol]))}
        for c in class_cols:
            vals = sorted(v for v in g[c].dropna().unique() if v != "")
            row[c] = vals[0] if vals else None
            row[f"{c}_conflict"] = len(vals) > 1
        out.append(row)
    return pd.DataFrame(out).sort_values("label").reset_index(drop=True), mapping


def union(series):
    return ",".join(sorted({t for v in series.astype(str) for t in v.split(",") if t and t != "nan"}))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--network", default=ROOT / "data/network")
    ap.add_argument("--backup", default=None, help="directory to copy the pre-merge artifacts into")
    args = ap.parse_args()
    net = Path(args.network)

    if args.backup:
        Path(args.backup).mkdir(parents=True, exist_ok=True)
        for f in EDGE_FILES + ["nodes_plants.parquet", "nodes_pollinators.parquet"]:
            shutil.copy2(net / f, Path(args.backup) / f)
        print(f"[backup] pre-merge artifacts -> {args.backup}")

    pl = pd.read_parquet(net / "nodes_plants.parquet")
    pol = pd.read_parquet(net / "nodes_pollinators.parquet")
    pl_new, pl_map = merge_nodes(pl, "plant_id", ["family"])
    pol_new, pol_map = merge_nodes(pol, "pollinator_id", ["family", "order"])

    agg = {"plant": "first", "pollinator": "first", "plant_rank": "first", "pollinator_rank": "first",
           "plant_family": "first", "pollinator_family": "first", "pollinator_order": "first",
           "tier": "min", "types": union, "sources": union, "role_source": union,
           "n_records": "sum", "n_observations": "sum", "n_inat": "sum",
           "first_year": "min", "last_year": "max"}

    for fname in EDGE_FILES:
        e = pd.read_parquet(net / fname)
        before = len(e)
        e["plant_id"] = e["plant_id"].map(pl_map)
        e["pollinator_id"] = e["pollinator_id"].map(pol_map)
        e = e.groupby(["plant_id", "pollinator_id"], as_index=False).agg(agg)
        e["n_sources"] = e["sources"].str.split(",").str.len()
        e.to_parquet(net / fname, index=False)
        print(f"[edges] {fname}: {before} -> {len(e)} ({before - len(e)} duplicate pairs collapsed)")
        if fname == EDGE_FILES[0]:
            pl_new["degree"] = pl_new["plant_id"].map(e.groupby("plant_id").size()).fillna(0).astype(int)
            pol_new["degree"] = pol_new["pollinator_id"].map(e.groupby("pollinator_id").size()).fillna(0).astype(int)

    pl_new.to_parquet(net / "nodes_plants.parquet", index=False)
    pol_new.to_parquet(net / "nodes_pollinators.parquet", index=False)
    for name, old, new in [("plants", pl, pl_new), ("pollinators", pol, pol_new)]:
        flags = [c for c in new.columns if c.endswith("_conflict")]
        print(f"[nodes] {name}: {len(old)} -> {len(new)}  "
              f"({int((new.n_merged > 1).sum())} merged groups)")
        for f in flags:
            lab = new.loc[new[f], "label"].tolist()
            print(f"        {f}: {len(lab)}" + (" -> " + ", ".join(lab) if 0 < len(lab) <= 8 else ""))


if __name__ == "__main__":
    main()
