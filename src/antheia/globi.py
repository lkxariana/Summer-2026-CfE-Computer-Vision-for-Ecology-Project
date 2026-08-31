import re
import pandas as pd

_PAREN = re.compile(r"\([^)]*\)")
_EPITHET = re.compile(r"^[a-z][a-z-]+$")


def binomial(name):
    """Reduces a taxon name to 'Genus species' (dropping authorship/subgenus), or None if not resolvable to species level."""
    if not isinstance(name, str):
        return None
    parts = _PAREN.sub(" ", name).split()
    if len(parts) >= 2 and parts[0][:1].isupper() and _EPITHET.match(parts[1]):
        return f"{parts[0]} {parts[1]}"
    return None


def build_edges(globi_csv, plants, polls):
    """Extracts orientation-corrected unique (plant, pollinator) edges from a GloBI record file.

    GloBI flower-visitation records are predominantly pollinator→plant oriented but a noise tail is
    reversed, so each record is oriented by species-set membership (exact name first, then binomial
    normalization). Records where neither orientation resolves are dropped and counted.

    Args:
        globi_csv: path to the raw GloBI CONUS records csv.
        plants: set of plant species names with feature coverage.
        polls: set of pollinator species names with feature coverage.

    Returns:
        (edges, stats): edges has columns plant, pollinator, n_records, types; stats is a dict of per-step record counts.
    """
    overlap = set(plants) & set(polls)
    if overlap:
        raise ValueError(f"plant/pollinator sets overlap ({len(overlap)} names) — orientation would be ambiguous")

    df = pd.read_csv(globi_csv, usecols=["sourceTaxonName", "targetTaxonName", "interactionTypeName"])
    df = df.dropna(subset=["sourceTaxonName", "targetTaxonName"])
    df = df[df["sourceTaxonName"] != df["targetTaxonName"]]
    src, tgt = df["sourceTaxonName"], df["targetTaxonName"]

    pl_by_binom = {binomial(x): x for x in plants}
    po_by_binom = {binomial(x): x for x in polls}
    pl_by_binom.pop(None, None)
    po_by_binom.pop(None, None)

    src_pl = src.where(src.isin(plants)).fillna(src.map(binomial).map(pl_by_binom))
    src_po = src.where(src.isin(polls)).fillna(src.map(binomial).map(po_by_binom))
    tgt_pl = tgt.where(tgt.isin(plants)).fillna(tgt.map(binomial).map(pl_by_binom))
    tgt_po = tgt.where(tgt.isin(polls)).fillna(tgt.map(binomial).map(po_by_binom))

    asis = src_pl.notna() & tgt_po.notna()
    swap = src_po.notna() & tgt_pl.notna() & ~asis
    stats = {
        "records_in": len(df),
        "records_as_is": int(asis.sum()),
        "records_swapped": int(swap.sum()),
        "records_dropped": int((~(asis | swap)).sum()),
        "dropped_genus_only_tgt": int((~(asis | swap) & tgt.map(binomial).isna()).sum()),
    }

    oriented = pd.concat([
        pd.DataFrame({"plant": src_pl[asis], "pollinator": tgt_po[asis], "type": df.loc[asis, "interactionTypeName"]}),
        pd.DataFrame({"plant": tgt_pl[swap], "pollinator": src_po[swap], "type": df.loc[swap, "interactionTypeName"]}),
    ], ignore_index=True)

    edges = (oriented.groupby(["plant", "pollinator"])
             .agg(n_records=("type", "size"), types=("type", lambda t: ",".join(sorted(set(t)))))
             .reset_index())
    stats["unique_edges"] = len(edges)
    stats["n_plants"] = edges["plant"].nunique()
    stats["n_pollinators"] = edges["pollinator"].nunique()
    return edges, stats
