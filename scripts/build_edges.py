import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

LAT, LON = (24.0, 49.5), (-125.0, -66.0)
TIER = {"visitsFlowersOf": "A", "pollinates": "A", "visits": "B", "interactsWith": "B"}
PLANT_K = {"Plantae", "Archaeplastida", "Viridiplantae"}
ANIMAL_K = {"Animalia", "Metazoa"}
PLANT_P = {"Tracheophyta", "Streptophyta", "Magnoliophyta", "Anthophyta"}
ANIMAL_P = {"Arthropoda", "Chordata"}
IMMATURE = ("larva", "caterpillar", "nymph", "pupa", "egg", "instar", "juvenile", "immature")
ID_PREF = ("GBIF", "COL", "WFO", "ITIS")

COLS = ["interactionTypeName", "decimalLatitude", "decimalLongitude", "sourceNamespace",
        "referenceCitation", "eventDate", "sourceLifeStageName",
        "sourceCatalogNumber", "sourceInstitutionCode",
        "sourceTaxonIds", "sourceTaxonName", "sourceTaxonRank", "sourceTaxonSpeciesName",
        "sourceTaxonGenusName", "sourceTaxonFamilyName", "sourceTaxonOrderName",
        "sourceTaxonKingdomName", "sourceTaxonPhylumName",
        "targetTaxonIds", "targetTaxonName", "targetTaxonRank", "targetTaxonSpeciesName",
        "targetTaxonGenusName", "targetTaxonFamilyName", "targetTaxonOrderName",
        "targetTaxonKingdomName", "targetTaxonPhylumName"]


def pick_id(ids):
    """First identifier matching the authority preference order."""
    if not isinstance(ids, str):
        return None
    toks = [t.strip() for t in ids.split("|")]
    for pref in ID_PREF:
        for t in toks:
            if t.startswith(pref + ":") and not t.endswith(":no:match"):
                return t
    return None


def role_of(kingdom, phylum, names_plant, names_animal, name):
    """Vectorised elsewhere; kept for reference in tests."""
    raise NotImplementedError


class Ledger:
    def __init__(self):
        self.rows = []

    def add(self, step, op, rin, rout, **kw):
        self.rows.append({"step": step, "operation": op, "records_in": rin,
                          "records_out": rout, **kw})
        extra = "  ".join(f"{k}={v:,}" if isinstance(v, int) else f"{k}={v}" for k, v in kw.items())
        print(f"[{step:>2}] {op:<28} in={rin:>12,} out={rout:>12,}  {extra}", flush=True)

    def frame(self):
        return pd.DataFrame(self.rows)


def main():
    ap = argparse.ArgumentParser(description="Build the CONUS plant-pollinator interaction network")
    ap.add_argument("--globi", default="artifacts/external/globi_2026-08-26.csv.gz")
    ap.add_argument("--out", default="data/network")
    ap.add_argument("--chunk", type=int, default=1_500_000)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    led = Ledger()
    t0 = time.time()

    # ---- Steps 1-4: stream, filter to region, tier types, assign roles, orient -----------------
    kept, n_read, n_conus, n_nocoord, n_typed, n_violation, n_ambig = [], 0, 0, 0, 0, 0, 0
    nocoord_rows = []
    for chunk in pd.read_csv(args.globi, usecols=COLS, chunksize=args.chunk,
                             low_memory=False, on_bad_lines="skip"):
        n_read += len(chunk)
        typed = chunk[chunk["interactionTypeName"].isin(TIER)]
        n_typed += len(typed)
        has_xy = typed["decimalLatitude"].notna() & typed["decimalLongitude"].notna()
        inbox = has_xy & typed["decimalLatitude"].between(*LAT) & typed["decimalLongitude"].between(*LON)
        n_conus += int(inbox.sum())
        n_nocoord += int((~has_xy).sum())
        for df, bucket in ((typed[inbox], kept), (typed[~has_xy], nocoord_rows)):
            if not len(df):
                continue
            d = df.copy()
            # Step 3: roles from kingdom, then phylum, recording which rule fired.
            sk, tk = d["sourceTaxonKingdomName"], d["targetTaxonKingdomName"]
            sp_, tp_ = d["sourceTaxonPhylumName"], d["targetTaxonPhylumName"]
            src_animal = sk.isin(ANIMAL_K)
            tgt_plant = tk.isin(PLANT_K)
            src_plant = sk.isin(PLANT_K)
            tgt_animal = tk.isin(ANIMAL_K)
            fb_src_animal = ~sk.isin(ANIMAL_K | PLANT_K) & sp_.isin(ANIMAL_P)
            fb_tgt_plant = ~tk.isin(ANIMAL_K | PLANT_K) & tp_.isin(PLANT_P)
            d["role_source"] = np.where(src_animal & tgt_plant, "kingdom",
                                np.where((src_animal | fb_src_animal) & (tgt_plant | fb_tgt_plant),
                                         "phylum", "unresolved"))
            # Step 4: canonical direction is visitor -> plant. Keep only rows consistent with it.
            ok = (src_animal | fb_src_animal) & (tgt_plant | fb_tgt_plant)
            viol = (src_plant & tgt_animal)
            n_violation += int(viol.sum())
            n_ambig += int((~ok & ~viol).sum())
            bucket.append(d[ok])
    led.add(1, "read snapshot", n_read, n_read)
    led.add(2, "interaction-type selection", n_read, n_typed)
    led.add(3, "region filter", n_typed, n_conus, no_coordinates=n_nocoord)
    df = pd.concat(kept, ignore_index=True) if kept else pd.DataFrame(columns=COLS)
    led.add(4, "role + orientation", n_conus, len(df),
            orientation_violations=n_violation, unresolved_roles=n_ambig)
    del kept

    # ---- Step 5: identifiers -------------------------------------------------------------------
    df["pollinator_id"] = df["sourceTaxonIds"].map(pick_id)
    df["plant_id"] = df["targetTaxonIds"].map(pick_id)
    before = len(df)
    df = df[df["pollinator_id"].notna() & df["plant_id"].notna()]
    led.add(5, "identifier resolution", before, len(df))

    # An authority identifier must denote exactly one taxon. Where an id resolves to more than one
    # name across the corpus (GloBI id lists occasionally carry several ids per record, and ambiguous
    # genera can align inconsistently), the id is not a safe node key: fall back to the taxon name.
    lab = pd.concat([
        pd.DataFrame({"id": df["plant_id"], "name": df["targetTaxonName"]}),
        pd.DataFrame({"id": df["pollinator_id"], "name": df["sourceTaxonName"]}),
    ])
    multi = set(lab.groupby("id")["name"].nunique().pipe(lambda x: x[x > 1]).index)
    if multi:
        df["plant_id"] = np.where(df["plant_id"].isin(multi), "NAME:" + df["targetTaxonName"].astype(str),
                                  df["plant_id"])
        df["pollinator_id"] = np.where(df["pollinator_id"].isin(multi),
                                       "NAME:" + df["sourceTaxonName"].astype(str), df["pollinator_id"])
    # A name or identifier shared across kingdoms denotes two different taxa (e.g. Lucilia is both an
    # Asteraceae genus and a blowfly genus). Role assignment has already established that one side is
    # a plant and the other an animal, so qualifying the key by kingdom separates them correctly.
    shared = set(df["plant_id"]) & set(df["pollinator_id"])
    if shared:
        df["plant_id"] = np.where(df["plant_id"].isin(shared),
                                  df["plant_id"] + "|" + df["targetTaxonKingdomName"].fillna("Plantae"),
                                  df["plant_id"])
        df["pollinator_id"] = np.where(df["pollinator_id"].isin(shared),
                                       df["pollinator_id"] + "|" + df["sourceTaxonKingdomName"].fillna("Animalia"),
                                       df["pollinator_id"])
    led.add(5.5, "identifier disambiguation", len(df), len(df),
            ambiguous_ids=len(multi), cross_kingdom_homonyms=len(shared))

    # ---- Step 6: lowest available rank (species else genus) ------------------------------------
    df["plant"] = df["targetTaxonSpeciesName"].fillna(df["targetTaxonGenusName"])
    df["pollinator"] = df["sourceTaxonSpeciesName"].fillna(df["sourceTaxonGenusName"])
    df["plant_rank"] = np.where(df["targetTaxonSpeciesName"].notna(), "species", "genus")
    df["pollinator_rank"] = np.where(df["sourceTaxonSpeciesName"].notna(), "species", "genus")
    before = len(df)
    df = df[df["plant"].notna() & df["pollinator"].notna()]
    led.add(6, "rank policy (lowest available)", before, len(df),
            plant_genus_rows=int((df["plant_rank"] == "genus").sum()),
            pollinator_genus_rows=int((df["pollinator_rank"] == "genus").sum()))

    # ---- Step 7: deduplicate -------------------------------------------------------------------
    before = len(df)
    df = df.drop_duplicates()
    after_exact = len(df)
    key1 = df["referenceCitation"].notna()
    a = df[key1].drop_duplicates(subset=["referenceCitation", "pollinator_id", "plant_id",
                                         "interactionTypeName"])
    rest = df[~key1]
    key2 = rest["sourceCatalogNumber"].notna() & rest["sourceInstitutionCode"].notna()
    b = rest[key2].drop_duplicates(subset=["sourceCatalogNumber", "sourceInstitutionCode",
                                           "pollinator_id", "plant_id"])
    c = rest[~key2].drop_duplicates(subset=["pollinator_id", "plant_id", "interactionTypeName",
                                            "decimalLatitude", "decimalLongitude", "eventDate"])
    df = pd.concat([a, b, c], ignore_index=True)
    led.add(7, "deduplication", before, len(df), after_exact_dedup=after_exact,
            removed_pct=f"{(before - len(df)) / max(before,1):.1%}")

    # ---- Step 8: life stage --------------------------------------------------------------------
    ls = df["sourceLifeStageName"].fillna("").str.lower()
    immature = ls.str.contains("|".join(IMMATURE), regex=True)
    before = len(df)
    df = df[~immature]
    led.add(8, "life-stage filter", before, len(df), immature_dropped=int(immature.sum()),
            lifestage_annotated=f"{(ls != '').mean():.1%}")

    # ---- Step 9: aggregate ---------------------------------------------------------------------
    df["tier"] = df["interactionTypeName"].map(TIER)
    yr = pd.to_datetime(df["eventDate"], errors="coerce", format="mixed", utc=True).dt.year
    this_year = pd.Timestamp.utcnow().year
    df["year"] = yr.where((yr >= 1700) & (yr <= this_year))
    df["is_inat"] = df["sourceNamespace"].fillna("").str.contains("inaturalist", case=False)
    g = df.groupby(["plant_id", "pollinator_id"], sort=False)
    edges = g.agg(
        plant=("plant", "first"), pollinator=("pollinator", "first"),
        plant_rank=("plant_rank", "first"), pollinator_rank=("pollinator_rank", "first"),
        plant_family=("targetTaxonFamilyName", "first"),
        pollinator_family=("sourceTaxonFamilyName", "first"),
        pollinator_order=("sourceTaxonOrderName", "first"),
        tier=("tier", "min"), types=("interactionTypeName", lambda s: ",".join(sorted(set(s)))),
        n_records=("plant", "size"), n_observations=("referenceCitation", "nunique"),
        n_sources=("sourceNamespace", "nunique"),
        sources=("sourceNamespace", lambda s: ",".join(sorted(set(s.dropna())))),
        n_inat=("is_inat", "sum"), first_year=("year", "min"), last_year=("year", "max"),
        role_source=("role_source", "first"),
    ).reset_index()
    led.add(9, "aggregate to edges", len(df), len(df), edges=len(edges),
            plants=edges["plant_id"].nunique(), pollinators=edges["pollinator_id"].nunique())

    # ---- Step 10: emit -------------------------------------------------------------------------
    edges.to_parquet(out / "edges.parquet", index=False)
    for side, idc, namec, rankc, famc in (("plants", "plant_id", "plant", "plant_rank", "plant_family"),
                                          ("pollinators", "pollinator_id", "pollinator",
                                           "pollinator_rank", "pollinator_family")):
        n = edges.groupby(idc).agg(label=(namec, "first"), rank=(rankc, "first"),
                                   family=(famc, "first"), degree=(idc, "size")).reset_index()
        if side == "pollinators":
            n = n.merge(edges.groupby(idc)["pollinator_order"].first().rename("order"),
                        left_on=idc, right_index=True, how="left")
        n.to_parquet(out / f"nodes_{side}.parquet", index=False)
    if nocoord_rows:
        nc = pd.concat(nocoord_rows, ignore_index=True)
        nc["plant"] = nc["targetTaxonSpeciesName"].fillna(nc["targetTaxonGenusName"])
        nc["pollinator"] = nc["sourceTaxonSpeciesName"].fillna(nc["sourceTaxonGenusName"])
        nc = nc[nc["plant"].notna() & nc["pollinator"].notna()][["plant", "pollinator",
                                                                 "interactionTypeName", "sourceNamespace"]]
        nc.drop_duplicates().to_parquet(out / "edges_no_coordinates.parquet", index=False)
        led.add(10, "metaweb (no coordinates)", len(nc), len(nc.drop_duplicates()))
    led.frame().to_csv(out / "yield_ledger.csv", index=False)
    print(f"\nwrote {len(edges):,} edges to {out} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
