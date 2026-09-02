import sys
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path("artifacts/v2")
PLANT_RANKS = {"species", "genus"}
e = pd.read_parquet(OUT / "edges.parquet")
pl = pd.read_parquet(OUT / "nodes_plants.parquet")
po = pd.read_parquet(OUT / "nodes_pollinators.parquet")


def test_no_self_pairs():
    assert (e.plant_id != e.pollinator_id).all()
    assert (e.plant != e.pollinator).all()


def test_edges_unique():
    assert not e.duplicated(subset=["plant_id", "pollinator_id"]).any()


def test_referential_integrity():
    assert set(e.plant_id) <= set(pl.plant_id)
    assert set(e.pollinator_id) <= set(po.pollinator_id)


def test_ranks_are_species_or_genus():
    assert set(e.plant_rank) <= PLANT_RANKS
    assert set(e.pollinator_rank) <= PLANT_RANKS


def test_node_sets_disjoint():
    """No identifier may denote both a plant and a pollinator (cross-kingdom homonyms)."""
    overlap = set(e.plant_id) & set(e.pollinator_id)
    detail = [(i, e[e.plant_id == i].plant.iloc[0], e[e.pollinator_id == i].pollinator.iloc[0])
              for i in list(overlap)[:5]]
    assert not overlap, f"{len(overlap)} identifiers on both sides: {detail}"


def test_tier_reflects_best_evidence():
    """tier is the best tier supporting an edge: A iff at least one tier-A record exists."""
    A = {"visitsFlowersOf", "pollinates"}
    has_a = e.types.apply(lambda t: bool(set(t.split(",")) & A))
    assert (e.tier == "A").equals(has_a), "tier does not match presence of tier-A evidence"
    b = e[e.tier == "B"]
    assert not b.types.apply(lambda t: bool(set(t.split(",")) & A)).any()


def test_counts_coherent():
    assert (e.n_records >= 1).all()
    assert (e.n_sources >= 1).all()
    assert (e.n_inat <= e.n_records).all()
    assert (e.n_observations <= e.n_records).all()


def test_years_plausible():
    y = e.dropna(subset=["first_year", "last_year"])
    assert (y.first_year <= y.last_year).all()
    # Museum specimen records legitimately predate 1900; future dates are errors.
    assert y.first_year.min() >= 1700, f"earliest year {y.first_year.min()}"
    assert y.last_year.max() <= pd.Timestamp.utcnow().year, f"future year {y.last_year.max()}"


def test_no_records_invented():
    led = pd.read_csv(OUT / "yield_ledger.csv")
    surviving = int(led[led.step == 8].records_out.iloc[0])
    assert e.n_records.sum() <= surviving, f"{e.n_records.sum():,} > {surviving:,}"


def test_sources_consistent_with_count():
    n = e.sources.fillna("").apply(lambda s: len([x for x in s.split(",") if x]))
    assert (n == e.n_sources).all()


def test_ledger_monotone():
    led = pd.read_csv(OUT / "yield_ledger.csv")
    for _, r in led.iterrows():
        assert r.records_out <= r.records_in, f"step {r.step} grew: {r.records_in}->{r.records_out}"


def test_node_degree_matches_edges():
    d = e.groupby("plant_id").size()
    m = pl.set_index("plant_id")["degree"]
    assert (d.sort_index() == m.sort_index()).all()


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except AssertionError as ex:
            failed += 1
            print(f"FAIL {fn.__name__}: {ex}")
    print(f"\n{len(fns)-failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
