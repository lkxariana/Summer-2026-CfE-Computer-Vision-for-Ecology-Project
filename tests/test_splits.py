"""Invariants for the frozen splits in data/splits/.

A split that leaks a test taxon into training inflates every score reported against it, silently.
"""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SP = ROOT / "data/splits"

U = json.load(open(ROOT / "data/network/modelled_universe.json"))
PLANTS = {p["label"] for p in U["plants"]}
POLLS = {p["label"] for p in U["pollinators"]}
E = pd.read_parquet(ROOT / "data/network/edges.parquet")
E = E[E["plant"].isin(PLANTS) & E["pollinator"].isin(POLLS)]
PL = json.load(open(SP / "plants_75_10_15.json"))
PO = json.load(open(SP / "pollinators_75_10_15.json"))


def test_partitions_are_disjoint():
    for s in (PL, PO):
        tr, va, te = set(s["train"]), set(s["val"]), set(s["test"])
        assert not (tr & va) and not (tr & te) and not (va & te)


def test_partitions_cover_the_universe():
    for s, universe in ((PL, PLANTS), (PO, POLLS)):
        assert set(s["train"]) | set(s["val"]) | set(s["test"]) | set(s["no_interactions"]) == universe


def test_every_split_taxon_has_interactions():
    for s, col in ((PL, "plant"), (PO, "pollinator")):
        deg = E[col].value_counts()
        for part in ("train", "val", "test"):
            assert deg.reindex(s[part]).fillna(0).gt(0).all()
        assert not set(s["no_interactions"]) & set(deg.index)


def test_degree_distributions_are_comparable():
    """Stratification exists so held-out taxa are not systematically rarer than trained ones."""
    for s, col in ((PL, "plant"), (PO, "pollinator")):
        deg = E[col].value_counts()
        med = [np.median(deg.reindex(s[p]).to_numpy()) for p in ("train", "val", "test")]
        assert max(med) <= 2 * min(med), f"{col} median degrees {med}"


def test_proportions_are_as_declared():
    for s in (PL, PO):
        n = sum(len(s[p]) for p in ("train", "val", "test"))
        assert abs(len(s["train"]) / n - 0.75) < 0.01
        assert abs(len(s["test"]) / n - 0.15) < 0.01


def test_source_holdouts_are_sole_supported():
    for f in ("holdout_webofline.json", "holdout_gbifusbees.json"):
        h = json.load(open(SP / f))
        pairs = {tuple(p) for p in h["pairs"]}
        got = E[E["sources"] == h["source"]]
        assert {tuple(r) for r in got[["plant", "pollinator"]].values} == pairs
        assert len(pairs) > 0


def test_temporal_split_is_ordered_and_disjoint():
    t = json.load(open(SP / "temporal_2020.json"))
    tr = {tuple(p) for p in t["train_pairs"]}
    te = {tuple(p) for p in t["test_pairs"]}
    assert not (tr & te)
    yr = E.set_index(["plant", "pollinator"])["first_year"]
    assert yr.reindex(list(tr)).max() <= t["cutoff"] < yr.reindex(list(te)).min()


def test_both_new_block_is_the_intersection():
    b = json.load(open(SP / "both_new.json"))
    assert b["test_plants"] == PL["test"] and b["test_pollinators"] == PO["test"]


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
