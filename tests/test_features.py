"""Invariants for the feature caches in data/features/.

Everything is indexed to modelled_universe.json. A row that silently shifts against that order
corrupts every downstream result without raising, so shape and ordering are checked first.
"""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FEAT = ROOT / "data/features"

U = json.load(open(ROOT / "data/network/modelled_universe.json"))
NP_, NQ = len(U["plants"]), len(U["pollinators"])
F = np.load(FEAT / "F.npy"); P = np.load(FEAT / "P.npy")
FCo = np.load(FEAT / "FCo.npy"); ACo = np.load(FEAT / "ACo.npy")
N = np.load(FEAT / "N.npy", mmap_mode="r")
Frs = np.load(FEAT / "Frs.npy"); Prs = np.load(FEAT / "Prs.npy")
grid = pd.read_parquet(FEAT / "grid.parquet")
cov = pd.read_parquet(FEAT / "coverage.parquet")
tax = pd.read_parquet(FEAT / "taxonomy.parquet")
rng = np.random.default_rng(0)


def test_shapes_match_universe():
    nc = len(grid)
    assert F.shape == (NP_, nc) and P.shape == (NQ, nc)
    assert FCo.shape == (NP_, 52) and ACo.shape == (NQ, 52)
    assert N.shape == (NP_, NQ)
    assert Frs.shape == (NP_,) and Prs.shape == (NQ,)


def test_grid_is_compact_and_ordered():
    assert (grid["col"].to_numpy() == np.arange(len(grid))).all()
    assert grid["cell_idx"].is_monotonic_increasing
    assert grid["cell_idx"].is_unique


def test_curves_normalised():
    for C in (FCo, ACo):
        tot = C.sum(1)
        assert np.all((np.isclose(tot, 1, atol=1e-5)) | np.isclose(tot, 0))
        assert (C >= 0).all() and not np.isnan(C).any()


def test_curve_support_matches_occupancy():
    """A taxon has a curve exactly when it has occupied cells -- both come from the same records."""
    assert ((FCo.sum(1) > 0) == (F.sum(1) > 0)).all()
    assert ((ACo.sum(1) > 0) == (P.sum(1) > 0)).all()


def test_range_sizes_match():
    assert (Frs == F.sum(1)).all() and (Prs == P.sum(1)).all()


def test_cooccurrence_matches_products():
    """N is F @ P.T; recompute a random sample of pairs directly."""
    pi = rng.integers(0, NP_, 200); qi = rng.integers(0, NQ, 200)
    assert (np.asarray(N[pi[:, None], qi[None, :]]) ==
            (F[pi].astype(np.int32) @ P[qi].astype(np.int32).T)).all()


def test_cooccurrence_bounded_by_ranges():
    pi = rng.integers(0, NP_, 100)
    assert (np.asarray(N[pi]).max(1) <= Frs[pi]).all()
    assert int(np.asarray(N[rng.integers(0, NP_, 100)]).max()) < np.iinfo(np.uint16).max


def test_observed_taxa_are_the_direct_ones():
    """Only taxa the models were fit on can carry records; a populated zero-shot row means the
    universe's feature_source disagrees with the caches."""
    src = np.array([p["feature_source"] for p in U["plants"]])
    assert F[src != "direct"].sum(1).mean() < 1.0
    assert (F[src == "direct"].sum(1) > 0).mean() > 0.95


def test_tables_align_with_universe():
    assert len(tax) == NP_ + NQ and len(cov) == NP_ + NQ
    assert tax["label"].tolist()[:NP_] == [p["label"] for p in U["plants"]]
    assert cov["label"].tolist()[NP_:] == [p["label"] for p in U["pollinators"]]
    assert tax["rank"].notna().all() and tax["genus"].notna().all()


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
