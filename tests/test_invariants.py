import sys
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from antheia.globi import binomial, build_edges
from antheia.pairs import kfold_plants, plant_split, sample_negatives


def test_week_formula():
    doy = pd.Series([1, 7, 8, 14, 15, 361, 365, 366])
    week = ((doy - 1) // 7).clip(0, 51)
    assert week.tolist() == [0, 0, 1, 1, 2, 51, 51, 51]


def test_bin_conventions():
    lat, lon = 34.7, -119.8
    b = f"{round(np.floor(lat / 0.5) * 0.5, 1)}_{round(np.floor(lon / 0.5) * 0.5, 1)}"
    assert b == "34.5_-120.0"
    assert round(24.75 - 0.25, 2) == 24.5  # SDM/PPE centroid -> bin corner offset


def test_binomial():
    assert binomial("Apis mellifera Linnaeus, 1758") == "Apis mellifera"
    assert binomial("Bombus (Pyrobombus) impatiens") == "Bombus impatiens"
    assert binomial("Achillea millefolium var. occidentalis") == "Achillea millefolium"
    assert binomial("Solidago") is None
    assert binomial(None) is None


def test_build_edges_orientation():
    plants = {"Planta alba", "Plantb bella"}
    polls = {"Bugx unus", "Bugy duo"}
    records = pd.DataFrame({
        "sourceTaxonName": ["Bugx unus", "Bugx unus", "Planta alba", "Bugy duo Smith, 1900",
                            "Mystery thing", "Planta alba"],
        "targetTaxonName": ["Planta alba", "Planta alba", "Bugy duo", "Plantb bella",
                            "Whatever else", "Planta alba"],
        "interactionTypeName": ["visitsFlowersOf", "pollinates", "visitedBy", "visits", "visits", "x"],
    })
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "globi.csv"
        records.to_csv(path, index=False)
        edges, stats = build_edges(path, plants, polls)

    assert stats["records_in"] == 5          # self-pair removed
    assert stats["records_as_is"] == 1
    assert stats["records_swapped"] == 3     # 2 exact + 1 via binomial normalization
    assert stats["records_dropped"] == 1
    assert stats["unique_edges"] == 3
    e = edges.set_index(["plant", "pollinator"])
    assert e.loc[("Planta alba", "Bugx unus"), "n_records"] == 2
    assert e.loc[("Planta alba", "Bugx unus"), "types"] == "pollinates,visitsFlowersOf"
    assert e.loc[("Plantb bella", "Bugy duo"), "n_records"] == 1

    try:
        build_edges(path, plants | {"Bugx unus"}, polls)
        assert False, "overlap must raise"
    except ValueError:
        pass


def _toy_edges(n_plants=8):
    rows = []
    for i in range(n_plants):
        for j in range(i + 1):
            rows.append({"plant": f"P{i}", "pollinator": f"B{j}"})
    return pd.DataFrame(rows)


def test_plant_split_and_kfold():
    edges = _toy_edges()
    split = plant_split(edges, seed=0, test_frac=0.5)
    all_plants = set(edges["plant"])
    assert set(split["train"]) | set(split["test"]) == all_plants
    assert not set(split["train"]) & set(split["test"])
    assert split == plant_split(edges, seed=0, test_frac=0.5)

    folds = kfold_plants(edges, k=2, seed=0)
    assert set(folds[0]) | set(folds[1]) == all_plants
    assert not set(folds[0]) & set(folds[1])


class _StubStore:
    def __init__(self):
        self.polls = [f"B{i}" for i in range(6)]
        self.plants = [f"P{i}" for i in range(4)]
        self.p2i = {s: i for i, s in enumerate(self.plants)}
        self.N_full = np.ones((4, 6), dtype=np.uint16)


def test_sample_negatives_deterministic_and_clean():
    store = _StubStore()
    pos = pd.DataFrame({"plant": ["P0", "P1", "P2"], "pollinator": ["B0", "B1", "B2"]})
    known = set(zip(pos["plant"], pos["pollinator"]))
    a = sample_negatives(pos, store.plants, store, 3, "mixed", 7, known)
    b = sample_negatives(pos, store.plants, store, 3, "mixed", 7, known)
    assert a.equals(b)
    assert len(a) == 9
    assert not set(zip(a["plant"], a["pollinator"])) & known
    assert len(a.drop_duplicates()) == len(a)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"{len(fns)} tests passed")
