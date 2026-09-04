"""Frozen evaluation splits. Written once, before any model runs, and never regenerated per model.

Cold-start link prediction is evaluated by holding out *taxa*, not interactions: a held-out plant
must be unseen during training, so that predicting its partners exercises the features rather than
its memorised row. This is Setting B of Stock et al. (2018) on the plant side, Setting C on the
pollinator side, and Setting D where the two coincide.

  plants_75_10_15.json      leave-plant-out, degree-stratified          (primary)
  pollinators_75_10_15.json leave-pollinator-out, degree-stratified
  both_new.json             the induced held-out x held-out block
  holdout_webofline.json    interactions supported only by Web of Life
  holdout_gbifusbees.json   interactions supported only by gbif-us-bees
  temporal_2020.json        first recorded <= 2020 for training, later for test

Degree stratification keeps the degree distribution of the test set comparable to training. Without
it a random split over a distribution this skewed puts nearly all high-degree taxa on one side, and
retrieval scores move with test-set degree regardless of the model.

Taxa with no modellable interaction cannot be evaluated and are recorded separately; they stay in
the universe, since they remain candidates a model may rank.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
BINS = [0, 1, 2, 5, 10, 30, np.inf]
FRACTIONS = (0.75, 0.10, 0.15)


def stratified_split(degree, rng):
    """75/10/15 within each degree stratum; returns (train, val, test) label lists."""
    out = ([], [], [])
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        members = degree[(degree > lo) & (degree <= hi)].index.to_numpy()
        rng.shuffle(members)
        n = len(members)
        a, b = int(round(n * FRACTIONS[0])), int(round(n * (FRACTIONS[0] + FRACTIONS[1])))
        for bucket, part in zip(out, (members[:a], members[a:b], members[b:])):
            bucket.extend(part.tolist())
    return [sorted(b) for b in out]


def write(path, payload):
    json.dump(payload, open(path, "w"), indent=1)
    print(f"[wrote] {path.name}: " +
          "  ".join(f"{k}={len(v)}" for k, v in payload.items() if isinstance(v, list)))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--network", default=ROOT / "data/network")
    ap.add_argument("--universe", default=ROOT / "data/network/modelled_universe.json")
    ap.add_argument("--out", default=ROOT / "data/splits")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    u = json.load(open(args.universe))
    plants = {p["label"] for p in u["plants"]}
    polls = {p["label"] for p in u["pollinators"]}
    e = pd.read_parquet(Path(args.network) / "edges.parquet")
    e = e[e["plant"].isin(plants) & e["pollinator"].isin(polls)].reset_index(drop=True)
    print(f"[edges] {len(e)} modellable interactions")

    for side, col, fname in [("plant", "plant", "plants_75_10_15.json"),
                             ("pollinator", "pollinator", "pollinators_75_10_15.json")]:
        deg = e[col].value_counts()
        universe = plants if side == "plant" else polls
        isolated = sorted(universe - set(deg.index))
        tr, va, te = stratified_split(deg, np.random.default_rng(SEED))
        write(out / fname, {"seed": SEED, "side": side, "stratum_edges": BINS[1:-1],
                            "train": tr, "val": va, "test": te, "no_interactions": isolated})

    pl = json.load(open(out / "plants_75_10_15.json"))
    po = json.load(open(out / "pollinators_75_10_15.json"))
    for name, key in [("val", "val"), ("test", "test")]:
        block = e[e["plant"].isin(set(pl[key])) & e["pollinator"].isin(set(po[key]))]
        print(f"[both_new] {name}: {len(block)} interactions")
    write(out / "both_new.json", {"note": "induced block; both taxa unseen (Stock et al. Setting D)",
                                  "val_plants": pl["val"], "val_pollinators": po["val"],
                                  "test_plants": pl["test"], "test_pollinators": po["test"]})

    for src, fname in [("globalbioticinteractions/web-of-life", "holdout_webofline.json"),
                       ("globalbioticinteractions/gbif-us-bees", "holdout_gbifusbees.json")]:
        sole = e[e["sources"] == src]
        write(out / fname, {"source": src,
                            "note": "interactions this source alone supports; removed from training",
                            "pairs": sole[["plant", "pollinator"]].values.tolist()})

    yr = e["first_year"]
    tr = e[yr.notna() & (yr <= 2020)]
    te = e[yr.notna() & (yr > 2020)]
    write(out / "temporal_2020.json",
          {"cutoff": 2020, "note": "prospective; interactions with no year are excluded from both",
           "train_pairs": tr[["plant", "pollinator"]].values.tolist(),
           "test_pairs": te[["plant", "pollinator"]].values.tolist(),
           "undated": int(yr.isna().sum())})


if __name__ == "__main__":
    main()
