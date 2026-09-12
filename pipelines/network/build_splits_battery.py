"""The four evaluation splits of the connectivity ladder (plan §1.4), frozen to data/splits/.

  cold_plant   the existing plants_75_10_15.json (S2): train plants x all pollinators -> val/test plants
  cold_poll    (S3) the frozen degree-stratified pollinators_75_10_15.json; evaluated pairs are
               train plants x val (test) pollinators, trained on all plants x train pollinators
  cold_both    (S4) plant split x pollinator split: trained on train x train, evaluated on val x val
  warm         (S1) an edge-level 85/15 hold-out among train plants x train pollinators (tier A+B
               edges for training, tier-A edges evaluated), seed 42; every zero pair among the same
               species is a candidate negative

Everything is derived from the frozen plant split plus one frozen pollinator split, so the four
splits are mutually consistent: a "train pollinator" means the same set in every split. Written as
one JSON per split with explicit train/val/test lists and, for `warm`, the held-out edge ids.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

from antheia.paths import REPO_ROOT as ROOT
SEED = 42


def stratified_split(items, degree, rng, frac=(0.75, 0.10, 0.15), bins=8):
    """Degree-stratified split: quantile bins of log-degree, each bin split by frac."""
    items = np.asarray(items); ld = np.log1p(np.asarray(degree, float))
    edges = np.unique(np.quantile(ld, np.linspace(0, 1, bins + 1)))
    strat = np.clip(np.searchsorted(edges, ld, side="right") - 1, 0, len(edges) - 2)
    out = {"train": [], "val": [], "test": []}
    for b in np.unique(strat):
        idx = rng.permutation(np.flatnonzero(strat == b))
        n = len(idx); a, c = int(round(frac[0] * n)), int(round((frac[0] + frac[1]) * n))
        out["train"] += items[idx[:a]].tolist(); out["val"] += items[idx[a:c]].tolist(); out["test"] += items[idx[c:]].tolist()
    return {k: sorted(v) for k, v in out.items()}


def main():
    rng = np.random.default_rng(SEED)
    out = ROOT / "data/splits"
    plants = json.load(open(out / "plants_75_10_15.json"))
    u = json.load(open(ROOT / "data/network/modelled_universe.json"))
    polls_all = [p["label"] for p in u["pollinators"]]
    e = pd.read_parquet(ROOT / "data/network/edges.parquet")
    plants = {k: v for k, v in plants.items() if isinstance(v, list)}
    e = e[e.pollinator.isin(set(polls_all)) & e.plant.isin(set(sum(plants.values(), [])))].reset_index(drop=True)

    # pollinator split: the frozen one already in the repo (seed 42, degree-stratified, 5 strata)
    polls = json.load(open(out / "pollinators_75_10_15.json"))
    polls = {k: v for k, v in polls.items() if isinstance(v, list)}

    # warm: edge-level hold-out among train plants x train pollinators
    tp, tq = set(plants["train"]), set(polls["train"])
    ww = e[e.plant.isin(tp) & e.pollinator.isin(tq)]
    pairs = ww[["plant", "pollinator"]].drop_duplicates().reset_index(drop=True)
    perm = rng.permutation(len(pairs))
    n_val, n_test = int(0.10 * len(pairs)), int(0.15 * len(pairs))
    warm = {"train_plants": sorted(tp), "train_polls": sorted(tq),
            "val_pairs": pairs.iloc[perm[:n_val]].values.tolist(),
            "test_pairs": pairs.iloc[perm[n_val:n_val + n_test]].values.tolist(),
            "train_pairs": pairs.iloc[perm[n_val + n_test:]].values.tolist()}
    json.dump(warm, open(out / "warm_edges_75_10_15.json", "w"))

    summary = {
        "cold_plant": {k: len(v) for k, v in plants.items()},
        "cold_poll": {k: len(v) for k, v in polls.items()},
        "cold_both": {"train": f"{len(plants['train'])} x {len(polls['train'])}",
                      "val": f"{len(plants['val'])} x {len(polls['val'])}",
                      "test": f"{len(plants['test'])} x {len(polls['test'])}"},
        "warm": {"species": f"{len(tp)} x {len(tq)}", "train_pairs": len(warm["train_pairs"]),
                 "val_pairs": n_val, "test_pairs": n_test},
    }
    # tier-A positives per evaluated universe, for the prevalence column
    A = e[e.tier == "A"]
    summary["val_tierA_positives"] = {
        "cold_plant": int(A.plant.isin(set(plants["val"])).sum()),
        "cold_poll": int((A.plant.isin(tp) & A.pollinator.isin(set(polls["val"]))).sum()),
        "cold_both": int((A.plant.isin(set(plants["val"])) & A.pollinator.isin(set(polls["val"]))).sum()),
    }
    json.dump(summary, open(out / "battery_summary.json", "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
