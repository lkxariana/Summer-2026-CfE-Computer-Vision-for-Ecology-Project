"""Prospective holdout: train on pairs first recorded before a cutoff year, predict the pairs first recorded from the cutoff on.

  python scripts/build_prospective_split.py --cutoff 2024

Writes data/splits/prospective_<cutoff>.json with: train pairs (first_year < cutoff, all tiers), eval pairs (tier A,
first_year >= cutoff), eval plants (those with >= 1 eval pair), and the two strata -- plants / pollinators with no
pre-cutoff record at all ("prospective cold"). Pairs with unknown first_year are kept for training only.
"""
import argparse, json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cutoff", type=int, default=2024)
    a = ap.parse_args()
    e = pd.read_parquet(ROOT / "data/network/edges.parquet")
    known = e.first_year.notna()
    train = e[~known | (e.first_year < a.cutoff)]
    ev = e[known & (e.first_year >= a.cutoff) & (e.tier == "A")]
    ev = ev[~ev.set_index(["plant", "pollinator"]).index.isin(train.set_index(["plant", "pollinator"]).index)]
    cold_plants = sorted(set(ev.plant) - set(train.plant)); cold_polls = sorted(set(ev.pollinator) - set(train.pollinator))
    out = dict(cutoff=a.cutoff, train_pairs=[list(x) for x in zip(train.plant, train.pollinator)],
               eval_pairs=[list(x) for x in zip(ev.plant, ev.pollinator)], eval_plants=sorted(set(ev.plant)),
               cold_plants=cold_plants, cold_polls=cold_polls,
               summary=dict(n_train=len(train), n_eval=len(ev), n_eval_plants=ev.plant.nunique(), n_cold_plants=len(cold_plants),
                            n_cold_polls=len(cold_polls), n_eval_pairs_with_cold_plant=int(ev.plant.isin(cold_plants).sum()),
                            n_eval_pairs_with_cold_poll=int(ev.pollinator.isin(cold_polls).sum())))
    p = ROOT / "data/splits" / f"prospective_{a.cutoff}.json"; json.dump(out, open(p, "w"))
    print(json.dumps(out["summary"], indent=1)); print("[wrote]", p.relative_to(ROOT))


if __name__ == "__main__":
    main()
