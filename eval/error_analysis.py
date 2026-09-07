"""Where the best retrieval model fails, and whether the spatio-temporal signal could have helped.

Ten architecture variants have failed to beat the boosted ranker. Before proposing an eleventh, this
asks what the failures actually look like: whether misses are near or catastrophic, which plants they
concentrate in, whether the missed partners carry spatio-temporal signal the model is not using, and
how much of the apparent error is positive-unlabelled rather than real.
"""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from antheia.baselines.taxo_spatial_temporal import TaxoSpatialTemporal
from antheia.store import UniverseStore


def main():
    store = UniverseStore(curves="modelled")
    split = json.load(open(ROOT / "data/splits/plants_75_10_15.json"))
    e = pd.read_parquet(ROOT / "data/network/edges.parquet")
    e = e[e.plant.isin(store.p2i) & e.pollinator.isin(store.q2i)]
    train = e[e.plant.isin(set(split["train"]))]
    test = e[e.plant.isin(set(split["val"]))]; test = test[test.tier == "A"]
    tp = sorted(set(test.plant))
    part = {sp: set(store.idx_polls(g.pollinator)) for sp, g in test.groupby("plant")}

    m = TaxoSpatialTemporal().fit(train, store)
    S = np.vstack([m.score_plant(store.p2i[sp]) for sp in tp])
    order = np.argsort(-S, axis=1)
    rank_of = np.empty_like(order)
    np.put_along_axis(rank_of, order, np.arange(S.shape[1])[None, :].repeat(len(tp), 0), axis=1)

    # 1. how bad are the misses?
    ranks = np.concatenate([rank_of[i, sorted(part[sp])] + 1 for i, sp in enumerate(tp)])
    print(f"[1] true-partner rank distribution ({len(ranks):,} held-out partners of 13,124 candidates)")
    for lo, hi in [(1, 10), (11, 50), (51, 200), (201, 1000), (1001, 13124)]:
        f = ((ranks >= lo) & (ranks <= hi)).mean()
        print(f"      rank {lo:>5}-{hi:<5} {100 * f:5.1f}%")
    print(f"      median rank {int(np.median(ranks))}")

    # 2. which plants fail?
    u = json.load(open(ROOT / "data/network/modelled_universe.json"))
    src = {p["label"]: p["feature_source"] for p in u["plants"]}
    tax = pd.read_parquet(ROOT / "data/features/taxonomy.parquet")
    fam = dict(zip(tax.label, tax.family))
    train_gen = {s.split()[0] for s in set(train.plant)}
    rows = []
    for i, sp in enumerate(tp):
        r = rank_of[i, sorted(part[sp])] + 1
        rows.append(dict(plant=sp, degree=len(part[sp]), nrec10=(r <= 10).sum() / min(len(r), 10),
                         source=src.get(sp), genus_seen=sp.split()[0] in train_gen,
                         family=fam.get(sp)))
    d = pd.DataFrame(rows)
    print(f"\n[2] recall@10 by plant stratum")
    print(f"      genus seen in training : {d[d.genus_seen].nrec10.mean():.4f}  (n={d.genus_seen.sum()})")
    print(f"      genus unseen           : {d[~d.genus_seen].nrec10.mean():.4f}  (n={(~d.genus_seen).sum()})")
    for s, g in d.groupby("source"):
        print(f"      feature source {s:<14}: {g.nrec10.mean():.4f}  (n={len(g)})")
    for lo, hi in [(1, 2), (3, 10), (11, 50), (51, 10000)]:
        g = d[(d.degree >= lo) & (d.degree <= hi)]
        if len(g):
            print(f"      degree {lo:>3}-{hi:<5}        : {g.nrec10.mean():.4f}  (n={len(g)})")

    # 3. do the missed partners carry spatio-temporal signal the model is not using?
    CO = store.plant_proj @ store.poll_proj.T
    found, missed = [], []
    for i, sp in enumerate(tp):
        p = store.p2i[sp]
        pct = pd.Series(CO[p]).rank(pct=True).to_numpy()
        for q in sorted(part[sp]):
            (found if rank_of[i, q] < 10 else missed).append(pct[q])
    print(f"\n[3] per-cell co-activity percentile of the partner, within its plant's candidates")
    print(f"      partners found in top 10 : {np.mean(found):.3f}  (n={len(found):,})")
    print(f"      partners missed          : {np.mean(missed):.3f}  (n={len(missed):,})")
    print(f"      -> if these are close, the signal does not distinguish the misses")

    # 4. how much of the top-10 'error' is plausible but unrecorded?
    q_gen = np.array([s.split()[0] for s in store.polls])
    q_fam = np.array([fam.get(s, "UNK") for s in store.polls])
    same_gen = same_fam = fp = 0
    for i, sp in enumerate(tp):
        true = part[sp]
        tg = {q_gen[q] for q in true}; tf = {q_fam[q] for q in true}
        for q in order[i, :10]:
            if q in true:
                continue
            fp += 1
            same_gen += q_gen[q] in tg
            same_fam += q_fam[q] in tf
    print(f"\n[4] top-10 predictions that are not recorded partners: {fp:,}")
    print(f"      congeneric with a recorded partner : {100 * same_gen / fp:5.1f}%")
    print(f"      confamilial with a recorded partner: {100 * same_fam / fp:5.1f}%")
    print(f"      -> an upper bound on how much of the error is unrecorded rather than wrong")
    d.to_csv(ROOT / "results/error_analysis_val.csv", index=False)


if __name__ == "__main__":
    main()
