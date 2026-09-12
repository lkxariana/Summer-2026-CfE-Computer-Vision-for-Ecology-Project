"""Run one (model, config, split, seed) of the connectivity ladder and write its bundle (plan §8).

  python -m antheia.eval.ladder --model embednet --name retriever --config '{"softmax_weight": 0.0}' \
         --split cold_plant --seed 42

Splits (plan §1.4): cold_plant (val plants x all pollinators), cold_poll (train plants x val
pollinators), cold_both (val plants x val pollinators), warm (held-out edges among train plants x
train pollinators; candidates = all train pollinators). Training edges are whatever the split leaves
in train (tiers A+B); evaluation positives are tier A.

Strata written for every query: genus_unseen (plant genus absent from training), zeroshot (plant's
feature_source is not "direct"), low_degree (<= 2 tier-A partners).
"""
import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

from antheia.paths import REPO_ROOT as ROOT
from antheia.baselines import REGISTRY
from antheia.bundle import write_bundle
from antheia.store import UniverseStore
from antheia import negpool

BASE_EMBED = dict(epochs=25)


def load_split(name, part):
    sp = ROOT / "data/splits"
    plants = {k: v for k, v in json.load(open(sp / "plants_75_10_15.json")).items() if isinstance(v, list)}
    polls = {k: v for k, v in json.load(open(sp / "pollinators_75_10_15.json")).items() if isinstance(v, list)}
    if name == "cold_plant":
        return dict(train_plants=set(plants["train"]), train_polls=None, eval_plants=sorted(plants[part]), cand_polls=None, held_pairs=None)
    if name == "cold_poll":
        return dict(train_plants=None, train_polls=set(polls["train"]), eval_plants=sorted(plants["train"]),
                    cand_polls=sorted(polls[part]), held_pairs=None)
    if name == "cold_both":
        return dict(train_plants=set(plants["train"]), train_polls=set(polls["train"]), eval_plants=sorted(plants[part]),
                    cand_polls=sorted(polls[part]), held_pairs=None)
    if name == "warm":
        w = json.load(open(sp / "warm_edges_75_10_15.json"))
        held = {tuple(x) for x in w[f"{part}_pairs"]}
        return dict(train_plants=set(w["train_plants"]), train_polls=set(w["train_polls"]),
                    eval_plants=sorted({p for p, _ in held}), cand_polls=sorted(w["train_polls"]), held_pairs=held,
                    drop_pairs={tuple(x) for x in w["val_pairs"] + w["test_pairs"]})
    if name.startswith("prospective"):
        w = json.load(open(sp / f"{name}.json"))
        held = {tuple(x) for x in w["eval_pairs"]}
        return dict(train_plants=None, train_polls=None, eval_plants=sorted(w["eval_plants"]), cand_polls=None, held_pairs=held,
                    drop_pairs=held, train_pairs={tuple(x) for x in w["train_pairs"]}, cold_plants=set(w["cold_plants"]))
    raise ValueError(name)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="REGISTRY key, e.g. embednet, routed, popularity")
    ap.add_argument("--name", default=None, help="bundle name (defaults to --model)")
    ap.add_argument("--config", default="{}", help="JSON of model kwargs")
    ap.add_argument("--split", default="cold_plant", choices=["cold_plant", "cold_poll", "cold_both", "warm", "prospective_2024"])
    ap.add_argument("--part", default="val", choices=["val", "test"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--topk", type=int, default=500, help="candidates saved per query (retriever role)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    cfg = json.loads(args.config)
    name = args.name or args.model
    store = UniverseStore(curves="modelled")
    sp = load_split(args.split, args.part)
    negpool.set_pool(store.idx_polls(sp["train_polls"]) if sp["train_polls"] is not None else None)

    e = pd.read_parquet(ROOT / "data/network/edges.parquet")
    e = e[e.plant.isin(store.p2i) & e.pollinator.isin(store.q2i)]
    train = e.copy()
    if sp["train_plants"] is not None:
        train = train[train.plant.isin(sp["train_plants"])]
    if sp["train_polls"] is not None:
        train = train[train.pollinator.isin(sp["train_polls"])]
    if sp.get("train_pairs") is not None:
        train = train[pd.Series(list(zip(train.plant, train.pollinator))).isin(sp["train_pairs"]).to_numpy()]
    if sp.get("drop_pairs"):
        keep = ~pd.Series(list(zip(train.plant, train.pollinator))).isin(sp["drop_pairs"]).to_numpy()
        train = train[keep]
    # leakage invariant: nothing evaluated may be trained on
    ev_plants = sp["eval_plants"]
    cand = sp["cand_polls"] or list(store.polls)
    ci = store.idx_polls(cand)
    if args.split in ("cold_plant", "cold_both"):
        assert not set(ev_plants) & set(train.plant), "evaluated plants leak into training"
    if args.split in ("cold_poll", "cold_both"):
        assert not set(cand) & set(train.pollinator), "evaluated pollinators leak into training"

    A = e[e.tier == "A"]
    if args.split == "warm" or args.split.startswith("prospective"):
        pos = {(p, q) for p, q in sp["held_pairs"] if (p, q) in set(zip(A.plant, A.pollinator))}
        part_of = {}
        for p, q in pos:
            part_of.setdefault(p, set()).add(store.q2i[q])
        assert not pos & set(zip(train.plant, train.pollinator)), "held-out pairs leak into training"
    else:
        A = A[A.plant.isin(set(ev_plants)) & A.pollinator.isin(set(cand))]
        part_of = {p: set(store.idx_polls(g.pollinator)) for p, g in A.groupby("plant")}
    ev_plants = [p for p in ev_plants if part_of.get(p)]
    col = {int(q): j for j, q in enumerate(ci)}
    Y = np.zeros((len(ev_plants), len(ci)), np.int8)
    for i, p in enumerate(ev_plants):
        Y[i, [col[q] for q in part_of[p]]] = 1
    exclude = None
    if args.split == "warm" or args.split.startswith("prospective"):
        # filtered ranking: an evaluated plant's known pairs (training edges and the other part's held pairs) are not negatives
        known = {}
        for p, q in zip(train.plant, train.pollinator):
            known.setdefault(p, set()).add(q)
        for p, q in sp["drop_pairs"]:
            if (p, q) not in pos:
                known.setdefault(p, set()).add(q)
        exclude = np.zeros(Y.shape, bool)
        for i, p in enumerate(ev_plants):
            js = [col[store.q2i[q]] for q in known.get(p, ()) if q in store.q2i and store.q2i[q] in col]
            exclude[i, js] = True
        assert not (exclude & (Y == 1)).any(), "held-out positives marked as excluded"
        print(f"[warm] {int(exclude.sum()):,} known pairs excluded from the candidate rankings", flush=True)
    print(f"[{args.split}/{args.part}] train edges {len(train):,}; eval {len(ev_plants)} plants x {len(ci)} candidates, "
          f"{int(Y.sum()):,} tier-A positives (prevalence {Y.mean():.5f})", flush=True)

    # strata
    train_gen = {s.split()[0] for s in set(train.plant)}
    u = json.load(open(ROOT / "data/network/modelled_universe.json"))
    src = {p["label"]: p.get("feature_source", "direct") for p in u["plants"]}
    strata = {"genus_unseen": np.array([p.split()[0] not in train_gen for p in ev_plants]),
              "zeroshot": np.array([src.get(p, "direct") != "direct" for p in ev_plants]),
              "low_degree": np.array([len(part_of[p]) <= 2 for p in ev_plants])}
    if sp.get("cold_plants") is not None:
        strata["prospective_cold"] = np.array([p in sp["cold_plants"] for p in ev_plants])

    t0 = time.time()
    Model = REGISTRY[args.model]
    kw = dict(BASE_EMBED, **cfg, seed=args.seed, device=args.device) if args.model == "embednet" else dict(cfg)
    try:
        m = Model(**kw).fit(train, store)
    except TypeError:
        m = Model().fit(train, store)
    S = np.vstack([m.score_plant(store.p2i[p])[ci] for p in ev_plants]).astype(np.float32)
    wall = time.time() - t0
    topk = np.argsort(-S, axis=1)[:, :args.topk]
    out, met = write_bundle(name, {"model": args.model, **cfg}, f"{args.split}/{args.part}", args.seed, S, Y, ev_plants, cand,
                            strata, candidates_topk=ci[topk], wall_s=wall, exclude=exclude)
    print(f"  AUPR {met['aupr']:.4f} (1:3 {met['aupr_at_0.25']:.3f}, 1:1 {met['aupr_at_0.5']:.3f})  AUROC {met['auroc']:.3f}  "
          f"nR@10 {met['nrecall_at_10']:.4f}  nR@50 {met['nrecall_at_50']:.4f}  unseen-genus nR@10 "
          f"{met.get('nrecall_at_10__genus_unseen', float('nan')):.4f}  ({wall:.0f}s)\n[bundle] {out}", flush=True)


if __name__ == "__main__":
    main()
