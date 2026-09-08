"""Run one fusion re-ranker arm (plan §3) and write its bundle.

Retrieve-then-re-rank: the retriever (an embedding-model config, normally the frozen retriever_v1) is
re-fit with the same seed -- its runs are bit-reproducible -- and scores every training plant and every
evaluated plant over all candidates. Its top-K per plant are the candidates the fusion model trains on
and re-scores. The bundle's score matrix is the re-ranked top-K with the retriever's scores, shifted
below the re-ranked block, everywhere else, so pooled metrics stay defined over all candidates.

  python eval/run_fusion.py --name M2.2_fusion --retriever-config '{...}' \
         --config '{"field_dir": ".../field_v2", "k": 128}' --split cold_plant --seed 42
"""
import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT))
from antheia.bundle import write_bundle
from antheia.embednet import EmbedRanker
from antheia.fusion import FusionReranker
from antheia.store import UniverseStore
from eval.run_ladder import BASE_EMBED, load_split


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--name", required=True)
    ap.add_argument("--retriever-config", required=True, help="JSON kwargs of the embedding-model retriever")
    ap.add_argument("--config", required=True, help="JSON FusionConfig kwargs (field_dir required)")
    ap.add_argument("--split", default="cold_plant", choices=["cold_plant", "cold_poll", "cold_both", "warm"])
    ap.add_argument("--part", default="val", choices=["val", "test"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--topk", type=int, default=500)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    rcfg = json.loads(args.retriever_config); fcfg = json.loads(args.config)
    store = UniverseStore(curves="modelled")
    sp = load_split(args.split, args.part)

    e = pd.read_parquet(ROOT / "data/network/edges.parquet")
    e = e[e.plant.isin(store.p2i) & e.pollinator.isin(store.q2i)]
    train = e.copy()
    if sp["train_plants"] is not None:
        train = train[train.plant.isin(sp["train_plants"])]
    if sp["train_polls"] is not None:
        train = train[train.pollinator.isin(sp["train_polls"])]
    if sp.get("drop_pairs"):
        keep = ~pd.Series(list(zip(train.plant, train.pollinator))).isin(sp["drop_pairs"]).to_numpy()
        train = train[keep]
    ev_plants = sp["eval_plants"]; cand = sp["cand_polls"] or list(store.polls); ci = store.idx_polls(cand)
    if args.split in ("cold_plant", "cold_both"):
        assert not set(ev_plants) & set(train.plant)
    if args.split in ("cold_poll", "cold_both"):
        assert not set(cand) & set(train.pollinator)
    A = e[e.tier == "A"]
    if args.split == "warm":
        held = {(p, q) for p, q in sp["held_pairs"]} & set(zip(A.plant, A.pollinator))
        part_of = {}
        for p, q in held:
            part_of.setdefault(p, set()).add(store.q2i[q])
    else:
        A = A[A.plant.isin(set(ev_plants)) & A.pollinator.isin(set(cand))]
        part_of = {p: set(store.idx_polls(g.pollinator)) for p, g in A.groupby("plant")}
    ev_plants = [p for p in ev_plants if part_of.get(p)]
    col = {int(q): j for j, q in enumerate(ci)}
    Y = np.zeros((len(ev_plants), len(ci)), np.int8)
    for i, p in enumerate(ev_plants):
        Y[i, [col[q] for q in part_of[p]]] = 1
    print(f"[{args.split}/{args.part}] train edges {len(train):,}; eval {len(ev_plants)} plants x {len(ci)} candidates, "
          f"{int(Y.sum()):,} positives", flush=True)

    # ---- retriever: fit, score train plants and eval plants over the candidate set --------------
    t0 = time.time()
    ret = EmbedRanker(**dict(BASE_EMBED, **rcfg, seed=args.seed, device=args.device)).fit(train, store)
    train_plants = sorted(set(train.plant)); tpi = store.idx_plants(train_plants)
    K = args.topk
    def topk_rows(idx):
        S = np.vstack([ret.score_plant(int(p))[ci] for p in idx]).astype(np.float32)
        order = np.argsort(-S, axis=1)[:, :K]
        return S, np.take_along_axis(S, order, 1), ci[order]                     # scores, topK scores, topK poll idx
    S_tr, s_tr, c_tr = topk_rows(tpi)
    S_ev, s_ev, c_ev = topk_rows(store.idx_plants(ev_plants))
    del S_tr
    # retriever recall@K on the evaluated plants (the re-ranker's ceiling)
    hits = sum(len(set(c_ev[i].tolist()) & set(ci[np.flatnonzero(Y[i])].tolist())) for i in range(len(ev_plants)))
    rec_k = hits / Y.sum()
    print(f"  retriever fit+scored in {time.time() - t0:.0f}s; recall@{K} on eval plants {rec_k:.3f}", flush=True)
    del ret; torch.cuda.empty_cache()

    # ---- fusion ---------------------------------------------------------------------------------
    t1 = time.time()
    fus = FusionReranker(**dict(fcfg, seed=args.seed, device=args.device)).fit(train, store, s_tr, c_tr, tpi)
    S = S_ev.copy()
    for i, p in enumerate(ev_plants):
        new, qidx = fus.rerank(store.p2i[p], s_ev[i], c_ev[i], len(ci))
        # everything outside the top-K keeps the retriever score, pushed below the re-ranked block
        outside = np.ones(len(ci), bool); cols = np.array([col[int(q)] for q in qidx]); outside[cols] = False
        S[i, outside] = S[i, outside] - (S[i, outside].max() - new.min()) - 1.0
        S[i, cols] = new
    wall = time.time() - t0
    strata = {"genus_unseen": np.array([p.split()[0] not in {s.split()[0] for s in train_plants} for p in ev_plants]),
              "low_degree": np.array([len(part_of[p]) <= 2 for p in ev_plants])}
    u = json.load(open(ROOT / "data/network/modelled_universe.json"))
    src = {p["label"]: p.get("feature_source", "direct") for p in u["plants"]}
    strata["zeroshot"] = np.array([src.get(p, "direct") != "direct" for p in ev_plants])
    cfg_all = {"model": "fusion", "retriever": rcfg, **fcfg, "topk": K}
    out, met = write_bundle(args.name, cfg_all, f"{args.split}/{args.part}", args.seed, S, Y, ev_plants, cand, strata,
                            extra={"retriever_recall_at_K": float(rec_k), "fusion_wall_s": time.time() - t1}, wall_s=wall)
    print(f"  AUPR {met['aupr']:.4f} (1:3 {met['aupr_at_0.25']:.3f}, 1:1 {met['aupr_at_0.5']:.3f})  AUROC {met['auroc']:.3f}  "
          f"nR@10 {met['nrecall_at_10']:.4f}  nR@50 {met['nrecall_at_50']:.4f}  unseen-genus nR@10 "
          f"{met.get('nrecall_at_10__genus_unseen', float('nan')):.4f}  ({wall:.0f}s)\n[bundle] {out}", flush=True)


if __name__ == "__main__":
    main()
