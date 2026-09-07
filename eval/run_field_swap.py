"""Learned field embedding against the SVD grid projection, pollinator side, same downstream model.

The pollinator SDM's species head (`data/features/poll_field.npy`) is a learned spatio-temporal
influence vector; `poll_surf_proj.npy` is the SVD projection of that same model's output surface.
Both are 256-D, so they can be swapped under an otherwise identical embedding model. The arms:

  reference          text + surface + pca + scale on both sides
  field replaces     pollinator surface -> field
  field added        pollinator surface + field
  field, no impute   as "field replaces", but text-imputed rows zeroed (only trained vectors)
  no-text surface    surface + pca + scale, no text on either side   (taxonomy-free control)
  no-text field      same, pollinator surface -> field

Per-plant nrecall@10 is kept for every arm so later seeds can be pooled, and each arm is split by
whether the plant's genus appears in training -- the stratum where the boosted ranker collapses.
"""
import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from antheia.embednet import EmbedRanker
from antheia.metrics import paired_bootstrap, retrieval_metrics
from antheia.store import UniverseStore

FULL = ("text", "surface", "pca", "scale")
NOTX = ("surface", "pca", "scale")
RUNS = [
    ("reference",        dict(blocks=FULL)),
    ("field replaces",   dict(blocks=FULL, blocks_q=("text", "field", "pca", "scale"))),
    ("field added",      dict(blocks=FULL, blocks_q=("text", "surface", "field", "pca", "scale"))),
    ("field, no impute", dict(blocks=FULL, blocks_q=("text", "field", "pca", "scale"), field_impute=False)),
    ("no-text surface",  dict(blocks=NOTX)),
    ("no-text field",    dict(blocks=NOTX, blocks_q=("field", "pca", "scale"))),
]


def joint_runs(field_dirs):
    """Arms for the joint two-kingdom field model: the same vector space on both sides, trained
    rows only (untrained taxa are zero and the per-block LayerNorm maps them to zero)."""
    runs = [("reference", dict(blocks=FULL))]
    for d in field_dirs:
        tag = Path(d).name.replace("joint_field_", "")
        runs.append((f"joint {tag}", dict(blocks=("text", "field", "pca", "scale"), field_path=str(d), field_impute=False)))
        runs.append((f"joint {tag} + surface", dict(blocks=("text", "surface", "field", "pca", "scale"),
                                                     field_path=str(d), field_impute=False)))
        runs.append((f"no-text joint {tag}", dict(blocks=("field", "pca", "scale"), field_path=str(d), field_impute=False)))
    return runs


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--only", nargs="*", default=None, help="subset of arm names")
    ap.add_argument("--preset", default="swap", choices=["swap", "joint"])
    ap.add_argument("--field-dirs", nargs="*", default=None, help="joint_field_<scheme> directories (preset joint)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    global RUNS
    if args.preset == "joint":
        RUNS = joint_runs(args.field_dirs)

    store = UniverseStore(curves="modelled")
    split = json.load(open(ROOT / "data/splits/plants_75_10_15.json"))
    e = pd.read_parquet(ROOT / "data/network/edges.parquet")
    e = e[e.plant.isin(store.p2i) & e.pollinator.isin(store.q2i)]
    train = e[e.plant.isin(set(split["train"]))]
    test = e[e.plant.isin(set(split["val"]))]; test = test[test.tier == "A"]
    tp = sorted(set(test.plant))
    part = {sp: set(store.idx_polls(g.pollinator)) for sp, g in test.groupby("plant")}
    train_gen = {s.split()[0] for s in set(train.plant)}
    seen = np.array([sp.split()[0] in train_gen for sp in tp])
    Y = np.zeros((len(tp), len(store.polls)), np.int8)
    for i, sp in enumerate(tp):
        Y[i, list(part[sp])] = 1
    print(f"[data] val {len(test):,} tier-A edges over {len(tp)} plants; genus seen {seen.sum()} / "
          f"unseen {(~seen).sum()}; seed {args.seed}", flush=True)

    ref, rows, per_plant = None, [], {}
    for label, kw in RUNS:
        if args.only and label not in args.only:
            continue
        cfg = dict(epochs=args.epochs, use_genus_context=False, use_tier_head=False,
                   seed=args.seed, device=args.device)
        cfg.update(kw)
        t0 = time.time()
        m = EmbedRanker(**cfg).fit(train, store)
        S = np.vstack([m.score_plant(store.p2i[sp]) for sp in tp])
        pp = pd.DataFrame([retrieval_metrics(S[i], part[sp], ks=(10, 50)) for i, sp in enumerate(tp)])
        pr = average_precision_score(Y.ravel(), S.ravel().astype(np.float64))
        nr = pp["nrecall@10"].to_numpy()
        per_plant[label] = nr
        line = (f"  {label:<18} nR@10 {nr.mean():.4f}  seen {nr[seen].mean():.4f}  unseen {nr[~seen].mean():.4f}  "
                f"nR@50 {pp['nrecall@50'].mean():.4f}  MAP {pp['ap'].mean():.4f}  PR {pr:.4f}")
        if ref is not None:
            d, lo, hi, pv = paired_bootstrap(nr, ref, 2000, 42)
            line += f"   vs ref {d:+.4f} [{lo:+.4f},{hi:+.4f}] p={pv:.3f}"
        print(line + f"  ({time.time() - t0:.0f}s)", flush=True)
        rows.append(dict(arm=label, seed=args.seed, nrecall10=nr.mean(), nrecall10_seen=nr[seen].mean(),
                         nrecall10_unseen=nr[~seen].mean(), nrecall50=pp["nrecall@50"].mean(),
                         map=pp["ap"].mean(), pr_auc=pr))
        if ref is None:
            ref = nr
        del m; torch.cuda.empty_cache()

    out = ROOT / "results"
    stem = "field_swap" if args.preset == "swap" else "joint_field_swap"
    pd.DataFrame(rows).to_csv(out / f"{stem}_val_tierA_s{args.seed}.csv", index=False)
    np.savez(out / f"{stem}_perplant_s{args.seed}.npz", plants=np.array(tp), seen=seen, **per_plant)
    print(f"[wrote] results/{stem}_val_tierA_s{args.seed}.csv")


if __name__ == "__main__":
    main()
