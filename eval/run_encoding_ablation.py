"""Table 3: what is lost when an axis is summarised into a scalar before modelling.

One vehicle throughout -- the same gradient-boosted pair ranker -- so that a row's difference comes
from the representation it is given and nothing else. Within a panel each row substitutes the
representation of one axis while a fixed control block holds the others constant. The temporal panel
always carries co-occurrence, because without it any phenological effect is readable as an abundance
proxy (Blanchet, Cazelles & Gravel 2020, *Ecology Letters* 23:1050).

Delta is the paired bootstrap difference against the row above, over the same held-out plants.

Every implicit row is paired with a **species-permutation control**: the same representation, of the
same dimensionality, with the vectors shuffled between species. A richer representation hands the
booster more columns, and a 52-week curve is close to a unique fingerprint, so a gain can come from
the extra capacity acting as a species identity code rather than from the axis carrying information.
Only the margin over the permuted control is attributable to the representation.
"""
import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from antheia.metrics import bootstrap_mean, paired_bootstrap, retrieval_metrics
from antheia.store import UniverseStore

KS = (10, 20)
EPS = 1e-12


def _pca(store, k=15, seed=42):
    f = TruncatedSVD(k, random_state=seed).fit_transform(store.F.astype(np.float32))
    p = TruncatedSVD(k, random_state=seed).fit_transform(store.P.astype(np.float32))
    return f / (np.linalg.norm(f, axis=1, keepdims=True) + EPS), \
           p / (np.linalg.norm(p, axis=1, keepdims=True) + EPS)


def overlap_stats(f, a):
    """Seven standard niche-overlap statistics between two normalised 52-week curves."""
    fs, as_ = f + EPS, a + EPS
    peak = np.abs(f.argmax(1) - a.argmax(1))
    return np.column_stack([
        np.minimum(f, a).sum(1),                                             # overlap coefficient
        1 - 0.5 * np.abs(f - a).sum(1),                                      # Schoener D
        (f * a).sum(1) / np.sqrt((f ** 2).sum(1) * (a ** 2).sum(1) + EPS),   # Pianka O
        np.sqrt(fs * as_).sum(1),                                            # Bhattacharyya
        np.sqrt(np.clip(1 - np.sqrt(fs * as_).sum(1), 0, 1)),                # Hellinger
        0.5 * ((fs * np.log(fs / ((fs + as_) / 2))).sum(1) +
               (as_ * np.log(as_ / ((fs + as_) / 2))).sum(1)),               # Jensen-Shannon
        np.minimum(peak, 52 - peak),                                         # circular peak lag
    ])


def build_rows(store):
    Fpca, Ppca = _pca(store)
    N = store.N_full
    cooc = lambda pi, qi: np.log1p(np.asarray(N[pi, qi], dtype=np.float64))[:, None]
    pop = lambda pi, qi: np.log1p(store.Prs[qi])[:, None]

    def jaccard(pi, qi):
        n = np.asarray(N[pi, qi], dtype=np.float64)
        return (n / (store.Frs[pi] + store.Prs[qi] - n + EPS))[:, None]

    spatial_control = lambda pi, qi: pop(pi, qi)
    temporal_control = lambda pi, qi: np.hstack([cooc(pi, qi), pop(pi, qi)])

    rng = np.random.default_rng(0)
    Fs, Ps = Fpca[rng.permutation(len(Fpca))], Ppca[rng.permutation(len(Ppca))]
    FCs = store.FC[rng.permutation(len(store.FC))]
    ACs = store.AC[rng.permutation(len(store.AC))]

    A = [("explicit", "shared occupied cells (count)", cooc),
         ("explicit", "range overlap (Jaccard)", jaccard),
         ("implicit", "occupancy embedding (PCA, 15D)",
          lambda pi, qi: np.hstack([Fpca[pi] * Ppca[qi], (Fpca[pi] * Ppca[qi]).sum(1)[:, None]])),
         ("control", "  same, species-permuted",
          lambda pi, qi: np.hstack([Fs[pi] * Ps[qi], (Fs[pi] * Ps[qi]).sum(1)[:, None]]))]
    B = [("explicit", "overlap coefficient sum min(f,a)",
          lambda pi, qi: np.minimum(store.FC[pi], store.AC[qi]).sum(1)[:, None]),
         ("explicit", "seven overlap statistics",
          lambda pi, qi: overlap_stats(store.FC[pi], store.AC[qi])),
         ("implicit", "raw 52-week curves",
          lambda pi, qi: np.hstack([store.FC[pi], store.AC[qi]])),
         ("control", "  same, species-permuted",
          lambda pi, qi: np.hstack([FCs[pi], ACs[qi]])),
         ("implicit", "per-week product f*a",
          lambda pi, qi: store.FC[pi] * store.AC[qi]),
         ("control", "  same, species-permuted",
          lambda pi, qi: FCs[pi] * ACs[qi])]
    return [("A - spatial", spatial_control, A), ("B - temporal", temporal_control, B)]


def run_row(store, feats, train, test_plants, partners, seed, n_neg, bootstrap):
    rng = np.random.default_rng(seed)
    pi, qi = store.idx_plants(train["plant"]), store.idx_polls(train["pollinator"])
    known = set(zip(pi.tolist(), qi.tolist()))
    np_, nq_ = np.repeat(pi, n_neg), rng.integers(0, len(store.polls), len(pi) * n_neg)
    keep = [i for i, (a, b) in enumerate(zip(np_, nq_)) if (a, b) not in known]
    np_, nq_ = np_[keep], nq_[keep]
    X = np.vstack([feats(pi, qi), feats(np_, nq_)])
    y = np.concatenate([np.ones(len(pi)), np.zeros(len(np_))])
    clf = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, min_samples_leaf=100,
                                         l2_regularization=5.0, random_state=seed).fit(X, y)
    nq = len(store.polls)
    S = np.empty((len(test_plants), nq), dtype=np.float32)
    for r, sp in enumerate(test_plants):
        S[r] = clf.predict_proba(feats(np.full(nq, store.p2i[sp]), np.arange(nq)))[:, 1]
    recs = [retrieval_metrics(S[r], partners[sp], ks=KS) for r, sp in enumerate(test_plants)]
    pp = pd.DataFrame(recs)
    Y = np.zeros(S.shape, np.int8)
    for r, sp in enumerate(test_plants):
        Y[r, list(partners[sp])] = 1
    m, lo, hi, _ = bootstrap_mean(pp["nrecall@10"].to_numpy(), bootstrap, seed)
    return pp, {"nrecall@10": m, "lo": lo, "hi": hi, "recall@10": pp["recall@10"].mean(),
                "map": pp["ap"].mean(), "ndcg@10": pp["ndcg@10"].mean(),
                "pr_auc": average_precision_score(Y.ravel(), S.ravel().astype(np.float64))}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--part", default="val", choices=["val", "test"])
    ap.add_argument("--eval-tier", default="A", choices=["A", "AB"])
    ap.add_argument("--n-neg", type=int, default=10)
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=ROOT / "results")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    store = UniverseStore(curves="modelled")
    split = json.load(open(ROOT / "data/splits/plants_75_10_15.json"))
    e = pd.read_parquet(ROOT / "data/network/edges.parquet")
    e = e[e["plant"].isin(store.p2i) & e["pollinator"].isin(store.q2i)]
    train = e[e["plant"].isin(set(split["train"]))]
    test = e[e["plant"].isin(set(split[args.part]))]
    if args.eval_tier == "A":
        test = test[test["tier"] == "A"]
    test_plants = sorted(set(test["plant"]))
    partners = {sp: set(store.idx_polls(g["pollinator"])) for sp, g in test.groupby("plant")}
    print(f"[data] train {len(train):,} | {args.part} {len(test):,} over {len(test_plants)} plants "
          f"| curves FC={store.curve_sources['FC']} AC={store.curve_sources['AC']}", flush=True)

    rows, prev = [], {}
    for panel, control, spec in build_rows(store):
        print(f"\n[{panel}]", flush=True)
        prev_pp = None
        for kind, label, block in spec:
            t0 = time.time()
            feats = lambda pi, qi, b=block, c=control: np.hstack([c(pi, qi), b(pi, qi)])
            pp, r = run_row(store, feats, train, test_plants, partners,
                            args.seed, args.n_neg, args.bootstrap)
            if prev_pp is not None and kind != "control":
                d, lo, hi, p = paired_bootstrap(pp["nrecall@10"].to_numpy(),
                                                prev_pp["nrecall@10"].to_numpy(),
                                                args.bootstrap, args.seed)
                r.update(delta=d, delta_lo=lo, delta_hi=hi, delta_p=p)
            r.update(panel=panel, kind=kind, representation=label)
            if kind == "control" and rows:
                d, lo, hi, pv = paired_bootstrap(prev_pp["nrecall@10"].to_numpy(),
                                                 pp["nrecall@10"].to_numpy(),
                                                 args.bootstrap, args.seed)
                r.update(delta=d, delta_lo=lo, delta_hi=hi, delta_p=pv)
            rows.append(r)
            if kind != "control":
                prev_pp = pp
            print(f"  {kind:<8} {label:<34} nR@10 {r['nrecall@10']:.4f} MAP {r['map']:.4f} "
                  f"PR {r['pr_auc']:.4f}" +
                  (f"  D {r['delta']:+.4f} p={r['delta_p']:.3f}" if "delta" in r else "") +
                  f"  ({time.time()-t0:.0f}s)", flush=True)

    df = pd.DataFrame(rows)[["panel", "kind", "representation", "nrecall@10", "lo", "hi",
                             "recall@10", "map", "ndcg@10", "pr_auc",
                             "delta", "delta_lo", "delta_hi", "delta_p"]]
    dest = out / f"encoding_ablation_{args.part}_tier{args.eval_tier}.csv"
    df.to_csv(dest, index=False)
    print(f"\n[wrote] {dest}")


if __name__ == "__main__":
    main()
