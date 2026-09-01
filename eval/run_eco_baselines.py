import argparse
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from antheia.config import load_config
from antheia.globi import binomial
from antheia.metrics import bootstrap_mean
from antheia.pairs import load_split
from antheia.store import FeatureStore
from antheia.taxonomy import family_map, genus


def pollinator_abundance(cfg, store):
    """Observation counts per pollinator species from the raw GBIF file (Vazquez/Dormann neutral model).

    For within-plant ranking the plant abundance term is constant, so only the pollinator side ranks.
    """
    cache = cfg["cache_dir"] / "poll_obs_counts.npy"
    if cache.exists():
        return np.load(cache)
    counts = {}
    path = cfg["data_root"] / "Plant Pollinator Initial Analysis" / "pollinator_observations_v2.csv"
    for chunk in pd.read_csv(path, usecols=["pollinator_species"], chunksize=2_000_000, low_memory=False):
        for sp, c in chunk["pollinator_species"].value_counts().items():
            counts[sp] = counts.get(sp, 0) + int(c)
    arr = np.array([counts.get(sp, 0) for sp in store.polls], dtype=np.float64)
    np.save(cache, arr)
    return arr


def main():
    ap = argparse.ArgumentParser(description="External ecological baselines: abundance, congeneric transfer, latent-trait SVD")
    ap.add_argument("--config", default=None)
    ap.add_argument("--rank", type=int, default=32, help="SVD latent dimension")
    args = ap.parse_args()

    cfg = load_config(args.config)
    store = FeatureStore(cfg)
    edges = pd.read_parquet(cfg["edges"])
    prov = pd.read_csv(cfg["edges"].parent / "edge_provenance_v1.csv")
    edges = edges.merge(prov, on=["plant", "pollinator"], how="left")
    edges["tier1"] = edges["n_inat"].notna() & (edges["n_inat"] < edges["n"])
    split = load_split(cfg["edges"].parent / "split_v1.json")
    seed, nboot = cfg["eval"]["seed"], cfg["eval"]["bootstrap_n"]
    train_pos = edges[edges["plant"].isin(split["train"])]
    test_pos = edges[edges["plant"].isin(split["test"])]
    n_po = len(store.polls)
    t0 = time.time()

    # --- taxonomy of plants: genus + family -----------------------------------------------------
    pfam = family_map(cfg["paths"]["globi"], set(store.plants))
    p_gen = np.array([genus(s) for s in store.plants])
    p_fam = np.array([pfam.get(s, "UNK") for s in store.plants])
    tr_pi = store.idx_plants(train_pos["plant"])
    tr_qi = store.idx_polls(train_pos["pollinator"])

    # --- 1. abundance null (pollinator observation counts) --------------------------------------
    abund = pollinator_abundance(cfg, store)
    print(f"abundance counts: {(abund > 0).mean():.1%} of pollinators covered ({time.time() - t0:.0f}s)", flush=True)

    # --- 2. congeneric transfer: partners of same-genus (then same-family) training plants -------
    gen_mat, fam_mat = {}, {}
    for g, q in zip(p_gen[tr_pi], tr_qi):
        gen_mat.setdefault(g, np.zeros(n_po))[q] += 1
    for f, q in zip(p_fam[tr_pi], tr_qi):
        fam_mat.setdefault(f, np.zeros(n_po))[q] += 1

    def congeneric(p):
        g, f = p_gen[p], p_fam[p]
        if g in gen_mat:
            return gen_mat[g] + 1e-6 * fam_mat.get(f, np.zeros(n_po))
        return fam_mat.get(f, np.zeros(n_po))

    n_gen_seen = np.mean([p_gen[store.p2i[sp]] in gen_mat for sp in test_pos["plant"].unique()])
    print(f"congeneric: {n_gen_seen:.1%} of test plants have a training congener", flush=True)

    # --- 3. latent-trait SVD with taxonomic imputation (Strydom-style RDPG) ----------------------
    A = csr_matrix((np.ones(len(tr_pi)), (tr_pi, tr_qi)), shape=(len(store.plants), n_po))
    U, S, Vt = svds(A.astype(np.float64), k=args.rank)
    U, S, Vt = U[:, ::-1], S[::-1], Vt[::-1]
    P_lat = U * S                       # plant latent traits (training plants only are informative)
    Q_lat = Vt.T                        # pollinator latent traits
    seen = np.zeros(len(store.plants), bool)
    seen[np.unique(tr_pi)] = True
    gen_lat, fam_lat = {}, {}
    for g in np.unique(p_gen[seen]):
        gen_lat[g] = P_lat[seen & (p_gen == g)].mean(0)
    for f in np.unique(p_fam[seen]):
        fam_lat[f] = P_lat[seen & (p_fam == f)].mean(0)
    global_lat = P_lat[seen].mean(0)

    def svd_score(p):
        v = gen_lat.get(p_gen[p], fam_lat.get(p_fam[p], global_lat))
        return Q_lat @ v

    # --- evaluate ------------------------------------------------------------------------------
    partners = {sp: set(store.idx_polls(g["pollinator"])) for sp, g in test_pos.groupby("plant")}
    t1 = {sp: set(store.idx_polls(g["pollinator"]))
          for sp, g in test_pos[test_pos["tier1"]].groupby("plant")}
    models = {
        "abundance null (obs counts)": lambda p: abund,
        "congeneric transfer": congeneric,
        f"latent-trait SVD (k={args.rank})": svd_score,
    }
    rows = []
    for name, fn in models.items():
        r10, r50, hit, t1r = [], [], [], []
        for sp, part in partners.items():
            p = store.p2i[sp]
            s = np.asarray(fn(p), dtype=np.float64) + np.random.default_rng(p).uniform(0, 1e-9, n_po)
            top = np.argpartition(-s, 50)[:50]
            top = top[np.argsort(-s[top])]
            r10.append(len(part & set(top[:10].tolist())) / len(part))
            r50.append(len(part & set(top.tolist())) / len(part))
            hit.append(float(len(part & set(top[:10].tolist())) > 0))
            if t1.get(sp):
                sm = s.copy()
                others = list(part - t1[sp])
                if others:
                    sm[others] = -np.inf
                tt = set(np.argpartition(-sm, 10)[:10].tolist())
                t1r.append(len(t1[sp] & tt) / len(t1[sp]))
        m, lo, hi, _ = bootstrap_mean(np.array(r10), nboot, seed)
        rows.append({"model": name, "recall@10": m, "lo": lo, "hi": hi,
                     "recall@50": np.mean(r50), "hit@10": np.mean(hit),
                     "t1_recall@10": np.mean(t1r)})
        print(f"  {name:<30} R@10 {m:.4f} [{lo:.4f},{hi:.4f}]  R@50 {np.mean(r50):.4f} "
              f"hit@10 {np.mean(hit):.4f}  T1 {np.mean(t1r):.4f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(cfg["edges"].parent / "eco_baselines_v1.csv", index=False)
    print(f"\nSaved. ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
