import argparse
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from antheia import models
from antheia.config import load_config
from antheia.metrics import bootstrap_mean
from antheia.pairs import load_split
from antheia.store import FeatureStore
from antheia.taxonomy import TaxonomyAffinity

SDM_DIR = Path("/scratch/ariana.l/Stage 6 Seed Testing")

# Apples-to-apples GBIF vs SDM. BOTH arms rank over the SAME restricted candidate universe
# (the 1,275 pollinators with SDM surfaces), so numbers here are NOT comparable to the
# full-universe (24,939-candidate) results elsewhere in EXPERIMENTS.md.


class SDMFeatures:
    def __init__(self, cfg, store):
        self.store = store
        self.species = list(np.load(cfg["cache_dir"] / "sdm_species.npy", allow_pickle=True))
        n_bins = len(store.common_bins)
        self.S = np.load(cfg["cache_dir"] / f"sdm_surfaces_{len(self.species)}x{n_bins}x52.npy",
                         mmap_mode="r")
        self.sp2row = {s: i for i, s in enumerate(self.species)}
        self.cand = np.array([store.q2i[s] for s in self.species])       # index into store.polls
        # range-averaged SDM curve per species (normalised), the old aggregated view
        A = np.asarray(self.S).sum(1).astype(np.float64)
        self.A_sdm = A / np.clip(A.sum(1, keepdims=True), 1e-12, None)

    def delta_sdm(self, p_idx, q_rows):
        """Range-averaged overlap: plant PPE curve vs SDM activity curve."""
        return np.minimum(self.store.FC[p_idx][None, :], self.A_sdm[q_rows]).sum(1)

    def delta_bilateral(self, p_idx, q_rows):
        """Per-cell overlap of plant flowering and SDM activity, averaged over shared cells."""
        st = self.store
        fsurf = np.asarray(st.surfaces[p_idx], dtype=np.float32)          # bins x 52
        pmask = st.F[p_idx].astype(bool)
        out = np.zeros(len(q_rows))
        for k, qr in enumerate(q_rows):
            qi = self.cand[qr]
            shared = np.flatnonzero(pmask & st.P[qi].astype(bool))
            if shared.size == 0:
                continue
            f = fsurf[shared]
            a = np.asarray(self.S[qr][shared], dtype=np.float32)
            fs, as_ = f.sum(1, keepdims=True), a.sum(1, keepdims=True)
            ok = (fs[:, 0] > 0) & (as_[:, 0] > 0)
            if not ok.any():
                continue
            f = f[ok] / fs[ok]
            a = a[ok] / as_[ok]
            out[k] = np.minimum(f, a).sum(1).mean()
        return out


def main():
    ap = argparse.ArgumentParser(description="GBIF vs SDM activity curves, and bilateral local Delta")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    store = FeatureStore(cfg)
    sdm = SDMFeatures(cfg, store)
    seed, nboot = cfg["eval"]["seed"], cfg["eval"]["bootstrap_n"]

    edges = pd.read_parquet(cfg["edges"])
    prov = pd.read_csv(cfg["edges"].parent / "edge_provenance_v1.csv")
    edges = edges.merge(prov, on=["plant", "pollinator"], how="left")
    edges["tier1"] = edges["n_inat"].notna() & (edges["n_inat"] < edges["n"])
    sdm_set = set(sdm.species)
    e = edges[edges["pollinator"].isin(sdm_set)].reset_index(drop=True)
    split = load_split(cfg["edges"].parent / "split_v1.json")
    train_pos = e[e["plant"].isin(split["train"])]
    test_pos = e[e["plant"].isin(split["test"])]
    print(f"restricted universe: {len(sdm.species):,} candidate pollinators")
    print(f"edges kept: {len(e):,}/{len(edges):,} | train {len(train_pos):,} | test {len(test_pos):,} "
          f"({test_pos['plant'].nunique()} plants)", flush=True)

    man = pd.read_csv(SDM_DIR / "species_manifest.csv")
    src = dict(zip(man["species"], man["model_source"]))
    known = set(zip(e["plant"], e["pollinator"]))
    rng = np.random.default_rng(seed)

    # within-plant contrasts over the restricted candidate set
    pi, qpos, qneg = [], [], []
    for plant, q in zip(train_pos["plant"], train_pos["pollinator"]):
        p = store.p2i[plant]
        qr = sdm.sp2row[q]
        for _ in range(10):
            r = int(rng.integers(len(sdm.species)))
            if r != qr and (plant, sdm.species[r]) not in known:
                pi.append(p); qpos.append(qr); qneg.append(r)
    pi, qpos, qneg = np.array(pi), np.array(qpos), np.array(qneg)
    print(f"contrasts: {len(pi):,}", flush=True)

    def feats(name, p_idx, q_rows):
        qi = sdm.cand[q_rows]
        if name == "n":
            return store.N_full[p_idx, qi].astype(np.float64)[:, None]
        if name == "tax":
            return aff.pairs(p_idx, qi)
        if name == "delta_gbif":
            return np.minimum(store.FC[p_idx], store.AC[qi]).sum(1)[:, None]
        if name == "delta_local_gbif":
            return store.delta_local_pairs(p_idx, qi)[:, None]
        if name == "delta_sdm":
            return np.array([sdm.delta_sdm(p, [r])[0] for p, r in zip(p_idx, q_rows)])[:, None]
        if name == "delta_bilateral":
            uniq = {}
            out = np.zeros(len(p_idx))
            for p in np.unique(p_idx):
                m = p_idx == p
                out[m] = sdm.delta_bilateral(p, q_rows[m])
            return out[:, None]
        raise KeyError(name)

    aff = TaxonomyAffinity(store, train_pos, cfg["paths"]["globi"])
    SPECS = {
        "N only": ["n"],
        "N + tax": ["n", "tax"],
        "N + tax + GBIF local": ["n", "tax", "delta_local_gbif"],
        "N + tax + SDM delta": ["n", "tax", "delta_sdm"],
        "N + tax + SDM bilateral": ["n", "tax", "delta_bilateral"],
        "N + tax + GBIF local + SDM bilat": ["n", "tax", "delta_local_gbif", "delta_bilateral"],
        "N + GBIF delta": ["n", "delta_gbif"],
        "N + GBIF local delta": ["n", "delta_local_gbif"],
        "N + SDM delta": ["n", "delta_sdm"],
        "N + SDM bilateral local delta": ["n", "delta_bilateral"],
        "N + GBIF local + SDM bilateral": ["n", "delta_local_gbif", "delta_bilateral"],
    }
    comps = sorted({c for v in SPECS.values() for c in v})
    t0 = time.time()
    Xp = {c: feats(c, pi, qpos) for c in comps}
    Xn = {c: feats(c, pi, qneg) for c in comps}
    print(f"train features built ({time.time()-t0:.0f}s)", flush=True)

    pipes = {}
    for name, spec in SPECS.items():
        d = np.hstack([Xp[c] for c in spec]) - np.hstack([Xn[c] for c in spec])
        pipe = Pipeline([("s", StandardScaler(with_mean=False)),
                         ("lr", LogisticRegression(max_iter=1000, random_state=seed, fit_intercept=False))])
        pipe.fit(np.vstack([d, -d]), np.concatenate([np.ones(len(d)), np.zeros(len(d))]))
        pipes[name] = pipe

    all_rows = np.arange(len(sdm.species))
    partners = {sp: {sdm.sp2row[q] for q in g["pollinator"]} for sp, g in test_pos.groupby("plant")}
    t1_partners = {sp: {sdm.sp2row[q] for q in g[g["tier1"] == True]["pollinator"]}
                   for sp, g in test_pos.groupby("plant")}
    head_rows = {sdm.sp2row[s] for s in sdm.species if src.get(s) == "head"}
    rec = {n: {"all": [], "t1": [], "head": []} for n in SPECS}
    t0 = time.time()
    for i, (sp, part) in enumerate(partners.items()):
        p = store.p2i[sp]
        blk = {c: feats(c, np.full(len(all_rows), p), all_rows) for c in comps}
        for name, spec in SPECS.items():
            s = pipes[name].decision_function(np.hstack([blk[c] for c in spec]))
            t10 = set(np.argpartition(-s, 10)[:10].tolist())
            rec[name]["all"].append(len(part & t10) / len(part))
            pt1 = t1_partners.get(sp, set())
            if pt1:
                sm = s.copy(); sm[list(part - pt1)] = -np.inf
                rec[name]["t1"].append(len(pt1 & set(np.argpartition(-sm, 10)[:10].tolist())) / len(pt1))
            ph = part & head_rows
            if ph:
                sm = s.copy(); sm[list(part - ph)] = -np.inf
                rec[name]["head"].append(len(ph & set(np.argpartition(-sm, 10)[:10].tolist())) / len(ph))
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(partners)} plants ({time.time()-t0:.0f}s)", flush=True)

    rows = []
    print(f"\n{'model':<34}{'R@10':>18}{'Tier-1':>10}{'fitted-head':>13}")
    for name in SPECS:
        m, lo, hi, _ = bootstrap_mean(np.array(rec[name]["all"]), nboot, seed)
        rows.append({"model": name, "r10": m, "lo": lo, "hi": hi,
                     "r10_tier1": np.mean(rec[name]["t1"]), "r10_head": np.mean(rec[name]["head"])})
        print(f"{name:<34}{m:>8.4f} [{lo:.3f},{hi:.3f}]{np.mean(rec[name]['t1']):>10.4f}"
              f"{np.mean(rec[name]['head']):>13.4f}")
    pd.DataFrame(rows).to_csv(cfg["edges"].parent / "sdm_v1.csv", index=False)
    print("\nNOTE: candidate universe is 1,275 (SDM-covered) — NOT comparable to full-universe numbers.")


if __name__ == "__main__":
    main()
