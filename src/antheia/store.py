import numpy as np
import pandas as pd


class FeatureStore:
    """Loads all per-species artifacts once, aligned to a fixed (plants, pollinators) universe.

    The universe is the coverage intersection: plants present in Vf, f_curves, V_delta 4D/15D and
    Vf_prob; pollinators present in Vp and a_curves. All arrays are row-aligned to the sorted
    universe lists, so every downstream lookup is integer indexing.
    """

    def __init__(self, cfg):
        p = cfg["paths"]
        self.cfg = cfg
        F_df = pd.read_csv(p["f_matrix"], index_col=0)
        P_df = pd.read_csv(p["p_matrix"], index_col=0)
        Vf = pd.read_csv(p["vf"], index_col=0)
        Vp = pd.read_csv(p["vp"], index_col=0)
        vd4 = pd.read_csv(p["vdelta4"], index_col=0)
        vd15 = pd.read_csv(p["vdelta15"], index_col=0)
        vfp = pd.read_csv(p["vf_prob"], index_col=0)
        ac = pd.read_csv(p["a_curves"], index_col=0)
        if p["f_curves"].exists():
            fc, self.f_curves_source = pd.read_csv(p["f_curves"], index_col=0), p["f_curves"].name
        else:
            fc, self.f_curves_source = pd.read_csv(p["f_curves_fallback"], index_col=0), p["f_curves_fallback"].name

        self.common_bins = [b for b in F_df.columns if b in set(P_df.columns)]
        self.plants = sorted(set(Vf.index) & set(fc.index) & set(vd4.index) & set(vd15.index) & set(vfp.index))
        self.polls = sorted(set(Vp.index) & set(ac.index))
        self.p2i = {s: i for i, s in enumerate(self.plants)}
        self.q2i = {s: i for i, s in enumerate(self.polls)}

        self.F = F_df.loc[self.plants, self.common_bins].to_numpy(np.int8)
        self.P = P_df.loc[self.polls, self.common_bins].to_numpy(np.int8)
        self.VF = Vf.loc[self.plants].to_numpy(np.float64)
        self.VP = Vp.loc[self.polls].to_numpy(np.float64)
        self.VD4 = vd4.loc[self.plants].to_numpy(np.float64)
        self.VD15 = vd15.loc[self.plants].to_numpy(np.float64)
        self.VFP = vfp.loc[self.plants].to_numpy(np.float64)
        self.FC = fc.loc[self.plants].to_numpy(np.float64)
        self.AC = ac.loc[self.polls].to_numpy(np.float64)
        self.Frs = self.F.sum(1).astype(np.float64)
        self.Prs = self.P.sum(1).astype(np.float64)
        self._N = None
        self._S = None

    @property
    def N_full(self):
        """Full shared-bin-count matrix (n_plants x n_pollinators, uint16), disk-cached."""
        if self._N is None:
            cache = self.cfg["cache_dir"] / f"N_full_{len(self.plants)}x{len(self.polls)}.npy"
            if cache.exists():
                self._N = np.load(cache)
            else:
                self._N = (self.F.astype(np.float32) @ self.P.astype(np.float32).T).astype(np.uint16)
                cache.parent.mkdir(parents=True, exist_ok=True)
                np.save(cache, self._N)
        return self._N

    @property
    def surfaces(self):
        """Per-plant PPE opportunity surfaces (n_plants x n_common_bins x 52, float16 memmap)."""
        if self._S is None:
            path = self.cfg["cache_dir"] / f"plant_surfaces_{len(self.plants)}x{len(self.common_bins)}x52.npy"
            if not path.exists():
                raise FileNotFoundError(f"{path} missing — run scripts/build_plant_surfaces.py first")
            self._S = np.load(path, mmap_mode="r")
        return self._S

    def delta_local_pairs(self, pi, qi):
        """Local temporal overlap: plant flowering curve restricted to the pair's shared bins vs the pollinator's activity curve."""
        out = np.zeros(len(pi))
        order = np.argsort(pi, kind="stable")
        sp = pi[order]
        starts = np.flatnonzero(np.r_[True, sp[1:] != sp[:-1]])
        for s, e in zip(starts, np.r_[starts[1:], len(sp)]):
            p = sp[s]
            rows = order[s:e]
            q = qi[rows]
            masks = (self.P[q] * self.F[p][None, :]).astype(np.float32)
            fl = masks @ self.surfaces[p].astype(np.float32)
            sums = fl.sum(1, keepdims=True)
            fl = np.divide(fl, sums, out=np.zeros_like(fl), where=sums > 0)
            out[rows] = np.minimum(fl, self.AC[q]).sum(1)
        return out

    def idx_plants(self, names):
        return np.fromiter((self.p2i[s] for s in names), dtype=np.int64, count=len(names))

    def idx_polls(self, names):
        return np.fromiter((self.q2i[s] for s in names), dtype=np.int64, count=len(names))

    def _component_pairs(self, name, pi, qi):
        if name == "vf":
            return self.VF[pi]
        if name == "vp":
            return self.VP[qi]
        if name == "vd4":
            return self.VD4[pi]
        if name == "vd15":
            return self.VD15[pi]
        if name == "vfp":
            return self.VFP[pi]
        if name == "n":
            return self.N_full[pi, qi].astype(np.float64)[:, None]
        if name == "delta":
            return np.minimum(self.FC[pi], self.AC[qi]).sum(1)[:, None]
        if name == "delta_local":
            return self.delta_local_pairs(pi, qi)[:, None]
        if name == "frs":
            return self.Frs[pi][:, None]
        if name == "prs":
            return self.Prs[qi][:, None]
        raise KeyError(name)

    def assemble(self, pi, qi, spec):
        """Feature matrix for pair index arrays (pi, qi) under a component spec tuple."""
        return np.hstack([self._component_pairs(c, pi, qi) for c in spec])

    def assemble_plant(self, pi, spec):
        """Feature matrix for one plant against every pollinator in the universe."""
        n_po = len(self.polls)
        blocks = []
        for c in spec:
            if c in ("vf", "vd4", "vd15", "vfp"):
                arr = {"vf": self.VF, "vd4": self.VD4, "vd15": self.VD15, "vfp": self.VFP}[c]
                blocks.append(np.broadcast_to(arr[pi], (n_po, arr.shape[1])))
            elif c == "vp":
                blocks.append(self.VP)
            elif c == "n":
                blocks.append(self.N_full[pi].astype(np.float64)[:, None])
            elif c == "delta":
                blocks.append(np.minimum(self.FC[pi][None, :], self.AC).sum(1)[:, None])
            elif c == "delta_local":
                masks = (self.P * self.F[pi][None, :]).astype(np.float32)
                fl = masks @ self.surfaces[pi].astype(np.float32)
                sums = fl.sum(1, keepdims=True)
                fl = np.divide(fl, sums, out=np.zeros_like(fl), where=sums > 0)
                blocks.append(np.minimum(fl, self.AC).sum(1)[:, None])
            elif c == "frs":
                blocks.append(np.full((n_po, 1), self.Frs[pi]))
            elif c == "prs":
                blocks.append(self.Prs[:, None])
            else:
                raise KeyError(c)
        return np.hstack(blocks)
