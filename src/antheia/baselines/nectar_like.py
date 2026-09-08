"""NECTAR-style plausibility: a genus constraint times the product of spatial and phenological overlap.

Baiotto, Cosma, ... & Guzman (bioRxiv 2026-04-01, "Local interaction networks reconstructed from global
biodiversity data improve pollinator restoration decision making") infer flower visitation in ecoregion k
as

    p_ijk = p_co-occ * p_co-phen = [ sum_s p_occ(i,s) p_occ(j,s) / area_k ] * [ |D_i ∩ D_j| / |D_j| ]

for pairs passing a taxonomic constraint (the pollinator has a record with the plant's genus; graph
embedding + phylogenetic transfer for pollinators with no records), then threshold at the 25th
percentile of confirmed pairs. Re-implemented here on our inputs, continental rather than per ecoregion:
spatial overlap is the inner product of the two week-summed presence surfaces, phenological overlap the
normalised overlap of the two cell-summed presence curves (active weeks = weeks above 10% of peak),
and the genus constraint is read from the training interactions. Two variants:

  NectarLike        genus-gated product (pairs failing the constraint get the product scaled by 1e-3, so
                    they rank below every gated pair but keep their internal order -- the thresholding
                    step of the original is not needed for ranking metrics)
  NectarLikeUngated the overlap product alone: what the inference rule contributes without the
                    interaction records
"""
import numpy as np

from antheia.baselines.base import Baseline

FEAT = "data/features"


class NectarLikeUngated(Baseline):
    name = "NECTAR-style overlap product (spatial x phenological), ungated"
    reference = "Baiotto et al. 2026 (bioRxiv), Eq. 1; re-implemented"
    gated = False

    def fit(self, edges, store):
        from pathlib import Path
        root = Path(__file__).resolve().parents[3]
        self.store = store
        Ps = np.load(root / FEAT / "plant_surf_space.npy").astype(np.float64)   # [P, cells]  sum_w presence
        Qs = np.load(root / FEAT / "poll_surf_space.npy").astype(np.float64)
        Pt = np.load(root / FEAT / "plant_surf_time.npy").astype(np.float64)    # [P, 52]     sum_c presence
        Qt = np.load(root / FEAT / "poll_surf_time.npy").astype(np.float64)
        self.Ps, self.Qs = Ps / 52.0, Qs / 52.0
        # active weeks: above 10% of the species' peak (NECTAR uses 5th-95th percentile onset/offset)
        self.Dp = Pt >= 0.10 * Pt.max(1, keepdims=True)
        self.Dq = Qt >= 0.10 * Qt.max(1, keepdims=True)
        if self.gated:
            pi = store.idx_plants(edges["plant"]); qi = store.idx_polls(edges["pollinator"])
            gen = np.array([s.split()[0] for s in store.plants])
            self.gen_ids = {g: i for i, g in enumerate(sorted(set(gen)))}
            self.plant_gen = np.array([self.gen_ids[g] for g in gen])
            self.G = np.zeros((len(self.gen_ids), len(store.polls)), bool)      # genus x pollinator recorded
            self.G[self.plant_gen[pi], qi] = True
        return self

    def score_plant(self, p):
        cooc = self.Qs @ self.Ps[p] / self.Ps.shape[1]                          # [Q] mean over cells of joint presence
        inter = (self.Dq & self.Dp[p][None, :]).sum(1)
        cophen = inter / np.maximum(self.Dq.sum(1), 1)                          # |D_i ∩ D_j| / |D_j|
        s = cooc * cophen
        if self.gated:
            gate = self.G[self.plant_gen[p]]
            s = np.where(gate, s, 1e-3 * s)
        return s


class NectarLike(NectarLikeUngated):
    name = "NECTAR-style plausibility (genus constraint x spatial x phenological overlap)"
    gated = True
