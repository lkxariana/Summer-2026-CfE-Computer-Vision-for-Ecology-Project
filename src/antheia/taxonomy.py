import numpy as np
import pandas as pd


def genus(name):
    return name.split()[0]


def family_map(globi_csv, plants):
    """Majority-vote plant→family from the GloBI record file's taxonomy columns."""
    df = pd.read_csv(globi_csv, usecols=["sourceTaxonName", "sourceTaxonFamilyName",
                                         "targetTaxonName", "targetTaxonFamilyName"])
    both = pd.concat([
        df[["sourceTaxonName", "sourceTaxonFamilyName"]].set_axis(["name", "family"], axis=1),
        df[["targetTaxonName", "targetTaxonFamilyName"]].set_axis(["name", "family"], axis=1),
    ])
    both = both[both["name"].isin(plants)].dropna()
    return both.groupby("name")["family"].agg(lambda s: s.mode().iat[0]).to_dict()


class TaxonomyAffinity:
    """Pollinator × plant-genus/family affinity tables built from training edges only.

    Encodes 'who visits what kind of plant': for an unseen test plant, its genus/family is known
    metadata, and each pollinator's affinity for that taxon is estimated from training-plant edges
    (no test-plant edges touch these tables, so the plant-grouped split is respected).
    """

    def __init__(self, store, train_edges, globi_csv):
        self.store = store
        gen_of = {p: genus(p) for p in store.plants}
        fam = family_map(globi_csv, set(store.plants))
        fam_of = {p: fam.get(p, "UNK") for p in store.plants}
        genera = sorted(set(gen_of.values()))
        families = sorted(set(fam_of.values()))
        g2i = {g: i for i, g in enumerate(genera)}
        f2i = {f: i for i, f in enumerate(families)}
        self.gi = np.array([g2i[gen_of[p]] for p in store.plants])
        self.fi = np.array([f2i[fam_of[p]] for p in store.plants])
        C_gen = np.zeros((len(store.polls), len(genera)), np.float32)
        C_fam = np.zeros((len(store.polls), len(families)), np.float32)
        for pl, po in zip(train_edges["plant"], train_edges["pollinator"]):
            q = store.q2i[po]
            C_gen[q, self.gi[store.p2i[pl]]] += 1
            C_fam[q, self.fi[store.p2i[pl]]] += 1
        deg = C_gen.sum(1, keepdims=True)
        self.tables = (np.log1p(C_gen), C_gen / (deg + 1), np.log1p(C_fam), C_fam / (deg + 1))
        self.n_features = 4

    def pairs(self, pi, qi):
        g, f = self.gi[pi], self.fi[pi]
        return np.stack([self.tables[0][qi, g], self.tables[1][qi, g],
                         self.tables[2][qi, f], self.tables[3][qi, f]], 1).astype(np.float64)

    def plant(self, p):
        g, f = self.gi[p], self.fi[p]
        return np.stack([self.tables[0][:, g], self.tables[1][:, g],
                         self.tables[2][:, f], self.tables[3][:, f]], 1).astype(np.float64)

    def genus_seen_in_train(self, p):
        return bool(self.tables[0][:, self.gi[p]].sum() > 0)
