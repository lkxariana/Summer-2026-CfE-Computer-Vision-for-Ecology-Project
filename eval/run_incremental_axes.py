import sys, json; sys.path.insert(0,'src'); sys.path.insert(0,'eval')
import numpy as np, pandas as pd
from antheia.store import UniverseStore
from run_encoding_ablation import run_row, _pca

store = UniverseStore(curves="modelled")
split = json.load(open('data/splits/plants_75_10_15.json'))
e = pd.read_parquet('data/network/edges.parquet')
e = e[e.plant.isin(store.p2i) & e.pollinator.isin(store.q2i)]
train = e[e.plant.isin(set(split['train']))]
test = e[e.plant.isin(set(split['val']))]; test = test[test.tier == 'A']
tp = sorted(set(test.plant)); part = {sp: set(store.idx_polls(g.pollinator)) for sp, g in test.groupby('plant')}

# taxonomic affinity from TRAINING edges only: how often does pollinator q visit plants of p's genus/family
gen = np.array([s.split()[0] for s in store.plants])
fam = np.array([store.family.get(s, "UNK") for s in store.plants])
gi = {g: i for i, g in enumerate(sorted(set(gen)))}; fi = {f: i for i, f in enumerate(sorted(set(fam)))}
GI = np.array([gi[g] for g in gen]); FI = np.array([fi[f] for f in fam])
Cg = np.zeros((len(store.polls), len(gi)), np.float32)
Cf = np.zeros((len(store.polls), len(fi)), np.float32)
for pl, po in zip(train.plant, train.pollinator):
    Cg[store.q2i[po], GI[store.p2i[pl]]] += 1
    Cf[store.q2i[po], FI[store.p2i[pl]]] += 1
gtot = Cg.sum(0) + 1e-9; ftot = Cf.sum(0) + 1e-9

def tax(pi, qi):
    g, f = GI[pi], FI[qi * 0 + pi]
    return np.column_stack([np.log1p(Cg[qi, g]), Cg[qi, g] / gtot[g],
                            np.log1p(Cf[qi, F := FI[pi]]), Cf[qi, F] / ftot[F]])

Fp, Pp = _pca(store)
N = store.N_full
pop = lambda pi, qi: np.log1p(store.Prs[qi])[:, None]
cooc = lambda pi, qi: np.log1p(np.asarray(N[pi, qi], dtype=np.float64))[:, None]
spatial = lambda pi, qi: np.hstack([cooc(pi, qi), Fp[pi] * Pp[qi]])
temporal = lambda pi, qi: np.hstack([store.FC[pi], store.AC[qi]])

rows = [
    ("popularity only",                 lambda pi, qi: pop(pi, qi)),
    ("taxonomy only",                   lambda pi, qi: np.hstack([pop(pi, qi), tax(pi, qi)])),
    ("taxonomy + spatial",              lambda pi, qi: np.hstack([pop(pi, qi), tax(pi, qi), spatial(pi, qi)])),
    ("taxonomy + temporal",             lambda pi, qi: np.hstack([pop(pi, qi), tax(pi, qi), temporal(pi, qi)])),
    ("taxonomy + spatial + temporal",   lambda pi, qi: np.hstack([pop(pi, qi), tax(pi, qi), spatial(pi, qi), temporal(pi, qi)])),
]
out = []
prev = None
from antheia.metrics import paired_bootstrap
for name, f in rows:
    pp, r = run_row(store, f, train, tp, part, 42, 10, 500)
    line = "  %-30s nR@10 %.4f  MAP %.4f  PR %.4f" % (name, r['nrecall@10'], r['map'], r['pr_auc'])
    if prev is not None:
        d, lo, hi, p = paired_bootstrap(pp['nrecall@10'].to_numpy(), prev['nrecall@10'].to_numpy(), 500, 42)
        line += "   vs taxonomy %+.4f p=%.3f" % (d, p)
    print(line, flush=True)
    out.append(dict(config=name, **{k: v for k, v in r.items() if k != 'lo'}))
    if name == "taxonomy only":
        prev = pp
pd.DataFrame(out).to_csv('results/incremental_axes_val_tierA.csv', index=False)
