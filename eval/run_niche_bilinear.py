import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch
ROOT = Path("/scratch/cher/Summer-2026-CfE-Computer-Vision-for-Ecology-Project")
sys.path.insert(0, str(ROOT / "src"))
from antheia.metrics import paired_bootstrap, retrieval_metrics
from antheia.pairnet import PairRanker
from antheia.store import UniverseStore
from sklearn.metrics import average_precision_score

store = UniverseStore(curves="modelled")
split = json.load(open(ROOT / "data/splits/plants_75_10_15.json"))
e = pd.read_parquet(ROOT / "data/network/edges.parquet")
e = e[e.plant.isin(store.p2i) & e.pollinator.isin(store.q2i)]
train = e[e.plant.isin(set(split["train"]))]
test = e[e.plant.isin(set(split["val"]))]; test = test[test.tier == "A"]
tp = sorted(set(test.plant))
part = {sp: set(store.idx_polls(g.pollinator)) for sp, g in test.groupby("plant")}
rows, ref = [], None
for rank in [0, 16, 64, 128]:
    t0 = time.time()
    m = PairRanker(epochs=20, bilinear_rank=rank).fit(train, store)
    S = np.vstack([m.score_plant(store.p2i[sp]) for sp in tp])
    pp = pd.DataFrame([retrieval_metrics(S[i], part[sp], ks=(10, 20)) for i, sp in enumerate(tp)])
    Y = np.zeros(S.shape, np.int8)
    for i, sp in enumerate(tp): Y[i, list(part[sp])] = 1
    pr = average_precision_score(Y.ravel(), S.ravel().astype(np.float64))
    line = "  bilinear_rank=%-4d nR@10 %.4f  MAP %.4f  PR %.4f" % (rank, pp["nrecall@10"].mean(), pp["ap"].mean(), pr)
    if ref is not None:
        d, lo, hi, pv = paired_bootstrap(pp["nrecall@10"].to_numpy(), ref["nrecall@10"].to_numpy(), 2000, 42)
        line += "   vs rank0 %+.4f [%+.4f,%+.4f] p=%.3f" % (d, lo, hi, pv)
    print(line + "  (%.0fs)" % (time.time() - t0), flush=True)
    rows.append(dict(bilinear_rank=rank, nrecall10=pp["nrecall@10"].mean(), map=pp["ap"].mean(), pr_auc=pr))
    if ref is None: ref = pp
    del m; torch.cuda.empty_cache()
pd.DataFrame(rows).to_csv(ROOT / "results/pairnet_bilinear_val.csv", index=False)
print("[wrote] results/pairnet_bilinear_val.csv")
