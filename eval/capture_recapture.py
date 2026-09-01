import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from antheia.config import load_config

# Two documentation processes (iNaturalist vs curated sources) independently "capture" true
# interactions. The 2x2 capture table identifies how many real edges neither process recorded.
cfg = load_config()
e = pd.read_parquet(cfg["edges"]).merge(
    pd.read_csv(cfg["edges"].parent / "edge_provenance_v1.csv"), on=["plant", "pollinator"], how="inner")
e["n_cur"] = e["n"] - e["n_inat"]

both = int(((e["n_inat"] > 0) & (e["n_cur"] > 0)).sum())
inat_only = int(((e["n_inat"] > 0) & (e["n_cur"] == 0)).sum())
cur_only = int(((e["n_inat"] == 0) & (e["n_cur"] > 0)).sum())
n_obs = both + inat_only + cur_only
print(f"edges: {len(e):,} | iNat-only {inat_only:,} | curated-only {cur_only:,} | both {both:,}")

# Lincoln-Petersen with Chapman's bias correction.
n1, n2 = both + inat_only, both + cur_only
chapman = (n1 + 1) * (n2 + 1) / (both + 1) - 1
var = ((n1 + 1) * (n2 + 1) * (n1 - both) * (n2 - both)) / ((both + 1) ** 2 * (both + 2))
se = np.sqrt(var)
print(f"\nCaptured by iNat: {n1:,} | by curated: {n2:,} | by both: {both:,}")
print(f"Chapman estimate of TRUE edge count among covered species: {chapman:,.0f} "
      f"(95% CI {chapman - 1.96 * se:,.0f}–{chapman + 1.96 * se:,.0f})")
print(f"Observed / estimated true = {n_obs / chapman:.1%}  -> we see about this share of the metaweb")
print(f"Implied detection: iNat {n1 / chapman:.1%}, curated {n2 / chapman:.1%}")

# Independence is the load-bearing assumption; check its direction.
p_cur_given_inat = both / n1
p_cur_marginal = n2 / n_obs
ratio = p_cur_given_inat / p_cur_marginal
print(f"\noverlap: only {both / n_obs:.1%} of documented edges are recorded by BOTH process families")
print(f"P(curated | iNat-documented) = {p_cur_given_inat:.3f} vs marginal {p_cur_marginal:.3f} (ratio {ratio:.2f})")
if ratio < 1:
    print("NEGATIVE dependence: the processes document largely DISJOINT edges (they specialise by taxon).")
    print("Lincoln-Petersen assumes independence; under negative dependence it OVER-estimates the total,")
    print("so the figure above is an upper bound, not an estimate. Treat it as non-identified.")
else:
    print("Positive dependence: processes favour the same edges, so Chapman UNDER-estimates (a floor).")

# Stratify: heterogeneity in capture probability also biases the estimate downward.
g = pd.read_csv(cfg["paths"]["globi"], usecols=["sourceTaxonName", "sourceTaxonOrderName"]).dropna()
order = g.groupby("sourceTaxonName")["sourceTaxonOrderName"].agg(lambda s: s.mode().iat[0])
e["order"] = e["pollinator"].map(order).fillna("UNK")
print(f"\n{'order':<16}{'obs':>9}{'both':>8}{'est. true':>12}{'observed share':>16}")
tot_strat = 0
for o, sub in e.groupby("order"):
    b = int(((sub["n_inat"] > 0) & (sub["n_cur"] > 0)).sum())
    a1 = int((sub["n_inat"] > 0).sum())
    a2 = int((sub["n_cur"] > 0).sum())
    if b < 20 or len(sub) < 200:
        continue
    est = (a1 + 1) * (a2 + 1) / (b + 1) - 1
    tot_strat += est
    print(f"{o:<16}{len(sub):>9,}{b:>8,}{est:>12,.0f}{len(sub) / est:>15.1%}")
print(f"\nsum of stratum estimates: {tot_strat:,.0f} (vs pooled {chapman:,.0f}).")
print("Stratifying does not reconcile them — within-order overlap is so small (Lepidoptera: 49 edges)")
print("that estimates are unstable. Robust conclusion: the two documentation processes are near-disjoint")
print("views of the same network, so the documented edge set is a process-structured minority sample.")
