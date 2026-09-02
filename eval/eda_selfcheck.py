import sys
from pathlib import Path
import numpy as np
import pandas as pd

e = pd.read_parquet("data/network/edges.parquet")
WIND = ["Poaceae", "Cyperaceae", "Juncaceae", "Typhaceae", "Betulaceae", "Fagaceae", "Pinaceae",
        "Cupressaceae", "Chenopodiaceae", "Amaranthaceae", "Plantaginaceae", "Urticaceae"]
INSECT = ["Asteraceae", "Lamiaceae", "Fabaceae", "Rosaceae", "Apiaceae", "Brassicaceae",
          "Onagraceae", "Ericaceae", "Boraginaceae", "Ranunculaceae"]

def deg_by_family(sub):
    d = sub.groupby(["plant_family", "plant"]).size().groupby("plant_family").agg(["mean", "size"])
    return d.rename(columns={"mean": "mean_degree", "size": "n_taxa"})

print("=" * 74)
print("Q1. DO WIND-POLLINATED PLANTS HAVE 'POLLINATORS'?")
print("=" * 74)
for tier, sub in (("A (flower visitation)", e[e.tier == "A"]), ("A+B (all)", e)):
    d = deg_by_family(sub)
    w = d.reindex(WIND).dropna()
    i = d.reindex(INSECT).dropna()
    print(f"\n{tier}:  wind-pollinated mean degree {w.mean_degree.mean():.1f} "
          f"vs insect-pollinated {i.mean_degree.mean():.1f}  (ratio {w.mean_degree.mean()/i.mean_degree.mean():.2f})")
    print("  wind families:  " + ", ".join(f"{k} {v:.1f}" for k, v in w.mean_degree.sort_values(ascending=False).head(5).items()))
    print("  insect families:" + ", ".join(f" {k} {v:.1f}" for k, v in i.mean_degree.sort_values(ascending=False).head(5).items()))
    po = sub[sub.plant_family == "Poaceae"]
    print(f"  Poaceae: {len(po):,} interactions ({100*len(po)/len(sub):.2f}% of tier), "
          f"{po.plant.nunique()} taxa, mean degree {len(po)/max(po.plant.nunique(),1):.1f}")

print("\n" + "=" * 74)
print("Q2. DOES THE FAUNA LOOK LIKE NORTH AMERICA?")
print("=" * 74)
for tier, sub in (("A", e[e.tier == "A"]), ("A+B", e)):
    v = (100 * sub.pollinator_order.value_counts(normalize=True)).head(6)
    print(f"  tier {tier:<4}: " + ", ".join(f"{k} {x:.0f}%" for k, x in v.items()))

print("\n" + "=" * 74)
print("Q5. WHERE DO GENUS-LEVEL RECORDS SIT?")
print("=" * 74)
g = e[e.plant_rank == "genus"].drop_duplicates("plant").plant_family.value_counts()
s = e[e.plant_rank == "species"].drop_duplicates("plant").plant_family.value_counts()
r = pd.DataFrame({"genus_taxa": g, "species_taxa": s}).fillna(0)
r["pct_genus"] = 100 * r.genus_taxa / (r.genus_taxa + r.species_taxa)
print("plant families with most genus-only taxa:")
print(r.sort_values("genus_taxa", ascending=False).head(6).to_string())
gq = e[e.pollinator_rank == "genus"].drop_duplicates("pollinator").pollinator_order.value_counts()
sq = e[e.pollinator_rank == "species"].drop_duplicates("pollinator").pollinator_order.value_counts()
rq = pd.DataFrame({"genus": gq, "species": sq}).fillna(0)
rq["pct_genus"] = 100 * rq.genus / (rq.genus + rq.species)
print("\npollinator orders by share identified only to genus:")
print(rq[rq.genus + rq.species > 100].sort_values("pct_genus", ascending=False).head(6).to_string())

print("\n" + "=" * 74)
print("Q6. DO KNOWN SPECIALIST RELATIONSHIPS APPEAR?")
print("=" * 74)
checks = [("squash bees on Cucurbita", ["Peponapis", "Xenoglossa"], "Cucurbita"),
          ("yucca moths on Yucca", ["Tegeticula", "Parategeticula"], "Yucca"),
          ("Habropoda on Vaccinium", ["Habropoda"], "Vaccinium"),
          ("Andrena on Salix", ["Andrena"], "Salix")]
for label, gen, host in checks:
    m = e[e.pollinator.str.startswith(tuple(g + " " for g in gen)) | e.pollinator.isin(gen)]
    m = m[m.plant.str.startswith(host)]
    print(f"  {label:<28} {len(m):>4} interactions"
          + (f"  e.g. {m.iloc[0].pollinator} x {m.iloc[0].plant}" if len(m) else "   *** ABSENT ***"))
