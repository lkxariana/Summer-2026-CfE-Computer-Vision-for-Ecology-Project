"""Data for the story figures: results/fig_arms.csv (retriever arms: universe / within-site / cold-pollinator AUPR) and
results/fig_regimes.csv (final vs v1 vs best comparison model per regime). Re-run before antheia.eval.figures."""
import sys
from pathlib import Path
import pandas as pd
from antheia.paths import REPO_ROOT as ROOT
from antheia.bundle import load_bundles, seed_pooled

ARMS = {"R3 (taxon+cell)": "M3.7_rgcn_sym", "final retriever (no taxon)": "A2_rgcn_sym_notaxon", "no cell nodes": "A1_rgcn_sym_nocell",
        "month-collapsed": "A3_rgcn_sym_monthcollapsed", "+ memory vector (R1)": "M3.11_rgcn_sym_warmres",
        "+ co-presence stat (R4)": "M3.10_rgcn_sym_pair_joint", "+ learned co-presence (R6c)": "M3.14_rgcn_sym_bilinear",
        "+ presence embed concat (R6b)": "M3.13_rgcn_sym_presconcat", "+ presence embed add (R6)": "M3.12s_rgcn_sym_pressurf",
        "co-occurrence negatives (R7)": "M3.15_rgcn_sym_coocneg", "attention aggregation": "M3.6_rgcn_attention",
        "plant-only rehearsal (v1)": "M3.1_rgcn", "+ degree encoding (R5, on v1)": "M3.9_rgcn_degree_enc",
        "+ co-presence stat (R4, on v1)": "M3.8_rgcn_pair_joint", "+ memory (R1, on v1)": "M3.3_rgcn_warmres",
        "final + opportunity term (A4)": "A4_rgcn_notaxon_pair", "A4, opportunity off within sites": "A4_rgcn_notaxon_pair_affinityonly"}


def main():
    df = load_bundles(); pooled = seed_pooled(df, group=("model", "split")).set_index(["model", "split"])
    get = lambda m, sp, k: float(pooled.loc[(m, sp), k]) if (m, sp) in pooled.index else float("nan")
    rows = []
    for a, m in ARMS.items():
        src = "A4_rgcn_notaxon_pair" if m.endswith("_affinityonly") else m      # the affinity-only row shares the universe numbers
        rows.append(dict(arm=a, model=m, cold_plant=get(src, "cold_plant/val", "aupr"), cold_poll=get(src, "cold_poll/val", "aupr"),
                         local=get(m, "localnet", "mean_aupr"), local_pL=get(m, "localnet", "mean_link_precision_at_L")))
    pd.DataFrame(rows).to_csv(ROOT / "results/fig_arms.csv", index=False)
    regime = []
    for sp, lab in [("cold_plant/val", "cold plant"), ("cold_poll/val", "cold pollinator"), ("cold_both/val", "cold both"), ("warm/val", "warm")]:
        sub = pooled.xs(sp, level="split"); base = sub[~sub.index.str.startswith(("M", "A"))]["aupr"]
        regime.append(dict(regime=lab, final=get("M5.0_system_notaxon", sp, "aupr"), system_v1=get("M2.12_fusion_on_rgcn", sp, "aupr"),
                           best_baseline=float(base.max()), best_baseline_name=base.idxmax(), chance=float(sub["prevalence"].iloc[0])))
    pd.DataFrame(regime).to_csv(ROOT / "results/fig_regimes.csv", index=False)
    print(pd.DataFrame(rows).round(3).to_string(index=False)); print(pd.DataFrame(regime).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
