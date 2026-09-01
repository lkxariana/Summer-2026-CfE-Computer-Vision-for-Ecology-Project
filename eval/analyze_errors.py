import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from antheia.config import load_config
from antheia.pairs import load_split
from antheia.store import FeatureStore

MODELS = ["rank_n", "rank_n_tax_ldelta", "gbm_geo_tuned", "twotower_none", "twotower_bioclip", "twotower_bioclip2"]


def main():
    ap = argparse.ArgumentParser(description="Error/segment/complementarity analysis over saved per-plant scores")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    store = FeatureStore(cfg)
    edges = pd.read_parquet(cfg["edges"])
    sc_dir = cfg["edges"].parent / "final_scores"

    df = None
    for m in MODELS:
        d = pd.read_csv(sc_dir / f"{m}.csv").rename(columns={"r10": f"r10_{m}", "r50": f"r50_{m}", "hit": f"hit_{m}"})
        df = d if df is None else df.merge(d, on="plant")

    deg = edges.groupby("plant").size().rename("degree")
    tier1 = edges.assign(t1=edges["plant"].map(lambda x: 1)).groupby("plant").size()
    df = df.merge(deg, left_on="plant", right_index=True)
    df["range_size"] = [store.Frs[store.p2i[p]] for p in df["plant"]]
    df["genus"] = df["plant"].str.split().str[0]
    train_plants = set(load_split(cfg["edges"].parent / "split_v1.json")["train"])
    train_genera = {p.split()[0] for p in train_plants}
    df["genus_seen"] = df["genus"].isin(train_genera)

    print("=== segment means (recall@10) ===")
    df["deg_q"] = pd.qcut(df["degree"], 4, labels=["Q1 low", "Q2", "Q3", "Q4 high"], duplicates="drop")
    df["range_q"] = pd.qcut(df["range_size"], 4, labels=["narrow", "R2", "R3", "wide"], duplicates="drop")
    for seg in ["deg_q", "range_q", "genus_seen"]:
        print(f"\nby {seg}:")
        print(df.groupby(seg, observed=True)[[f"r10_{m}" for m in MODELS]].mean()
              .rename(columns=lambda c: c.replace("r10_", "")).to_string(float_format=lambda x: f"{x:.3f}"))
        print("  n:", df.groupby(seg, observed=True).size().to_dict())

    print("\n=== complementarity (hit@10) ===")
    best = "twotower_bioclip"
    for m in MODELS:
        if m == best:
            continue
        a, b = df[f"hit_{best}"], df[f"hit_{m}"]
        print(f"{best} vs {m:<20} both {int((a * b).sum()):>4} | only_best {int((a * (1 - b)).sum()):>4} | "
              f"only_other {int(((1 - a) * b).sum()):>4} | neither {int(((1 - a) * (1 - b)).sum()):>4}")

    pool = ["gbm_geo_tuned", "twotower_bioclip"]
    oracle_hit = df[[f"hit_{m}" for m in pool]].max(1).mean()
    oracle_r10 = df[[f"r10_{m}" for m in pool]].max(1).mean()
    print(f"\noracle over {pool}: hit@10 {oracle_hit:.4f} (best single {df[f'hit_{best}'].mean():.4f}), "
          f"r10 {oracle_r10:.4f} — upper bound for a 2-model ensemble")
    all_oracle = df[[f"hit_{m}" for m in MODELS]].max(1).mean()
    print(f"oracle over all {len(MODELS)} models: hit@10 {all_oracle:.4f}")

    print("\n=== hardest plants (all models miss, highest degree) ===")
    miss = df[df[[f"hit_{m}" for m in MODELS]].sum(1) == 0].nlargest(10, "degree")
    print(miss[["plant", "degree", "range_size", "genus_seen"]].to_string(index=False))
    print(f"\nplants missed by every model: {len(df[df[[f'hit_{m}' for m in MODELS]].sum(1) == 0])} / {len(df)}")
    print(f"plants hit by every model:    {len(df[df[[f'hit_{m}' for m in MODELS]].sum(1) == len(MODELS)])} / {len(df)}")

    df.to_csv(cfg["edges"].parent / "error_analysis_v1.csv", index=False)
    print(f"\nSaved {cfg['edges'].parent / 'error_analysis_v1.csv'}")


if __name__ == "__main__":
    main()
