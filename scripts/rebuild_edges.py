import argparse
import datetime
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from antheia.config import load_config
from antheia.globi import build_edges
from antheia.store import FeatureStore


def main():
    ap = argparse.ArgumentParser(description="Rebuild the orientation-corrected GloBI edge list")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    store = FeatureStore(cfg)
    edges, stats = build_edges(cfg["paths"]["globi"], set(store.plants), set(store.polls))

    out = cfg["edges"]
    out.parent.mkdir(parents=True, exist_ok=True)
    edges.to_parquet(out, index=False)

    deg = edges.groupby("plant").size()
    card = out.with_name(out.stem + "_card.md")
    with open(card, "w") as f:
        f.write(f"# Edge list {out.stem}\n\n")
        f.write(f"Built {datetime.date.today()} from `{cfg['paths']['globi']}`.\n")
        f.write(f"Coverage universe: {len(store.plants)} plants x {len(store.polls)} pollinators ")
        f.write(f"(f_curves source: `{store.f_curves_source}`).\n")
        f.write("Orientation corrected by species-set membership with binomial-name fallback; ")
        f.write("see `src/antheia/globi.py`.\n\n")
        f.write("| stat | value |\n|---|---|\n")
        for k, v in stats.items():
            f.write(f"| {k} | {v:,} |\n")
        f.write(f"| connectance | {stats['unique_edges'] / (stats['n_plants'] * stats['n_pollinators']):.4%} |\n")
        f.write(f"| plant degree median / mean / max | {deg.median():.0f} / {deg.mean():.1f} / {deg.max()} |\n")
        f.write(f"| edges with n_records >= 2 | {(edges['n_records'] >= 2).sum():,} |\n\n")
        f.write("Top pollinators: " + ", ".join(edges["pollinator"].value_counts().head(5).index) + "\n\n")
        f.write("Top plants: " + ", ".join(edges["plant"].value_counts().head(5).index) + "\n")

    print(f"Saved {len(edges):,} edges -> {out}")
    for k, v in stats.items():
        print(f"  {k}: {v:,}")


if __name__ == "__main__":
    main()
