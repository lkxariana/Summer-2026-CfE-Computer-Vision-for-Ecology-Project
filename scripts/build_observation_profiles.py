import argparse
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from antheia.config import load_config
from antheia.store import FeatureStore

BASE = "datasets/imageomics/TreeOfLife-200M-Embeddings/bioclip-2_float16"
COLS = ["scientific_name", "source_dataset", "basisOfRecord", "publisher"]


def main():
    ap = argparse.ArgumentParser(description="Per-species observation-process profiles from TOL-200M metadata (no embeddings)")
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--end", type=int, required=True)
    args = ap.parse_args()

    cfg = load_config()
    store = FeatureStore(cfg)
    targets = set(store.plants) | set(store.polls)
    fs = HfFileSystem()
    rows = defaultdict(lambda: defaultdict(int))
    n_failed = 0
    t0 = time.time()

    for i in range(args.start, args.end):
        path = f"{BASE}/train-{i:05d}-of-00666.parquet"
        tb = None
        for attempt in range(6):
            try:
                with HfFileSystem().open(path, "rb") as fh:
                    tb = pq.ParquetFile(fh).read(columns=COLS)
                break
            except Exception as ex:
                if attempt == 5:
                    print(f"  shard {i} FAILED after retries: {type(ex).__name__}", flush=True)
                else:
                    time.sleep(2 ** attempt + random.random())
        if tb is None:
            n_failed += 1
            continue
        d = tb.to_pandas()
        d = d[d["scientific_name"].isin(targets)]
        if not len(d):
            continue
        d["basisOfRecord"] = d["basisOfRecord"].fillna("unknown").str.lower()
        for sp, g in d.groupby("scientific_name"):
            r = rows[sp]
            r["n_images"] += len(g)
            for k, v in g["source_dataset"].fillna("unknown").value_counts().items():
                r[f"src::{k}"] += int(v)
            for k, v in g["basisOfRecord"].value_counts().items():
                r[f"bor::{k}"] += int(v)
            r["n_publishers"] = max(r["n_publishers"], g["publisher"].nunique())
        if (i - args.start + 1) % 20 == 0:
            print(f"  {i - args.start + 1}/{args.end - args.start} shards, {len(rows)} species ({time.time() - t0:.0f}s)", flush=True)

    out = cfg["cache_dir"] / "obs_parts"
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame.from_dict(rows, orient="index").fillna(0).astype(int)
    df.index.name = "species"
    df.to_parquet(out / f"obs_{args.start:05d}_{args.end:05d}.parquet")
    n_img = int(df["n_images"].sum()) if "n_images" in df else 0
    print(f"shards {args.start}-{args.end}: {len(df)} species, {n_img:,} images, {n_failed} shards failed ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
