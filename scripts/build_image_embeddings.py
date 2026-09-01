import argparse
import sys
import time
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from antheia.config import load_config
from antheia.store import FeatureStore

REPO = "datasets/imageomics/TreeOfLife-200M-Embeddings/bioclip-2_float16"
N_SHARDS = 666
DIM = 768


def main():
    ap = argparse.ArgumentParser(description="Per-species mean BioCLIP-2 IMAGE embeddings from TreeOfLife-200M")
    ap.add_argument("--config", default=None)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=N_SHARDS)
    args = ap.parse_args()

    cfg = load_config(args.config)
    store = FeatureStore(cfg)
    wanted = {s: i for i, s in enumerate(store.plants + store.polls)}
    n_sp = len(wanted)
    sums = np.zeros((n_sp, DIM), np.float64)
    counts = np.zeros(n_sp, np.int64)
    fs = HfFileSystem()
    t0 = time.time()

    for i in range(args.start, args.end):
        path = f"{REPO}/train-{i:05d}-of-{N_SHARDS:05d}.parquet"
        try:
            with fs.open(path, "rb") as fh:
                pf = pq.ParquetFile(fh)
                names = pf.read(columns=["scientific_name"]).column(0).to_pylist()
                idx = np.array([wanted.get(n, -1) for n in names])
                if not (idx >= 0).any():
                    continue
                # Only pull embeddings for row groups that actually contain target species.
                offset, rg_hits = 0, []
                for rg in range(pf.metadata.num_row_groups):
                    n_rows = pf.metadata.row_group(rg).num_rows
                    if (idx[offset:offset + n_rows] >= 0).any():
                        rg_hits.append((rg, offset, n_rows))
                    offset += n_rows
                for rg, offset, n_rows in rg_hits:
                    tb = pf.read_row_group(rg, columns=["emb"])
                    sub = idx[offset:offset + n_rows]
                    hit = np.flatnonzero(sub >= 0)
                    emb = np.stack(tb.column(0).to_numpy(zero_copy_only=False)[hit]).astype(np.float64)
                    np.add.at(sums, sub[hit], emb)
                    np.add.at(counts, sub[hit], 1)
        except Exception as ex:
            print(f"  shard {i} FAILED: {type(ex).__name__} {str(ex)[:120]}", flush=True)
            continue
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{args.end} shards | species covered {int((counts > 0).sum()):,}/{n_sp:,} "
                  f"| {time.time() - t0:.0f}s", flush=True)

    part = cfg["cache_dir"] / "img_parts"
    part.mkdir(parents=True, exist_ok=True)
    np.save(part / f"sums_{args.start:05d}_{args.end:05d}.npy", sums.astype(np.float32))
    np.save(part / f"counts_{args.start:05d}_{args.end:05d}.npy", counts)
    print(f"\nshards {args.start}-{args.end}: species covered {int((counts > 0).sum()):,} | "
          f"images {int(counts.sum()):,} | {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
