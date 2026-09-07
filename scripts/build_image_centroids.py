"""Per-taxon image-embedding centroids from TreeOfLife-200M, for the universe taxa only.

The text embeddings we already use encode what BioCLIP associates with a *name*. An image centroid
encodes average *appearance* -- size, colour, pilosity, proboscis-length correlates -- which is the
morphological trait signal classical pollination ecology relies on and which no other input we have
carries. Corolla depth, the canonical trait, is recorded for fewer than fifty species worldwide, so
an image centroid is the only route to trait matching at this scale.

These are ViT-H/14 embeddings from BioCLIP 2.5 (1024-d), a different model from the ViT-L/14 text
encoder used elsewhere here (768-d), so the two are not in a shared space and the centroid is not a
re-encoding of the name.

The archive is ordered by kingdom -- the low shards are BIOSCAN insects, the high shards Plantae --
so shards must be strided across the range or a contiguous sample returns one kingdom.

The full set is 933 shards and roughly 455 GB. Only rows whose species or genus is in the universe
are kept, so a sample of shards is streamed and centroids accumulated; coverage is reported so the
sample size can be chosen against it rather than guessed.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
REPO = "imageomics/TreeOfLife-200M-Embeddings"
PATTERN = "bioclip-2.5-vith14_float16/train-{:05d}-of-00933.parquet"
DIM = 1024


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shards", type=int, default=32)
    ap.add_argument("--stride", action="store_true",
                    help="spread shards across the whole range; the archive is ordered by kingdom, "
                         "so contiguous shards return a single kingdom")
    ap.add_argument("--universe", default=ROOT / "data/network/modelled_universe.json")
    ap.add_argument("--out", default=ROOT / "data/features")
    args = ap.parse_args()

    u = json.load(open(args.universe))
    plants = [p["label"] for p in u["plants"]]
    polls = [p["label"] for p in u["pollinators"]]
    want = {s: i for i, s in enumerate(plants + polls)}          # species-rank labels
    genus_of = {}
    for i, s in enumerate(plants + polls):
        genus_of.setdefault(s.split()[0], []).append(i)
    n = len(want)
    tot = np.zeros((n, DIM), np.float64)
    cnt = np.zeros(n, np.int64)

    order = (np.linspace(0, 932, args.shards).astype(int) if args.stride
             else np.arange(args.shards))
    for k in order:
        f = hf_hub_download(REPO, PATTERN.format(k), repo_type="dataset")
        pf = pq.ParquetFile(f)
        for g in range(pf.num_row_groups):
            d = pf.read_row_group(g, columns=["emb", "genus", "species", "scientific_name"]).to_pandas()
            sci = d["scientific_name"].fillna("")
            binom = np.where(d["species"].notna() & d["genus"].notna(),
                             d["genus"].fillna("") + " " + d["species"].fillna(""), sci)
            rows = [want.get(b, -1) for b in binom]
            emb = np.stack(d["emb"].to_numpy())
            for r, v in zip(rows, emb):
                if r >= 0:
                    tot[r] += v
                    cnt[r] += 1
        Path(f).unlink(missing_ok=True)                          # 488 MB each; do not accumulate
        cov = int((cnt > 0).sum())
        print(f"  shard {k + 1}/{args.shards}: {cov}/{n} taxa covered ({100 * cov / n:.1f}%)", flush=True)

    C = np.zeros_like(tot, dtype=np.float32)
    nz = cnt > 0
    C[nz] = (tot[nz] / cnt[nz, None]).astype(np.float32)
    out = Path(args.out)
    np.save(out / "image_centroids.npy", C)
    np.save(out / "image_centroid_counts.npy", cnt)
    print(f"[image] {int(nz.sum())}/{n} taxa with a centroid "
          f"(plants {int(nz[:len(plants)].sum())}/{len(plants)}, "
          f"pollinators {int(nz[len(plants):].sum())}/{len(polls)})")
    print(f"[wrote] {out}/image_centroids.npy")


if __name__ == "__main__":
    main()
