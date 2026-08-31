import argparse
import os
import sys
import time
from pathlib import Path
import numpy as np

os.environ.setdefault("HF_HOME", "/scratch/cher/hf_cache")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from antheia.config import load_config
from antheia.store import FeatureStore


def embed(names, model, tokenizer, device, batch):
    import torch
    out = []
    with torch.no_grad():
        for i in range(0, len(names), batch):
            toks = tokenizer([f"a photo of {n}." for n in names[i:i + batch]]).to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                e = model.encode_text(toks)
            e = e / e.norm(dim=-1, keepdim=True)
            out.append(e.float().cpu().numpy())
            if (i // batch) % 20 == 0:
                print(f"  {i}/{len(names)}", flush=True)
    return np.vstack(out).astype(np.float16)


def main():
    ap = argparse.ArgumentParser(description="BioCLIP text-tower embeddings for all species names")
    ap.add_argument("--config", default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--model", default="hf-hub:imageomics/bioclip")
    args = ap.parse_args()

    import open_clip
    import torch

    cfg = load_config(args.config)
    store = FeatureStore(cfg)
    print(f"embedding {len(store.plants)} plants + {len(store.polls)} pollinators on {args.device}", flush=True)

    t0 = time.time()
    model, _, _ = open_clip.create_model_and_transforms(args.model)
    tokenizer = open_clip.get_tokenizer(args.model)
    model = model.to(args.device).eval()
    print(f"model loaded ({time.time() - t0:.0f}s)", flush=True)

    cache = cfg["cache_dir"]
    cache.mkdir(parents=True, exist_ok=True)
    np.save(cache / "bioclip_text_plants.npy", embed(store.plants, model, tokenizer, args.device, args.batch))
    print("plants done", flush=True)
    np.save(cache / "bioclip_text_polls.npy", embed(store.polls, model, tokenizer, args.device, args.batch))
    print(f"Saved embeddings to {cache} ({time.time() - t0:.0f}s total)")


if __name__ == "__main__":
    main()
