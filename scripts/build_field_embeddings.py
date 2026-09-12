"""Pollinator species vectors from the trained spatio-temporal SDM, aligned to the modelled universe.

The pollinator SDM (`pipelines/sdm/build_deliverable.py`) is a SINR: a shared encoder over
(Fourier lat/lon, Climplicit monthly climate, Fourier week) and a per-species linear head. Each
row of `cls.weight` is a species' learned spatio-temporal influence -- the direction in encoder
space along which that species is present. Until now only the SDM's *output* surfaces were used,
projected on an SVD grid basis; this exports the *embedding* itself.

  poll_field.npy         [n_polls, 256] float32   head rows for observed species, ridge-imputed
                                                   from BioCLIP-2 text for the rest (LE-SINR's idea)
  poll_field_direct.npy  [n_polls] bool            True where the row is a trained head vector

The held-out R^2 of the text->field ridge is printed: it is the share of the field embedding that
taxonomy (via the text encoder) already explains, i.e. how much of this feature is not identity.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from antheia.paths import DATA_ROOT, HF_CACHE

ROOT = Path(__file__).resolve().parents[1]
SDM = Path(str(DATA_ROOT) + "/pollinator_sdm")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--head", default=SDM / "deliverable_universe/model_head.pt")
    ap.add_argument("--occ", default=SDM / "pollinator_occ_gbifv3.npz")
    ap.add_argument("--text", default=str(DATA_ROOT) + "/text_embeddings/polls_bioclip2.pt")
    ap.add_argument("--universe", default=ROOT / "data/network/modelled_universe.json")
    ap.add_argument("--out", default=ROOT / "data/features")
    ap.add_argument("--alpha", type=float, default=10.0)
    args = ap.parse_args()

    polls = [p["label"] for p in json.load(open(args.universe))["pollinators"]]
    q2i = {s: i for i, s in enumerate(polls)}
    W = torch.load(args.head, map_location="cpu", weights_only=True)["cls.weight"].numpy()  # [5822, 256]
    occ_names = np.load(args.occ)["names"]
    assert len(occ_names) == len(W), (len(occ_names), W.shape)

    field = np.zeros((len(polls), W.shape[1]), np.float32)
    direct = np.zeros(len(polls), bool)
    for i, name in enumerate(occ_names):
        j = q2i.get(str(name))
        if j is not None:
            field[j] = W[i]
            direct[j] = True
    print(f"[align] {direct.sum()}/{len(polls)} universe pollinators have a trained head vector", flush=True)

    td = torch.load(args.text, map_location="cpu", weights_only=False)
    assert list(td["names"]) == polls, "text embeddings are not in universe order"
    T = td["embeddings"].numpy().astype(np.float64)
    T /= np.linalg.norm(T, axis=1, keepdims=True) + 1e-12

    # how much of the field embedding is already in the text embedding (held-out, direct species)
    Xd, Yd = T[direct], field[direct].astype(np.float64)
    pred = np.zeros_like(Yd)
    for tr, te in KFold(5, shuffle=True, random_state=0).split(Xd):
        pred[te] = Ridge(alpha=args.alpha).fit(Xd[tr], Yd[tr]).predict(Xd[te])
    w = Yd.var(0)
    r2 = 1 - ((Yd - pred) ** 2).sum(0) / ((Yd - Yd.mean(0)) ** 2).sum(0)
    print(f"[text->field] held-out variance-weighted R^2 = {float((w * r2).sum() / w.sum()):.3f}  "
          f"(cosine to truth, mean {np.mean(np.sum(pred * Yd, 1) / (np.linalg.norm(pred, axis=1) * np.linalg.norm(Yd, axis=1) + 1e-12)):.3f})",
          flush=True)

    field[~direct] = Ridge(alpha=args.alpha).fit(Xd, Yd).predict(T[~direct]).astype(np.float32)
    out = Path(args.out)
    np.save(out / "poll_field.npy", field)
    np.save(out / "poll_field_direct.npy", direct)
    print(f"[wrote] {out}/poll_field.npy {field.shape}, poll_field_direct.npy "
          f"({(~direct).sum()} rows imputed from text)", flush=True)


if __name__ == "__main__":
    main()
