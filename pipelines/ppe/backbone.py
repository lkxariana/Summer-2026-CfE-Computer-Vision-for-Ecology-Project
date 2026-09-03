import json
import sys
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from scipy.spatial import cKDTree
from pipelines.config import resolve, resolve_models_dir

Z_DYN = 192


def _model_class(cfg):
    root = str(resolve_models_dir(cfg))
    sys.path.insert(0, root)
    try:
        for m in [m for m in list(sys.modules) if m == "models" or m.startswith("models.")]:
            del sys.modules[m]
        from models.cross_modal_vae import CrossModalVAE
    finally:
        sys.path.remove(root)
    return CrossModalVAE


def load_e98(cfg, device="cpu"):
    """Must build from the bwei model code (vendor copy / HF mirror): the cherd-tree class
    silently drops dynamic_encoder.null_species_emb, the parameter species dropout trains."""
    ckpt = torch.load(resolve(cfg, "ppe_ckpt"), map_location="cpu", weights_only=False)
    mcfg = OmegaConf.to_container(OmegaConf.create(ckpt["hyper_parameters"]).model, resolve=True)
    mcfg["species_embedding_path"] = str(resolve(cfg, "species_embeddings"))
    model = _model_class(cfg)(mcfg)
    state = {k[len("model."):]: v for k, v in ckpt["state_dict"].items() if k.startswith("model.")}
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert not missing and not unexpected, (missing, unexpected)
    for p in model.parameters():
        p.requires_grad = False
    return model.to(device).eval()


def encode_zdyn(model, species_id, prism, alphaearth, device="cpu", batch=4096):
    """z_dynamic [N,192] from (vocab species_id, prism [N,365,7], alphaearth [N,64]) — the only
    inputs with weights in e98; doy / lat-lon / climplicit stay None."""
    out = np.empty((len(species_id), Z_DYN), np.float32)
    with torch.no_grad():
        for i in range(0, len(species_id), batch):
            s = slice(i, i + batch)
            _, zd = model.forward_field(
                torch.as_tensor(np.asarray(species_id[s]), dtype=torch.long, device=device),
                torch.nan_to_num(torch.as_tensor(np.asarray(prism[s], np.float32), device=device)),
                torch.as_tensor(np.asarray(alphaearth[s], np.float32), device=device))
            out[s] = zd.float().cpu().numpy()
    return out


def load_vocab(cfg):
    return json.load(open(resolve(cfg, "species_vocab")))


def load_grid(cfg):
    return pd.read_csv(resolve(cfg, "grid_centroids"))


def load_prism_weekly(cfg):
    """(windows [K,365,7], {(cell_idx, week): row}) from out_dir, falling back to the HF copy."""
    p = cfg["paths"]["out_dir"] / "prism_weekly.npz"
    npz = np.load(p if p.exists() else resolve(cfg, "prism_weekly"))
    by_key = {(int(c), int(w)): i for i, (c, w) in enumerate(zip(npz["cell_idx"], npz["week"]))}
    return npz["window"], by_key


def alphaearth_by_cell(cfg, grid):
    ae = pd.read_parquet(resolve(cfg, "alphaearth"))
    _, nn = cKDTree(ae[["lat", "lon"]].values).query(grid[["centroid_lat", "centroid_lon"]].values)
    return ae[[f"emb_{i:02d}" for i in range(64)]].values[nn].astype(np.float32)
