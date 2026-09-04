import os
from pathlib import Path
import yaml
from huggingface_hub import hf_hub_download, snapshot_download

REPO_ROOT = Path(__file__).resolve().parents[1]

HF_FALLBACKS = {
    "ppe_ckpt": ("ppe", "last.ckpt"),
    "species_embeddings": ("ppe", "species_embeddings_v2.pt"),
    "species_vocab": ("ppe", "species_vocab.json"),
    "alphaearth": ("ppe", "inputs/alphaearth_2017.parquet"),
    "grid_centroids": ("ppe", "inputs/grid_centroids_0.5deg.csv"),
    "plant_events": ("ppe", "inputs/plant_flowering_events.parquet"),
    "prism_weekly": ("ppe", "inputs/prism_weekly.npz"),
}


def load_config(path=None):
    """Loads configs/pipelines.yaml; any paths entry is overridable with env PIPELINES_<KEY>."""
    path = Path(path) if path else REPO_ROOT / "configs" / "pipelines.yaml"
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for k, v in cfg["paths"].items():
        cfg["paths"][k] = Path(os.environ.get(f"PIPELINES_{k.upper()}", v))
    return cfg


def resolve(cfg, key):
    """Local path if present, else download the HF fallback (see configs/pipelines.yaml hf:)."""
    p = cfg["paths"].get(key)
    if p is not None and Path(p).exists():
        return Path(p)
    if key in HF_FALLBACKS:
        repo, fn = HF_FALLBACKS[key]
        return Path(hf_hub_download(cfg["hf"][repo], fn))
    raise FileNotFoundError(f"{key}: {p} missing and no HF fallback")


def resolve_models_dir(cfg):
    """Directory whose models/ holds the e98 CrossModalVAE code (local vendor copy or HF)."""
    p = Path(cfg["paths"]["ppe_models_dir"])
    if (p / "models" / "cross_modal_vae.py").exists():
        return p
    return Path(snapshot_download(cfg["hf"]["ppe"], allow_patterns="models/*"))
