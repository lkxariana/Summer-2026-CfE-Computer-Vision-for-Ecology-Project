import os
from pathlib import Path
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_config(path=None):
    """Loads a yaml config and resolves all paths to absolute Paths (env ANTHEIA_DATA_ROOT overrides data_root)."""
    path = Path(path) if path else REPO_ROOT / "configs" / "default.yaml"
    with open(path) as f:
        cfg = yaml.safe_load(f)
    root = Path(os.environ.get("ANTHEIA_DATA_ROOT", cfg["data_root"]))
    cfg["data_root"] = root
    cfg["paths"] = {k: root / v for k, v in cfg["paths"].items()}
    cfg["edges"] = REPO_ROOT / cfg["edges"]
    cfg["cache_dir"] = REPO_ROOT / cfg["cache_dir"]
    return cfg
