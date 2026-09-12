"""Every location outside the repository, in one place.

Set ANTHEIA_DATA to the root that holds the frozen inputs (text embeddings, occurrence fields, SDM outputs, caches);
everything else is derived. ANTHEIA_RUNS moves the bundle directory (default: <repo>/runs, gitignored).
"""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("ANTHEIA_DATA", "/scratch/cher/antheia-data"))

TEXT_DIR = DATA_ROOT / "text_embeddings"                       # plants_bioclip2.pt, polls_bioclip2.pt
SDM_DIR = DATA_ROOT / "pollinator_sdm"                          # occurrences, deliverable surfaces, joint fields
FIELD_DIR = SDM_DIR / "joint_field" / "field_v2"                # the joint presence field used by the final model
PLANT_OCC = DATA_ROOT / "plant_occ"
OPPORTUNITY_SURFACE = DATA_ROOT / "opportunity_surface_e98"     # PPE flowering-opportunity surface (parquet parts)
HF_CACHE = Path(os.environ.get("HF_HOME", "/scratch/cher/hf_cache"))

FEATURES = REPO_ROOT / "data" / "features"                      # built by pipelines/, gitignored
NETWORK = REPO_ROOT / "data" / "network"
SPLITS = REPO_ROOT / "data" / "splits"                          # frozen, committed
RUNS = Path(os.environ.get("ANTHEIA_RUNS", REPO_ROOT / "runs"))  # bundles, gitignored
