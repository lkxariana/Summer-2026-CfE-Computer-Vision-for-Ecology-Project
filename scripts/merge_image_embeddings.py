import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from antheia.config import load_config
from antheia.store import FeatureStore

cfg = load_config()
store = FeatureStore(cfg)
part = cfg["cache_dir"] / "img_parts"
n_sp, dim = len(store.plants) + len(store.polls), 768
sums = np.zeros((n_sp, dim), np.float64)
counts = np.zeros(n_sp, np.int64)
files = sorted(part.glob("sums_*.npy"))
for f in files:
    sums += np.load(f).astype(np.float64)
    counts += np.load(str(f).replace("sums_", "counts_"))
print(f"merged {len(files)} partitions")

cov = counts > 0
mean = np.zeros_like(sums)
mean[cov] = sums[cov] / counts[cov][:, None]
norms = np.linalg.norm(mean, axis=1, keepdims=True)
mean = np.divide(mean, norms, out=np.zeros_like(mean), where=norms > 0)
n_pl = len(store.plants)
np.save(cfg["cache_dir"] / "bioclip2_img_plants.npy", mean[:n_pl].astype(np.float16))
np.save(cfg["cache_dir"] / "bioclip2_img_polls.npy", mean[n_pl:].astype(np.float16))
np.save(cfg["cache_dir"] / "bioclip2_img_counts.npy", counts)
print(f"plants covered {int(cov[:n_pl].sum()):,}/{n_pl:,} "
      f"({cov[:n_pl].mean():.1%}) | pollinators {int(cov[n_pl:].sum()):,}/{n_sp - n_pl:,} "
      f"({cov[n_pl:].mean():.1%}) | images {int(counts.sum()):,}")
print(f"median images/covered species: {np.median(counts[cov]):.0f}")
