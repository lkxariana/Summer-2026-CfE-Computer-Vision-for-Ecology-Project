"""Pollinator-side negative-sampling pool (protocol invariant, 2026-09-09).

On the cold-pollinator, cold-both and warm splits the evaluated pollinators must be *unseen*: they may not appear in
training even as sampled negatives, and the re-ranker's hard negatives must come from the training pollinators, not from
the evaluation candidate set. Every sampler that draws pollinator negatives draws from this pool. run_ladder / run_fusion set
it from the split's training pollinators; when unset (cold-plant, local networks) the pool is every pollinator, so those
results are unchanged.
"""
import numpy as np

_POOL = None


def set_pool(idx):
    global _POOL
    _POOL = None if idx is None else np.asarray(idx, np.int64)


def pool(n_total):
    return np.arange(n_total) if _POOL is None else _POOL


def sample(rng, n, n_total):
    p = pool(n_total)
    return p[rng.integers(0, len(p), n)]
