# 03 — Methodology

ANTHEIA frames plant-pollinator interaction prediction as binary link prediction over a species-pair feature space. The pipeline has three components: (1) constructing spatial existence embeddings for plants and pollinators, (2) computing phenological temporal features from the PPE opportunity surface and GBIF activity curves, and (3) training a logistic regression classifier on the combined feature vector.

---

## 1. Spatial Existence Matrices

For each species, ANTHEIA first asks: which 0.5° × 0.5° CONUS bins has this species been observed in?

**Plant existence matrix F** (6,466 species × 3,162 bins)
- Built from PhenoField flowering records, binned to 0.5° resolution
- Binary: F[species, bin] = 1 if the species has a flowering observation in that bin, 0 otherwise
- Sparsity: 0.980

**Pollinator existence matrix P** (24,939 species × 3,162 bins)
- Built from `pollinator_observations_v2.csv`, binned to 0.5° resolution
- Binary: P[species, bin] = 1 if the species has an occurrence record in that bin, 0 otherwise
- Sparsity: 0.985

Both matrices are restricted to the same 3,162 common CONUS bins before any downstream computation.

**Bin format:** `"lat_lon"` string, e.g. `"34.5_-120.0"`, bin center at 0.5° resolution.

---

## 2. PCA Embeddings

Each existence matrix is independently compressed to 15 dimensions via PCA, yielding a per-species geographic distribution embedding.

**Vf** (plant embedding, 15D)
- PCA fit on F matrix: `PCA(n_components=15, svd_solver='randomized', random_state=42)`
- Variance explained: 39.9%
- Lower variance explained than Vp reflects known iNaturalist observation bias toward human-populated areas, compressing plant distribution structure

**Vp** (pollinator embedding, 15D)
- PCA fit on P matrix, same settings
- Variance explained: 46.2%

The embeddings capture geographic distribution structure: species with similar spatial ranges receive similar vectors. This allows the model to generalize across species pairs that share range characteristics even if they have never been directly observed together.

---

## 3. Shared Bin Count N

For a given (plant, pollinator) pair, N is the count of 0.5° bins in which both species have been observed:

```
N = F_common[plant_idx] @ P_common[pollinator_idx]
```

where `F_common` and `P_common` are both restricted to the 3,162 common bins. N is a scalar feature encoding raw spatial co-occurrence intensity — the Galiana co-occurrence count.

**Critical:** always use the common-bin-restricted matrices for this computation. Using the full F and P arrays directly causes silent dimension mismatches.

---

## 4. Phenological Temporal Features

The Spatial Baseline uses only [Vf, Vp, N]. The ANTHEIA models add temporal features derived from the PPE opportunity surface (plant side) and GBIF activity curves (pollinator side).

### Flowering curves f_curves

Per-plant 52-week flowering probability curves, built from the full PPE opportunity surface (6,697 parquet files). For each plant species:

1. Stream all parquet files from `/scratch/ariana.l/ppe-outputs/opportunity_surface/part_*.parquet`
2. Filter to target species, groupby week, take mean `norm` value
3. Normalize to sum = 1 across 52 weeks

**Week formula:** `week = (doy - 1) // 7`, clipped to [0, 51]. Do not use `doy // 7`.

f_curves represent climate-driven flowering probability — independent of where observers happened to be — making them a genuine phenological signal rather than a proxy for observation effort.

### Activity curves a_curves

Per-pollinator 52-week activity curves, built from `pollinator_observations_v2.csv`. For each pollinator species:

1. Chunked processing of occurrence records, applying the same week formula
2. Count observations per (species, week)
3. Normalize to sum = 1 across 52 weeks

a_curves are derived from opportunistic citizen science observations, which introduces observation-density bias. This is the core asymmetry motivating SDM integration — see `docs/06-extensions-and-next-steps.md`.

### Temporal overlap scalar Δ

For a given (plant, pollinator) pair:

```
Δ = Σ_t min(f̃_plant(t), ã_pollinator(t))
```

where `f̃` and `ã` are the normalized 52-week curves. This is the coefficient of overlapping (Ridout & Linkie 2009) — the shared area under the two normalized curves. Δ = 0 means phenologically incompatible; Δ = 1 means perfectly synchronized.

Δ encodes per-pair phenological alignment directly: a value the model cannot reconstruct from species embeddings alone, since Vf and Vp encode geographic distribution, not timing.

### Spatiotemporal plant embedding V_δ

An alternative to the scalar Δ: PCA compression of the full per-plant PPE opportunity surface matrix (bins × weeks) to 4D or 15D, yielding a spatiotemporal embedding that captures how flowering probability varies across both space and time.

- **V_δ (4D):** 4-component PCA of the opportunity surface — adds spatial structure to the temporal signal but loses the plant-pollinator alignment that Δ directly encodes
- **V_δ (15D):** 15-component PCA — 48.6% of spatiotemporal variance explained; best PR-AUC result across all models

### PMf — continuous plant existence matrix

An alternative to the binary F matrix: replacing each binary cell with the PPE flowering probability at that bin, averaged across weeks, then PCA'd to 15D. This is `Vf_prob` — a continuous spatiotemporal representation of the plant existence distribution.

---

## 5. Feature Vectors

| **Model** | **Feature Vector** | **Dim** |
|---|---|---|
| Spatial Baseline | Vf + Vp + N | 31D |
| ANTHEIA-Scalar | Vf + Vp + N + Δ | 32D |
| ANTHEIA-4D | Vf + Vp + N + V_δ (4D) | 35D |
| ANTHEIA-15D | Vf + Vp + N + V_δ (15D) | 46D |
| ANTHEIA-PMf | Vf_prob + Vp + N | 31D |

*ANTHEIA-Scalar is the core model. ANTHEIA-4D and ANTHEIA-15D extend it with richer spatiotemporal plant embeddings. ANTHEIA-PMf replaces the binary plant existence matrix with a continuous PPE flowering probability matrix.*

All models use the same Vp (pollinator spatial embedding) and N (shared bin count). The models differ only in how the plant side is represented and whether temporal information is included.

---

## 6. Pair Universe and Label Construction

The shared pair universe is constructed by intersecting species coverage across all five models:

- **Plants:** species present in Vf, f_curves, V_δ (4D), V_δ (15D), and Vf_prob → 6,348 plant species
- **Pollinators:** species present in Vp and a_curves → 24,939 pollinator species
- **Positive pairs:** GloBI interactions within this coverage → 139 pairs
- **Negative pairs:** randomly sampled non-interacting pairs at 3:1 ratio → 417 pairs
- **Total:** 556 pairs

A single 80/20 stratified train/test split is shared across all five models for fair comparison. Negative pairs are resampled per seed; the train/test split is fixed.

**Negative sampling:** `np.random.default_rng(seed)` — not `np.random.seed()`. The `default_rng` interface is required for reproducibility across the 5 seeds [42, 0, 1, 2, 3].

---

## 7. Classifier

Logistic regression on the per-pair feature vector, fit independently for each model and each seed.

```python
LogisticRegression(max_iter=1000, random_state=seed)
```

Evaluated on the held-out 20% test set:
- **ROC-AUC:** ranking metric; measures how well the model separates positive from negative pairs
- **PR-AUC:** precision-recall metric; more sensitive to performance on positive pairs, and more informative under class imbalance

Results reported as 5-seed mean ± std.

The choice of logistic regression is deliberate: it is a linear classifier operating on fixed feature vectors, which means any performance gain from adding a feature (e.g. Δ) reflects genuine signal in that feature rather than increased model capacity. The ablation results are therefore interpretable as direct feature importance measurements.
