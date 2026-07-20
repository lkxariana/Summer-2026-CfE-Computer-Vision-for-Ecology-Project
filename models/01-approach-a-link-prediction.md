> Part of [Models](README.md). Prev: [Initial analysis](00-initial-analysis.md) · Next: [Approach B](02-approach-b-sdm-overlap.md).

# Approach A — link prediction (matrix completion)

## Intuition

Like Netflix predicting which movies you'll like from the ones you've rated. Here it's a **plants × pollinators grid** where a cell means "do they interact?" — a few cells known from GloBI, most blank. Describe each species by a short vector (a summary of *where it lives*, so species found in the same places get similar vectors), add the timing-overlap number `Δ` for the pair, and train a classifier on the known cells to fill in the blanks. The classifier *is* the model; everything else just makes its features. PPE's only job here is `Δ` — the feature that says "even though these two co-occur in space, are they active at the same time of year?"

## Steps

1. **Labels**: GloBI flower-visitation pairs → label `1`. Sample non-recorded pairs → label `0`.
2. **Per-species vector**: grid CONUS; build a presence matrix `M` (species × cells) from GBIF; reduce it (PCA, or any dimensionality reduction) → a short vector `v_s` per species.
3. **Timing `Δ`**: overlap of the flowering curve (PPE) and the activity curve (GBIF) for the pair — the scalar from the initial analysis.
4. *(optional)* traits / phylogeny.
5. **Feature vector per pair**: `φ = [v_P, v_A, Δ, traits]`.
6. **Train** a classifier (logistic / random forest / small NN) on the labeled pairs, class-balanced to ~25% positive.
7. **Predict** every pair → the completed matrix of interaction probabilities.

**Pair-level vs spatial.** The basic model predicts per *pair* ("interact anywhere"). To predict per *location* (and run the spatial hold-out), condition the features on place: `Δ(P,A,ℓ)` varies across space, so each cell becomes `P(interact | P, A, ℓ)`. Held-out regions test it — the observed co-occurrence count `N` is undefined where you never sampled (so that arm collapses), while PPE's `Δ` is defined everywhere (so it holds). That gap is the headline.

## Math

- Presence matrix `M ∈ {0,1}^{S×G}` (S species, G grid cells). PCA → `V ∈ ℝ^{S×k}`, with `v_s` = row `s`.
  *(Equivalent co-occurrence view: factor `C = M Mᵀ`, where `C_{ij}` = number of cells shared by species `i` and `j`.)*
- `Δ_{PA} = Σ_t min( f̃_P(t), ã_A(t) ) ∈ [0,1]` (normalized flowering and activity curves).
- Feature vector `φ_{PA} = [ v_P ; v_A ; Δ_{PA} ; traits_{PA} ]`.
- Classifier `ŷ_{PA} = g(φ_{PA}) ∈ [0,1]`, trained by minimizing log-loss over labeled pairs.
- Spatial version: `φ_{PA}(ℓ)` with `Δ_{PA}(ℓ)` → `ŷ_{PA}(ℓ) = g(φ_{PA}(ℓ))`.

## Pseudocode

```python
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier

# Step 2: per-species vectors from spatial co-occurrence
M = presence_matrix(gbif, grid)             # (S species, G cells), 0/1
V = PCA(n_components=15).fit_transform(M)   # (S, 15)
vec = dict(zip(species, V))                 # species -> 15-dim vector

# Step 3 + 5: feature vector for a pair
def features(P, A):
    delta = overlap(ppe_flower(P), circular_kde(gbif_doy[A]))   # timing term Δ
    return concat([vec[P], vec[A], [delta]])                    # + traits (optional)

# Step 6: train on GloBI positives + sampled negatives (~25% positive)
X = [features(P, A) for (P, A) in labeled_pairs]
y = [label for (_, _, label) in labeled_pairs]
clf = RandomForestClassifier().fit(X, y)

# Step 7: fill in any unknown pair
# clf.predict_proba(features(P, A))[1]  ->  P(interact)
```

## Is PPE helping? The comparison

Train the **same classifier twice** on the **same pairs**, changing only whether the timing feature is included:
- **Without PPE (A2):** features = `[v_P, v_A, N]` (co-occurrence vectors + Galiana count).
- **With PPE (A3):** features = `[v_P, v_A, N, Δ]` — one extra column, the timing overlap.

Inputs needed: per-species co-occurrence vectors `v_P, v_A` (GBIF presence matrix → PCA); Galiana `N` (shared-cell count); `Δ` (PPE flowering × bee activity overlap); labels (GloBI = 1, sampled non-pairs = 0); a held-out set of pairs.

Calculation: train each version, predict the held-out pairs, take PR-AUC / ROC-AUC — one number each. **PPE helps if `score(A3) > score(A2)`.** Decisive: under **spatial-block** and **cold-start** hold-outs, `N` is undefined at unseen places (A2 collapses) while `Δ` is defined everywhere (A3 holds). Interpretable check: among pairs with high `N` but low `Δ`, count the false positives A3 removes.
