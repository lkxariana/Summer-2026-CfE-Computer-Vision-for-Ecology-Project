> Part of [Models](README.md). Prev: [Approach A](01-approach-a-link-prediction.md).

# Approach B — SDM-overlap (availability fields)

## Intuition

Don't predict the interaction directly — predict where and when each species is **available**, then overlap. The plant's **flowering field** (PPE) says where/when it's blooming; the pollinator's **activity field** (a learned SDM on GBIF) says where/when it's flying. Their overlap is the opportunity field. You need no interaction labels to *build* it — only to *check* it. It's the same core idea as the initial analysis, with two upgrades: the pollinator side is a proper learned field (so it generalizes to places with few records, instead of a flat regional curve), and you keep the full spatiotemporal surface rather than collapsing to one number.

## Steps

1. **Plant field** `f_P(ℓ,t)`: frozen PPE backbone + probe.
2. **Pollinator field** `a_A(ℓ,t)`: a presence-only SDM on `A`'s GBIF points. Variants (the ablation):
   - **B0** raw KDE of dates (no SDM) — baseline;
   - **B1** coordinates-only INR (SINR-style);
   - **B2** + climate covariates;
   - **B3** + PPE representation as input (does the plant-trained representation transfer?);
   - **B4** joint SINR across many pollinators (joint vs stacked).
3. **Opportunity field**: overlap of the two fields over `(ℓ,t)`.
4. **Validate (presence-only)**: held-out co-observations should rank above effort-matched background (Boyce / AUC-against-background); spatial and cold-start hold-outs; the plant field separately checked against **USA-NPN** (the one clean gold standard).

## Math

- `f_P(ℓ,t)` from PPE; `a_A(ℓ,t) = σ(g_A(ℓ,t))` from the SDM `g_A`.
- SDM trained presence-only (assume-negative loss): minimize
  `Σ_{presences} −log σ(g_A) + Σ_{background} −log(1 − σ(g_A))`,
  with background drawn from observer effort (target-group background).
- Opportunity: `opp(ℓ,t) = f_P(ℓ,t) · a_A(ℓ,t)` (product), or `Σ_t min(f̃_P, ã_A)` per cell if you want the circular overlap as the per-location score.
- Validation: Boyce / AUC of `opp` over held-out positives vs background; plant field scored against NPN.

## Pseudocode

```python
# Step 1: plant flowering field (frozen PPE)
f = ppe_flower(P, cells, days)              # (n_cells, 365)

# Step 2: pollinator availability field (presence-only SDM)
sdm = train_sdm(
    gbif_points[A],                         # presence-only occurrences
    features=[coords, climate, ppe_repr],   # B1 coords / B2 +climate / B3 +PPE representation
    loss="assume_negative",                 # presence-background, no pseudo-negatives
)
a = sdm.predict(cells, days)                # (n_cells, 365)

# Step 3: opportunity field
opp = f * a                                 # or circular-overlap per cell

# Step 4: validate (presence-only) + plant field vs NPN
score = boyce(opp, heldout_interactions[(P, A)], background)
plant_check = boyce(f, npn_flowering[P], npn_background)
```

## Is PPE helping? Two comparisons

**(1) PPE flowering as the availability field.** Build the opportunity field twice, the **same pollinator field both times**, changing only flowering:
- With PPE: `opp_PPE = f_PPE(ℓ,t) · a_field(ℓ,t)`.
- Baseline: `opp_date = f_date(ℓ,t) · a_field(ℓ,t)` (`f_date` from the plant's observation dates).

Score Boyce / AUC-against-background over held-out interactions for each; PPE helps if `opp_PPE` ranks higher, decisive at unseen locations. Also the **term ablation** — compare `f·a` vs `f`-only vs `a`-only; both terms should matter (if `a`-only ties, flowering adds nothing; if `f`-only ties, the pollinator side adds nothing).

**(2) PPE representation as input to the pollinator SDM (B3 vs B2).** Build the bee's distribution model twice:
- **B2:** inputs `[coords, climate]`.
- **B3:** inputs `[coords, climate, PPE representation]`.

Inputs needed: the bee's GBIF points + background; climate; PPE's representation at those points. Calculation: train each, predict the bee's held-out occurrences, take a presence-only score (Boyce / AUC-against-background) — one number each. **PPE's representation transfers if `score(B3) > score(B2)`.**

Separately, the plant flowering field is checked directly against **USA-NPN** (the one clean gold standard).
