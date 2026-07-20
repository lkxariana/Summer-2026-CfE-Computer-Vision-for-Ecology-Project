> Part of [Models](README.md). Next: [Approach A](01-approach-a-link-prediction.md).

# Initial analysis — the computed opportunity surface (go/no-go)

## Intuition

A bee can only visit a flower when the flower is **blooming** and the bee is **flying**, in the **same place**. PPE tells you when a plant is flowering (anywhere, any day of year); the bee's sighting dates tell you when it is active. Multiply the two and you get an *opportunity* surface — high only where both are present at once. If this is real signal, the places and dates where the pair was actually recorded interacting should land where opportunity is high. The decisive test: PPE should get this right even at locations with **no records**, where a naive "observed dates" baseline is blind. If it cannot beat that baseline, the premise is weak and you stop.

This is the simplest possible model — nothing is trained except the flowering probe and a kernel density. It also introduces the three building blocks (flowering curve, activity curve, overlap) that Approaches A and B reuse.

## Steps

1. Pick a handful of well-documented CONUS pairs (plant `P`, bee `A`).
2. **Flowering curve** `f_P(ℓ,t)`: query the frozen PPE backbone + probe across day-of-year at location `ℓ`.
3. **Activity curve** `a_A(t)`: kernel density of `A`'s GBIF sighting dates (pooled regionally), using a **circular** kernel (the year wraps).
4. **Opportunity**: `opp(ℓ,t) = f_P(ℓ,t) · a_A(t)`.
5. **Validate (presence-only)**: held-out co-observations of the pair should score higher opportunity than effort-matched background points — Boyce index or AUC-against-background — *including at held-out locations*.
6. **Baseline**: repeat the check with an observation-date window instead of PPE. PPE must win, especially at unseen locations.

## Math

- `f_P(ℓ,t) ∈ [0,1]`, `t = 1…365` (PPE probe).
- `a_A(t) ≥ 0`, normalized so `Σ_t a_A(t) = 1` (circular KDE of sighting dates).
- `opp(ℓ,t) = f_P(ℓ,t) · a_A(t)`.
- Validation, given held-out positives `{(ℓ_i,t_i)}` and background `{(ℓ_j,t_j)}` drawn from observer effort:
  `AUC = P( opp(positive) > opp(background) )`, or the Boyce index of `opp` over positives vs background. Restrict positives/background to **unobserved cells** for the decisive spatial test.

## Pseudocode

```python
for (P, A) in well_documented_pairs:
    f = ppe_flower(P, cells, days)          # (n_cells, 365) flowering prob, from PPE
    a = circular_kde(gbif_doy[A], days)     # (365,) activity over the year, sums to 1
    opp = f * a[None, :]                    # (n_cells, 365) opportunity surface

    pos = heldout_interactions[(P, A)]      # (cell, day) of held-out co-observations
    bg  = sample_background(effort)         # (cell, day) ~ where/when observers were active

    if boyce(opp, pos, bg) <= boyce(date_window(P, A), pos, bg):
        stop("PPE does not beat the date baseline — premise weak")
    # decisive: also evaluate with pos/bg restricted to cells with no records for this pair
```

## Is PPE helping? The comparison

Build the **same opportunity surface twice**, changing only the flowering term:
- **With PPE:** `opp_PPE(ℓ,t) = f_PPE(ℓ,t) · a(t)` — flowering from PPE.
- **Without PPE (baseline):** `opp_date(ℓ,t) = f_date(ℓ,t) · a(t)` — `f_date` is a window/KDE of the plant's own observation dates (when it was photographed flowering). The bee activity `a(t)` is **identical** in both; only flowering changes.

Inputs needed: PPE flowering field; the plant's observation flowering dates (to build `f_date`); the bee's GBIF dates (for `a`); held-out interaction records (test positives); effort-matched background points.

Calculation: for each surface, compute the Boyce index (or AUC-against-background) over the held-out interactions vs background — one number each. **PPE helps if `Boyce(opp_PPE) > Boyce(opp_date)`.** Decisive test: restrict positives and background to cells with **no records of the pair** — there `f_date` is flat/blind, so only PPE can rank them.
