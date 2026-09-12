# Evaluation protocol

Fixed before the test split was scored. Every number in the paper follows it.

## Task

Cold-start retrieval. For a plant held out of training, rank all 13,124 candidate pollinators in the
modelled universe. This is Setting B of Stock et al. (2018, *Neural Computation* 30:2245–2283): the
plant has no training interactions, so only its features can carry the prediction. Held-out taxa are
drawn by a degree-stratified 75/10/15 split, frozen in `data/splits/` and never regenerated per model.

## Primary metric — normalised recall@10

Recall@k is capped below 1 whenever a plant has more than k recorded partners: a plant with 40
partners scores at most 0.25 at k=10 however good the ranking. 29% of held-out plants exceed k=10,
so plain recall@10 partly measures the degree distribution of the test set rather than the model.
We therefore report

    nrecall@k = |relevant ∩ top-k| / min(|relevant|, k)

which is the achievable fraction, and equals recall@k for plants with at most k partners. This is the
standard remedy for the truncation bias of recall at fixed cutoff (Järvelin & Kekäläinen 2002, *ACM
TOIS* 20:422–446). Plain recall@{10,20} is reported alongside for comparability, together with
nDCG@{10,20} and the median rank of the first true partner.

Under positive-unlabelled data every recall figure is a lower bound: a correctly predicted but
unrecorded interaction counts as a miss.

## Two deliverables, two discrimination metrics

The model is asked to do two jobs, and they impose different demands on its scores.

A **retrieval system** answers "given this plant, which pollinators should I look for" — one plant at
a time, so only the ordering within a plant matters. **MAP**, the mean over held-out plants of the
average precision of the full ranking, is the cutoff-free summary of that, and sits alongside
nrecall@10 as a per-plant average.

A **predicted metaweb** is the whole plant-by-pollinator matrix thresholded once and analysed as a
network, which is what link prediction is usually used for in this literature (Strydom et al. 2022,
*Methods Ecol Evol* 13:2308). A single global threshold requires scores to be comparable between
plants. **Pooled PR-AUC** measures that, and a method can rank well per plant while being unusable
this way.

The two are reported together, and where they disagree the disagreement is the result: congeneric
transfer has the best MAP and the third-best PR-AUC because its score — the share of a plant's
congeners visiting a pollinator — is on a scale that depends on how many congeners the plant has.

Their baselines differ and the values are not comparable to each other. Random ranking gives MAP
0.0022 here, a macro-average over plants of differing prevalence, against pooled PR-AUC 0.0013, the
network's connectance. MAP also weights every plant equally whatever its degree.

## Pooled PR-AUC at true connectance

All candidates are scored for retrieval anyway, so the pooled set is the complete (held-out plant ×
candidate) block and needs no negative sampling; the positive rate is the network's own connectance,
stated with each table. ROC-AUC appears only in the supplement: at a positive rate near 0.1% it is
insensitive to the negative class (Poisot 2023, *Methods Ecol. Evol.* 14:1333–1345).

## Evidence tier

Models train on all interactions. Scoring is restricted to **Tier A** — those supported by a
flower-visitation term (`visitsFlowersOf`, `pollinates`) rather than a general-association term
(`visits`, `interactsWith`). Training on both keeps the maximum signal; evaluating on Tier A means a
miss is a miss against evidence that the interaction is pollination rather than co-occurrence on a
plant. Tier A+B results are reported in the supplement, and the ranking of methods is identical under
both.

## Uncertainty

Bootstrap over held-out plants, 1,000 resamples. Model-versus-model differences use a paired
bootstrap on per-plant scores: every method is scored on the identical plants, and between-plant
variance is large enough to hide real differences in an unpaired test.

## Evaluation sets

| set | definition |
|---|---|
| held-out plants | the frozen leave-plant-out test split |
| expert field networks | interactions only Web of Life supports, removed from training; scored on the plants left with no training interaction |
| specimen records | the same for `gbif-us-bees` |

The two source holdouts test transfer to independently assembled data, and stay cold-start so they
are comparable with the primary set.
