> Part of [Idea 6](README.md). Next: [PPE integration](02-ppe-integration.md).

# Overview: contribution, approach, and positioning

## One-line contribution

PPE's continuous phenology field predicts plant–pollinator interaction *opportunity* across space and time — including at locations and dates with no records — validated presence-only against held-out co-observations, beating GDD / observation-date / EO baselines. The contribution is the **input**: a validated, spatially-continuous, temporally-resolved phenological opportunity term.

## Approach

Use PPE to rank where and when interaction is phenologically *possible* for a **known** pair, then validate that held-out co-observations land in high-opportunity space-time. PPE is most reliable at the flowering **peak**, and held-out co-observations concentrate near peaks (same conspicuousness bias) — so the evaluation lives where PPE is trustworthy.

The alternative — using PPE to assert phenological *non*-overlap (forbidden links) — is suboptimal. Those calls fall at the window **edges**, where PPE is least reliable: its backbone is trained on iNaturalist (Pheno3M), which over-represents conspicuous peak bloom and under-samples the shoulders, so asserting non-overlap there risks manufacturing false negatives. Ranking opportunity uses PPE in the direction it is reliable; asserting absence does not.

## Where this lives (the framework we build on)

This is **SDM extended to biotic interactions** — specifically the sub-genre of *predicting interaction networks across space*. The lineage and how each piece is used:

- **Wisz et al. 2013** (Biol. Reviews) and **Dormann et al. 2018** ("Biotic interactions in SDM: 10 questions…", Glob. Ecol. Biogeogr., https://doi.org/10.1111/geb.12759) — foundational + cautionary. Dormann is the methods-hygiene cite; existing methods "must assume biotic interactions are constant in space and time." PPE removes that assumption — that is the gap.
- **Poisot et al. 2021** (Phil. Trans. R. Soc. B, https://doi.org/10.1098/rstb.2021.0063) — "A roadmap toward predicting species interaction networks (across space and time)." The direct framework. They predict the species pool first, then interactions, and demonstrate exactly our move: applying the model to pairs never observed to co-occur identified 1546 new possible interactions (48% between pairs with no observed co-occurrence). They use latent variables; we supply a phenology field. Their result sets the bar the spatial-hold-out ablation must clear.
- **Blanchet et al. 2020** (Ecol. Lett., "Co-occurrence is not evidence of ecological interactions") and **Coelho et al. 2024** (Nat. Ecol. Evol., https://doi.org/10.1038/s41559-023-02254-y) — the framing constraint. Coelho quantifies it: only ~20% of co-occurrences are real interactions. This is why the claim must be *opportunity*, never *interaction strength*. Naming this distinction correctly is how you signal you know the literature.

Prior work predicts opportunity from latent factors, climate, or static SDMs that hold phenology constant in space and time. **PPE's contribution is the temporally-resolved, spatially-continuous, validated phenological opportunity term those approaches lack** — the phenology-aware opportunity term static interaction-SDMs are missing.
