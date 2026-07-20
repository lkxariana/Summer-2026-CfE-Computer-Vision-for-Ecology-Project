"""
04_compare_to_original.py -- Stage 2, Step 1, Part 4

Merge the PPE-based edge summary against the original (observation-based)
Step 8 result and compute comparison statistics: bin-count ratio, weighted-
mean overlap shift, correlation, and the pairs with the largest disagreement.
"""

import pandas as pd
from pathlib import Path

DATA_DIR = Path("/scratch/ariana.l/Plant Pollinator Initial Analysis")
OUTPUTS_DIR = Path("/scratch/ariana.l/Part2_PhenoField_Predictions/step1_ppe_overlap_outputs")


def compare(edge_summary_ppe, original):
    comparison = edge_summary_ppe.merge(
        original[["species", "pollinator_species", "n_bins", "overlap_mean", "overlap_weighted_mean"]],
        on=["species", "pollinator_species"],
        how="left",
        suffixes=("_ppe", "_orig"),
    )

    comparison["weighted_mean_diff"] = (
        comparison["overlap_weighted_mean_ppe"] - comparison["overlap_weighted_mean_orig"]
    )
    comparison["n_bins_ratio"] = comparison["n_bins_ppe"] / comparison["n_bins_orig"]

    print("=== Coverage: n_bins, PPE vs. original ===")
    print(comparison[["n_bins_ppe", "n_bins_orig", "n_bins_ratio"]].describe())

    print("\n=== Weighted mean overlap: PPE vs. original ===")
    print(comparison[["overlap_weighted_mean_ppe", "overlap_weighted_mean_orig", "weighted_mean_diff"]].describe())

    print("\n=== Correlation between PPE and original weighted means ===")
    print(comparison[["overlap_weighted_mean_ppe", "overlap_weighted_mean_orig"]].corr())

    print("\n=== Pairs with the largest PPE vs. original disagreement ===")
    top_disagreements = comparison.reindex(
        comparison["weighted_mean_diff"].abs().sort_values(ascending=False).index
    )[["jaccard_rank", "species", "pollinator_species", "overlap_weighted_mean_orig",
       "overlap_weighted_mean_ppe", "weighted_mean_diff", "n_bins_orig", "n_bins_ppe"]]
    print(top_disagreements.head(15))

    return comparison


if __name__ == "__main__":
    edge_summary_ppe = pd.read_csv(OUTPUTS_DIR / "edge_summary_ppe.csv")
    original = pd.read_csv(DATA_DIR / "step8_edge_overlap_summary.csv")

    comparison = compare(edge_summary_ppe, original)

    output_path = OUTPUTS_DIR / "ppe_vs_original_comparison.csv"
    comparison.to_csv(output_path, index=False)
    print(f"\nSaved comparison to {output_path}")
