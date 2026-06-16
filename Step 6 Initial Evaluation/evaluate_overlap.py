"""
Step 6: Initial Evaluation — Does Timing Matter?

Evaluates whether temporal (phenological) information adds discriminative
signal beyond spatial co-occurrence alone, using five metrics from the
project brief.

Outputs:
  - overlap_distribution_threshold.png
  - spatial_vs_temporal_scatter.png
  - bin_size_sensitivity_plot.png
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr

# ── Load saved results ────────────────────────────────────────────────────────

df_results = pd.read_csv('/scratch/ariana.l/overlap_coefficients.csv')
st_df      = pd.read_csv('/scratch/ariana.l/spatial_vs_temporal.csv')
sens_df    = pd.read_csv('/scratch/ariana.l/bin_size_sensitivity.csv')

# Recompute headline variables
THRESHOLD   = 0.3
total       = len(df_results)
low_overlap = (df_results['overlap'] < THRESHOLD).sum()
pct         = 100 * low_overlap / total
edge_means  = df_results.groupby(['plant', 'pollinator'])['overlap'].mean()
low_edges   = (edge_means < THRESHOLD).sum()
pct_edges   = 100 * low_edges / len(edge_means)

print("=" * 60)
print("METRIC 2 — Headline Result")
print("=" * 60)
print(f"Total spatially co-occurring edge×bin combinations: {total}")
print(f"With temporal overlap < {THRESHOLD}: {low_overlap} ({pct:.1f}%)")
print(f"Unique edges with mean overlap < {THRESHOLD}: "
      f"{low_edges} out of {len(edge_means)} ({pct_edges:.1f}%)")
print()
print("Overlap value distribution:")
print(f"  Median : {df_results['overlap'].median():.3f}")
print(f"  Mean   : {df_results['overlap'].mean():.3f}")
print(f"  < 0.3  : {(df_results['overlap'] < 0.3).mean() * 100:.1f}%")
print(f"  < 0.5  : {(df_results['overlap'] < 0.5).mean() * 100:.1f}%")
print(f"  < 0.7  : {(df_results['overlap'] < 0.7).mean() * 100:.1f}%")


# ── Figure 1 — Overlap distribution with threshold ───────────────────────────

fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(df_results['overlap'], bins=30, color='steelblue',
        edgecolor='white', alpha=0.85)
ax.axvline(THRESHOLD, color='crimson', linestyle='--', linewidth=2,
           label=f'Threshold = {THRESHOLD} ({pct:.1f}% below)')
ax.axvline(df_results['overlap'].median(), color='black', linestyle=':',
           linewidth=1.5,
           label=f"Median = {df_results['overlap'].median():.3f}")
ax.set_xlabel('Temporal Overlap Coefficient')
ax.set_ylabel('Count (edge × bin combinations)')
ax.set_title(
    'Distribution of Phenological Overlap Across Spatially Co-occurring Pairs\n'
    f'{low_overlap} of {total} ({pct:.1f}%) fall below the {THRESHOLD} threshold'
)
ax.legend(loc='upper right')
plt.tight_layout()
plt.savefig('/scratch/ariana.l/overlap_distribution_threshold.png',
            dpi=150, bbox_inches='tight')
print("Saved: overlap_distribution_threshold.png")


# ── Figure 2 — Spatial vs. temporal divergence scatter ───────────────────────

sp_corr, sp_p = spearmanr(st_df['spatial_jaccard'], st_df['temporal_overlap'])
pe_corr, pe_p = pearsonr(st_df['spatial_jaccard'],  st_df['temporal_overlap'])

print()
print("=" * 60)
print("METRIC 3 — Spatial vs. Temporal Divergence")
print("=" * 60)
print(f"Spearman ρ = {sp_corr:.2f}  (p = {sp_p:.2f})")
print(f"Pearson  r = {pe_corr:.2f}  (p = {pe_p:.2f})")

fig, ax = plt.subplots(figsize=(9, 7))
ax.scatter(st_df['spatial_jaccard'], st_df['temporal_overlap'],
           s=st_df['n_shared_bins'] * 1.2, alpha=0.6,
           color='steelblue', edgecolor='black', linewidth=0.8)

sp_median = st_df['spatial_jaccard'].median()
tp_median = st_df['temporal_overlap'].median()
ax.axvline(sp_median, color='gray', linestyle='--', alpha=0.5, linewidth=1)
ax.axhline(tp_median, color='gray', linestyle='--', alpha=0.5, linewidth=1)

for _, row in st_df.iterrows():
    label = (f"{row['plant'].split()[0][:3]}."
             f"×{row['pollinator'].split()[0][:3]}.")
    ax.annotate(label,
                (row['spatial_jaccard'], row['temporal_overlap']),
                fontsize=7, alpha=0.85,
                xytext=(5, 5), textcoords='offset points')

ax.set_xlabel('Spatial Overlap (Jaccard index of bins)')
ax.set_ylabel('Temporal Overlap (mean across shared bins)')
ax.set_title(
    f'Spatial vs. Temporal Overlap per Edge (n={len(st_df)})\n'
    f'Spearman ρ = {sp_corr:.2f} (p={sp_p:.2f})  •  '
    f'Pearson r = {pe_corr:.2f} (p={pe_p:.2f})'
)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.grid(alpha=0.3)
ax.text(0.95, 0.05,
        'High spatial,\nlow temporal\n(spatial-only models\nget these wrong)',
        ha='right', va='bottom', fontsize=8, style='italic', color='crimson',
        alpha=0.8,
        bbox=dict(boxstyle='round,pad=0.5', facecolor='mistyrose',
                  edgecolor='crimson', alpha=0.5))
plt.tight_layout()
plt.savefig('/scratch/ariana.l/spatial_vs_temporal_scatter.png',
            dpi=150, bbox_inches='tight')
print("Saved: spatial_vs_temporal_scatter.png")


# ── Figure 3 — Sensitivity to bin size ───────────────────────────────────────

fig, ax1 = plt.subplots(figsize=(9, 5))
color1 = 'steelblue'
ax1.plot(sens_df['bin_size'], sens_df['pct_below_0.3'],
         'o-', color=color1, linewidth=2, markersize=8, label='% below 0.3')
ax1.axhline(20, color='crimson', linestyle='--', linewidth=1.5,
            label='15–20% significance threshold')
ax1.set_xlabel('Spatial Bin Size (degrees)')
ax1.set_ylabel('% of edge×bin combinations with overlap < 0.3', color=color1)
ax1.tick_params(axis='y', labelcolor=color1)
ax1.set_ylim(0, 70)

ax2 = ax1.twinx()
color2 = 'darkorange'
ax2.plot(sens_df['bin_size'], sens_df['median_overlap'],
         's--', color=color2, linewidth=1.5, markersize=7, label='Median overlap')
ax2.set_ylabel('Median Overlap Coefficient', color=color2)
ax2.tick_params(axis='y', labelcolor=color2)
ax2.set_ylim(0, 1)

idx = sens_df[sens_df['bin_size'] == 1.5].index[0]
ax1.annotate('1.5° (primary analysis)',
             xy=(1.5, sens_df.loc[idx, 'pct_below_0.3']),
             xytext=(1.8, 48), fontsize=9,
             arrowprops=dict(arrowstyle='->', color='gray'), color='gray')

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=9)
ax1.set_title(
    'Sensitivity of Temporal Mismatch Signal to Spatial Bin Size\n'
    'Signal persists across all bin sizes tested'
)
plt.tight_layout()
plt.savefig('/scratch/ariana.l/bin_size_sensitivity_plot.png',
            dpi=150, bbox_inches='tight')
print("Saved: bin_size_sensitivity_plot.png")
print()
print("Step 6 complete.")
