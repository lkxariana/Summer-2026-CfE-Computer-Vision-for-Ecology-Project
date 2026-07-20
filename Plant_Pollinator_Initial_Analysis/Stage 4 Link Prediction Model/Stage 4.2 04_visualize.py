"""
04_visualize.py
Stage 4 Visualization — Combined figure (v4)

Three-row layout:
  Row 0: Ground Truth | A2 (static) | PPE Delta x 4 seasons
  Row 1: A2 latitude profile (static bar) | PPE Delta Hovmoller (lat x week)
  Row 2: Mean A3 - A2 difference map (full year)

Requires pred_df and delta_df from 02 and 03, and gt from 01.
Outputs: antheia_combined_figure_v4.png
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

PLANT = 'Achillea millefolium'
SPATIAL_WEEKS = {
    13: 'Spring\n(Wk13)',
    26: 'Summer\n(Wk26)',
    39: 'Fall\n(Wk39)',
    52: 'Winter\n(Wk52)'
}

# --- Hovmoller: delta (latitude × week) ---
delta_df['lat_band'] = np.floor(delta_df['lat']).astype(int)
hovmoller_delta = delta_df.groupby(['lat_band', 'week'])['delta'].mean().unstack(fill_value=np.nan)

# --- A2 latitude profile (static) ---
a2_profile = pred_df.groupby('lat')['pred_a2'].mean().sort_index()

# --- Layout ---
fig = plt.figure(figsize=(26, 16), facecolor='white')
gs = gridspec.GridSpec(
    3, 7,
    height_ratios=[1.2, 1.2, 1.0],
    width_ratios=[1, 1, 1, 1, 1, 1, 0.05],
    hspace=0.45, wspace=0.15
)

CONUS_LON = (-125, -66)
CONUS_LAT = (24, 50)
CMAP_PRED = 'RdYlGn'
CMAP_DELTA = 'YlOrRd'
CMAP_DIFF = 'RdBu_r'

# ── Row 0: Spatial maps ──────────────────────────────────────────────────────

# Col 0: Ground Truth (static)
ax_gt = fig.add_subplot(gs[0, 0])
ax_gt.set_facecolor('white')
ax_gt.scatter(gt['decimalLongitude'], gt['decimalLatitude'],
              c='#2d6a4f', s=6, alpha=0.5)
ax_gt.set_xlim(*CONUS_LON); ax_gt.set_ylim(*CONUS_LAT)
ax_gt.set_aspect('equal'); ax_gt.set_xticks([]); ax_gt.set_yticks([])
ax_gt.set_title('Ground Truth\n(GloBI)', fontsize=10, fontweight='bold')

# Col 1: A2 (static, week-independent)
ax_a2 = fig.add_subplot(gs[0, 1])
ax_a2.set_facecolor('white')
a2_data = pred_df[pred_df['week'] == 1]
ax_a2.scatter(a2_data['lon'], a2_data['lat'],
              c=a2_data['pred_a2'], cmap=CMAP_PRED,
              s=6, vmin=0.6, vmax=1.0, alpha=0.8)
ax_a2.set_xlim(*CONUS_LON); ax_a2.set_ylim(*CONUS_LAT)
ax_a2.set_aspect('equal'); ax_a2.set_xticks([]); ax_a2.set_yticks([])
ax_a2.set_title('A2 — Spatial\nCo-occurrence', fontsize=10, fontweight='bold')

# Cols 2–5: PPE Delta across 4 seasons
for col, (week, label) in enumerate(SPATIAL_WEEKS.items()):
    ax = fig.add_subplot(gs[0, col + 2])
    ax.set_facecolor('white')
    week_data = delta_df[delta_df['week'] == week]
    ax.scatter(week_data['lon'], week_data['lat'],
               c=week_data['delta'], cmap=CMAP_DELTA,
               s=6, vmin=0, vmax=delta_df['delta'].max(), alpha=0.8)
    ax.set_xlim(*CONUS_LON); ax.set_ylim(*CONUS_LAT)
    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f'PPE Δ\n{label}', fontsize=10, fontweight='bold')

# ── Row 1: A2 latitude profile + Delta Hovmoller ─────────────────────────────

# A2 latitude profile (static bar chart — no time dimension)
ax_a2_prof = fig.add_subplot(gs[1, :2])
ax_a2_prof.set_facecolor('white')
ax_a2_prof.barh(a2_profile.index, a2_profile.values,
                color='#2d6a4f', alpha=0.7, height=0.8)
ax_a2_prof.set_xlabel('Mean P(interaction)', fontsize=9)
ax_a2_prof.set_ylabel('Latitude (°N)', fontsize=9)
ax_a2_prof.set_title('A2 — Latitude Profile\n(Static, No Temporal Dimension)',
                      fontsize=10, fontweight='bold')
ax_a2_prof.set_xlim(0.6, 1.0)
ax_a2_prof.set_ylim(24, 50)
ax_a2_prof.axvline(a2_profile.mean(), color='red',
                   linestyle='--', linewidth=1, alpha=0.7)

# PPE Delta Hovmoller (latitude x week)
ax_hov = fig.add_subplot(gs[1, 2:6])
ax_hov.set_facecolor('white')
month_ticks = [1, 5, 9, 14, 18, 22, 27, 31, 35, 40, 44, 48, 52]
month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', '']

ax_hov.imshow(
    hovmoller_delta.sort_index(ascending=False),
    aspect='auto', cmap=CMAP_DELTA,
    vmin=0, vmax=hovmoller_delta.values.max(),
    extent=[1, 52, hovmoller_delta.index.min() - 0.5,
            hovmoller_delta.index.max() + 0.5]
)
ax_hov.set_xlabel('Month', fontsize=9)
ax_hov.set_ylabel('Latitude (°N)', fontsize=9)
ax_hov.set_title('PPE Δ = min(flowering, activity) — Latitude × Week',
                  fontsize=10, fontweight='bold')
ax_hov.set_xticks(month_ticks)
ax_hov.set_xticklabels(month_labels, fontsize=8)

# Hovmoller colorbar
cax_hov = fig.add_subplot(gs[1, 6])
sm_delta = plt.cm.ScalarMappable(
    cmap=CMAP_DELTA,
    norm=mcolors.Normalize(0, hovmoller_delta.values.max())
)
sm_delta.set_array([])
plt.colorbar(sm_delta, cax=cax_hov, label='PPE Δ')

# ── Row 2: Mean A3 − A2 difference map (full year) ───────────────────────────

ax_diff = fig.add_subplot(gs[2, :6])
ax_diff.set_facecolor('white')
diff_mean = pred_df.groupby(['lat', 'lon']).apply(
    lambda x: (x['pred_a3'] - x['pred_a2']).mean()
).reset_index(name='diff_mean')

ax_diff.scatter(
    diff_mean['lon'], diff_mean['lat'],
    c=diff_mean['diff_mean'], cmap=CMAP_DIFF,
    s=8, vmin=-0.3, vmax=0.3, alpha=0.9
)
ax_diff.set_xlim(*CONUS_LON); ax_diff.set_ylim(*CONUS_LAT)
ax_diff.set_aspect('equal'); ax_diff.set_xticks([]); ax_diff.set_yticks([])
ax_diff.set_title('Mean A3 − A2 Difference (Full Year)',
                  fontsize=11, fontweight='bold')

# Difference colorbar
cax_diff = fig.add_axes([0.92, 0.08, 0.015, 0.22])
sm_diff = plt.cm.ScalarMappable(cmap=CMAP_DIFF, norm=mcolors.Normalize(-0.3, 0.3))
sm_diff.set_array([])
plt.colorbar(sm_diff, cax=cax_diff, label='ΔP (A3 − A2)')

fig.patch.set_facecolor('white')
fig.suptitle('Achillea millefolium — ANTHEIA Interaction Predictions',
             fontsize=15, fontweight='bold', y=1.01)

plt.savefig(BASE + "antheia_combined_figure_v4.png",
            dpi=150, bbox_inches='tight', facecolor='white')
plt.show()
print("Saved!")
