import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from antheia.config import load_config

BLUE, ORANGE = "#2a78d6", "#eb6834"          # validated categorical slots 1-2
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#d8d7d2", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 9, "legend.fontsize": 7.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 8, "axes.edgecolor": GRID,
    "axes.labelcolor": INK2, "text.color": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.linewidth": 0.8, "font.family": "DejaVu Sans",
})


def despine(ax, keep=("bottom",)):
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(s in keep)
    ax.tick_params(length=0)


def fig1(cfg, out):
    """The ladder: recall@10 per model on all labels vs curated Tier-1 labels."""
    t = pd.read_csv(cfg["edges"].parent / "final_table_v1.csv").set_index("model")
    ens = pd.read_csv(cfg["edges"].parent / "ensemble_v1.csv").set_index("model")
    rows = [
        ("degree null", t.loc["degree_rank"]), ("shared bins (N)", t.loc["rank_n"]),
        ("N + taxonomy + local $\\Delta$", t.loc["rank_n_tax_ldelta"]),
        ("GBM (geometry+phenology)", t.loc["gbm_geo_tuned"]),
        ("two-tower", t.loc["twotower_none"]),
        ("two-tower + BioCLIP", t.loc["twotower_bioclip"]),
    ]
    labels = [r[0] for r in rows] + ["ensemble (GBM + tower)"]
    all_m = [r[1]["r10_all"] for r in rows] + [ens.loc["ens_rank_avg", "recall@10"]]
    all_lo = [r[1]["lo"] for r in rows] + [ens.loc["ens_rank_avg", "lo"]]
    all_hi = [r[1]["hi"] for r in rows] + [ens.loc["ens_rank_avg", "hi"]]
    t1_m = [r[1]["r10_t1"] for r in rows] + [ens.loc["ens_rank_avg", "t1_recall@10"]]

    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    ax.grid(axis="x", color=GRID, lw=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    ax.hlines(y, all_m, t1_m, color=GRID, lw=1.6, zorder=1)
    ax.errorbar(all_m, y, xerr=[np.array(all_m) - all_lo, np.array(all_hi) - all_m],
                fmt="o", ms=6.5, color=BLUE, ecolor=BLUE, elinewidth=1.4, capsize=0,
                zorder=3, label="all labels", markeredgecolor=SURFACE, markeredgewidth=1.0)
    ax.plot(t1_m, y, "o", ms=6.5, color=ORANGE, zorder=3, label="curated (Tier-1) labels",
            markeredgecolor=SURFACE, markeredgewidth=1.0)
    for yy, a, b in zip(y, all_m, t1_m):
        ax.annotate(f"{a:.3f}", (a, yy), xytext=(0, -11), textcoords="offset points",
                    ha="center", color=INK2, fontsize=6.8)
        ax.annotate(f"{b:.3f}", (b, yy), xytext=(0, 7), textcoords="offset points",
                    ha="center", color=INK2, fontsize=6.8)
    ax.set_yticks(y, labels)
    ax.set_ylim(-0.7, len(labels) - 0.3)
    ax.set_xlabel("recall@10  (true partner in top 10 of 24,939 candidates)")
    ax.set_title("Cold-start ranking for unseen plants", loc="left", color=INK, fontweight="bold")
    ax.legend(frameon=False, loc="lower right", ncols=1)
    despine(ax)
    fig.tight_layout()
    fig.savefig(out / "fig1_ladder.pdf")
    fig.savefig(out / "fig1_ladder.png", dpi=300)
    plt.close(fig)


def fig2(cfg, out):
    """Process dependence: probe AUC per feature family, and within-order transfer."""
    probe = pd.read_csv(cfg["edges"].parent / "provenance_probe_v1.csv")
    ax_ab = pd.read_csv(cfg["edges"].parent / "axis_ablation_v1.csv").set_index("stratum")

    keep = ["taxonomy_affinity", "range_sizes", "delta+local", "n_only",
            "vf (plant PCA-15)", "vp (pollinator PCA-15)", "bioclip_poll (512D)"]
    nice = {"taxonomy_affinity": "taxonomy affinity", "range_sizes": "range sizes",
            "delta+local": "$\\Delta$ + local $\\Delta$", "n_only": "shared bins (N)",
            "vf (plant PCA-15)": "plant occupancy PCA", "vp (pollinator PCA-15)": "pollinator occupancy PCA",
            "bioclip_poll (512D)": "BioCLIP name emb."}
    p_ = probe.set_index("family").loc[keep]

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.1), gridspec_kw={"width_ratios": [1.0, 1.0]})

    ax = axes[0]
    y = np.arange(len(p_))
    ax.grid(axis="x", color=GRID, lw=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    ax.barh(y, p_["auc_mean"], height=0.6, color=BLUE, edgecolor=SURFACE, linewidth=1.0)
    ax.axvline(0.5, color=INK2, lw=1.0, ls="-")
    ax.axvline(0.824, color=ORANGE, lw=1.4, ls="--")
    ax.annotate("pollinator family alone", (0.824, -0.80), xytext=(-4, 0),
                textcoords="offset points", ha="right", va="center", color=ORANGE, fontsize=6.8)
    ax.annotate("chance", (0.5, len(p_) - 0.42), xytext=(3, 0), textcoords="offset points",
                ha="left", va="top", color=INK2, fontsize=6.8)
    for yy, v in zip(y, p_["auc_mean"]):
        ax.annotate(f"{v:.2f}", (v, yy), xytext=(3, 0), textcoords="offset points",
                    va="center", color=INK2, fontsize=6.8)
    ax.set_yticks(y, [nice[k] for k in p_.index])
    ax.set_xlim(0.45, 1.0)
    ax.set_ylim(-1.0, len(p_) - 0.35)
    ax.set_xlabel("AUC predicting the documenting source")
    ax.set_title("(a) Features encode who observed", loc="left", color=INK, fontweight="bold")
    despine(ax)

    ax = axes[1]
    models = [("rank_vp_only", "pollinator occupancy PCA"), ("rank_full_vp", "occupancy + N + $\\Delta$"),
              ("rank_n", "shared bins (N)"), ("rank_n_tax_ldelta", "N + taxonomy + local $\\Delta$"),
              ("gbm_geo", "GBM (geometry+phenology)")]
    cur = [ax_ab.loc["Hymenoptera, curated", m] for m, _ in models]
    inat = [ax_ab.loc["Hymenoptera, iNat-only", m] for m, _ in models]
    y = np.arange(len(models))
    ax.grid(axis="x", color=GRID, lw=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    ax.hlines(y, inat, cur, color=GRID, lw=1.6, zorder=1)
    ax.plot(inat, y, "o", ms=6.5, color=BLUE, zorder=3, label="iNaturalist",
            markeredgecolor=SURFACE, markeredgewidth=1.0)
    ax.plot(cur, y, "o", ms=6.5, color=ORANGE, zorder=3, label="curated",
            markeredgecolor=SURFACE, markeredgewidth=1.0)
    for yy, a, b in zip(y, inat, cur):
        ax.annotate(f"{b / a:.2f}×", (max(a, b), yy), xytext=(6, 0), textcoords="offset points",
                    va="center", color=INK2, fontsize=6.8)
    ax.set_yticks(y, [n for _, n in models])
    ax.set_xlim(0, 0.78)
    ax.set_ylim(-1.4, len(models) - 0.3)
    ax.set_xlabel("recall@10 within Hymenoptera only")
    ax.set_title("(b) Transfer across sources", loc="left", color=INK, fontweight="bold")
    ax.legend(frameon=False, loc="lower center", ncols=2, bbox_to_anchor=(0.5, -0.02),
              handletextpad=0.3, columnspacing=1.2)
    despine(ax)

    fig.tight_layout()
    fig.savefig(out / "fig2_process.pdf")
    fig.savefig(out / "fig2_process.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    cfg = load_config()
    out = cfg["edges"].parent / "figures"
    out.mkdir(exist_ok=True)
    fig1(cfg, out)
    fig2(cfg, out)
    print(f"wrote figures to {out}")
