from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import BoxStyle, FancyBboxPatch, Patch

from antheia.paths import REPO_ROOT as REPO
OUT = REPO / "results" / "figures"

# categorical slots 1, 2, 3, 7 of the documented palette (validated, see report)
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"
SURFACE = "#ffffff"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
CSS = 1.0 / 96.0                                  # one CSS px, in inches
BAR_CAP, BAR_GAP, BAR_ROUND = 24 * CSS, 2 * CSS, 4 * CSS

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "font.size": 8,
    "axes.labelsize": 8, "axes.titlesize": 9.5, "legend.fontsize": 7.5,
    "xtick.labelsize": 8, "ytick.labelsize": 7.5, "text.color": INK,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.7, "pdf.fonttype": 42,
})

REGIMES = ["cold plant", "cold pollinator", "cold both", "warm"]
BASELINE_NAMES = {
    "baseline_widedeep": "Wide & Deep",
    "baseline_pair_gbm": "pair GBM",
    "baseline_antheia_scalar": "ANTHEIA v1",
    "baseline_svd_taxonomic": "SVD + taxonomy",
}
SERIES = [
    ("best comparison model", "best_baseline", BLUE),
    ("system v1 (plant-only rehearsal)", "system_v1", ORANGE),
    ("final system (symmetric rehearsal)", "final", AQUA),
]
FAMILIES = [
    ("structure ablations", BLUE, "o",
     ["R3 (taxon+cell)", "final retriever (no taxon)", "no cell nodes", "month-collapsed",
      "attention aggregation"]),
    ("continental priors added", ORANGE, "s",
     ["+ memory vector (R1)", "+ co-presence stat (R4)", "+ learned co-presence (R6c)",
      "+ presence embed concat (R6b)", "+ presence embed add (R6)",
      "final + opportunity term (A4)", "A4, opportunity off within sites"]),
    ("objective changes", AQUA, "D", ["co-occurrence negatives (R7)"]),
    ("plant-only rehearsal (v1) and its variants", VIOLET, "^",
     ["plant-only rehearsal (v1)", "+ degree encoding (R5, on v1)",
      "+ co-presence stat (R4, on v1)", "+ memory (R1, on v1)"]),
]
SHORT = {
    "R3 (taxon+cell)": "R3 (taxon+cell)",
    "final retriever (no taxon)": "final (no taxon)",
    "no cell nodes": "no cell nodes",
    "month-collapsed": "month-collapsed",
    "attention aggregation": "attention agg.",
    "+ memory vector (R1)": "+ memory (R1)",
    "+ co-presence stat (R4)": "+ co-presence (R4)",
    "+ learned co-presence (R6c)": "+ learned co-pres. (R6c)",
    "+ presence embed concat (R6b)": "+ presence concat (R6b)",
    "+ presence embed add (R6)": "+ presence add (R6)",
    "co-occurrence negatives (R7)": "co-occ. negatives (R7)",
    "plant-only rehearsal (v1)": "plant-only rehearsal (v1)",
    "+ degree encoding (R5, on v1)": "+ degree (R5, on v1)",
    "+ co-presence stat (R4, on v1)": "+ co-presence (R4, on v1)",
    "+ memory (R1, on v1)": "+ memory (R1, on v1)",
    "final + opportunity term (A4)": "final + opportunity (A4)",
    "A4, opportunity off within sites": "A4, opportunity off in sites",
}


def despine(ax, keep=("bottom", "left")):
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(s in keep)
    ax.tick_params(length=0, pad=4)


def save(fig, stem):
    fig.savefig(OUT / f"{stem}.png", dpi=300)
    fig.savefig(OUT / f"{stem}.pdf")
    plt.close(fig)


def inches(ax, x, y):
    """Data coordinates to figure inches (dpi-invariant, so the same geometry renders in the pdf)."""
    px = ax.transData.transform((x, y))
    return px / ax.figure.dpi


def rounded_bar(ax, cx, width, top, base, color, zorder=3):
    """A bar in figure inches: 4px-rounded data end, square at the baseline (the rounding is clipped away)."""
    patch = FancyBboxPatch(
        (cx - width / 2, base), width, top - base,
        boxstyle=BoxStyle("Round", pad=0, rounding_size=BAR_ROUND),
        transform=ax.figure.dpi_scale_trans, facecolor=color, edgecolor="none",
        mutation_aspect=1, zorder=zorder,
    )
    ax.add_artist(patch)
    patch.set_clip_path(ax.patch)
    return patch


def fig_regimes(df):
    """Grouped bars of AUPR per cold-start regime, log y, with a per-group chance mark."""
    df = df.set_index("regime").loc[REGIMES]
    fig = plt.figure(figsize=(7, 3.2), dpi=300)
    ax = fig.add_axes([0.085, 0.225, 0.9, 0.545])
    ax.set_yscale("log")
    ax.set_xlim(-0.5, len(REGIMES) - 0.5)
    ax.set_ylim(1.5e-4, 0.95)
    ax.set_xticks(range(len(REGIMES)), REGIMES)
    ax.set_yticks([1e-3, 1e-2, 1e-1], ["0.001", "0.01", "0.1"])
    ax.set_yticks([], minor=True)
    ax.set_ylabel("AUPR at network prevalence (log scale)")
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    despine(ax)
    ax.set_title(
        "Cold-start regimes: the final system leads all three cold regimes; "
        "matrix factorisation keeps warm",
        loc="left", color=INK, pad=24, fontsize=8.6,
    )
    ax.legend(
        handles=[Patch(facecolor=c, edgecolor="none", label=n) for n, _, c in SERIES],
        loc="lower left", bbox_to_anchor=(0, 1.005, 1, 0.1), ncols=3,
        frameon=False, labelcolor=INK2, handlelength=0.85, handleheight=0.85,
        handletextpad=0.5, columnspacing=1.6, borderpad=0,
    )

    fig.canvas.draw()
    unit = inches(ax, 1, 1)[0] - inches(ax, 0, 1)[0]
    width = min(BAR_CAP, 0.26 * unit)
    pitch = width + BAR_GAP
    base = ax.get_window_extent().y0 / fig.dpi - 0.25          # below the axes: rounding clipped off
    for g, regime in enumerate(REGIMES):
        row = df.loc[regime]
        cx0 = inches(ax, g, 1)[0]
        for i, (_, col, color) in enumerate(SERIES):
            cx = cx0 + (i - 1) * pitch
            top = inches(ax, g, row[col])[1]
            rounded_bar(ax, cx, width, top, base, color)
            ax.text(cx, top + 0.045, f"{row[col]:.3f}", transform=fig.dpi_scale_trans,
                    ha="center", va="bottom", rotation=90, fontsize=6.6, color=INK2,
                    zorder=6)
        ax.text(cx0, ax.get_window_extent().y0 / fig.dpi - 0.27,
                f"best: {BASELINE_NAMES[row['best_baseline_name']]}",
                transform=fig.dpi_scale_trans, ha="center", va="top",
                fontsize=6.4, color=MUTED, zorder=6)
        y_ch = inches(ax, g, row["chance"])[1]
        x0, x1 = cx0 - pitch - width / 2 - 0.04, cx0 + pitch + width / 2 + 0.04
        ax.add_artist(Line2D(
            [x0, x1], [y_ch, y_ch], transform=fig.dpi_scale_trans, color=INK2, lw=0.9,
            dashes=(3, 2), zorder=5,
            path_effects=[pe.withStroke(linewidth=2.9, foreground=SURFACE)],
        ))
        if g == 0:
            ax.text(x1 + 0.03, y_ch, "chance", transform=fig.dpi_scale_trans, ha="left",
                    va="center", fontsize=6.5, color=INK2, zorder=6,
                    path_effects=[pe.withStroke(linewidth=2.5, foreground=SURFACE)])
    save(fig, "fig2_regimes")


def _align(ann, dx, dy):
    ann.set_ha("center" if abs(dx) < 4 else ("left" if dx > 0 else "right"))
    ann.set_va("center" if abs(dy) < 3 else ("bottom" if dy > 0 else "top"))


def place_labels(ax, pts, texts, obstacles):
    """Greedily places point labels at the first candidate offset that clears earlier labels, markers and the axes edge."""
    fig = ax.figure
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    cands = [(7, 0), (-7, 0), (0, 6), (0, -7), (6, 4), (-6, 4), (6, -5), (-6, -5),
             (13, 0), (-13, 0), (0, 12), (0, -13), (11, 6), (-11, 6), (11, -7), (-11, -7),
             (0, 18), (0, -19)]
    taken = list(obstacles)
    frame = ax.get_window_extent()
    for (x, y), label in zip(pts, texts):
        ann = ax.annotate(label, (x, y), xytext=cands[0], textcoords="offset points",
                          fontsize=6.0, color=INK2, ha="center", va="bottom", zorder=6,
                          path_effects=[pe.withStroke(linewidth=2.2, foreground=SURFACE)])
        best, best_cost = cands[0], None
        for k, (dx, dy) in enumerate(cands):
            _align(ann, dx, dy)
            ann.set_position((dx, dy))
            bb = ann.get_window_extent(rend).expanded(1.06, 1.25)
            cost = sum(max(0.0, b.intersection(bb, b).width * b.intersection(bb, b).height)
                       for b in taken if b.overlaps(bb))
            if not frame.containsx(bb.x0) or not frame.containsx(bb.x1):
                cost += 4000
            if not frame.containsy(bb.y0) or not frame.containsy(bb.y1):
                cost += 4000
            cost += 12 * k
            if best_cost is None or cost < best_cost:
                best, best_cost = (dx, dy), cost
        dx, dy = best
        _align(ann, dx, dy)
        ann.set_position((dx, dy))
        taken.append(ann.get_window_extent(rend).expanded(1.06, 1.25))


def fig_tradeoff(df):
    """Universe vs within-site AUPR for every retriever arm, coloured by arm family."""
    df = df.set_index("arm")
    fig = plt.figure(figsize=(5, 4.5), dpi=300)
    ax = fig.add_axes([0.145, 0.105, 0.835, 0.735])
    ax.set_xlim(0.1420, 0.2060)
    ax.set_ylim(0.1725, 0.2320)
    ax.set_xticks([0.15, 0.16, 0.17, 0.18, 0.19, 0.20], ["0.15", "0.16", "0.17", "0.18", "0.19", "0.20"])
    ax.set_yticks([0.18, 0.19, 0.20, 0.21, 0.22], ["0.18", "0.19", "0.20", "0.21", "0.22"])
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.set_xlabel("universe AUPR (cold plant)")
    ax.set_ylabel("within-site mean AUPR (91 networks)")
    despine(ax)
    ax.set_title("Every continental prior raises universe AUPR\nand lowers within-site AUPR",
                 loc="left", color=INK, pad=26)

    base = df.loc["R3 (taxon+cell)"]
    ax.axvline(base["cold_plant"], color=AXIS, lw=0.7, zorder=1)
    ax.axhline(base["local"], color=AXIS, lw=0.7, zorder=1)
    ax.annotate("base (R3)", (base["cold_plant"], ax.get_ylim()[0]), xytext=(4, 4),
                textcoords="offset points", fontsize=6.2, color=MUTED, ha="left", va="bottom",
                path_effects=[pe.withStroke(linewidth=2.2, foreground=SURFACE)])

    pts, texts, obstacles = [], [], []
    for name, color, marker, arms in FAMILIES:
        sub = df.loc[arms]
        ax.scatter(sub["cold_plant"], sub["local"], s=58, marker=marker, c=color,
                   edgecolors=SURFACE, linewidths=1.5, zorder=4, label=name)
        for arm, r in sub.iterrows():
            pts.append((r["cold_plant"], r["local"]))
            texts.append(SHORT[arm])
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.005, 1, 0.12), ncols=2, frameon=False,
              labelcolor=INK2, fontsize=6.6, handletextpad=0.25, columnspacing=1.0,
              borderpad=0, scatterpoints=1)
    fig.canvas.draw()
    for (x, y) in pts:
        px, py = ax.transData.transform((x, y))
        obstacles.append(matplotlib.transforms.Bbox.from_bounds(px - 6, py - 6, 12, 12))
    place_labels(ax, pts, texts, obstacles)
    save(fig, "fig4_tradeoff")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    fig_regimes(pd.read_csv(REPO / "results" / "fig_regimes.csv"))
    fig_tradeoff(pd.read_csv(REPO / "results" / "fig_arms.csv"))
    print(f"wrote figures to {OUT}")
