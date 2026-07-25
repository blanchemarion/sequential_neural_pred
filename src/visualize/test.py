import json
import textwrap
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle


# ============================================================
# USER SETTINGS
# ============================================================

USE_PROMPT_VALUES = True
#JSON_PATH = Path("scores_cache_90_810_4split_submetric_viz.json")

OUTDIR = Path("nethobench_slide_figures")
OUTDIR.mkdir(exist_ok=True)

SAVE_PNG = True
SAVE_PDF = True
SAVE_SVG = True
DPI = 300


# ============================================================
# DATA
# ============================================================

models = ["VAR", "1_step", "Sequifier", "AR", "TF", "TF_QL_KL"]

families = [
    "Distribution",
    "Temporal",
    "Relational",
    "Geometry",
    "State dynamics",
    "Composite",
]

# Values from the manuscript/table pasted in your message
prompt_values = {
    "Distribution":  [0.350, 0.414, 0.412, 0.445, 0.448, 0.456],
    "Temporal":      [0.336, 0.511, 0.679, 0.483, 0.527, 0.543],
    "Relational":    [0.633, 0.652, 0.629, 0.601, 0.730, 0.712],
    "Geometry":      [0.769, 0.766, 0.731, 0.697, 0.728, 0.773],
    "State dynamics":[0.337, 0.487, 0.554, 0.519, 0.540, 0.563],
    "Composite":     [0.488, 0.565, 0.595, 0.548, 0.597, 0.609],
}


def load_values_from_json(json_path):
    """
    Load family-level Nethobench scores from your score cache.
    This assumes the JSON structure has a top-level key 'means'.
    """
    with open(json_path, "r") as f:
        cache = json.load(f)

    means = cache["means"]

    model_map = {
        "VAR": "VAR",
        "1_step": "1_step",
        "Sequifier": "sequifier",
        "AR": "AR",
        "TF": "TF",
        "TF_QL_KL": "TF_QL_0.08_KL_0.02",
    }

    family_key_map = {
        "Distribution": "family_distribution",
        "Temporal": "family_temporal_spectral",
        "Relational": "family_relational",
        "Geometry": "family_geometry",
        "State dynamics": "family_state_dynamics",
        "Composite": "composite_score",
    }

    values = {}
    for fam in families:
        values[fam] = [
            means[model_map[m]][family_key_map[fam]]
            for m in models
        ]

    return values


if USE_PROMPT_VALUES:
    values = prompt_values
else:
    values = load_values_from_json(JSON_PATH)

score_matrix = np.array([values[fam] for fam in families])


# ============================================================
# STYLE HELPERS
# ============================================================

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 17,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 12,
    "figure.titlesize": 19,
})


def save_figure(fig, stem):
    if SAVE_PNG:
        fig.savefig(OUTDIR / f"{stem}.png", dpi=DPI, bbox_inches="tight")
    if SAVE_PDF:
        fig.savefig(OUTDIR / f"{stem}.pdf", bbox_inches="tight")
    if SAVE_SVG:
        fig.savefig(OUTDIR / f"{stem}.svg", bbox_inches="tight", format="svg")


# ============================================================
# 1) FAMILY-LEVEL HEATMAP
# ============================================================

def plot_family_heatmap(score_matrix, values):
    fig = plt.figure(figsize=(15.5, 7.2), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[4.4, 1.45])

    ax = fig.add_subplot(gs[0, 0])
    ax_callout = fig.add_subplot(gs[0, 1])
    ax_callout.axis("off")

    im = ax.imshow(
        score_matrix,
        cmap="YlGnBu",
        vmin=0.30,
        vmax=0.80,
        aspect="auto",
    )

    ax.set_xticks(np.arange(len(models)))
    ax.set_xticklabels(models, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(families)))
    ax.set_yticklabels(families)

    ax.set_title(
        "Family-level Nethobench scores reveal structural fingerprints",
        loc="left",
        pad=18,
        fontweight="bold",
    )

    # Cell grid
    ax.set_xticks(np.arange(-0.5, len(models), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(families), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Cell values
    for i in range(score_matrix.shape[0]):
        for j in range(score_matrix.shape[1]):
            val = score_matrix[i, j]
            ax.text(
                j, i, f"{val:.3f}",
                ha="center", va="center",
                fontsize=11,
                color="black",
                fontweight="bold" if val >= np.max(score_matrix[i]) - 1e-12 else "normal",
            )

    # Highlight best and second-best in each row
    best_color = "#111111"
    second_color = "#444444"

    for i, row in enumerate(score_matrix):
        order = np.argsort(row)
        second_idx = order[-2]
        best_idx = order[-1]

        # Best: solid outline
        ax.add_patch(Rectangle(
            (best_idx - 0.5, i - 0.5),
            1, 1,
            fill=False,
            edgecolor=best_color,
            linewidth=3.0,
        ))

        # Second best: dashed outline
        ax.add_patch(Rectangle(
            (second_idx - 0.5, i - 0.5),
            1, 1,
            fill=False,
            edgecolor=second_color,
            linewidth=2.2,
            linestyle="--",
        ))

    # Small legend for outlines
    ax.text(
        0.00, -0.16,
        "Solid outline = best in family     Dashed outline = second best",
        transform=ax.transAxes,
        fontsize=11,
        color="#333333",
    )

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Nethobench score", rotation=90)

    # Callout values
    temporal_sequifier = values["Temporal"][models.index("Sequifier")]
    relational_tf = values["Relational"][models.index("TF")]
    composite_tfqlkl = values["Composite"][models.index("TF_QL_KL")]

    callouts = [
        {
            "title": "Sequifier",
            "body": f"Best temporal structure\nTRJDIST = {temporal_sequifier:.3f}",
            "color": "#D37A00",
            "y": 0.78,
        },
        {
            "title": "TF",
            "body": f"Best relational structure\nRelational = {relational_tf:.3f}",
            "color": "#B23A8A",
            "y": 0.51,
        },
        {
            "title": "TF_QL_KL",
            "body": f"Best composite / balance\nComposite = {composite_tfqlkl:.3f}",
            "color": "#6B4AA5",
            "y": 0.24,
        },
    ]

    for c in callouts:
        ax_callout.text(
            0.03, c["y"],
            c["title"],
            transform=ax_callout.transAxes,
            fontsize=16,
            fontweight="bold",
            color=c["color"],
            va="top",
        )
        ax_callout.text(
            0.03, c["y"] - 0.075,
            c["body"],
            transform=ax_callout.transAxes,
            fontsize=12.5,
            color="#222222",
            va="top",
            linespacing=1.35,
            bbox=dict(
                boxstyle="round,pad=0.55",
                facecolor=c["color"],
                alpha=0.10,
                edgecolor=c["color"],
                linewidth=1.5,
            ),
        )

    save_figure(fig, "01_family_level_heatmap")
    return fig


# ============================================================
# 2) CONTRAST PANEL
# ============================================================

def get_score(family, model):
    return values[family][models.index(model)]


contrasts = [
    {
        "title": "1. Prediction formulation",
        "comparison": ("1_step", "TF"),
        "message": "Sequence prediction improves long-rollout structure, especially relational realism.",
        "metrics": ["Composite", "Relational", "Temporal"],
    },
    {
        "title": "2. Architecture",
        "comparison": ("1_step", "Sequifier"),
        "message": "A richer next-token architecture better preserves trajectory flow and state dynamics.",
        "metrics": ["Temporal", "State dynamics", "Composite"],
    },
    {
        "title": "3. Training regime",
        "comparison": ("TF", "AR"),
        "message": "Self-feedback training does not automatically improve closed-loop neural realism.",
        "metrics": ["Composite", "Relational", "Geometry"],
    },
    {
        "title": "4. Loss function",
        "comparison": ("TF", "TF_QL_KL"),
        "message": "Distribution-aware losses improve balance, but introduce structural trade-offs.",
        "metrics": ["Distribution", "Geometry", "State dynamics", "Composite", "Relational"],
    },
]


def plot_contrast_panel():
    """
    Slide-ready 2x2 contrast panel: dumbbell scores per metric, numeric labels at points,
    per-panel legend for the two models (reference vs comparison).
    """

    fig, axes = plt.subplots(
        2, 2,
        figsize=(16, 11.0),
        sharex=True
    )

    # Generous margins: suptitle above axes+titles, wide column gutter (deltas vs y-labels),
    # tall row gap (no takeaway vs next title clash), deep footer (x ticks vs fig text).
    fig.subplots_adjust(
        left=0.11,
        right=0.97,
        top=0.80,
        bottom=0.28,
        wspace=0.52,
        hspace=0.88
    )

    fig.suptitle(
        "Forecasting design choices leave different structural signatures",
        x=0.10,
        y=0.97,
        ha="left",
        fontsize=17,
        fontweight="bold"
    )

    # Scores are drawn in [xmin, ~0.78]; x-axis ticks end at 0.8; extra span reserves
    # in-axes space for Δ labels without spilling into the neighboring subplot.
    xmin = 0.30
    xlim_right = 0.875

    before_color = "#6F6F6F"
    improve_color = "#2E8B57"
    degrade_color = "#C44E52"
    grid_color = "#D8D8D8"
    text_dark = "#222222"
    text_mid = "#555555"

    clean_contrasts = [
        {
            "ax": axes[0, 0],
            "title": "Prediction formulation",
            "comparison": ("1_step", "TF"),
            "metrics": ["Composite", "Relational", "Temporal"],
            "takeaway": "Sequence training mainly improves relational structure.",
        },
        {
            "ax": axes[0, 1],
            "title": "Architecture",
            "comparison": ("1_step", "Sequifier"),
            "metrics": ["Temporal", "State dynamics", "Composite"],
            "takeaway": "A richer next-token model better preserves trajectory flow.",
        },
        {
            "ax": axes[1, 0],
            "title": "Training regime",
            "comparison": ("TF", "AR"),
            "metrics": ["Composite", "Relational", "Geometry"],
            "takeaway": "Self-feedback does not automatically improve realism.",
        },
        {
            "ax": axes[1, 1],
            "title": "Loss function",
            "comparison": ("TF", "TF_QL_KL"),
            "metrics": ["Distribution", "Geometry", "State dynamics", "Composite", "Relational"],
            "takeaway": "Distribution-aware loss improves balance, with trade-offs.",
        },
    ]

    max_metric_rows = max(len(c["metrics"]) for c in clean_contrasts)

    label_bbox = dict(boxstyle="round,pad=0.18", fc="white", ec="#D0D0D0", linewidth=0.5, alpha=0.94)

    def point_offsets(v0, v1):
        """Horizontal separation for numeric labels (offset points)."""
        sep = abs(v1 - v0)
        if sep < 0.018:
            return -20, 20
        if sep < 0.035:
            return -16, 16
        return -10, 10

    for idx, item in enumerate(clean_contrasts):
        ax = item["ax"]
        col = idx % 2
        before, after = item["comparison"]
        metrics = item["metrics"]
        n = len(metrics)
        y_base = (max_metric_rows - n) / 2.0
        y_positions = np.arange(n)[::-1] + y_base

        ax.set_xlim(xmin, xlim_right)
        ax.set_ylim(-0.95, max_metric_rows - 0.2)

        # Right-hand panels: put metric names on the outer (right) edge so they do not
        # collide with deltas from the left-hand column in the center gutter.
        if col == 1:
            ax.yaxis.tick_right()
            ax.tick_params(axis="y", labelleft=False, labelright=True)

        # Panel title only; model names appear in the per-panel legend below.
        ax.set_title(
            item["title"],
            loc="left",
            fontsize=12.5,
            fontweight="bold",
            pad=5,
            color=text_dark
        )

        deltas_panel = []
        for y, metric in zip(y_positions, metrics):
            v0 = get_score(metric, before)
            v1 = get_score(metric, after)
            delta = v1 - v0
            deltas_panel.append(delta)
            delta_color = improve_color if delta >= 0 else degrade_color

            # Dumbbell line
            ax.plot(
                [v0, v1],
                [y, y],
                color=delta_color,
                linewidth=2.5,
                solid_capstyle="round",
                zorder=2
            )

            # Points
            ax.scatter(v0, y, s=115, color=before_color, edgecolor="white", linewidth=0.9, zorder=3)
            ax.scatter(v1, y, s=115, color=delta_color, edgecolor="white", linewidth=0.9, zorder=4)

            before_dx, after_dx = point_offsets(v0, v1)

            ax.annotate(
                f"{v0:.3f}",
                xy=(v0, y),
                xytext=(before_dx, -18),
                textcoords="offset points",
                ha="right" if before_dx < 0 else "left",
                va="top",
                fontsize=7.5,
                color=before_color,
                bbox=label_bbox,
                zorder=5,
            )

            ax.annotate(
                f"{v1:.3f}",
                xy=(v1, y),
                xytext=(after_dx, -18),
                textcoords="offset points",
                ha="left" if after_dx > 0 else "right",
                va="top",
                fontsize=7.5,
                color=delta_color,
                bbox=label_bbox,
                zorder=5,
            )

            # Delta in reserved x-span (data coords)
            ax.text(
                xlim_right - 0.002,
                y,
                f"{delta:+.3f}",
                ha="right",
                va="center",
                fontsize=9.5,
                fontweight="bold",
                color=delta_color,
                clip_on=False,
                zorder=5,
            )

        if all(d >= 0 for d in deltas_panel):
            after_legend_color = improve_color
        elif all(d < 0 for d in deltas_panel):
            after_legend_color = degrade_color
        else:
            after_legend_color = "#454545"

        legend_handles = [
            Line2D(
                [0],
                [0],
                linestyle="none",
                marker="o",
                markersize=6.5,
                markerfacecolor=before_color,
                markeredgecolor="white",
                markeredgewidth=0.55,
                label=before,
            ),
            Line2D(
                [0],
                [0],
                linestyle="none",
                marker="o",
                markersize=6.5,
                markerfacecolor=after_legend_color,
                markeredgecolor="white",
                markeredgewidth=0.55,
                label=after,
            ),
        ]
        ax.legend(
            handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.02),
            ncol=2,
            bbox_transform=ax.transAxes,
            frameon=True,
            fancybox=False,
            fontsize=8,
            labelcolor=text_dark,
            handletextpad=0.45,
            columnspacing=1.1,
            borderpad=0.35,
            edgecolor="#CCCCCC",
        ).get_frame().set_linewidth(0.55)

        ax.set_yticks(y_positions)
        ax.set_yticklabels(metrics, fontsize=9.5)

        ax.set_xticks([0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
        ax.set_xticklabels(["0.3", "0.4", "0.5", "0.6", "0.7", "0.8"])
        ax.grid(axis="x", color=grid_color, linewidth=1.0)
        ax.set_axisbelow(True)

        for spine in ["top", "right", "left"]:
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color("#BFBFBF")

        ax.tick_params(axis="y", length=0, labelsize=9.5)
        ax.tick_params(axis="x", colors=text_mid, labelsize=9, pad=4)
        if idx in [2, 3]:
            ax.tick_params(axis="x", pad=9)

        if idx in [0, 1]:
            ax.tick_params(axis="x", labelbottom=False)

        ax.text(
            0.5,
            -0.15,
            textwrap.fill(item["takeaway"], width=62),
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=7.8,
            color="#333333",
            clip_on=False,
            linespacing=1.1,
        )

    fig.text(
        0.53,
        0.108,
        "Netbench family score",
        ha="center",
        fontsize=12,
        fontweight="bold"
    )

    fig.text(
        0.10,
        0.034,
        "Gray = reference model   Green = improved   Red = degraded",
        ha="left",
        fontsize=9,
        color=text_mid
    )

    save_figure(fig, "02_design_choice_contrast_panel_clean")
    return fig

# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    fig1 = plot_family_heatmap(score_matrix, values)
    fig2 = plot_contrast_panel()

    plt.show()

    print(f"Saved figures to: {OUTDIR.resolve()}")