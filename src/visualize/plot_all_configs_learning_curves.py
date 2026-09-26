import torch
import numpy as np

import matplotlib.pyplot as plt

from pathlib import Path
import re
from typing import Dict, List, Tuple, Optional, Any

from itertools import combinations

from cns_plotting import setup_cnsplots_style


setup_cnsplots_style({"text.usetex": False})


def _ensure_list_of_dicts(x):
    if x is None:
        return None
    if len(x) == 0:
        return []
    if not isinstance(x[0], dict):
        return [{"total": float(v)} for v in x]
    # cast tensors -> float
    out = []
    for h in x:
        hh = {}
        for k, v in h.items():
            if torch.is_tensor(v):
                hh[k] = float(v.detach().cpu().item())
            else:
                hh[k] = float(v)
        out.append(hh)
    return out



# ----------------------------
# Helpers
# ----------------------------
def to_float(x):
    if torch.is_tensor(x):
        return float(x.detach().cpu().item())
    return float(x)

def ema_smooth(y: np.ndarray, alpha: float = 0.15) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    if len(y) == 0:
        return y
    out = np.empty_like(y)
    out[0] = y[0]
    for i in range(1, len(y)):
        out[i] = alpha * y[i] + (1 - alpha) * out[i - 1]
    return out


def short_label(folder_name: str) -> str:
    base = Path(folder_name).name
    if base.startswith("checkpoints_"):
        base = base[len("checkpoints_"):]  # remove prefix once

    # Special-case labels for paper figures (keeps legends compact/readable).
    # These run folder names are used by `plot_regime_validation_2x2_paper()`.
    if base.startswith("TF_QTL_"):
        return base  # keep full "TF_QTL_0.08_KL_0.02" (and any future variants)
    if base.lower().startswith("ar_kv_"):
        return "AR"
    if base.lower().startswith("tf_"):
        return "TF"

    parts = base.split("_")
    if not parts:
        return base

    head = parts[0]          # OS / TF / AR
    label = head

    # optional variant directly after the head (e.g., AR_KV, OS_SSM)
    if len(parts) > 1:
        second = parts[1]
        if (not second.isdigit()) and (second != "var"):
            label = f"{head}_{second}"

    # optional var suffix anywhere (e.g., AR_var_05 -> AR_var_05, AR_KV_var_05 -> AR_KV_var_05)
    if "var" in parts:
        i = parts.index("var")
        if i + 1 < len(parts):
            label += f"_var_{parts[i + 1]}"

    return label


def load_history(checkpoint_folder: str | Path):
    """
    Returns:
      mode: str (e.g. "TL" or "AR")
      train_norm: List[dict]
      val_norm:   List[dict]
      train_raw:  Optional[List[dict]]
      val_raw:    Optional[List[dict]]
    """
    checkpoint_folder = Path(checkpoint_folder)
    checkpoint_path = checkpoint_folder / "final_model.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"{checkpoint_path} not found")

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    mode = ckpt.get("mode", None)  # "TL" or "AR" in your saver

    train_norm = ckpt.get("train_losses", None)
    val_norm   = ckpt.get("val_losses", None)

    if train_norm is None or val_norm is None:
        raise KeyError(
            f"train_losses/val_losses not found in {checkpoint_path}. "
            f"Available keys: {list(ckpt.keys())}"
        )

    train_raw = ckpt.get("train_losses_raw", None)
    val_raw   = ckpt.get("val_losses_raw", None)

    # Backward compat: floats -> dicts
    def normalize_list(hist):
        if hist is None:
            return None
        if len(hist) > 0 and not isinstance(hist[0], dict):
            hist = [{"total": float(x)} for x in hist]
        return [{k: to_float(v) for k, v in h.items()} for h in hist]

    train_norm = normalize_list(train_norm)
    val_norm   = normalize_list(val_norm)
    train_raw  = normalize_list(train_raw)
    val_raw    = normalize_list(val_raw)

    return mode, train_norm, val_norm, train_raw, val_raw


def available_prefixes_from_val(val_hist: List[Dict[str, float]]) -> List[str]:
    """Return prefixes detected from *_total keys, e.g. ['os','tf','ar','ar_ssm','ar_kv']."""
    if not val_hist:
        return []
    keys = val_hist[0].keys()
    prefs = [k[:-len("_total")] for k in keys if k.endswith("_total")]
    # stable order: put common ones first if present, then the rest sorted
    common = [p for p in ["os", "tf", "ar"] if p in prefs]
    rest = sorted([p for p in prefs if p not in common])
    return common + rest


def normalize_mode_to_prefix(mode: Optional[str]) -> Optional[str]:
    if mode is None:
        return None
    m = str(mode).strip().lower().replace("-", "_")
    # If you ever store TL but keys are tf_*
    if m in ("tl", "teacher_forcing", "teacherforcing"):
        return "tf"
    return m


def get_primary_prefix(mode: Optional[str], val_hist: List[Dict[str, float]]) -> Optional[str]:
    """
    Picks the 'primary' validation prefix.

    - If ckpt['mode'] matches a detected prefix, use it.
    - Else if it's an AR variant (starts with 'ar') and any ar* prefix exists, prefer:
      exact match -> 'ar' -> first ar_* found.
    - Else fall back to first detected prefix, or None if old-format (unprefixed).
    """
    if not val_hist:
        return None

    keys = val_hist[0].keys()
    prefs = available_prefixes_from_val(val_hist)
    if not prefs:
        # old style: 'total' exists (or nothing we recognize)
        return None

    mp = normalize_mode_to_prefix(mode)

    # exact match to *_total key
    if mp is not None and f"{mp}_total" in keys:
        return mp

    # AR fallback logic
    if mp is not None and mp.startswith("ar"):
        if "ar_total" in keys:
            return "ar"
        ar_variants = [p for p in prefs if p.startswith("ar_")]
        if ar_variants:
            return ar_variants[0]

    # otherwise: pick first detected prefix
    return prefs[0]


def get_series(hist: List[Dict[str, float]], comp: str, prefix: Optional[str]) -> np.ndarray:
    """
    - prefix None: old format uses 'total', 'mae', ...
    - prefix like 'ar_kv': reads 'ar_kv_total' for total, or 'ar_kv_mae' for mae, etc.
    """
    if not hist:
        return np.asarray([], dtype=float)

    if prefix is None:
        return np.asarray([h.get(comp, np.nan) for h in hist], dtype=float)

    key = f"{prefix}_total" if comp == "total" else f"{prefix}_{comp}"
    return np.asarray([h.get(key, np.nan) for h in hist], dtype=float)



# ----------------------------
# Plotter
# ----------------------------

def plot_regime_validation_2x4(
    run_folders,
    save_dir="learning_curves",
    out_name_svg="regime_validation_2x2.svg",
    out_name_png="regime_validation_2x2.png",
    which="norm",                 # "norm" or "raw"
    smooth_alpha=0.12,
    run_colors=None,
    annotate_endpoints=False,
):
    """
    2×4 figure:
      rows   : Matched validation (primary) / Closed-loop rollout (ar)
      columns: MAE / Shape / Cross / Var loss
    Reuses the exact same checkpoint histories and metric extraction rules.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    metrics = ("mae", "shape", "cross", "var")
    metric_titles = ("MAE", "Shape loss", "Cross loss", "Var loss")
    regimes = (("primary", "Matched validation"), ("ar", "Closed-loop rollout"))
    n_metrics = len(metrics)
    n_regimes = len(regimes)

    # Build one canonical run table using the same loading path as existing plots
    runs = []
    for folder in run_folders:
        mode, tr_norm, va_norm, tr_raw, va_raw = load_history(folder)
        if which == "norm":
            va = va_norm
        elif which == "raw":
            if va_raw is None:
                raise KeyError(
                    f"{folder} has no val_losses_raw. Use which='norm' or ensure raw history is saved."
                )
            va = va_raw
        else:
            raise ValueError("which must be 'norm' or 'raw'")

        name = Path(folder).name
        runs.append((name, mode, va))

    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # Precompute smoothed series for all panels
    panel_series = {}  # (ri, ci, run_idx) -> (epochs, y, color, label)
    max_epoch = 0
    legend_handles, legend_labels = [], []

    for run_idx, (name, mode, va) in enumerate(runs):
        label = short_label(name)
        color = (
            (run_colors.get(label) if run_colors is not None else None)
            or (run_colors.get(name) if run_colors is not None else None)
            or color_cycle[run_idx % len(color_cycle)]
        )

        if label not in legend_labels:
            # Proxy handle for one shared legend
            h = plt.Line2D([], [], color=color, linewidth=2.4, label=label)
            legend_handles.append(h)
            legend_labels.append(label)

        for ri, (regime_key, _regime_title) in enumerate(regimes):
            vp = get_primary_prefix(mode, va) if regime_key == "primary" else regime_key
            if regime_key != "primary" and va and (f"{vp}_total" not in va[0]):
                available = available_prefixes_from_val(va)
                raise ValueError(
                    f"val_regime='{regime_key}' missing in {name}. Available prefixes: {available}"
                )

            for ci, metric in enumerate(metrics):
                y = get_series(va, metric, prefix=vp)
                y = ema_smooth(y, alpha=smooth_alpha) if len(y) else y
                epochs = np.arange(1, len(y) + 1)
                panel_series[(ri, ci, run_idx)] = (epochs, y, color, label)

                if len(epochs):
                    max_epoch = max(max_epoch, int(epochs[-1]))

    # Paper-friendly style
    plt.rcParams.update({
        "axes.facecolor": "white",
        "figure.facecolor": "white",
    })

    fig, axes = plt.subplots(
        n_regimes,
        n_metrics,
        figsize=(13.5, 6.2),
        sharex="row",
        constrained_layout=False,
    )

    for ci, t in enumerate(metric_titles):
        axes[0, ci].set_title(t, fontsize=12, fontweight="semibold", pad=8)

    for ri in range(n_regimes):
        for ci in range(n_metrics):
            ax = axes[ri, ci]
            ax.set_xlim(1, max(2, max_epoch))
            ax.grid(False)
            ax.tick_params(labelsize=10)
        _rk, regime_title = regimes[ri]
        axes[ri, 0].set_ylabel(f"{regime_title}\nLoss", fontsize=10)

    for run_idx in range(len(runs)):
        for ri in range(n_regimes):
            for ci in range(n_metrics):
                ax = axes[ri, ci]
                epochs, y, color, _label = panel_series[(ri, ci, run_idx)]
                if len(epochs) == 0:
                    continue
                ax.plot(epochs, y, linewidth=2.4, color=color, alpha=0.96)

                finite_idx = np.where(np.isfinite(y))[0]
                if finite_idx.size:
                    j = int(finite_idx[-1])
                    ax.scatter(epochs[j], y[j], s=20, color=color, zorder=6)
                    if annotate_endpoints:
                        ax.annotate(
                            f"{y[j]:.3g}",
                            (epochs[j], y[j]),
                            xytext=(4, 0),
                            textcoords="offset points",
                            fontsize=8,
                            color=color,
                            va="center",
                        )

    for ci in range(n_metrics):
        axes[n_regimes - 1, ci].set_xlabel("Epoch", fontsize=11)

    # One shared legend outside panel area (top-center)
    fig.legend(
        legend_handles,
        legend_labels,
        ncol=max(1, len(legend_labels)),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        frameon=False,
        fontsize=10,
        handlelength=2.5,
        columnspacing=1.6,
    )

    fig.subplots_adjust(left=0.12, right=0.98, top=0.88, bottom=0.14, wspace=0.26, hspace=0.22)

    out_svg = save_dir / out_name_svg
    out_png = save_dir / out_name_png
    
    fig.savefig(out_svg, format="svg", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out_png, dpi=400, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    print(f"Saved: {out_svg}")
    print(f"Saved: {out_png}")
    return out_svg, out_png



# -----------------------
# MAIN
# -----------------------
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    run_folders = [  
        "checkpoints_ar_kv_2",
        "checkpoints_tf_2",
        "checkpoints_TF_QTL_0.08_KL_0.02",
        #"checkpoints_VAR_BASELINE_90_90",
    ]
    run_colors = {
        "checkpoints_ar_kv_2" : "#2AA876",
        "checkpoints_tf_2": "#A23B72",
        "checkpoints_TF_QTL_0.08_KL_0.02": "#6A4C93",
        #"checkpoints_VAR_BASELINE_90_90": "#3E7CB1",
    }

    plot_regime_validation_2x4(
        run_folders,
        save_dir="learning_curves",
        out_name_svg="regime_validation_2x2.svg",
        out_name_png="regime_validation_2x2.png",
        which="norm",
        smooth_alpha=0.12,
        run_colors=run_colors,
        annotate_endpoints=False,
    )



if __name__ == "__main__":
    main()

