"""Evaluate saved checkpoints only; paired training-seed panels in cnsplots style.

Run from any directory with the project's Python environment::

    python src/visualize/checkpoint_seed_experiment.py --inventory
    python src/visualize/checkpoint_seed_experiment.py

Defaults: six models, seeds 101/102/103, requested epochs 25/75/150, 50 distinct
validation sequence IDs, 90 context + 810 forecast bins. Missing checkpoints
are listed explicitly and never replaced. Panel e combines the six models;
panels f/g compare all models at fixed epoch 150.
Composite is the local nethobench structural composite. Normalized MSE
is mean over regions of forecast MSE / pooled validation forecast variance.
Neither the numerical values nor metric definition of an external experiment
can be inferred from its illustration. No training or parameter fitting occurs.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import random
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "nethobench"))

import numpy as np
import pandas as pd

FOLDERS = {
    "TF": "TF",
    "TF_TQL_0.08_KL_0.02": "TF_QTL_0.08_KL_0.02",
    "AR_KV": "AR_KV",
    "1_step": "1_step",
    "GRU_AR": "GRU_AR_90_90",
    "cDMM_SSM": "cDMM_SSM_z12_90_90",
}
LABELS = {"TF": "TF", "TF_TQL_0.08_KL_0.02": "TF_KL_QL", "AR_KV": "AR",
          "1_step": "1_step", "GRU_AR": "RNN", "cDMM_SSM": "SSM"}
COLORS = {"TF": "#3C5488", "TF_TQL_0.08_KL_0.02": "#8491B4",
          "AR_KV": "#E64B35", "1_step": "#B09C85", "GRU_AR": "#4DBBD5",
          "cDMM_SSM": "#00A087"}
INFERENCE_SOURCES = {
    "GRU_AR": ROOT / "src/infer/inference_gru_ar.py",
    "cDMM_SSM": ROOT / "src/infer/inference_cdmm_ssm.py",
}


def fingerprint(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_plot(fig, path):
    """Render beside the destination, then replace it after the file is closed."""
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.stem}_", suffix=path.suffix, dir=path.parent)
    os.close(fd)
    try:
        fig.savefig(temporary, dpi=300)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def saved_score_table(output, manifest):
    """Load plotted scores only after matching them to the saved per-run scores."""
    table = pd.read_csv(output / "seed_scores.csv")
    runs = manifest.get("runs", [])
    if len(table) != len(runs) or table.empty:
        raise ValueError("Saved score table and run manifest have different lengths")
    if table.duplicated(["model", "seed", "epoch"]).any():
        raise ValueError("Duplicate model/seed/epoch scores")
    for row in table.itertuples(index=False):
        cache = output / "cache" / f"{row.model}_seed{row.seed}_epoch{row.epoch}" / "scores.json"
        saved = json.loads(cache.read_text(encoding="utf-8"))
        scores = saved["scores"]
        if not any(run == saved["metadata"] for run in runs):
            raise ValueError(f"Score provenance missing from manifest: {cache}")
        if not np.allclose([row.composite, row.normalized_mse],
                           [scores["FINAL_COMPOSITE_SCORE"], scores["normalized_mse"]],
                           rtol=1e-12, atol=1e-12):
            raise ValueError(f"Score table disagrees with saved scores: {cache}")
    return table


def sign_flip(differences):
    """Exact two-sided paired randomization test of the mean difference."""
    d = np.asarray(differences, dtype=float)
    if not len(d) or not np.isfinite(d).all():
        raise ValueError("Sign-flip test requires finite paired observations")
    null = np.asarray(list(itertools.product((-1, 1), repeat=len(d)))) @ d
    return float(np.mean(np.abs(null) >= abs(d.sum()) - 1e-12))


def holm(pvalues):
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    adjusted[order] = np.minimum(1, np.maximum.accumulate(p[order] * np.arange(len(p), 0, -1)))
    return adjusted


def validation_paths(config):
    """Honor recorded validation files; old TF seed101 predates path metadata."""
    split = config.get("split_seed", config.get("random_seed"))
    stem = f"data100_ba16_Tin90_Tout{config['T_out']}_splitseed{split}"
    paths = []
    for key, prefix in [("processed_val_examples_path", "processed_val"),
                        ("processed_val_seq_indices_path", "processed_val_seq_indices")]:
        recorded = config.get(key)
        path = Path(recorded) if recorded else ROOT / "data_processed" / f"{prefix}_{stem}.npy"
        if not path.is_absolute():
            path = ROOT / path
        if not path.is_file() and recorded:
            path = ROOT / "data_processed" / Path(recorded).name
        if not path.is_file():
            # Early checkpoints recorded *_seed101; later preprocessing renamed
            # that same fixed split to *_splitseed101. Values are cross-checked.
            path = ROOT / "data_processed" / f"{prefix}_{stem}.npy"
        if not path.is_file():
            raise FileNotFoundError(f"Missing saved validation data: {path}")
        paths.append(path)
    return paths


def validation_windows(paths, count, horizon, selection_seed, selected=None):
    examples = np.load(paths[0], mmap_mode="r")
    ids = np.load(paths[1])
    needed = 90 + horizon
    chunks = int(np.ceil(needed / examples.shape[2]))
    windows = {}
    # Widefield preparation partitions each sequence into consecutive disjoint
    # 180-bin chunks. Use its first complete window; never pad the ground truth.
    for sid in np.unique(ids):
        rows = np.flatnonzero(ids == sid)
        if len(rows) < chunks:
            continue
        window = np.asarray(examples[rows[:chunks]]).transpose(0, 2, 1).reshape(-1, examples.shape[1])[:needed]
        if np.isfinite(window).all():
            windows[int(sid)] = window
    if selected is None:
        if len(windows) < count:
            raise ValueError(f"Need {count} complete distinct validation sequences; found {len(windows)}")
        selected = np.sort(np.random.default_rng(selection_seed).choice(sorted(windows), count, replace=False))
    if any(int(sid) not in windows for sid in selected):
        raise ValueError("Selected sequences are not available in every model's validation split")
    return np.stack([windows[int(sid)] for sid in selected]), np.asarray(selected)


def load_model(name, path, device):
    import torch
    if name == "GRU_AR":
        from infer.inference_gru_ar import create_inference_model
        return create_inference_model(path, device)[0]
    if name == "cDMM_SSM":
        from infer.inference_cdmm_ssm import create_inference_model
        return create_inference_model(path, device)[0]
    from models.model_KV_cached import create_model_cached
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    cfg = checkpoint["config"]
    model = create_model_cached(**{key: cfg[key] for key in
                                ("n_vars", "d_model", "n_heads", "n_layers", "d_ff", "dropout", "T_in", "T_out")}, device=device)
    state = {key.removeprefix("_orig_mod."): value for key, value in checkpoint["model_state_dict"].items()}
    model.load_state_dict(state, strict=True)
    return model.eval()


def forecast(model, name, context, horizon, device, batch_size, evaluation_seed, ssm_num_samples):
    """Use the same rollout and primary prediction as each src/infer entry point."""
    import torch
    if name == "GRU_AR":
        from infer.inference_gru_ar import generate_long_forecast
    elif name == "cDMM_SSM":
        from infer.inference_cdmm_ssm import generate_long_outputs
    else:
        from infer.inference_all_regimes import generate_long_sequence

    batches = []
    with torch.inference_mode():
        # The SSM inference entry point evaluates one sequence at a time with
        # one generator shared across sequences. Batching changes its draws.
        step = 1 if name == "cDMM_SSM" else batch_size
        generator = None
        if name == "cDMM_SSM":
            generator = torch.Generator(device=device)
            generator.manual_seed(int(evaluation_seed))
        for start in range(0, len(context), step):
            x = torch.as_tensor(context[start:start + step], dtype=torch.float32, device=device)
            if name == "GRU_AR":
                pred = generate_long_forecast(model, x, horizon=horizon)
            elif name == "cDMM_SSM":
                _, samples = generate_long_outputs(
                    model, x, horizon=horizon, num_samples=ssm_num_samples,
                    generator=generator,
                )
                pred = samples[0]
            else:
                sequence = generate_long_sequence(
                    model, x, 90 + horizon, 90, int(model.T_out), device
                )
                pred = sequence[:, 90:]
            batches.append(pred.cpu().numpy())
    output = np.concatenate(batches)
    if output.shape != (len(context), horizon, context.shape[-1]) or not np.isfinite(output).all():
        raise ValueError(f"Invalid forecast for {name}: {output.shape}")
    return output


def paired_plot(values, labels, colors, ylabel, title, letter, path, connect_mean=False):
    import matplotlib.pyplot as plt
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(max(5.2, len(labels) * 1.45), 3.65))
    fig.subplots_adjust(left=.18 if len(labels) == 3 else .12, right=.98, bottom=.25, top=.74)
    jitter = np.linspace(-.065, .065, values.shape[0])
    for i, row in enumerate(values):
        ax.plot(x + jitter[i], row, color="#CFD6DC", lw=1.35, zorder=1)
    means = values.mean(axis=0)
    sem = values.std(axis=0, ddof=1) / np.sqrt(len(values))
    if connect_mean:
        ax.plot(x, means, color=colors[0], lw=2.2, zorder=2)
    for j, color in enumerate(colors):
        ax.scatter(j + jitter, values[:, j], s=40, color=color, edgecolors="none", zorder=3)
        ax.errorbar(j, means[j], yerr=sem[j], fmt="none", ecolor="#19232D", elinewidth=2, capsize=0, zorder=4)
        ax.plot([j - .14, j + .14], [means[j]] * 2, color="#19232D", lw=2.5, zorder=5)
    ax.set_xticks(x, labels)
    ax.set_ylabel(ylabel)
    if connect_mean:
        ax.set_xlabel("Training epoch")
    ax.set_xlim(-.35, len(labels) - .65)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(False)
    ax.margins(y=.2)
    fig.text(.015, .97, letter, fontsize=20, weight="bold", va="top")
    fig.text(.085, .97, title, fontsize=16, va="top")
    for extension in ("png", "svg", "pdf"):
        save_plot(fig, Path(f"{path}.{extension}"))
    plt.close(fig)


def epoch_plot(table, args):
    """One panel with three seed scores per model and available epoch."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8.2, 4.3))
    fig.subplots_adjust(left=.12, right=.98, bottom=.19, top=.77)
    offsets = np.linspace(-.33, .33, len(args.models))
    for name, offset in zip(args.models, offsets):
        part = table[table.model == name]
        xmean, ymean = [], []
        for j, epoch in enumerate(args.epochs):
            scores = part.loc[part.epoch == epoch].set_index("seed").reindex(args.seeds)["composite"]
            for k, value in enumerate(scores):
                if np.isfinite(value):
                    ax.scatter(j + offset + (k - (len(args.seeds) - 1) / 2) * .035, value,
                               color=COLORS[name], s=32, zorder=3)
            if scores.notna().all():
                xmean.append(j + offset)
                ymean.append(scores.mean())
                sem = scores.sem()
                ax.errorbar(j + offset, scores.mean(), yerr=sem, fmt="none",
                            ecolor=COLORS[name], elinewidth=1.5, capsize=2, zorder=4)
        if xmean:
            ax.plot(xmean, ymean, color=COLORS[name], lw=1.5, alpha=.8,
                    label=LABELS[name], zorder=2)
        else:
            ax.plot([], [], color=COLORS[name], label=LABELS[name])
    for j, epoch in enumerate(args.epochs):
        if not (table.epoch == epoch).any():
            ax.text(j, .04, "No checkpoint", transform=ax.get_xaxis_transform(),
                    ha="center", fontsize=10, color="#7A3030")
    ax.set_xticks(np.arange(len(args.epochs)), [str(e) for e in args.epochs])
    ax.set_xlim(-.6, len(args.epochs) - .4)
    ax.set_xlabel("Training epoch")
    ax.set_ylabel("Composite")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(False)
    ax.margins(y=.18)
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(.5, 1.27),
              frameon=False, fontsize=10)
    fig.text(.015, .98, "e", fontsize=20, weight="bold", va="top")
    fig.text(.085, .98, "Composite by checkpoint epoch", fontsize=16, va="top")
    for extension in ("png", "svg", "pdf"):
        save_plot(fig, args.output / f"e_composite.{extension}")
    plt.close(fig)


def report(table, args):
    from cns_plotting import setup_cnsplots_style
    expected = pd.MultiIndex.from_product(
        [args.models, args.seeds, sorted(set(args.epochs + [args.comparison_epoch]))],
        names=["model", "seed", "epoch"],
    )
    indexed = table.set_index(["model", "seed", "epoch"])
    if not indexed.index.is_unique:
        raise ValueError("Duplicate model/seed/epoch scores")
    selected_scores = indexed.reindex(expected)[["composite", "normalized_mse"]]
    table = selected_scores.reset_index()
    setup_cnsplots_style({"font.family": "Arial", "font.size": 13, "axes.labelsize": 15,
                          "xtick.labelsize": 12, "ytick.labelsize": 12, "svg.fonttype": "none",
                          "savefig.bbox": None})
    summary = table.groupby(["model", "epoch"], sort=False)[["composite", "normalized_mse"]].agg(["mean", "sem"])
    summary.to_csv(args.output / "mean_sem.csv")
    table.to_csv(args.output / "panel_source_scores.csv", index=False)
    epoch_plot(table, args)
    tests = []
    for name in args.models:
        wide = table[table.model == name].pivot(index="seed", columns="epoch", values="composite").loc[args.seeds, args.epochs]
        family = []
        for a, b in itertools.combinations(args.epochs, 2):
            diff = (wide[b] - wide[a]).to_numpy()
            if not np.isfinite(diff).all():
                continue
            family.append(dict(model=name, epoch_a=a, epoch_b=b, mean_b_minus_a=float(diff.mean()),
                               b_exceeds_a=int((diff > 0).sum()), ties=int((diff == 0).sum()), n_seeds=len(diff), p=sign_flip(diff)))
        for row, adjusted in zip(family, holm([r["p"] for r in family])):
            row["p_holm_within_model"] = adjusted
        tests.extend(family)
    pd.DataFrame(tests).to_csv(args.output / "checkpoint_contrasts.csv", index=False)
    comparison = table[table.epoch == args.comparison_epoch]
    comparison_complete = np.isfinite(comparison[["composite", "normalized_mse"]].to_numpy()).all()
    for metric, letter, ylabel in [("composite", "f", "Composite"), ("normalized_mse", "g", "Normalized MSE")]:
        if not comparison_complete:
            print(f"Panel {letter} unavailable: epoch {args.comparison_epoch} scores are incomplete", file=sys.stderr)
            continue
        wide = comparison.pivot(index="seed", columns="model", values=metric).loc[args.seeds, args.models]
        title = f"{'Composite scores' if metric == 'composite' else 'Pointwise forecast error'} at epoch {args.comparison_epoch}\nunder the same evaluation protocol"
        paired_plot(wide.to_numpy(), [LABELS[m] for m in args.models], [COLORS[m] for m in args.models],
                    ylabel, title, letter, args.output / f"{letter}_{metric}")
    (args.output / "interpretation.txt").write_text(
        "Saved checkpoints only; no retraining. Sequifier and VAR excluded.\n"
        f"Training seeds: {args.seeds}; sampled epochs: {args.epochs}; comparison epoch: {args.comparison_epoch}.\n"
        f"{args.num_sequences} distinct validation sequence IDs; 90 context + {args.horizon} generated bins.\n"
        "One score per training seed on the entire shared validation set; no SEM over validation splits.\n"
        "Seed SEM describes training variability, not animal or session uncertainty.\n"
        "Panel e is one combined plot; a missing epoch has no points and is labeled.\n"
        "Models are not matched for capacity or compute.\n"
        "cDMM uses the first stochastic sample from src/infer/inference_cdmm_ssm.py; it is not the reference diagonal SSM.\n"
        "Composite: nethobench.calculate_neuro_composites FINAL_COMPOSITE_SCORE.\n"
        "Normalized MSE: mean_r(mean_sequence,time((pred-gt)^2) / var_sequence,time(gt)).\n"
        "Exact two-sided paired sign-flip tests; Holm across checkpoint contrasts within each model.\n"
        "Three seeds imply a minimum attainable two-sided p-value of 0.25.\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint-root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / "output/checkpoint_seed_experiment")
    parser.add_argument("--models", nargs="+", choices=list(FOLDERS), default=list(FOLDERS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[101, 102, 103])
    parser.add_argument("--epochs", nargs="+", type=int, default=[25, 75, 150])
    parser.add_argument("--comparison-epoch", type=int, default=150)
    parser.add_argument("--num-sequences", type=int, default=50)
    parser.add_argument("--horizon", type=int, default=810, help="Generated bins, excluding 90 context bins")
    parser.add_argument("--evaluation-seed", type=int, default=102)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=15)
    parser.add_argument("--ssm-num-samples", type=int, default=None,
                        help="SSM samples per context; defaults to src/infer's globals.json setting")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--inventory", action="store_true")
    parser.add_argument("--plots-only", action="store_true")
    parser.add_argument("--force", action="store_true", help="Recompute cached predictions and scores")
    args = parser.parse_args(argv)
    if len(args.seeds) < 2 or len(set(args.seeds)) != len(args.seeds):
        parser.error("Use at least two distinct training seeds")
    if len(set(args.epochs)) != len(args.epochs) or len(set(args.models)) != len(args.models):
        parser.error("Models and epochs must be unique")
    if min(args.num_sequences, args.horizon, args.batch_size, args.threads) < 1:
        parser.error("Counts must be positive")
    if args.ssm_num_samples is not None and args.ssm_num_samples < 1:
        parser.error("--ssm-num-samples must be positive")
    paths = {}
    missing = []
    for name, seed in itertools.product(args.models, args.seeds):
        folder = args.checkpoint_root / f"checkpoints_{FOLDERS[name]}_seed{seed}"
        available = sorted(int(p.stem.rsplit("_", 1)[1]) for p in folder.glob("checkpoint_fixed_epoch_*.pt"))
        if args.inventory:
            print(f"{name} seed {seed}: {available}")
        for epoch in sorted(set(args.epochs + [args.comparison_epoch])):
            path = folder / f"checkpoint_fixed_epoch_{epoch}.pt"
            if path.is_file():
                paths[name, seed, epoch] = path
            else:
                missing.append(dict(model=name, seed=seed, epoch=epoch, checkpoint=str(path),
                                    available_epochs=",".join(map(str, available))))
    if args.inventory:
        return
    args.output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(missing, columns=["model", "seed", "epoch", "checkpoint", "available_epochs"]).to_csv(
        args.output / "missing_checkpoints.csv", index=False)
    for row in missing:
        print(f"Missing {row['checkpoint']}; available epochs: {row['available_epochs']}. No substitution.", file=sys.stderr)
    if args.plots_only:
        manifest_path = args.output / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"No completed run manifest: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        report(saved_score_table(args.output, manifest), args)
        return
    import torch
    if "cDMM_SSM" in args.models and args.ssm_num_samples is None:
        from infer.inference_cdmm_ssm import parse_args as ssm_parse_args
        args.ssm_num_samples = int(ssm_parse_args([]).num_samples)
    from nethobench.neuro.metrics.composites import calculate_neuro_composites
    from threadpoolctl import threadpool_limits
    torch.set_num_threads(args.threads)
    selected = None
    reference = None
    rows = []
    manifest = {"arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, "runs": []}
    for (name, seed, epoch), path in paths.items():
        print(f"Evaluating {name}, seed {seed}, epoch {epoch}", flush=True)
        cfg = torch.load(path, map_location="cpu", weights_only=False)["config"]
        if int(cfg["T_in"]) != 90:
            raise ValueError("This protocol requires a 90-bin context")
        data_paths = validation_paths(cfg)
        windows, selected = validation_windows(data_paths, args.num_sequences, args.horizon, args.evaluation_seed, selected)
        if reference is None:
            reference = windows
            np.save(args.output / "validation_sequence_ids.npy", selected)
        elif not np.allclose(reference, windows, rtol=1e-5, atol=1e-6):
            raise ValueError("Validation values/normalization differ across checkpoints; cannot treat them as matched data")
        gt = windows[:, 90:].astype(np.float64)
        metadata = {"checkpoint": str(path.resolve()), "checkpoint_sha256": fingerprint(path),
                    "window_sha256": hashlib.sha256(windows.tobytes()).hexdigest(),
                    "metric_source_sha256": fingerprint(ROOT / "nethobench/nethobench/neuro/metrics/composites.py"),
                    "script_sha256": fingerprint(Path(__file__)), "evaluation_seed": args.evaluation_seed,
                    "inference_source_sha256": fingerprint(INFERENCE_SOURCES.get(
                        name, ROOT / "src/infer/inference_all_regimes.py")),
                    "device": args.device, "batch_size": args.batch_size,
                    "ssm_num_samples": args.ssm_num_samples if name == "cDMM_SSM" else None,
                    "torch_version": torch.__version__}
        cache = args.output / "cache" / f"{name}_seed{seed}_epoch{epoch}"
        cache.mkdir(parents=True, exist_ok=True)
        score_path = cache / "scores.json"
        cached = json.loads(score_path.read_text()) if score_path.exists() else None
        if not args.force and cached and cached["metadata"] == metadata:
            scores = cached["scores"]
        else:
            random.seed(args.evaluation_seed)
            np.random.seed(args.evaluation_seed)
            torch.manual_seed(args.evaluation_seed)
            forecast_meta = cache / "forecast_metadata.json"
            if (not args.force and forecast_meta.exists() and (cache / "forecast.npy").exists()
                    and json.loads(forecast_meta.read_text()) == metadata):
                pred = np.load(cache / "forecast.npy")
            else:
                model = load_model(name, path, torch.device(args.device))
                pred = forecast(model, name, windows[:, :90], args.horizon,
                                args.device, args.batch_size, args.evaluation_seed,
                                args.ssm_num_samples)
                np.save(cache / "forecast.npy", pred)
                forecast_meta.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
                del model
            print("  Scoring forecast...", flush=True)
            with threadpool_limits(limits=args.threads):
                scores = {k: float(v) for k, v in calculate_neuro_composites(gt, pred.astype(np.float64)).items()}
            variance = gt.var(axis=(0, 1))
            if np.any(variance <= 1e-12):
                raise ValueError("Normalized MSE undefined for a constant validation region")
            scores["normalized_mse"] = float(np.mean(np.mean((pred - gt) ** 2, axis=(0, 1)) / variance))
            if not np.isfinite([scores["FINAL_COMPOSITE_SCORE"], scores["normalized_mse"]]).all():
                raise ValueError("Nonfinite primary scores")
            score_path.write_text(json.dumps({"metadata": metadata, "scores": scores}, indent=2), encoding="utf-8")
        rows.append(dict(model=name, seed=seed, epoch=epoch, composite=scores["FINAL_COMPOSITE_SCORE"], normalized_mse=scores["normalized_mse"]))
        manifest["runs"].append(metadata)
        pd.DataFrame(rows).to_csv(args.output / "seed_scores.csv", index=False)
        (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    report(pd.DataFrame(rows), args)
    print(f"Saved results to {args.output}")


if __name__ == "__main__":
    main()
