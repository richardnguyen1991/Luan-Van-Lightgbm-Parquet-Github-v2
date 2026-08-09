"""Headless thesis-ready visualizations for the LightGBM baseline."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import gc
import json
import math
import shutil
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from sklearn.metrics import auc, precision_recall_curve, roc_curve


COLOR_PALETTE = (
    "#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00",
    "#56B4E9", "#000000", "#7A3E00", "#5D3A9B", "#008080",
)
LINE_STYLES = ("-", "--", "-.", ":")
MARKERS = ("o", "s", "^", "D", "v", "P", "X", "<", ">", "*")
AXIS_FONT_SIZE = 10
TITLE_FONT_SIZE = 11
MAX_CURVE_POINTS = 5000
ArtifactCallback = Callable[[Path, str], None]


def _read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _context(run_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[str], str, str, int]:
    run_config = _read_json(run_dir / "config" / "run_config.json")
    history = _read_json(run_dir / "metrics" / "history.json")
    mapping = _read_json(run_dir / "config" / "label_mapping.json")
    class_names = [name for name, _ in sorted(mapping.items(), key=lambda item: int(item[1]))]
    final_iteration = max((int(record["iteration"]) for record in history), default=0)
    return (
        run_config,
        history,
        class_names,
        str(run_config["model_name"]),
        str(run_config.get("run_id", run_dir.name)),
        final_iteration,
    )


def _title(base: str, model: str, run_id: str, final_iteration: int, suffix: str = "") -> str:
    progress = "" if final_iteration == 100 else f" | current_iteration={final_iteration}"
    extra = f" | {suffix}" if suffix else ""
    return f"{base} | {model} | {run_id} | final_iteration=100{progress}{extra}"


def _style_axis(axis: plt.Axes, grid: bool = True) -> None:
    axis.tick_params(axis="both", labelsize=AXIS_FONT_SIZE)
    for label in axis.get_xticklabels():
        label.set_rotation(45)
        label.set_ha("right")
    if grid:
        axis.grid(True, linestyle="--", alpha=0.4)


def _line_style(index: int, many: bool = False) -> dict[str, Any]:
    return {
        "color": COLOR_PALETTE[index % len(COLOR_PALETTE)],
        "linestyle": LINE_STYLES[index % len(LINE_STYLES)] if many else LINE_STYLES[(index // len(COLOR_PALETTE)) % len(LINE_STYLES)],
        "marker": MARKERS[index % len(MARKERS)] if many else MARKERS[index % 2],
        "markevery": 0.12,
        "markersize": 3.2,
        "linewidth": 1.35,
    }


def _notify(paths: Sequence[Path], category: str, callback: ArtifactCallback | None) -> None:
    if callback:
        for path in paths:
            callback(path, category)


def _save_figure(
    fig: plt.Figure,
    run_dir: Path,
    name: str,
    table: pd.DataFrame,
    callback: ArtifactCallback | None = None,
    csv_index: bool = False,
) -> list[Path]:
    figures = run_dir / "figures"
    metrics = run_dir / "metrics"
    figures.mkdir(parents=True, exist_ok=True)
    metrics.mkdir(parents=True, exist_ok=True)
    png, pdf, csv_path = figures / f"{name}.png", figures / f"{name}.pdf", metrics / f"{name}.csv"
    fig.tight_layout()
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight", metadata={"CreationDate": None, "ModDate": None})
    table.to_csv(csv_path, index=csv_index, float_format="%.6f")
    plt.close(fig)
    gc.collect()
    _notify((png, pdf), "figures", callback)
    _notify((csv_path,), "metrics", callback)
    return [png, pdf, csv_path]


def _downsample(*arrays: np.ndarray) -> tuple[np.ndarray, ...]:
    length = len(arrays[0])
    if length <= MAX_CURVE_POINTS:
        return tuple(np.asarray(value) for value in arrays)
    indices = np.unique(np.linspace(0, length - 1, MAX_CURVE_POINTS, dtype=np.int64))
    return tuple(np.asarray(value)[indices] for value in arrays)


def plot_learning_curves(run_dir: Path, callback: ArtifactCallback | None = None) -> list[Path]:
    _, history, _, model, run_id, final_iteration = _context(run_dir)
    table = pd.DataFrame(history)
    required = {
        "iteration", "session_id", "train_multi_logloss", "val_multi_logloss",
        "train_multi_error", "val_multi_error", "train_macro_f1", "val_macro_f1",
    }
    missing = sorted(required.difference(table.columns))
    if missing:
        raise ValueError(f"history.json lacks learning-curve fields: {missing}")
    changes = [
        int(table.iloc[index]["iteration"])
        for index in range(1, len(table))
        if table.iloc[index]["session_id"] != table.iloc[index - 1]["session_id"]
    ]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
    panels = (
        ("train_multi_logloss", "val_multi_logloss", "Multi-logloss", "Multi-logloss", False),
        ("train_multi_error", "val_multi_error", "Accuracy", "Accuracy", True),
        ("train_macro_f1", "val_macro_f1", "Macro-F1", "Macro-F1", False),
    )
    for axis, (train_col, val_col, panel, ylabel, invert) in zip(axes, panels):
        train_values = 1.0 - table[train_col] if invert else table[train_col]
        val_values = 1.0 - table[val_col] if invert else table[val_col]
        axis.plot(table["iteration"], train_values, label="Train", **_line_style(0))
        axis.plot(table["iteration"], val_values, label="Validation", **_line_style(1))
        for index, iteration in enumerate(changes):
            axis.axvline(iteration, color="#666666", linestyle="--", linewidth=1.0, label="Resume" if index == 0 else None)
        axis.set_title(panel, fontsize=TITLE_FONT_SIZE)
        axis.set_xlabel("Boosting iteration", fontsize=AXIS_FONT_SIZE)
        axis.set_ylabel(ylabel, fontsize=AXIS_FONT_SIZE)
        axis.legend(fontsize=9)
        _style_axis(axis)
    fig.suptitle(_title("Learning Curves", model, run_id, final_iteration), fontsize=TITLE_FONT_SIZE)
    plot_table = table.copy()
    plot_table["train_accuracy"] = 1.0 - plot_table["train_multi_error"]
    plot_table["val_accuracy"] = 1.0 - plot_table["val_multi_error"]
    paths = _save_figure(fig, run_dir, "learning_curves", plot_table, callback)
    history_csv = run_dir / "metrics" / "history.csv"
    table.to_csv(history_csv, index=False, float_format="%.6f")
    _notify((history_csv,), "metrics", callback)
    return [*paths, history_csv]


def plot_lr_schedule(run_dir: Path, callback: ArtifactCallback | None = None) -> list[Path]:
    _, history, _, model, run_id, final_iteration = _context(run_dir)
    table = pd.DataFrame(history)[["iteration", "learning_rate"]]
    fig, axis = plt.subplots(figsize=(8.5, 5.2))
    axis.plot(table["iteration"], table["learning_rate"], label="Learning rate", **_line_style(0))
    axis.set_xlabel("Boosting iteration", fontsize=AXIS_FONT_SIZE)
    axis.set_ylabel("Learning rate", fontsize=AXIS_FONT_SIZE)
    axis.set_title(_title("Learning-Rate Schedule", model, run_id, final_iteration), fontsize=TITLE_FONT_SIZE)
    _style_axis(axis)
    return _save_figure(fig, run_dir, "lr_schedule", table, callback)


def plot_iteration_time(run_dir: Path, callback: ArtifactCallback | None = None) -> list[Path]:
    run_config, history, _, model, run_id, final_iteration = _context(run_dir)
    table = pd.DataFrame(history)[["iteration", "session_id", "iteration_seconds", "checkpoint_seconds"]]
    sessions = list(dict.fromkeys(table["session_id"].tolist()))
    colors = {session: COLOR_PALETTE[index % len(COLOR_PALETTE)] for index, session in enumerate(sessions)}
    fig, axis = plt.subplots(figsize=(11, 5.6))
    axis.bar(table["iteration"], table["iteration_seconds"], color=[colors[value] for value in table["session_id"]], edgecolor="black", linewidth=0.2, label="Boosting iteration")
    checkpoint_rows = table[table["checkpoint_seconds"] > 0]
    axis.scatter(checkpoint_rows["iteration"], checkpoint_rows["checkpoint_seconds"], marker="D", color="#000000", s=25, label="Checkpoint block")
    average = float(table["iteration_seconds"].mean()) if len(table) else 0.0
    remaining = max(0, 100 - final_iteration)
    safe_seconds = float(run_config["session"]["maximum_hours"]) * 3600 - float(run_config["session"]["stop_before_minutes"]) * 60
    sessions_needed = math.ceil(remaining * average / safe_seconds) if average and safe_seconds > 0 else 0
    axis.set_xlabel("Boosting iteration", fontsize=AXIS_FONT_SIZE)
    axis.set_ylabel("Seconds", fontsize=AXIS_FONT_SIZE)
    axis.set_title(_title("Iteration and Checkpoint Time", model, run_id, final_iteration, f"estimated remaining sessions={sessions_needed}"), fontsize=TITLE_FONT_SIZE)
    axis.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    _style_axis(axis)
    return _save_figure(fig, run_dir, "iteration_time", table, callback)


def plot_class_distribution(run_dir: Path, callback: ArtifactCallback | None = None) -> list[Path]:
    _, _, class_names, model, run_id, final_iteration = _context(run_dir)
    manifest = _read_json(run_dir / "config" / "sample_manifest.json")
    rows = [
        {"split": split, "class": name, "count": int(manifest["split"]["class_counts"][split].get(name, 0))}
        for split in ("train", "validation", "test") for name in class_names
    ]
    table = pd.DataFrame(rows)
    x = np.arange(len(class_names))
    width = 0.25
    fig, axis = plt.subplots(figsize=(max(10, len(class_names) * 0.72), 6))
    for index, split in enumerate(("train", "validation", "test")):
        values = table.loc[table["split"] == split, "count"].to_numpy()
        axis.bar(x + (index - 1) * width, values, width, label=split.title(), color=COLOR_PALETTE[index], edgecolor="black", linewidth=0.25)
    axis.set_yscale("log")
    axis.set_xticks(x, class_names)
    axis.set_xlabel("Class", fontsize=AXIS_FONT_SIZE)
    axis.set_ylabel("Samples (log scale)", fontsize=AXIS_FONT_SIZE)
    axis.set_title(_title("Class Distribution", model, run_id, final_iteration), fontsize=TITLE_FONT_SIZE)
    axis.legend(fontsize=9)
    _style_axis(axis)
    return _save_figure(fig, run_dir, "class_distribution", table, callback)


def _annotate_matrix(axis: plt.Axes, matrix: np.ndarray, normalized: bool) -> None:
    count = matrix.shape[0]
    if count > 15:
        return
    size = 7 if count > 10 else 9
    threshold = float(np.nanmax(matrix)) / 2 if matrix.size else 0.0
    for row in range(count):
        for column in range(count):
            value = matrix[row, column]
            text = f"{value:.4f}" if normalized else str(int(value))
            axis.text(column, row, text, ha="center", va="center", fontsize=size, color="white" if value > threshold else "black")


def plot_confusion_matrices(run_dir: Path, confusion: np.ndarray, callback: ArtifactCallback | None = None) -> list[Path]:
    _, _, names, model, run_id, final_iteration = _context(run_dir)
    if confusion.shape != (len(names), len(names)):
        raise ValueError("Confusion matrix shape differs from label_mapping.json")
    raw = pd.DataFrame(confusion.astype(np.int64), index=names, columns=names)
    with np.errstate(divide="ignore", invalid="ignore"):
        normalized_values = confusion.astype(np.float64) / confusion.sum(axis=1, keepdims=True)
    normalized_values = np.nan_to_num(normalized_values, nan=0.0, posinf=0.0, neginf=0.0)
    normalized = pd.DataFrame(normalized_values, index=names, columns=names)
    size = max(8.5, len(names) * 0.58)
    fig, axis = plt.subplots(figsize=(size, size * 0.88))
    image = axis.imshow(normalized_values, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    fig.colorbar(image, ax=axis)
    _annotate_matrix(axis, normalized_values, True)
    axis.set_xticks(range(len(names)), names)
    axis.set_yticks(range(len(names)), names)
    axis.set_xlabel("Predicted class", fontsize=AXIS_FONT_SIZE)
    axis.set_ylabel("True class", fontsize=AXIS_FONT_SIZE)
    axis.set_title(_title("Row-Normalized Confusion Matrix", model, run_id, final_iteration), fontsize=TITLE_FONT_SIZE)
    _style_axis(axis, False)
    generated = _save_figure(fig, run_dir, "confusion_matrix_normalized", normalized, callback, csv_index=True)
    normalized_png = run_dir / "figures" / "confusion_matrix.png"
    normalized_pdf = run_dir / "figures" / "confusion_matrix.pdf"
    shutil.copyfile(run_dir / "figures" / "confusion_matrix_normalized.png", normalized_png)
    shutil.copyfile(run_dir / "figures" / "confusion_matrix_normalized.pdf", normalized_pdf)
    _notify((normalized_png, normalized_pdf), "figures", callback)
    generated.extend((normalized_png, normalized_pdf))

    fig, axis = plt.subplots(figsize=(size, size * 0.88))
    masked = np.ma.masked_less_equal(confusion, 0)
    image = axis.imshow(masked, cmap="viridis", norm=LogNorm(vmin=1, vmax=max(1, int(confusion.max()))), aspect="auto")
    fig.colorbar(image, ax=axis)
    _annotate_matrix(axis, confusion, False)
    axis.set_xticks(range(len(names)), names)
    axis.set_yticks(range(len(names)), names)
    axis.set_xlabel("Predicted class", fontsize=AXIS_FONT_SIZE)
    axis.set_ylabel("True class", fontsize=AXIS_FONT_SIZE)
    axis.set_title(_title("Raw Confusion Matrix (Log Scale)", model, run_id, final_iteration), fontsize=TITLE_FONT_SIZE)
    _style_axis(axis, False)
    generated.extend(_save_figure(fig, run_dir, "confusion_matrix_raw", raw, callback, csv_index=True))
    raw_alias = run_dir / "metrics" / "confusion_matrix.csv"
    raw.to_csv(raw_alias, index=True)
    _notify((raw_alias,), "metrics", callback)
    return [*generated, raw_alias]


def plot_roc_curves(run_dir: Path, y_true: np.ndarray, y_prob: np.ndarray, callback: ArtifactCallback | None = None) -> list[Path]:
    _, _, names, model, run_id, final_iteration = _context(run_dir)
    rows: list[dict[str, Any]] = []
    grid = np.linspace(0, 1, 2001)
    macro_sum = np.zeros_like(grid)
    macro_count = 0
    fig, axis = plt.subplots(figsize=(10.5, 7))
    many = len(names) > 6
    for class_index, name in enumerate(names):
        y_true_binary = (np.asarray(y_true) == class_index).astype(np.int8)
        if y_true_binary.min() == y_true_binary.max():
            del y_true_binary
            gc.collect()
            continue
        y_prob_c = np.asarray(y_prob[:, class_index], dtype=np.float32)
        fpr, tpr, thresholds = roc_curve(y_true_binary, y_prob_c)
        score = auc(fpr, tpr)
        draw_fpr, draw_tpr = _downsample(fpr, tpr)
        axis.plot(draw_fpr, draw_tpr, label=f"{name} ({score:.4f})", **_line_style(class_index, many))
        rows.extend({"class": name, "fpr": float(x), "tpr": float(y), "threshold": float(t)} for x, y, t in zip(fpr, tpr, thresholds))
        macro_sum += np.interp(grid, fpr, tpr)
        macro_count += 1
        del y_true_binary, y_prob_c, fpr, tpr, thresholds, draw_fpr, draw_tpr
        gc.collect()
    y_true_binary = np.equal.outer(np.asarray(y_true), np.arange(len(names))).astype(np.int8).ravel()
    y_prob_c = np.asarray(y_prob, dtype=np.float32).ravel()
    fpr, tpr, thresholds = roc_curve(y_true_binary, y_prob_c)
    micro_score = auc(fpr, tpr)
    draw_fpr, draw_tpr = _downsample(fpr, tpr)
    axis.plot(draw_fpr, draw_tpr, label=f"Micro-average ({micro_score:.4f})", color="#000000", linestyle="--", linewidth=2)
    rows.extend({"class": "micro-average", "fpr": float(x), "tpr": float(y), "threshold": float(t)} for x, y, t in zip(fpr, tpr, thresholds))
    del y_true_binary, y_prob_c, fpr, tpr, thresholds, draw_fpr, draw_tpr
    gc.collect()
    if macro_count:
        macro_tpr = macro_sum / macro_count
        macro_score = auc(grid, macro_tpr)
        axis.plot(grid, macro_tpr, label=f"Macro-average ({macro_score:.4f})", color="#666666", linestyle="-.", linewidth=2)
        rows.extend({"class": "macro-average", "fpr": float(x), "tpr": float(y), "threshold": np.nan} for x, y in zip(grid, macro_tpr))
    axis.plot([0, 1], [0, 1], color="#444444", linestyle=":", linewidth=1.2, label="Random Guess")
    axis.set_xlabel("False Positive Rate", fontsize=AXIS_FONT_SIZE)
    axis.set_ylabel("True Positive Rate", fontsize=AXIS_FONT_SIZE)
    axis.set_title(_title("One-vs-Rest ROC Curves", model, run_id, final_iteration, f"plot samples={len(y_true):,}"), fontsize=TITLE_FONT_SIZE)
    axis.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    _style_axis(axis)
    return _save_figure(fig, run_dir, "roc_curves", pd.DataFrame(rows), callback)


def plot_pr_curves(run_dir: Path, y_true: np.ndarray, y_prob: np.ndarray, callback: ArtifactCallback | None = None) -> list[Path]:
    _, _, names, model, run_id, final_iteration = _context(run_dir)
    rows: list[dict[str, Any]] = []
    fig, axis = plt.subplots(figsize=(10.5, 7))
    many = len(names) > 6
    for class_index, name in enumerate(names):
        y_true_binary = (np.asarray(y_true) == class_index).astype(np.int8)
        if not y_true_binary.any():
            del y_true_binary
            gc.collect()
            continue
        y_prob_c = np.asarray(y_prob[:, class_index], dtype=np.float32)
        precision, recall, thresholds = precision_recall_curve(y_true_binary, y_prob_c)
        draw_recall, draw_precision = _downsample(recall, precision)
        axis.plot(draw_recall, draw_precision, label=name, **_line_style(class_index, many))
        padded_thresholds = np.append(thresholds, np.nan)
        rows.extend({"class": name, "recall": float(x), "precision": float(y), "threshold": float(t)} for x, y, t in zip(recall, precision, padded_thresholds))
        del y_true_binary, y_prob_c, precision, recall, thresholds, padded_thresholds, draw_recall, draw_precision
        gc.collect()
    y_true_binary = np.equal.outer(np.asarray(y_true), np.arange(len(names))).astype(np.int8).ravel()
    y_prob_c = np.asarray(y_prob, dtype=np.float32).ravel()
    precision, recall, thresholds = precision_recall_curve(y_true_binary, y_prob_c)
    draw_recall, draw_precision = _downsample(recall, precision)
    axis.plot(draw_recall, draw_precision, label="Micro-average", color="#000000", linestyle="--", linewidth=2)
    padded_thresholds = np.append(thresholds, np.nan)
    rows.extend({"class": "micro-average", "recall": float(x), "precision": float(y), "threshold": float(t)} for x, y, t in zip(recall, precision, padded_thresholds))
    del y_true_binary, y_prob_c, precision, recall, thresholds, padded_thresholds, draw_recall, draw_precision
    gc.collect()
    axis.set_xlabel("Recall", fontsize=AXIS_FONT_SIZE)
    axis.set_ylabel("Precision", fontsize=AXIS_FONT_SIZE)
    axis.set_title(_title("One-vs-Rest Precision-Recall Curves", model, run_id, final_iteration, f"plot samples={len(y_true):,}"), fontsize=TITLE_FONT_SIZE)
    axis.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    _style_axis(axis)
    return _save_figure(fig, run_dir, "pr_curves", pd.DataFrame(rows), callback)


def plot_per_class_metrics(run_dir: Path, per_class: pd.DataFrame, callback: ArtifactCallback | None = None) -> list[Path]:
    _, _, _, model, run_id, final_iteration = _context(run_dir)
    order = per_class.sort_values(["support", "class"], ascending=[True, True]).reset_index(drop=True)
    y = np.arange(len(order))
    height = 0.24
    fig, axis = plt.subplots(figsize=(11, max(6, len(order) * 0.48)))
    for offset, column, label, color in ((-height, "f1", "F1", 0), (0, "precision", "Precision", 1), (height, "recall", "Recall", 2)):
        bars = axis.barh(y + offset, order[column], height, label=label, color=COLOR_PALETTE[color], edgecolor="black", linewidth=0.25)
        for bar, value in zip(bars, order[column]):
            axis.text(min(float(value) + 0.008, 1.01), bar.get_y() + bar.get_height() / 2, f"{float(value):.4f}", va="center", fontsize=8)
    axis.set_yticks(y, [f"{row['class']} (n={int(row['support']):,})" for _, row in order.iterrows()])
    axis.set_xlim(0, 1.05)
    axis.set_xlabel("Score", fontsize=AXIS_FONT_SIZE)
    axis.set_ylabel("Class (support)", fontsize=AXIS_FONT_SIZE)
    axis.set_title(_title("Per-Class F1 / Precision / Recall", model, run_id, final_iteration), fontsize=TITLE_FONT_SIZE)
    axis.legend(fontsize=9)
    _style_axis(axis)
    return _save_figure(fig, run_dir, "per_class_f1", order, callback)


def plot_feature_importance(
    run_dir: Path,
    table: pd.DataFrame,
    name: str,
    value_column: str,
    xlabel: str,
    callback: ArtifactCallback | None = None,
    error_column: str | None = None,
) -> list[Path]:
    _, _, _, model, run_id, final_iteration = _context(run_dir)
    top = table.sort_values(value_column, ascending=False).head(30).sort_values(value_column, ascending=True)
    fig, axis = plt.subplots(figsize=(10.5, max(6, len(top) * 0.32)))
    errors = top[error_column] if error_column and error_column in top else None
    axis.barh(top["feature"], top[value_column], xerr=errors, color=COLOR_PALETTE[0], edgecolor="black", linewidth=0.25, capsize=2)
    axis.set_xlabel(xlabel, fontsize=AXIS_FONT_SIZE)
    axis.set_ylabel("Feature", fontsize=AXIS_FONT_SIZE)
    axis.set_title(_title(name.replace("_", " ").title(), model, run_id, final_iteration), fontsize=TITLE_FONT_SIZE)
    _style_axis(axis)
    return _save_figure(fig, run_dir, name, table, callback)


def generate_incremental_reports(run_dir: str | Path, callback: ArtifactCallback | None = None) -> list[Path]:
    root = Path(run_dir)
    return [
        *plot_learning_curves(root, callback),
        *plot_lr_schedule(root, callback),
        *plot_iteration_time(root, callback),
    ]


def generate_final_figures(
    run_dir: str | Path,
    y_true_plot: np.ndarray,
    y_prob_plot: np.ndarray,
    confusion: np.ndarray,
    per_class: pd.DataFrame,
    importance_tables: Mapping[str, pd.DataFrame],
    callback: ArtifactCallback | None = None,
) -> list[Path]:
    root = Path(run_dir)
    generated = [
        *plot_learning_curves(root, callback),
        *plot_lr_schedule(root, callback),
        *plot_confusion_matrices(root, confusion, callback),
        *plot_roc_curves(root, y_true_plot, y_prob_plot, callback),
        *plot_pr_curves(root, y_true_plot, y_prob_plot, callback),
        *plot_per_class_metrics(root, per_class, callback),
        *plot_class_distribution(root, callback),
        *plot_iteration_time(root, callback),
    ]
    specs = {
        "feature_importance_gain": ("gain", "Total gain", None),
        "feature_importance_split": ("split_count", "Split count", None),
        "permutation_importance": ("mean_decrease", "Macro-F1 decrease", "std_decrease"),
        "shap_feature_importance": ("mean_abs_shap", "Mean absolute SHAP contribution", None),
    }
    for name, (value, xlabel, error) in specs.items():
        generated.extend(plot_feature_importance(root, importance_tables[name], name, value, xlabel, callback, error))
    return generated
