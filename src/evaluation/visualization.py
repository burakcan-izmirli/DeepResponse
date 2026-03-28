"""Visualization for model evaluation."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _build_plot_prefix(
    *,
    split_type: str,
    trainable_layers: int,
    data_source: str,
    fold_idx: int | str | None,
) -> str:
    """Build file-name prefix for saved plots."""
    split_type = str(split_type)
    trainable_layers = int(trainable_layers)
    data_source = str(data_source)

    if fold_idx not in {"", None}:
        return f"{data_source}_{split_type}_stl{trainable_layers}_fold{fold_idx}"
    return f"{data_source}_{split_type}_stl{trainable_layers}"


def visualize_results(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metrics: dict[str, float],
    *,
    split_type: str = "unknown",
    trainable_layers: int = 0,
    data_source: str = "unknown",
    fold_idx: int | str | None = None,
    output_dir: Path | str | None = None,
) -> None:
    """Generate and save scatter/error distribution plots."""
    if y_true.size == 0 or y_pred.size == 0:
        logging.warning("Visualization skipped because y_true/y_pred is empty.")
        return

    prefix = _build_plot_prefix(
        split_type=split_type,
        trainable_layers=trainable_layers,
        data_source=data_source,
        fold_idx=fold_idx,
    )

    out_dir = Path(output_dir) if output_dir is not None else Path(".")
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.scatter(y_true, y_pred, alpha=0.5, s=20)
        min_val = min(float(y_true.min()), float(y_pred.min()))
        max_val = max(float(y_true.max()), float(y_pred.max()))
        ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2)
        ax.set_xlabel("True pIC50", fontsize=12)
        ax.set_ylabel("Predicted pIC50", fontsize=12)
        ax.set_title(
            f"Prediction vs True Values\nR2={metrics['r2']:.4f}, PCC={metrics['pcc']:.4f}",
            fontsize=14,
        )
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / f"{prefix}_scatter_plot.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 6))
        errors = y_pred - y_true
        ax.hist(errors, bins=50, edgecolor="black", alpha=0.7)
        ax.axvline(x=0, color="r", linestyle="--", linewidth=2)
        ax.set_xlabel("Prediction Error (Predicted - True)", fontsize=12)
        ax.set_ylabel("Frequency", fontsize=12)
        ax.set_title(
            f"Error Distribution\nMean Error={np.mean(errors):.4f}, Std={np.std(errors):.4f}",
            fontsize=14,
        )
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / f"{prefix}_histogram.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    except Exception as exc:
        logging.warning("Visualization failed for prefix '%s' (%s).", prefix, exc)
