"""Metric computation and evaluation reporting for model predictions."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)

from config.constants import BINARY_THRESHOLD
from src.evaluation.visualization import visualize_results


def compute_metrics(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    binary_threshold: float = BINARY_THRESHOLD,
) -> dict[str, float]:
    """Compute regression and binary classification metrics."""
    y_true = _convert_1d_numpy(y_true)
    y_pred = _convert_1d_numpy(y_pred)

    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError(
            f"y_true and y_pred length mismatch: {y_true.shape[0]} != {y_pred.shape[0]}"
        )

    valid_mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not valid_mask.all():
        y_true = y_true[valid_mask]
        y_pred = y_pred[valid_mask]

    _nan = float("nan")
    n_samples = y_true.shape[0]
    if n_samples == 0:
        return dict.fromkeys(
            ("mse", "rmse", "mae", "r2", "pcc", "pcc_p", "scc", "scc_p",
             "accuracy", "precision", "recall", "f1", "mcc"),
            _nan,
        )

    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))

    if n_samples > 2:
        pcc, pcc_p = pearsonr(y_true, y_pred)
        scc, scc_p = spearmanr(y_true, y_pred)
        pcc, pcc_p, scc, scc_p = float(pcc), float(pcc_p), float(scc), float(scc_p)
    else:
        pcc = pcc_p = scc = scc_p = _nan

    y_true_binary = (y_true >= binary_threshold).astype(int)
    y_pred_binary = (y_pred >= binary_threshold).astype(int)
    accuracy = float(accuracy_score(y_true_binary, y_pred_binary))
    precision = float(precision_score(y_true_binary, y_pred_binary, zero_division=0))
    recall = float(recall_score(y_true_binary, y_pred_binary, zero_division=0))
    f1 = float(f1_score(y_true_binary, y_pred_binary, zero_division=0))
    mcc = float(matthews_corrcoef(y_true_binary, y_pred_binary))

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "pcc": pcc,
        "pcc_p": pcc_p,
        "scc": scc,
        "scc_p": scc_p,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mcc": mcc,
    }


def evaluate_test_metrics(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    comet: object | None = None,
    *,
    split_type: str = "unknown",
    trainable_encoder_layers: int = 0,
    data_source: str = "unknown",
    fold_idx: int | str | None = None,
    output_dir: Path | str | None = None,
) -> dict[str, float]:
    """Evaluate predictions and optionally log/visualize metrics."""
    y_true_array = _convert_1d_numpy(y_true)
    y_pred_array = _convert_1d_numpy(y_pred)
    metrics = compute_metrics(y_true_array, y_pred_array)

    logging.info(
        "Test results: RMSE=%.4f, MAE=%.4f, R2=%.4f, PCC=%.4f, SCC=%.4f",
        metrics["rmse"], metrics["mae"], metrics["r2"], metrics["pcc"], metrics["scc"],
    )

    if comet is not None:
        comet.log_metrics(metrics)

    visualize_results(
        y_true_array,
        y_pred_array,
        metrics,
        split_type=split_type,
        trainable_layers=trainable_encoder_layers,
        data_source=data_source,
        fold_idx=fold_idx,
        output_dir=output_dir,
    )

    return metrics


def _convert_1d_numpy(values) -> np.ndarray:
    """Flatten supported inputs into a 1D numpy array."""
    if isinstance(values, (pd.DataFrame, pd.Series)):
        values = values.to_numpy()
    return np.asarray(values).reshape(-1)
