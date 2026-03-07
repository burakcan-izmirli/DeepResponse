"""Logging and timing utilities for the DeepResponse pipeline."""

from __future__ import annotations

import logging
import os
import sys
import time
from argparse import Namespace
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterator

from config.constants import DEFAULT_BOUNDED_OUTPUT_MODE, DIR_LOGS
from config.defaults import DefaultConfig

_CONSOLE_FORMAT = "[%(asctime)s] %(levelname)s - %(message)s"
_CONSOLE_DATE_FORMAT = "%H:%M:%S"
_FILE_FORMAT = "%(asctime)s | %(levelname)s | %(message)s"
_FILE_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(args: Namespace) -> None:
    """Initialize stdout/file logging with a unique run id."""
    log_dir = Path(DIR_LOGS)
    log_dir.mkdir(parents=True, exist_ok=True)

    run_stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_id = f"{run_stamp}_pid{os.getpid()}"

    data_source = getattr(args, "data_source", "unknown")
    split_type = getattr(args, "split_type", "unknown")
    evaluation_source = getattr(args, "evaluation_source", None)

    log_parts = [data_source]
    if evaluation_source:
        log_parts.append(f"to_{evaluation_source}")
    log_parts.extend([split_type, run_id])
    log_path = log_dir / f"{'_'.join(map(str, log_parts))}.log"

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(
        logging.Formatter(_CONSOLE_FORMAT, datefmt=_CONSOLE_DATE_FORMAT)
    )

    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(
        logging.Formatter(_FILE_FORMAT, datefmt=_FILE_DATE_FORMAT)
    )

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    logging.captureWarnings(True)
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("tensorflow").setLevel(logging.ERROR)

    logging.info("Logging to %s", log_path)


@contextmanager
def execution_timer() -> Iterator[None]:
    """Track total execution time for a run."""
    start_time = time.time()
    try:
        yield
    finally:
        duration = time.time() - start_time
        logging.info(
            "Total execution time: %.2f seconds (%.2f minutes)",
            duration,
            duration / 60.0,
        )


def log_configuration(args: Namespace) -> None:
    """Log effective CLI/default configuration values in a structured block."""
    defaults = DefaultConfig()

    learning_rate = getattr(args, "learning_rate", defaults.learning_rate)
    unfreeze_epoch = getattr(args, "unfreeze_epoch", defaults.unfreeze_epoch)
    unfreeze_lr_factor = getattr(
        args, "unfreeze_lr_factor", defaults.unfreeze_lr_factor
    )
    if unfreeze_epoch >= 0:
        effective_lr = f"{learning_rate} -> {learning_rate * unfreeze_lr_factor}"
    else:
        effective_lr = learning_rate

    requested_norm = getattr(args, "cell_feature_normalization", "auto")
    effective_norm = "zscore" if requested_norm == "auto" else requested_norm

    requested_residual = bool(
        getattr(args, "residual_target", defaults.residual_target)
    )
    residual_label = str(requested_residual)

    sections = {
        "Data": {
            "data_source": getattr(args, "data_source", defaults.data_source),
            "evaluation_source": getattr(
                args, "evaluation_source", defaults.evaluation_source
            )
            or "None",
            "split_type": getattr(args, "split_type", defaults.split_type),
        },
        "Training": {
            "learning_rate": learning_rate,
            "effective_learning_rate": effective_lr,
            "weight_decay": getattr(args, "weight_decay", defaults.weight_decay),
            "epochs": getattr(args, "epochs", defaults.epochs),
            "batch_size": getattr(args, "batch_size", defaults.batch_size),
            "random_state": getattr(args, "random_state", defaults.random_state),
            "n_splits": getattr(args, "n_splits", defaults.n_splits),
            "patience": getattr(args, "patience", defaults.patience),
            "mixed_precision": bool(getattr(args, "use_amp", defaults.use_amp)),
        },
        "Model": {
            "trainable_encoder_layers": getattr(
                args, "trainable_encoder_layers", defaults.trainable_encoder_layers
            ),
            "encoder_pooling": getattr(
                args, "encoder_pooling", defaults.encoder_pooling
            ),
            "cell_embed_dim": getattr(args, "cell_embed_dim", defaults.cell_embed_dim),
            "latent_dim": getattr(args, "latent_dim", defaults.latent_dim),
            "rank_dim": getattr(args, "rank_dim", defaults.rank_dim),
            "hidden_dim": getattr(args, "hidden_dim", defaults.hidden_dim),
            "dropout": getattr(args, "dropout", defaults.dropout),
            "force_cell_blind": getattr(
                args, "force_cell_blind", defaults.force_cell_blind
            ),
            "fusion_type": getattr(args, "fusion_type", defaults.fusion_type),
            "modality_dropout_drug": getattr(
                args, "modality_dropout_drug", defaults.modality_dropout_drug
            ),
            "modality_dropout_cell": getattr(
                args, "modality_dropout_cell", defaults.modality_dropout_cell
            ),
            "modality_dropout_schedule": getattr(
                args, "modality_dropout_schedule", defaults.modality_dropout_schedule
            ),
            "modality_dropout_final_scale": getattr(
                args,
                "modality_dropout_final_scale",
                defaults.modality_dropout_final_scale,
            ),
            "bounded_output": defaults.bounded_output,
            "bounded_output_mode": getattr(
                defaults, "bounded_output_mode", DEFAULT_BOUNDED_OUTPUT_MODE
            ),
            "bounded_output_tau": defaults.bounded_output_tau,
            "ranking_weight": getattr(args, "ranking_weight", defaults.ranking_weight),
            "ranking_group_mode": getattr(
                args, "ranking_group_mode", defaults.ranking_group_mode
            ),
            "unfreeze_epoch": unfreeze_epoch,
            "unfreeze_layers": getattr(
                args, "unfreeze_layers", defaults.unfreeze_layers
            ),
            "unfreeze_lr_factor": unfreeze_lr_factor,
        },
        "Pipeline": {
            "cache_datasets": getattr(args, "cache_datasets", defaults.cache_datasets),
            "cell_feature_normalization": (
                effective_norm
                if requested_norm == effective_norm
                else f"{requested_norm} -> {effective_norm}"
            ),
            "residual_target": residual_label,
            "drug_identity_for_split": "smiles->name fallback (fixed)",
            "random_split_grouping": "(drug identity, cell) pair-grouped (fixed)",
            "cross_domain_adaptation": "domain-wise zscore (fixed)",
        },
        "Logging": {
            "use_comet": getattr(args, "use_comet", defaults.use_comet),
        },
    }

    lines = ["==================== DeepResponse Configuration ===================="]
    for section_name, entries in sections.items():
        lines.append(f"[{section_name}]")
        key_width = max(len(key) for key in entries)
        for key, value in entries.items():
            lines.append(f"  {key:<{key_width}} : {value}")
    lines.append("====================================================================")
    config_string = "\n".join(lines)
    logging.info("\n" + config_string)
