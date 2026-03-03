"""CLI argument parser for DeepResponse"""

from __future__ import annotations

import argparse
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from typing import Any

from config.defaults import DefaultConfig
from config.constants import (
    ALLOWED_CHECKPOINT_METRICS,
    DATA_SOURCES,
    ENCODER_POOLING_CHOICES,
    FUSION_TYPES,
    MODALITY_DROPOUT_SCHEDULES,
    RANKING_GROUP_MODES,
    SPLIT_TYPES,
)


def _parse_bool(value: Any) -> bool:
    return str(value).lower() == "true"


def argument_parser() -> argparse.Namespace:
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        description="DeepResponse: Large Scale Prediction of Cancer Cell Line Drug Response",
        epilog="For more information, visit: [https://github.com/burakcan-izmirli/DeepResponse](https://github.com/burakcan-izmirli/DeepResponse)",
    )

    defaults = DefaultConfig()

    tracking_group = parser.add_argument_group("Experiment Tracking")
    dataset_group = parser.add_argument_group("Dataset & Splitting")
    training_group = parser.add_argument_group("Training Hyperparameters")
    model_group = parser.add_argument_group("Model Architecture")
    advanced_group = parser.add_argument_group("Advanced Techniques")

    tracking_group.add_argument(
        "-uc",
        "--use_comet",
        default=defaults.use_comet,
        type=_parse_bool,
        help="Whether to use Comet ML for experiment tracking",
    )

    dataset_group.add_argument(
        "-ds",
        "--data_source",
        default=defaults.data_source,
        type=str,
        choices=sorted(DATA_SOURCES),
        help="Primary data source for training",
    )
    dataset_group.add_argument(
        "-es",
        "--evaluation_source",
        default=defaults.evaluation_source,
        type=str,
        choices=sorted(DATA_SOURCES),
        help="Evaluation data source for cross-domain validation",
    )
    dataset_group.add_argument(
        "-st",
        "--split_type",
        default=defaults.split_type,
        type=str,
        choices=sorted(SPLIT_TYPES),
        help="Dataset splitting strategy",
    )
    dataset_group.add_argument(
        "--n_splits",
        default=defaults.n_splits,
        type=int,
        help="Number of folds used for split configuration (>= 1).",
    )

    training_group.add_argument(
        "-rs",
        "--random_state",
        "--seed",
        dest="random_state",
        default=defaults.random_state,
        type=int,
        help="Random seed for reproducibility (>= 0)",
    )
    training_group.add_argument(
        "-bs",
        "--batch_size",
        default=defaults.batch_size,
        type=int,
        help="Training batch size (> 0)",
    )
    training_group.add_argument(
        "-e",
        "--epochs",
        dest="epochs",
        default=defaults.epochs,
        type=int,
        help="Total number of training epochs (> 0)",
    )
    training_group.add_argument(
        "-lr",
        "--learning_rate",
        default=defaults.learning_rate,
        type=float,
        help="Optimizer learning rate (> 0)",
    )
    training_group.add_argument(
        "--weight_decay",
        default=defaults.weight_decay,
        type=float,
        help="Weight decay coefficient (>= 0)",
    )
    training_group.add_argument(
        "--patience",
        default=defaults.patience,
        type=int,
        help="Early stopping patience in epochs (> 0)",
    )
    training_group.add_argument(
        "--ranking_weight",
        default=defaults.ranking_weight,
        type=float,
        help="Weight of pairwise ranking auxiliary loss (>= 0)",
    )

    model_group.add_argument(
        "-stl",
        "--trainable_encoder_layers",
        default=defaults.trainable_encoder_layers,
        type=int,
        help=(
            "Number of trainable layers in the drug encoder "
            "(-1 for all, 0 for frozen, >0 for specific count)"
        ),
    )
    model_group.add_argument(
        "--encoder_pooling",
        default=defaults.encoder_pooling,
        type=str,
        choices=sorted(ENCODER_POOLING_CHOICES),
        help="Pooling strategy for encoder embeddings",
    )
    model_group.add_argument(
        "--cell_embed_dim",
        default=defaults.cell_embed_dim,
        type=int,
        help="Cell encoder embedding dimension",
    )
    model_group.add_argument(
        "--latent_dim",
        default=defaults.latent_dim,
        type=int,
        help="Latent dimension in factorized response head",
    )
    model_group.add_argument(
        "--rank_dim",
        default=defaults.rank_dim,
        type=int,
        help="Rank dimension in factorized bilinear interaction",
    )
    model_group.add_argument(
        "--hidden_dim",
        default=defaults.hidden_dim,
        type=int,
        help="Hidden dimension in response MLP",
    )
    model_group.add_argument(
        "--dropout",
        default=defaults.dropout,
        type=float,
        help="Dropout rate for model head and cell encoder",
    )
    model_group.add_argument(
        "--force_cell_blind",
        default=defaults.force_cell_blind,
        type=_parse_bool,
        help="Zero-out cell embeddings before fusion to run a cell-blind control experiment",
    )
    model_group.add_argument(
        "--fusion_type",
        default=defaults.fusion_type,
        type=str,
        choices=sorted(FUSION_TYPES),
        help="Fusion head type for combining drug and cell embeddings",
    )
    advanced_group.add_argument(
        "--modality_dropout_drug",
        default=defaults.modality_dropout_drug,
        type=float,
        help="Training-time dropout probability for drug embeddings",
    )
    advanced_group.add_argument(
        "--modality_dropout_cell",
        default=defaults.modality_dropout_cell,
        type=float,
        help="Training-time dropout probability for cell embeddings",
    )
    advanced_group.add_argument(
        "--modality_dropout_schedule",
        default=defaults.modality_dropout_schedule,
        type=str,
        choices=sorted(MODALITY_DROPOUT_SCHEDULES),
        help="Schedule for modality dropout probabilities over training",
    )
    advanced_group.add_argument(
        "--modality_dropout_final_scale",
        default=defaults.modality_dropout_final_scale,
        type=float,
        help="Final scaling factor for warmup-decay modality dropout schedule",
    )
    advanced_group.add_argument(
        "--unfreeze_epoch",
        default=defaults.unfreeze_epoch,
        type=int,
        help="Epoch at which to unfreeze top encoder layers (-1 to disable)",
    )
    advanced_group.add_argument(
        "--unfreeze_layers",
        default=defaults.unfreeze_layers,
        type=int,
        help="Number of top encoder layers to unfreeze at unfreeze_epoch",
    )
    advanced_group.add_argument(
        "--unfreeze_lr_factor",
        default=defaults.unfreeze_lr_factor,
        type=float,
        help="Learning rate factor applied to base LR after unfreezing (multiplied)",
    )

    advanced_group.add_argument(
        "--cache_datasets",
        default=defaults.cache_datasets,
        type=_parse_bool,
        help="Cache preprocessed datasets in memory for speed",
    )
    advanced_group.add_argument(
        "--checkpoint_metric",
        default=defaults.checkpoint_metric,
        type=str,
        choices=sorted(ALLOWED_CHECKPOINT_METRICS),
        help="Checkpoint selection metric for final evaluation model loading",
    )
    advanced_group.add_argument(
        "--hard_validation",
        default=defaults.hard_validation,
        type=_parse_bool,
        help="Use harder validation split for drug-based stratified strategies",
    )
    advanced_group.add_argument(
        "--ood_weighting",
        default=defaults.ood_weighting,
        type=_parse_bool,
        help="Enable rare-drug + OOD-aware sample weighting during training",
    )
    advanced_group.add_argument(
        "--residual_target",
        default=defaults.residual_target,
        type=_parse_bool,
        help="Train on residual target y-mean(y_train) and restore by +mean(y_train) at inference",
    )
    advanced_group.add_argument(
        "--ranking_group_mode",
        default=defaults.ranking_group_mode,
        type=str,
        choices=sorted(RANKING_GROUP_MODES),
        help="Grouping strategy for pairwise ranking regularization",
    )

    args = parser.parse_args()
    _validate_parsed_args(parser, args)
    return args


def _validate_parsed_args(parser: ArgumentParser, args: argparse.Namespace) -> None:
    if args.learning_rate <= 0:
        parser.error(f"learning_rate must be positive, got: {args.learning_rate}")
    if args.batch_size <= 0:
        parser.error(f"batch_size must be positive, got: {args.batch_size}")
    if args.epochs <= 0:
        parser.error(f"epochs must be positive, got: {args.epochs}")
    if args.random_state < 0:
        parser.error(f"random_state must be non-negative, got: {args.random_state}")
    if args.weight_decay < 0:
        parser.error(f"weight_decay must be non-negative, got: {args.weight_decay}")
    if args.patience <= 0:
        parser.error(f"patience must be positive, got: {args.patience}")
    if args.ranking_weight < 0:
        parser.error(f"ranking_weight must be non-negative, got: {args.ranking_weight}")
    if not (0.0 <= args.dropout < 1.0):
        parser.error(f"dropout must be in [0.0, 1.0), got: {args.dropout}")
    if not (0.0 <= args.modality_dropout_drug < 1.0):
        parser.error(
            "modality_dropout_drug must be in [0.0, 1.0), "
            f"got: {args.modality_dropout_drug}"
        )
    if not (0.0 <= args.modality_dropout_cell < 1.0):
        parser.error(
            "modality_dropout_cell must be in [0.0, 1.0), "
            f"got: {args.modality_dropout_cell}"
        )
    if not (0.0 < args.modality_dropout_final_scale <= 1.0):
        parser.error(
            "modality_dropout_final_scale must be in (0.0, 1.0], "
            f"got: {args.modality_dropout_final_scale}"
        )

    if args.n_splits <= 0:
        parser.error(f"n_splits must be positive, got: {args.n_splits}")
    if args.trainable_encoder_layers < -1:
        parser.error(
            "trainable_encoder_layers must be >= -1 "
            f"(got: {args.trainable_encoder_layers})"
        )

    if args.unfreeze_epoch >= 0 and args.unfreeze_epoch >= args.epochs:
        parser.error("unfreeze_epoch must be less than total epochs or -1 to disable.")
    if args.unfreeze_layers < 0:
        parser.error("unfreeze_layers must be >= 0")
    if args.unfreeze_lr_factor <= 0:
        parser.error("unfreeze_lr_factor must be > 0")

    if args.split_type == "cross_domain" and args.evaluation_source is None:
        parser.error("evaluation_source is required when split_type is 'cross_domain'")
    if args.evaluation_source == args.data_source and args.split_type == "cross_domain":
        parser.error(
            "evaluation_source must be different from data_source for cross_domain split"
        )

    if args.residual_target and args.split_type not in {
        "drug_stratified",
        "drug_cell_stratified",
    }:
        parser.error(
            "residual_target is currently supported only for "
            "drug_stratified and drug_cell_stratified."
        )

    print(f"✓ Argument validation passed on {args.data_source} data")
