"""Strategy resolution and instantiation for the DeepResponse pipeline."""

from __future__ import annotations

from config.constants import (
    DEFAULT_BOUNDED_OUTPUT_CENTER,
    DEFAULT_BOUNDED_OUTPUT_MIN_SCALE,
    DEFAULT_BOUNDED_OUTPUT_MODE,
    DEFAULT_BOUNDED_OUTPUT_SCALE,
    DEFAULT_BOUNDED_OUTPUT_STD_FACTOR,
    DEFAULT_LAYERWISE_LR_DECAY,
    DEFAULT_LAYERWISE_LR_MIN_SCALE,
    PROJECT_ROOT,
)
from src.comet.skip_comet_strategy import SkipCometStrategy
from src.comet.use_comet_strategy import UseCometStrategy
from src.dataset.cell_stratified_dataset_strategy import CellStratifiedDatasetStrategy
from src.dataset.cross_domain_dataset_strategy import CrossDomainDatasetStrategy
from src.dataset.drug_cell_stratified_dataset_strategy import DrugCellStratifiedDatasetStrategy
from src.dataset.drug_stratified_dataset_strategy import DrugStratifiedDatasetStrategy
from src.dataset.random_split_dataset_strategy import RandomSplitDatasetStrategy
from src.training import RandomSplitTrainingStrategy, StratifiedSplitTrainingStrategy


class StrategyResolver:
    """Resolve and instantiate strategies based on parsed CLI arguments."""

    def __init__(self, args) -> None:
        self.args = args

    @property
    def use_comet(self):
        return self.args.use_comet

    @property
    def data_source(self):
        return self.args.data_source

    @property
    def evaluation_source(self):
        return self.args.evaluation_source

    @property
    def split_type(self):
        return self.args.split_type

    @property
    def random_state(self):
        return self.args.random_state

    @property
    def batch_size(self):
        return self.args.batch_size

    @property
    def epochs(self):
        return getattr(self.args, "epochs", getattr(self.args, "epoch", 100))

    @property
    def learning_rate(self):
        return self.args.learning_rate

    @property
    def trainable_encoder_layers(self):
        return getattr(self.args, "trainable_encoder_layers", 6)

    @property
    def encoder_pooling(self):
        return getattr(self.args, "encoder_pooling", "mean")

    @property
    def n_splits(self):
        return getattr(self.args, "n_splits", 5)

    @property
    def weight_decay(self):
        return getattr(self.args, "weight_decay", 1e-4)

    @property
    def unfreeze_epoch(self):
        return getattr(self.args, "unfreeze_epoch", -1)

    @property
    def unfreeze_layers(self):
        return getattr(self.args, "unfreeze_layers", 0)

    @property
    def unfreeze_lr_factor(self):
        return getattr(self.args, "unfreeze_lr_factor", 1.0)

    @property
    def layerwise_lr_decay(self):
        return float(getattr(self.args, "layerwise_lr_decay", DEFAULT_LAYERWISE_LR_DECAY))

    @property
    def layerwise_lr_min_scale(self):
        return float(getattr(self.args, "layerwise_lr_min_scale", DEFAULT_LAYERWISE_LR_MIN_SCALE))

    @property
    def patience(self):
        return getattr(self.args, "patience", 20)

    @property
    def use_amp(self):
        return getattr(self.args, "use_amp", True)

    @property
    def ranking_weight(self):
        return float(getattr(self.args, "ranking_weight", 0.0))

    @property
    def dropout(self):
        return getattr(self.args, "dropout", 0.1)

    @property
    def cell_embed_dim(self):
        return getattr(self.args, "cell_embed_dim", 256)

    @property
    def latent_dim(self):
        return getattr(self.args, "latent_dim", 128)

    @property
    def rank_dim(self):
        return getattr(self.args, "rank_dim", 64)

    @property
    def hidden_dim(self):
        return getattr(self.args, "hidden_dim", 1024)

    @property
    def force_cell_blind(self):
        return bool(getattr(self.args, "force_cell_blind", False))

    @property
    def fusion_type(self):
        return getattr(self.args, "fusion_type", "film_bilinear")

    @property
    def modality_dropout_drug(self):
        return float(getattr(self.args, "modality_dropout_drug", 0.0))

    @property
    def modality_dropout_cell(self):
        return float(getattr(self.args, "modality_dropout_cell", 0.0))

    @property
    def modality_dropout_schedule(self):
        return getattr(self.args, "modality_dropout_schedule", "warmup_decay")

    @property
    def modality_dropout_final_scale(self):
        return float(getattr(self.args, "modality_dropout_final_scale", 0.25))

    @property
    def bounded_output(self):
        return getattr(self.args, "bounded_output", "none")

    @property
    def bounded_output_mode(self):
        return getattr(self.args, "bounded_output_mode", DEFAULT_BOUNDED_OUTPUT_MODE)

    @property
    def bounded_output_center(self):
        return float(getattr(self.args, "bounded_output_center", DEFAULT_BOUNDED_OUTPUT_CENTER))

    @property
    def bounded_output_scale(self):
        return float(getattr(self.args, "bounded_output_scale", DEFAULT_BOUNDED_OUTPUT_SCALE))

    @property
    def bounded_output_tau(self):
        return float(getattr(self.args, "bounded_output_tau", 1.0))

    @property
    def bounded_output_std_factor(self):
        return float(getattr(self.args, "bounded_output_std_factor", DEFAULT_BOUNDED_OUTPUT_STD_FACTOR))

    @property
    def bounded_output_min_scale(self):
        return float(getattr(self.args, "bounded_output_min_scale", DEFAULT_BOUNDED_OUTPUT_MIN_SCALE))

    @property
    def ranking_group_mode(self):
        return getattr(self.args, "ranking_group_mode", "auto")

    @property
    def checkpoint_metric(self):
        return getattr(self.args, "checkpoint_metric", "auto")

    @property
    def residual_target(self):
        return bool(getattr(self.args, "residual_target", False))

    def get_effective_learning_rate(self):
        """Return learning rate string, showing the post-unfreeze rate if applicable."""
        lr = self.learning_rate
        if (
            getattr(self.args, "unfreeze_epoch", -1) >= 0
            and getattr(self.args, "unfreeze_layers", 0) > 0
            and getattr(self.args, "unfreeze_lr_factor", 1.0) not in (None, 1.0)
        ):
            return f"{lr} -> {lr * self.args.unfreeze_lr_factor}"
        return lr

    def should_use_ranking_regularization(self) -> bool:
        """Return True if ranking regularization should be applied for this run."""
        return (
            self.split_type in {"cell_stratified", "drug_stratified", "drug_cell_stratified"}
            and self.ranking_weight > 0
        )

    def get_comet_strategy(self):
        """Return the appropriate Comet logging strategy."""
        return UseCometStrategy() if self.use_comet else SkipCometStrategy()

    def get_dataset_path(self) -> str:
        """Return the primary dataset CSV path for the configured data source."""
        return str(
            PROJECT_ROOT / "dataset_creator" / self.data_source / "processed" / "drug_response_features.csv"
        )

    def get_evaluation_dataset_path(self) -> str:
        """Return the evaluation dataset CSV path for cross-domain runs."""
        if self.evaluation_source is None:
            raise ValueError("evaluation_source must be provided for cross_domain split type.")
        return str(
            PROJECT_ROOT / "dataset_creator" / self.evaluation_source / "processed" / "drug_response_features.csv"
        )

    def get_split_strategy(self) -> dict:
        """Instantiate and return the dataset and training strategy pair."""
        split_type = self.split_type
        hard_validation = getattr(self.args, "hard_validation", True)
        ood_weighting = getattr(self.args, "ood_weighting", True)
        residual_target = self.residual_target
        dataset_path = self.get_dataset_path()

        dataset_kwargs = dict(
            n_splits=self.n_splits,
            hard_validation=hard_validation,
            ood_weighting=ood_weighting,
            residual_target=residual_target,
        )

        if split_type == "random":
            return {
                "dataset": RandomSplitDatasetStrategy(dataset_path, **dataset_kwargs),
                "training": RandomSplitTrainingStrategy(),
            }
        if split_type == "cell_stratified":
            return {
                "dataset": CellStratifiedDatasetStrategy(dataset_path, **dataset_kwargs),
                "training": StratifiedSplitTrainingStrategy(),
            }
        if split_type == "drug_stratified":
            return {
                "dataset": DrugStratifiedDatasetStrategy(dataset_path, **dataset_kwargs),
                "training": StratifiedSplitTrainingStrategy(),
            }
        if split_type == "drug_cell_stratified":
            return {
                "dataset": DrugCellStratifiedDatasetStrategy(dataset_path, **dataset_kwargs),
                "training": StratifiedSplitTrainingStrategy(),
            }
        if split_type == "cross_domain":
            return {
                "dataset": CrossDomainDatasetStrategy(
                    dataset_path,
                    self.get_evaluation_dataset_path(),
                    **dataset_kwargs,
                ),
                "training": RandomSplitTrainingStrategy(),
            }
        raise ValueError(f"Unknown split_type: {split_type!r}")
