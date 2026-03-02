"""Default config for the DeepResponse CLI."""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class DefaultConfig:
    # --- 1) Project and Runtime ---
    use_comet: bool = True
    use_amp: bool = True
    random_state: int = 42
    epochs: int = 100

    # --- 2) Data and Split ---
    data_source: str = "depmap"
    evaluation_source: Optional[str] = None
    split_type: str = "random"
    n_splits: int = 5

    # --- 3) Model Architecture ---
    # Baseline uses full encoder fine-tuning from step 0.
    trainable_encoder_layers: int = 12
    encoder_pooling: str = "mean"
    cell_embed_dim: int = 256
    latent_dim: int = 128
    rank_dim: int = 64
    hidden_dim: int = 1024
    dropout: float = 0.1
    force_cell_blind: bool = False
    fusion_type: str = "film_bilinear"
    modality_dropout_drug: float = 0.0
    modality_dropout_cell: float = 0.0
    modality_dropout_schedule: str = "warmup_decay"
    modality_dropout_final_scale: float = 0.25
    bounded_output: str = "none"
    bounded_output_tau: float = 1.0

    # --- 4) Training and Optimization ---
    batch_size: int = 64
    learning_rate: float = 5e-5
    weight_decay: float = 1e-3
    patience: int = 20
    ranking_weight: float = 0.0
    checkpoint_metric: str = "val_loss"
    cache_datasets: bool = False
    hard_validation: bool = True
    ood_weighting: bool = True
    residual_target: bool = False
    ranking_group_mode: str = "auto"

    # --- 5) Fine-Tuning ---
    # -1 disables staged unfreezing; with trainable_encoder_layers=12 this means
    # all configured layers are trainable from the beginning.
    unfreeze_epoch: int = -1
    unfreeze_layers: int = 12
    unfreeze_lr_factor: float = 0.1
