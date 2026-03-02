"""Project-wide constants for options, splitting, training, and evaluation."""

from typing import Set

# --- 1. Argument Validations & Allowed Choices ---
DATA_SOURCES: Set[str] = {"depmap", "gdsc", "ccle"}
SPLIT_TYPES: Set[str] = {
    "random",
    "cell_stratified",
    "drug_stratified",
    "drug_cell_stratified",
    "cross_domain",
}
ENCODER_POOLING_CHOICES: Set[str] = {
    "mean",
    "cls",
    "max",
}
FUSION_TYPES: Set[str] = {
    "concat",
    "film_bilinear",
    "xattn_dcn_residual",
}
BOUNDED_OUTPUT_CHOICES: Set[str] = {
    "none",
    "tanh",
}
BOUNDED_OUTPUT_MODES: Set[str] = {
    "train_stats_fixed",
    "global",
}
MODALITY_DROPOUT_SCHEDULES: Set[str] = {
    "constant",
    "warmup_decay",
}
RANKING_GROUP_MODES: Set[str] = {
    "auto",
    "cell",
    "drug",
}
ALLOWED_CHECKPOINT_METRICS: Set[str] = {"auto", "val_loss", "val_r2"}

# --- 1b. Internal Fixed Defaults (not exposed in public CLI) ---
DEFAULT_BOUNDED_OUTPUT_MODE: str = "train_stats_fixed"
DEFAULT_BOUNDED_OUTPUT_CENTER: float = 0.0
DEFAULT_BOUNDED_OUTPUT_SCALE: float = 10.0
DEFAULT_BOUNDED_OUTPUT_STD_FACTOR: float = 3.0
DEFAULT_BOUNDED_OUTPUT_MIN_SCALE: float = 1.0
DEFAULT_LAYERWISE_LR_DECAY: float = 1.0
DEFAULT_LAYERWISE_LR_MIN_SCALE: float = 1.0

# --- 2. Dataset Split Defaults ---
BINARY_THRESHOLD: float = 6.0
TEST_SPLIT_RATIO: float = 0.10
VALIDATION_SPLIT_RATIO: float = 0.10
DEFAULT_NUM_WORKERS: int = 4

# --- 3. Training & Optimization ---
GRAD_CLIP_NORM: float = 1.0
EARLY_STOP_MIN_DELTA: float = 1e-5
SAMPLE_WEIGHT_EPS: float = 1e-8
COSINE_ETA_MIN_SCALE: float = 0.01
COSINE_ETA_MIN_FLOOR: float = 1e-7
ONECYCLE_PCT_START: float = 0.3
ONECYCLE_FINAL_DIV_FACTOR: float = 1e4
ONECYCLE_DIV_FACTOR_RANDOM: float = 25.0
ONECYCLE_DIV_FACTOR_STRATIFIED: float = 100.0
CACHE_EMBEDDING_BATCH_MIN: int = 16
CACHE_EMBEDDING_BATCH_MAX: int = 256

# --- 4. Paths & Artifacts ---
DIR_LOGS: str = "logs"
DIR_CHECKPOINTS: str = "checkpoints"
