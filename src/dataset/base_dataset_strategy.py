"""Base dataset strategy."""
import concurrent.futures
import logging
import os
from abc import ABC, abstractmethod
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from config.constants import DEFAULT_NUM_WORKERS, VALIDATION_SPLIT_RATIO
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

CELL_FEATURES_FILENAME = "cell_line_features.npz"
GENE_AXIS_FILENAME = "gene_axis.csv"
PARALLEL_LOADER_WORKERS = 3


def collate_fn(batch):
    """Custom collate function for drug response data."""
    smiles = [item["smiles"] for item in batch]
    cell_features = torch.stack([item["cell_features"] for item in batch])
    responses = torch.stack([item["response"] for item in batch])

    result = {
        "smiles": smiles,
        "cell_features": cell_features,
        "response": responses,
    }

    if "sample_weight" in batch[0]:
        result["sample_weight"] = torch.stack([item["sample_weight"] for item in batch])

    if "group_id" in batch[0]:
        result["group_id"] = torch.stack([item["group_id"] for item in batch])

    if "drug_embedding" in batch[0]:
        result["drug_embedding"] = torch.stack(
            [item["drug_embedding"] for item in batch]
        )

    return result


class DrugCellResponseDataset(Dataset):
    """PyTorch Dataset for paired drug-cell response data."""

    def __init__(
        self,
        smiles: np.ndarray,
        cell_features: np.ndarray,
        targets: np.ndarray,
        sample_weights: np.ndarray = None,
        group_ids: np.ndarray = None,
        cached_drug_embeddings: Optional[Dict[str, torch.Tensor]] = None,
    ):
        self.smiles = smiles
        self.cell_features = cell_features
        self.targets = targets
        self.sample_weights = sample_weights
        self.group_ids = group_ids
        self.cached_drug_embeddings = cached_drug_embeddings or {}

    def __len__(self):
        # Required by Dataset/DataLoader to determine dataset size.
        return len(self.targets)

    def __getitem__(self, idx):
        smiles = self.smiles[idx]
        if isinstance(smiles, bytes):
            smiles = smiles.decode("utf-8")

        result = {
            "smiles": str(smiles),
            "cell_features": torch.tensor(self.cell_features[idx], dtype=torch.float32),
            "response": torch.tensor(self.targets[idx], dtype=torch.float32),
        }

        if self.sample_weights is not None:
            result["sample_weight"] = torch.tensor(
                self.sample_weights[idx], dtype=torch.float32
            )

        if self.group_ids is not None:
            result["group_id"] = torch.tensor(self.group_ids[idx], dtype=torch.long)

        cached_embedding = self.cached_drug_embeddings.get(str(smiles))
        if cached_embedding is not None:
            if torch.is_tensor(cached_embedding):
                result["drug_embedding"] = cached_embedding.detach().to(
                    dtype=torch.float32
                )
            else:
                result["drug_embedding"] = torch.tensor(
                    cached_embedding, dtype=torch.float32
                )

        return result


class BaseDatasetStrategy(ABC):
    def __init__(
        self,
        data_path,
        evaluation_data_path=None,
        hard_validation=True,
        ood_weighting=True,
        residual_target=False,
        n_splits=1,
    ):
        """Initialize base dataset strategy state."""
        self.data_path = data_path
        self.evaluation_data_path = evaluation_data_path
        self.n_splits = max(1, int(n_splits))
        self.hard_validation = hard_validation
        self.ood_weighting = ood_weighting
        self.residual_target = residual_target

    def _select_model_inputs(self, dataset_df: pd.DataFrame) -> pd.DataFrame:
        """Select model input columns for DataLoader construction."""
        return dataset_df[["drug_name", "cell_line_name"]].copy()

    def _attach_cell_features(self, dataset_df, base_dir):
        features_path = os.path.join(base_dir, CELL_FEATURES_FILENAME)
        features_npz = np.load(features_path, allow_pickle=True)
        features_lookup = {
            key: features_npz[key]
            for key in features_npz.files
            if not key.startswith("__")
        }

        dataset_df = dataset_df.copy()
        dataset_df["cell_line_features"] = dataset_df["cell_line_name"].map(
            features_lookup
        )
        return dataset_df

    def _read_dataset(self, path):
        dataset_df = pd.read_csv(path)
        if "cell_line_features" not in dataset_df.columns:
            dataset_df = self._attach_cell_features(dataset_df, os.path.dirname(path))
        if dataset_df.empty:
            raise ValueError(f"Dataset is empty after loading: {path}")
        return dataset_df

    def read_and_shuffle_dataset(
        self, random_state: Optional[int]
    ) -> Dict[str, Optional[pd.DataFrame]]:
        """Read and shuffle the main dataset."""
        try:
            dataset_raw = self._read_dataset(self.data_path)
        except FileNotFoundError:
            logging.error("Dataset file not found at: %s", self.data_path)
            raise

        dataset_raw = dataset_raw.sample(frac=1, random_state=random_state).reset_index(
            drop=True
        )
        logging.info("Dataset loaded with %d samples.", len(dataset_raw))
        return {"dataset": dataset_raw, "evaluation_dataset": None}

    def _load_gene_axis(self, data_path):
        axis_path = os.path.join(os.path.dirname(data_path), GENE_AXIS_FILENAME)
        if os.path.exists(axis_path):
            gene_df = pd.read_csv(axis_path)
            gene_col = gene_df.columns[0]
            genes = gene_df[gene_col].astype(str).tolist()
            return [g for g in genes if g]

        features_path = os.path.join(
            os.path.dirname(data_path), CELL_FEATURES_FILENAME
        )
        features_npz = np.load(features_path, allow_pickle=True)
        return [str(g) for g in features_npz["__gene_axis__"].tolist() if str(g)]

    @staticmethod
    def _map_feature_lookup(cell_features_lookup, transform):
        if hasattr(cell_features_lookup, "apply"):
            return cell_features_lookup.apply(transform)
        if isinstance(cell_features_lookup, dict):
            return {key: transform(value) for key, value in cell_features_lookup.items()}
        raise TypeError(
            f"Unsupported cell feature lookup type: {type(cell_features_lookup).__name__}"
        )

    def _align_cell_feature_lookup(
        self, cell_features_lookup, source_axis, target_axis
    ):
        index_map = {gene: i for i, gene in enumerate(source_axis)}
        indices = [index_map[gene] for gene in target_axis if gene in index_map]
        if not indices:
            raise ValueError(
                "Gene axis intersection is empty; cannot align cell features."
            )

        def transform(arr):
            if not isinstance(arr, np.ndarray):
                return arr
            if arr.ndim != 2:
                raise ValueError(
                    f"Cell feature array must be 2D, got shape {arr.shape}."
                )
            if arr.shape[0] == len(source_axis):
                return arr[indices, :]
            if arr.shape[1] == len(source_axis):
                return arr[:, indices]
            raise ValueError(
                f"Cell feature axis length mismatch for shape {arr.shape}; expected one dimension to equal "
                f"{len(source_axis)}."
            )

        return self._map_feature_lookup(cell_features_lookup, transform)

    @staticmethod
    def _normalize_smiles_for_identity(smiles_value):
        if pd.isna(smiles_value):
            return None
        text = str(smiles_value).strip()
        if not text or text.lower() in {"nan", "none"}:
            return None
        try:
            mol = Chem.MolFromSmiles(text)
        except Exception:
            mol = None
        if mol is not None:
            try:
                return Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
            except Exception:
                pass
        return text

    def _get_drug_identity_series(self, dataset_df):
        drug_name_identity = dataset_df["drug_name"].astype(str)
        if "smiles" not in dataset_df.columns:
            return drug_name_identity

        smiles_identity = dataset_df["smiles"].apply(
            self._normalize_smiles_for_identity
        )
        return smiles_identity.fillna(drug_name_identity).astype(str)

    def _with_drug_identity(self, dataset_df):
        out_df = dataset_df.copy()
        out_df["drug_identity"] = self._get_drug_identity_series(out_df)
        return out_df

    def _pair_group_series(self, dataset_df):
        with_identity = self._with_drug_identity(dataset_df)
        return (
            with_identity["drug_identity"].astype(str)
            + "||"
            + with_identity["cell_line_name"].astype(str)
        )

    @staticmethod
    def _smiles_kgram_tokens(smiles, k=3):
        text = BaseDatasetStrategy._normalize_smiles_for_identity(smiles)
        if text is None:
            return set()
        if len(text) <= k:
            return {text}
        return {text[idx : idx + k] for idx in range(len(text) - k + 1)}

    def _build_identity_smiles_lookup(self, dataset_df):
        if "smiles" not in dataset_df.columns:
            return {}
        with_identity = self._with_drug_identity(dataset_df)
        smiles_df = with_identity[["drug_identity", "smiles"]].copy()
        smiles_df["smiles_norm"] = smiles_df["smiles"].apply(
            self._normalize_smiles_for_identity
        )
        smiles_df = smiles_df.dropna(subset=["smiles_norm"])
        if smiles_df.empty:
            return {}

        counts = (
            smiles_df.groupby(["drug_identity", "smiles_norm"])
            .size()
            .reset_index(name="count")
            .sort_values(
                ["drug_identity", "count", "smiles_norm"], ascending=[True, False, True]
            )
        )
        top_smiles = counts.drop_duplicates(subset=["drug_identity"], keep="first")
        return dict(
            zip(
                top_smiles["drug_identity"].astype(str),
                top_smiles["smiles_norm"].astype(str),
            )
        )

    def _compute_identity_hardness_scores(
        self, dataset_df, candidate_identities, reference_identities
    ):
        candidate_ids = [str(identity) for identity in candidate_identities]
        reference_ids = [str(identity) for identity in reference_identities]
        if not candidate_ids:
            return {}

        smiles_lookup = self._build_identity_smiles_lookup(dataset_df)
        if not smiles_lookup:
            logging.warning(
                "SMILES lookup unavailable while computing identity hardness; using zero hardness scores."
            )
            return {identity: 0.0 for identity in candidate_ids}

        all_ids = sorted(set(candidate_ids) | set(reference_ids))
        fp_lookup = {}
        for identity in all_ids:
            smiles = smiles_lookup.get(identity)
            if not smiles:
                continue
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
            fp_lookup[identity] = AllChem.GetMorganFingerprintAsBitVect(
                mol, radius=2, nBits=2048
            )

        token_lookup = {
            identity: self._smiles_kgram_tokens(smiles_lookup.get(identity), k=3)
            for identity in all_ids
        }

        hardness_scores = {}
        for candidate in candidate_ids:
            ref_pool = [ref_id for ref_id in reference_ids if ref_id != candidate]
            if not ref_pool:
                hardness_scores[candidate] = 0.0
                continue

            similarities = []
            if candidate in fp_lookup:
                cand_fp = fp_lookup[candidate]
                similarities = [
                    float(DataStructs.TanimotoSimilarity(cand_fp, fp_lookup[ref_id]))
                    for ref_id in ref_pool
                    if ref_id in fp_lookup
                ]

            if not similarities:
                cand_tokens = token_lookup.get(candidate, set())
                if cand_tokens:
                    for ref_id in ref_pool:
                        ref_tokens = token_lookup.get(ref_id, set())
                        if not ref_tokens:
                            continue
                        union_size = len(cand_tokens | ref_tokens)
                        if union_size == 0:
                            continue
                        similarities.append(len(cand_tokens & ref_tokens) / union_size)

            max_similarity = max(similarities) if similarities else 0.0
            hardness_scores[candidate] = float(np.clip(1.0 - max_similarity, 0.0, 1.0))

        return hardness_scores

    def _select_hard_validation_identities(
        self,
        dataset_df: pd.DataFrame,
        train_val_identities: list[str],
        requested_val_count: int,
        target_val_rows: int,
        rng: np.random.Generator,
    ) -> tuple[set[str], set[str], int, Dict[str, float]]:
        """Select hard validation identities using similarity-aware ranking."""
        normalized_identities = [str(identity) for identity in train_val_identities]
        if len(normalized_identities) < 2:
            raise ValueError(
                "Train/validation identity pool must contain at least 2 identities."
            )

        seed_val_count = min(
            max(1, int(requested_val_count)),
            len(normalized_identities) - 1,
        )
        shuffled_ids = rng.permutation(normalized_identities)
        seed_train_ids = set(shuffled_ids[seed_val_count:])

        hardness_scores = self._compute_identity_hardness_scores(
            dataset_df,
            normalized_identities,
            list(seed_train_ids),
        )
        identity_row_counts = (
            dataset_df["drug_identity"].astype(str).value_counts().to_dict()
        )
        ranked_ids = sorted(
            normalized_identities,
            key=lambda identity: (
                hardness_scores.get(identity, 0.0),
                rng.random(),
            ),
            reverse=True,
        )

        val_identities: set[str] = set()
        val_rows = 0
        for identity in ranked_ids:
            if len(normalized_identities) - len(val_identities) <= 1:
                break
            val_identities.add(identity)
            val_rows += int(identity_row_counts.get(identity, 0))
            if val_rows >= target_val_rows and len(val_identities) >= seed_val_count:
                break

        train_identities = set(normalized_identities) - val_identities
        return train_identities, val_identities, val_rows, hardness_scores

    def _encode_cell_identities(
        self,
        dataset_df: pd.DataFrame,
        x_train: pd.DataFrame,
        x_val: pd.DataFrame,
        x_test: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Encode cell_line_name values as integer IDs for ranking metadata."""
        all_cells = pd.Index(dataset_df["cell_line_name"].astype(str).unique())
        cell_to_id = {cell_name: idx for idx, cell_name in enumerate(all_cells)}

        def encode(split_df: pd.DataFrame) -> np.ndarray:
            encoded = split_df["cell_line_name"].astype(str).map(cell_to_id)
            if encoded.isna().any():
                raise ValueError("Found cell lines not present in dataset cell index.")
            return encoded.to_numpy(dtype=np.int32)

        return encode(x_train), encode(x_val), encode(x_test)

    def _compute_train_sample_weights(
        self,
        train_df,
        rare_alpha=0.5,
        ood_beta=0.5,
        clip_min=0.5,
        clip_max=3.0,
    ):
        if not self.ood_weighting:
            return None

        with_identity = self._with_drug_identity(train_df)
        if with_identity.empty:
            return None

        counts = with_identity["drug_identity"].value_counts().astype(float)
        if counts.empty:
            return None

        median_count = float(np.median(counts.values))
        median_count = max(median_count, 1.0)

        rare_factor = (
            with_identity["drug_identity"]
            .map(
                lambda identity: (median_count / max(float(counts.loc[identity]), 1.0))
                ** rare_alpha
            )
            .astype(float)
            .to_numpy()
        )

        identities = counts.index.astype(str).tolist()
        hardness = self._compute_identity_hardness_scores(
            with_identity, identities, identities
        )
        hardness_values = np.array(
            [hardness.get(identity, 0.0) for identity in identities], dtype=np.float32
        )
        if hardness_values.size > 0 and float(np.ptp(hardness_values)) > 1e-8:
            hardness_norm = (hardness_values - hardness_values.min()) / np.ptp(
                hardness_values
            )
        else:
            hardness_norm = np.zeros_like(hardness_values)
        ood_factor_by_identity = {
            identity: 1.0 + (ood_beta * float(hardness_score))
            for identity, hardness_score in zip(identities, hardness_norm)
        }
        ood_factor = (
            with_identity["drug_identity"]
            .astype(str)
            .map(ood_factor_by_identity)
            .astype(float)
            .to_numpy()
        )

        sample_weights = rare_factor * ood_factor
        finite_mask = np.isfinite(sample_weights) & (sample_weights > 0)
        if not finite_mask.any():
            logging.warning(
                "Computed invalid sample weights; falling back to uniform weights."
            )
            return np.ones(len(with_identity), dtype=np.float32)

        mean_weight = float(sample_weights[finite_mask].mean())
        if mean_weight <= 0:
            mean_weight = 1.0
        sample_weights = sample_weights / mean_weight
        sample_weights = np.clip(sample_weights, clip_min, clip_max).astype(np.float32)

        logging.info(
            "Applied rare/OOD weighting on training data: identities=%d, alpha=%.2f, beta=%.2f, "
            "mean=%.3f, min=%.3f, max=%.3f",
            len(identities),
            rare_alpha,
            ood_beta,
            float(sample_weights.mean()),
            float(sample_weights.min()),
            float(sample_weights.max()),
        )
        return sample_weights

    def _apply_global_mean_residual_targets(
        self,
        x_train,
        y_train,
        x_val,
        y_val,
        x_test,
        y_test,
    ):
        """Build residual targets using the training-set global mean."""
        target_col = y_train.columns[0]

        y_train_numeric = pd.to_numeric(
            y_train[target_col], errors="coerce"
        ).reset_index(drop=True)
        if y_train_numeric.isna().all():
            raise ValueError(
                "Residual target cannot be computed: all training targets are NaN."
            )

        target_mean = float(y_train_numeric.mean())

        def _residual_frame(y_df):
            y_numeric = (
                pd.to_numeric(y_df[target_col], errors="coerce")
                .fillna(target_mean)
                .to_numpy(dtype=np.float32)
            )
            y_res = y_df.copy()
            y_res[target_col] = y_numeric - target_mean
            return y_res

        y_train_residual = _residual_frame(y_train)
        y_val_residual = _residual_frame(y_val)
        y_test_residual = _residual_frame(y_test)

        logging.info(
            "Residual target enabled: global_train_mean=%.4f",
            target_mean,
        )

        residual_metadata = {
            "residual_target": True,
            "target_mean": target_mean,
            "fallback_global_mean": target_mean,
        }
        return y_train_residual, y_val_residual, y_test_residual, residual_metadata

    def _validate_r2_prerequisites(
        self,
        y_train: pd.DataFrame,
        y_val: pd.DataFrame,
        y_test: pd.DataFrame,
    ) -> None:
        """Validate target distributions for stable R2 computation."""
        datasets = [("training", y_train), ("validation", y_val), ("test", y_test)]

        for name, y_data in datasets:
            y_values = y_data["pic50"].values

            if len(y_values) < 2:
                raise ValueError(f"{name} set has < 2 samples: cannot calculate R2")

            if np.var(y_values) == 0:
                logging.warning("%s set has zero variance in targets", name)

            if np.isnan(y_values).any():
                raise ValueError(f"{name} set contains NaN values")

            logging.info(
                "%s set: %d samples, mean=%.3f, std=%.3f, range=[%.3f, %.3f]",
                name,
                len(y_values),
                float(np.mean(y_values)),
                float(np.std(y_values)),
                float(np.min(y_values)),
                float(np.max(y_values)),
            )

    def _stack_cell_features(self, cell_features_lookup, cell_line_names=None):
        if hasattr(cell_features_lookup, "loc"):
            if cell_line_names is None:
                selected = cell_features_lookup
            else:
                name_index = pd.Index(cell_line_names)
                selected = cell_features_lookup.loc[
                    cell_features_lookup.index.intersection(name_index)
                ]
            arrays = [arr for arr in selected.values if isinstance(arr, np.ndarray)]
        elif isinstance(cell_features_lookup, dict):
            if cell_line_names is None:
                values = cell_features_lookup.values()
            else:
                selected_keys = pd.Index(cell_line_names).intersection(
                    pd.Index(cell_features_lookup.keys())
                )
                values = [cell_features_lookup[key] for key in selected_keys]
            arrays = [arr for arr in values if isinstance(arr, np.ndarray)]
        else:
            raise TypeError(
                f"Unsupported cell feature lookup type: {type(cell_features_lookup).__name__}"
            )
        if not arrays:
            raise ValueError("No cell line features available for normalization.")
        return np.stack(arrays, axis=0)

    def _compute_cell_feature_stats(self, cell_features_lookup, cell_line_names=None):
        stacked = self._stack_cell_features(
            cell_features_lookup, cell_line_names
        ).astype(np.float32, copy=False)

        # Warning-free NaN handling: avoid np.nanmean/np.nanstd on all-NaN slices.
        valid_mask = np.isfinite(stacked)
        valid_count = valid_mask.sum(axis=0).astype(np.float32)

        safe_values = np.where(valid_mask, stacked, 0.0).astype(np.float32, copy=False)
        sum_values = safe_values.sum(axis=0, dtype=np.float64)
        mean = np.divide(
            sum_values,
            valid_count,
            out=np.zeros_like(sum_values, dtype=np.float64),
            where=valid_count > 0,
        )

        centered = np.where(valid_mask, stacked - mean, 0.0).astype(
            np.float32, copy=False
        )
        sq_sum = np.square(centered, dtype=np.float64).sum(axis=0, dtype=np.float64)
        variance = np.divide(
            sq_sum,
            valid_count,
            out=np.zeros_like(sq_sum, dtype=np.float64),
            where=valid_count > 0,
        )
        variance = np.clip(variance, a_min=0.0, a_max=None)
        std = np.sqrt(variance)

        mean = np.nan_to_num(mean, nan=0.0, posinf=0.0, neginf=0.0).astype(
            np.float32, copy=False
        )
        std = np.nan_to_num(std, nan=1.0, posinf=1.0, neginf=1.0).astype(
            np.float32, copy=False
        )
        mean[valid_count == 0] = 0.0
        std[(valid_count <= 1) | (std < 1e-6)] = 1.0
        return mean, std

    def _apply_cell_feature_transform(self, cell_features_lookup, mean, std):
        def transform(arr):
            if not isinstance(arr, np.ndarray):
                return arr
            filled = np.where(np.isfinite(arr), arr, mean).astype(np.float32)
            scaled = (filled - mean) / std
            return np.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0).astype(
                np.float32
            )

        return self._map_feature_lookup(cell_features_lookup, transform)

    @staticmethod
    def _relative_val_size_for_kfold(n_splits: int) -> float:
        outer_test_ratio = 1.0 / float(max(2, n_splits))
        relative = VALIDATION_SPLIT_RATIO / max(1e-6, 1.0 - outer_test_ratio)
        return float(min(max(relative, 1e-3), 0.5))

    @staticmethod
    def _fold_seed(random_state: Optional[int], fold_idx: int) -> Optional[int]:
        if random_state is None:
            return None
        return int(random_state) + int(fold_idx)

    @staticmethod
    def _clone_cell_feature_lookup(
        cell_features_lookup: Union[pd.Series, Dict[str, np.ndarray]],
    ) -> Union[pd.Series, Dict[str, np.ndarray]]:
        if hasattr(cell_features_lookup, "apply"):
            return cell_features_lookup.apply(
                lambda value: (
                    np.array(value, copy=True)
                    if isinstance(value, np.ndarray)
                    else np.asarray(value).copy()
                )
            )
        return {
            key: (
                np.array(value, copy=True)
                if isinstance(value, np.ndarray)
                else np.asarray(value).copy()
            )
            for key, value in cell_features_lookup.items()
        }

    @abstractmethod
    def split_dataset(self, dataset, *args, **kwargs): ...

    @abstractmethod
    def prepare_dataset(self, dataset_dict, split_type, batch_size, random_state): ...

    def create_drug_cell_dataset(self, dataset_raw):
        drug_smiles_lookup = (
            dataset_raw[["drug_name", "smiles"]]
            .drop_duplicates(subset="drug_name")
            .set_index("drug_name")["smiles"]
        )

        cell_features_lookup = (
            dataset_raw[["cell_line_name", "cell_line_features"]]
            .drop_duplicates(subset="cell_line_name")
            .set_index("cell_line_name")["cell_line_features"]
        )
        return drug_smiles_lookup, cell_features_lookup

    def create_data_loader(
        self,
        x_data_df,
        y_data_df,
        batch_size,
        cell_features_lookup,
        drug_smiles_lookup,
        is_training=True,
        sample_weights=None,
        group_ids=None,
        return_filtered_data=False,
        original_y_data_df=None,
        num_workers=DEFAULT_NUM_WORKERS,
    ):
        """Create DataLoader from prepared split inputs."""
        logging.info(
            "Creating DataLoader. Training: %s, Samples: %d",
            is_training,
            len(x_data_df),
        )

        if len(x_data_df) != len(y_data_df):
            raise ValueError(
                f"Mismatch in lengths of x_data_df ({len(x_data_df)}) and y_data_df ({len(y_data_df)})."
            )

        def _reset_as_dataframe(values, default_col_name="target"):
            if isinstance(values, pd.DataFrame):
                return values.reset_index(drop=True)
            if isinstance(values, pd.Series):
                col_name = values.name if values.name else default_col_name
                return values.reset_index(drop=True).to_frame(name=col_name)

            arr = np.asarray(values)
            if arr.ndim == 0:
                arr = np.expand_dims(arr, axis=0)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            return pd.DataFrame(arr)

        # Prepare data using vectorized operations for efficiency
        x_data_df_reset = x_data_df.reset_index(drop=True)
        y_data_df_reset = _reset_as_dataframe(y_data_df, default_col_name="pic50")
        input_rows = len(x_data_df_reset)

        def _optional_aligned_series(values, field_name):
            if values is None:
                return None
            values_arr = np.asarray(values).reshape(-1)
            if len(values_arr) != len(x_data_df_reset):
                raise ValueError(
                    f"Mismatch in lengths of x_data_df ({len(x_data_df_reset)}) and {field_name} ({len(values_arr)})."
                )
            return pd.Series(values_arr)

        sample_weights_series = _optional_aligned_series(
            sample_weights, "sample_weights"
        )
        group_ids_series = _optional_aligned_series(group_ids, "group_ids")

        valid_indices = x_data_df_reset["cell_line_name"].isin(
            cell_features_lookup.index
        ) & x_data_df_reset["drug_name"].isin(drug_smiles_lookup.index)

        x_data_filtered = x_data_df_reset[valid_indices].reset_index(drop=True)
        y_data_filtered = y_data_df_reset[valid_indices].reset_index(drop=True)
        sample_weights_filtered = (
            sample_weights_series[valid_indices].to_numpy(dtype=np.float32)
            if sample_weights_series is not None
            else None
        )
        group_ids_filtered = (
            group_ids_series[valid_indices].to_numpy(dtype=np.int32)
            if group_ids_series is not None
            else None
        )

        filtered_out = input_rows - len(x_data_filtered)
        if filtered_out > 0:
            logging.warning(
                "Skipped %d entries due to missing cell features or drug SMILES.",
                filtered_out,
            )
        if filtered_out > 0 and not return_filtered_data and not is_training:
            logging.warning(
                "create_data_loader filtered out %d/%d rows while return_filtered_data=False. "
                "External labels may be misaligned for this split.",
                filtered_out,
                input_rows,
            )

        if x_data_filtered.empty:
            raise ValueError(
                "No valid data pairs found after filtering. Cannot proceed with an empty dataset."
            )

        cell_features = np.array(
            x_data_filtered["cell_line_name"].map(cell_features_lookup).tolist()
        )
        drug_smiles = np.array(
            x_data_filtered["drug_name"].map(drug_smiles_lookup).tolist()
        )
        targets = y_data_filtered.to_numpy(dtype=np.float32)

        # Ensure targets have the correct shape
        if targets.ndim == 1:
            targets = targets.reshape(-1, 1)

        # Create Dataset
        dataset = DrugCellResponseDataset(
            smiles=drug_smiles,
            cell_features=cell_features,
            targets=targets,
            sample_weights=sample_weights_filtered,
            group_ids=group_ids_filtered,
        )

        # Create DataLoader
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=is_training,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=torch.cuda.is_available(),
        )

        drug_shape = ()
        cell_shape = cell_features.shape[1:]

        if return_filtered_data:
            y_original_df = _reset_as_dataframe(
                original_y_data_df if original_y_data_df is not None else y_data_df,
                default_col_name="target",
            )
            y_original_filtered = y_original_df[valid_indices].reset_index(drop=True)
            return (
                drug_shape,
                cell_shape,
                dataloader,
                x_data_filtered,
                y_original_filtered,
            )

        return drug_shape, cell_shape, dataloader

    def _create_parallel_data_loaders(
        self,
        x_train,
        y_train,
        x_val,
        y_val,
        x_test,
        y_test,
        batch_size,
        train_cell_features,
        train_drug_smiles,
        eval_cell_features,
        eval_drug_smiles,
        y_test_actual,
        train_sample_weights=None,
        train_group_ids=None,
        val_group_ids=None,
        test_group_ids=None,
        val_cell_features=None,
        val_drug_smiles=None,
    ):
        """Create train/validation/test dataloaders in parallel."""
        val_cell_features = (
            train_cell_features if val_cell_features is None else val_cell_features
        )
        val_drug_smiles = train_drug_smiles if val_drug_smiles is None else val_drug_smiles

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=PARALLEL_LOADER_WORKERS
        ) as executor:
            future_train = executor.submit(
                self.create_data_loader,
                x_train,
                y_train,
                batch_size,
                train_cell_features,
                train_drug_smiles,
                is_training=True,
                sample_weights=train_sample_weights,
                group_ids=train_group_ids,
            )
            future_val = executor.submit(
                self.create_data_loader,
                x_val,
                y_val,
                batch_size,
                val_cell_features,
                val_drug_smiles,
                is_training=False,
                group_ids=val_group_ids,
            )
            future_test = executor.submit(
                self.create_data_loader,
                x_test,
                y_test,
                batch_size,
                eval_cell_features,
                eval_drug_smiles,
                is_training=False,
                group_ids=test_group_ids,
                return_filtered_data=True,
                original_y_data_df=y_test_actual,
            )

            drug_shape, cell_shape, train_dataset = future_train.result()
            _, _, valid_dataset = future_val.result()
            _, _, test_dataset, _, y_test_filtered = future_test.result()

        return (
            (drug_shape, cell_shape),
            train_dataset,
            valid_dataset,
            test_dataset,
            y_test_filtered,
        )
