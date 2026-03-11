"""Drug-cell stratified dataset strategy."""

import logging
from typing import Any, Dict, Iterator, Optional, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split

from config.constants import TEST_SPLIT_RATIO, VALIDATION_SPLIT_RATIO
from src.dataset.base_dataset_strategy import BaseDatasetStrategy

MIN_SPLIT_SAMPLES = 50
MIN_TRAIN_VAL_MULTIPLIER = 6
MIN_TRAIN_MULTIPLIER = 5
STRICT_TEST_CANDIDATE_FRACS = tuple(round(x, 3) for x in np.linspace(0.15, 0.45, 31))
STRICT_TEST_FALLBACK_FRAC = 0.30
STRICT_VAL_CANDIDATE_FRACS = tuple(round(x, 3) for x in np.linspace(0.10, 0.35, 26))
STRICT_VAL_FALLBACK_FRAC = 0.25


class DrugCellStratifiedDatasetStrategy(BaseDatasetStrategy):
    """Drug-cell stratified dataset strategy."""

    @staticmethod
    def _build_disjoint_masks(
        dataset_df: pd.DataFrame,
        train_drugs,
        holdout_drugs,
        train_cells,
        holdout_cells,
    ) -> Tuple[pd.Series, pd.Series]:
        holdout_mask = dataset_df["drug_identity"].isin(holdout_drugs) & dataset_df[
            "cell_line_name"
        ].isin(holdout_cells)
        train_mask = dataset_df["drug_identity"].isin(train_drugs) & dataset_df[
            "cell_line_name"
        ].isin(train_cells)
        return train_mask, holdout_mask

    @staticmethod
    def _to_str_set(values) -> Set[str]:
        return {str(value) for value in values}

    def _build_strict_test_holdout(
        self,
        dataset_df: pd.DataFrame,
        frac: float,
        train_drugs,
        test_drugs,
        train_cells,
        test_cells,
    ) -> Dict[str, Any]:
        train_val_mask, test_mask = self._build_disjoint_masks(
            dataset_df,
            train_drugs=train_drugs,
            holdout_drugs=test_drugs,
            train_cells=train_cells,
            holdout_cells=test_cells,
        )
        test_rows = int(test_mask.sum())
        train_val_rows = int(train_val_mask.sum())
        return {
            "frac": float(frac),
            "train_drugs": self._to_str_set(train_drugs),
            "test_drugs": self._to_str_set(test_drugs),
            "train_cells": self._to_str_set(train_cells),
            "test_cells": self._to_str_set(test_cells),
            "test_rows": test_rows,
            "train_val_rows": train_val_rows,
            "used_rows": int(test_rows + train_val_rows),
        }

    def _select_strict_test_holdout(
        self,
        dataset_df: pd.DataFrame,
        unique_drugs,
        unique_cells,
        random_state: Optional[int],
    ) -> Dict[str, Any]:
        target_test_rows = int(len(dataset_df) * TEST_SPLIT_RATIO)
        best = None
        cell_seed = self._fold_seed(random_state, 1)

        for frac in STRICT_TEST_CANDIDATE_FRACS:
            train_drugs, test_drugs = train_test_split(
                unique_drugs, test_size=frac, random_state=random_state
            )
            train_cells, test_cells = train_test_split(
                unique_cells, test_size=frac, random_state=cell_seed
            )
            candidate = self._build_strict_test_holdout(
                dataset_df,
                frac,
                train_drugs,
                test_drugs,
                train_cells,
                test_cells,
            )
            if candidate["test_rows"] >= target_test_rows and (
                best is None or candidate["used_rows"] > best["used_rows"]
            ):
                best = candidate

        if best is not None:
            return best

        fallback = self._build_strict_test_holdout(
            dataset_df,
            STRICT_TEST_FALLBACK_FRAC,
            *train_test_split(
                unique_drugs,
                test_size=STRICT_TEST_FALLBACK_FRAC,
                random_state=random_state,
            ),
            *train_test_split(
                unique_cells,
                test_size=STRICT_TEST_FALLBACK_FRAC,
                random_state=cell_seed,
            ),
        )
        logging.warning(
            "Could not reach strict %.1f%% test ratio; using fallback holdout fraction %.2f with %d strict-test rows.",
            TEST_SPLIT_RATIO * 100.0,
            STRICT_TEST_FALLBACK_FRAC,
            fallback["test_rows"],
        )
        return fallback

    def _select_strict_train_val_masks(
        self, train_val_df: pd.DataFrame, random_state: Optional[int]
    ) -> Dict[str, Any]:
        target_val_rows = int(len(train_val_df) * VALIDATION_SPLIT_RATIO)
        unique_drugs = train_val_df["drug_identity"].unique()
        unique_cells = train_val_df["cell_line_name"].unique()
        best = None
        cell_seed = self._fold_seed(random_state, 1)

        for frac in STRICT_VAL_CANDIDATE_FRACS:
            train_drugs, val_drugs = train_test_split(
                unique_drugs, test_size=frac, random_state=random_state
            )
            train_cells, val_cells = train_test_split(
                unique_cells, test_size=frac, random_state=cell_seed
            )
            train_mask, val_mask = self._build_disjoint_masks(
                train_val_df,
                train_drugs=train_drugs,
                holdout_drugs=val_drugs,
                train_cells=train_cells,
                holdout_cells=val_cells,
            )
            train_rows = int(train_mask.sum())
            val_rows = int(val_mask.sum())

            if val_rows >= MIN_SPLIT_SAMPLES and train_rows >= (
                MIN_SPLIT_SAMPLES * MIN_TRAIN_MULTIPLIER
            ):
                if best is None or abs(val_rows - target_val_rows) < abs(
                    best["val_rows"] - target_val_rows
                ):
                    best = {
                        "frac": float(frac),
                        "train_mask": train_mask,
                        "val_mask": val_mask,
                        "train_rows": train_rows,
                        "val_rows": val_rows,
                    }

        if best is not None:
            return best

        train_drugs, val_drugs = train_test_split(
            unique_drugs, test_size=STRICT_VAL_FALLBACK_FRAC, random_state=random_state
        )
        train_cells, val_cells = train_test_split(
            unique_cells, test_size=STRICT_VAL_FALLBACK_FRAC, random_state=cell_seed
        )
        train_mask, val_mask = self._build_disjoint_masks(
            train_val_df,
            train_drugs=train_drugs,
            holdout_drugs=val_drugs,
            train_cells=train_cells,
            holdout_cells=val_cells,
        )
        fallback = {
            "frac": STRICT_VAL_FALLBACK_FRAC,
            "train_mask": train_mask,
            "val_mask": val_mask,
            "train_rows": int(train_mask.sum()),
            "val_rows": int(val_mask.sum()),
        }
        logging.warning(
            "Strict disjoint validation fallback: frac=%.2f, train=%d, val=%d",
            STRICT_VAL_FALLBACK_FRAC,
            fallback["train_rows"],
            fallback["val_rows"],
        )
        return fallback

    def _build_xy_triplet(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, ...]:
        x_train = self._select_model_inputs(train_df)
        y_train = train_df[["pic50"]]
        x_val = self._select_model_inputs(val_df)
        y_val = val_df[["pic50"]]
        x_test = self._select_model_inputs(test_df)
        y_test = test_df[["pic50"]]
        return x_train, x_val, x_test, y_train, y_val, y_test

    def split_dataset(
        self, dataset_df: pd.DataFrame, random_state: Optional[int]
    ) -> Tuple[pd.DataFrame, ...]:
        """Create strict disjoint train/validation/test splits."""
        logging.info("Splitting dataset for drug-cell stratified split.")

        dataset_df = self._with_drug_identity(dataset_df)
        unique_drugs = dataset_df["drug_identity"].unique()
        unique_cells = dataset_df["cell_line_name"].unique()
        logging.info(
            "Total unique drug identities: %d, Total unique cells: %d",
            len(unique_drugs),
            len(unique_cells),
        )

        best = self._select_strict_test_holdout(
            dataset_df,
            unique_drugs,
            unique_cells,
            random_state,
        )

        logging.info(
            "Selected strict test holdout fraction %.3f -> strict test rows=%d, disjoint train/val pool=%d, total used=%d/%d",
            best["frac"],
            best["test_rows"],
            best["train_val_rows"],
            best["used_rows"],
            len(dataset_df),
        )

        test_df = dataset_df[
            dataset_df["drug_identity"].isin(best["test_drugs"])
            & dataset_df["cell_line_name"].isin(best["test_cells"])
        ].copy()

        train_val_df = dataset_df[
            dataset_df["drug_identity"].isin(best["train_drugs"])
            & dataset_df["cell_line_name"].isin(best["train_cells"])
        ].copy()

        if len(test_df) < MIN_SPLIT_SAMPLES:
            raise ValueError(
                f"Test set too small: {len(test_df)} samples (minimum: {MIN_SPLIT_SAMPLES})"
            )
        if len(train_val_df) < MIN_SPLIT_SAMPLES * MIN_TRAIN_VAL_MULTIPLIER:
            raise ValueError(
                f"Train/validation pool too small after strict holdout: {len(train_val_df)} samples"
            )

        split_seed = self._fold_seed(random_state, 2)
        train_df, val_df = self._strict_disjoint_train_val_split(
            train_val_df, split_seed
        )

        if len(val_df) < MIN_SPLIT_SAMPLES:
            raise ValueError(
                f"Validation set too small: {len(val_df)} samples (minimum: {MIN_SPLIT_SAMPLES})"
            )
        if len(train_df) < MIN_SPLIT_SAMPLES * MIN_TRAIN_MULTIPLIER:
            raise ValueError(
                f"Training set too small: {len(train_df)} samples (minimum: {MIN_SPLIT_SAMPLES * MIN_TRAIN_MULTIPLIER})"
            )

        self._validate_stratification_integrity(
            train_df,
            val_df,
            test_df,
            best["test_drugs"],
            best["test_cells"],
        )

        x_train, x_val, x_test, y_train, y_val, y_test = self._build_xy_triplet(
            train_df,
            val_df,
            test_df,
        )

        logging.info(
            "Final split sizes: Train=%d, Validation=%d, Test=%d",
            len(x_train),
            len(x_val),
            len(x_test),
        )

        total_samples = len(x_train) + len(x_val) + len(x_test)
        train_pct = len(x_train) / total_samples * 100
        val_pct = len(x_val) / total_samples * 100
        test_pct = len(x_test) / total_samples * 100

        logging.info(
            "Achieved ratios: Train=%.1f%%, Validation=%.1f%%, Test=%.1f%%",
            train_pct,
            val_pct,
            test_pct,
        )

        self._validate_r2_prerequisites(y_train, y_val, y_test)
        return x_train, x_val, x_test, y_train, y_val, y_test

    def _iter_drug_cell_kfold_splits(
        self, dataset_df: pd.DataFrame, random_state: Optional[int]
    ) -> Iterator[Tuple[Any, ...]]:
        dataset_df = self._with_drug_identity(dataset_df)
        unique_drugs = np.asarray(dataset_df["drug_identity"].unique())
        unique_cells = np.asarray(dataset_df["cell_line_name"].unique())
        requested = max(1, int(self.n_splits))

        n_folds = min(requested, len(unique_drugs), len(unique_cells))
        if n_folds < 2:
            x_train, x_val, x_test, y_train, y_val, y_test = self.split_dataset(
                dataset_df, random_state
            )
            yield 1, 1, x_train, x_val, x_test, y_train, y_val, y_test
            return

        if n_folds != requested:
            logging.warning(
                "Requested n_splits=%d but only %d drug folds and %d cell folds are possible. Using n_splits=%d.",
                requested,
                len(unique_drugs),
                len(unique_cells),
                n_folds,
            )

        drug_splitter = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        cell_seed = self._fold_seed(random_state, 1)
        cell_splitter = KFold(n_splits=n_folds, shuffle=True, random_state=cell_seed)
        logging.info(
            "Native drug-cell-stratified K-Fold enabled: n_splits=%d over %d drug identities and %d cells.",
            n_folds,
            len(unique_drugs),
            len(unique_cells),
        )

        fold_pairs = zip(drug_splitter.split(unique_drugs), cell_splitter.split(unique_cells))
        for fold_idx, (drug_fold, cell_fold) in enumerate(fold_pairs, start=1):
            train_drug_idx, test_drug_idx = drug_fold
            train_cell_idx, test_cell_idx = cell_fold

            train_drugs = set(unique_drugs[train_drug_idx].tolist())
            test_drugs = set(unique_drugs[test_drug_idx].tolist())
            train_cells = set(unique_cells[train_cell_idx].tolist())
            test_cells = set(unique_cells[test_cell_idx].tolist())

            test_df = dataset_df[
                dataset_df["drug_identity"].isin(test_drugs)
                & dataset_df["cell_line_name"].isin(test_cells)
            ].copy()
            train_val_df = dataset_df[
                dataset_df["drug_identity"].isin(train_drugs)
                & dataset_df["cell_line_name"].isin(train_cells)
            ].copy()

            if len(test_df) < MIN_SPLIT_SAMPLES:
                raise ValueError(
                    f"Fold {fold_idx}/{n_folds} test set too small: {len(test_df)} samples"
                )
            if len(train_val_df) < MIN_SPLIT_SAMPLES * MIN_TRAIN_VAL_MULTIPLIER:
                raise ValueError(
                    f"Fold {fold_idx}/{n_folds} train/validation pool too small: {len(train_val_df)} samples"
                )

            fold_seed = self._fold_seed(random_state, fold_idx)
            split_seed = None if fold_seed is None else int(fold_seed) + 2
            train_df, val_df = self._strict_disjoint_train_val_split(
                train_val_df, split_seed
            )

            if len(val_df) < MIN_SPLIT_SAMPLES:
                raise ValueError(
                    f"Fold {fold_idx}/{n_folds} validation set too small: {len(val_df)} samples"
                )
            if len(train_df) < MIN_SPLIT_SAMPLES * MIN_TRAIN_MULTIPLIER:
                raise ValueError(
                    f"Fold {fold_idx}/{n_folds} training set too small: {len(train_df)} samples"
                )

            self._validate_stratification_integrity(
                train_df,
                val_df,
                test_df,
                test_drugs,
                test_cells,
            )

            x_train, x_val, x_test, y_train, y_val, y_test = self._build_xy_triplet(
                train_df,
                val_df,
                test_df,
            )

            total_samples = len(x_train) + len(x_val) + len(x_test)
            train_pct = len(x_train) / total_samples * 100.0
            val_pct = len(x_val) / total_samples * 100.0
            test_pct = len(x_test) / total_samples * 100.0
            logging.info(
                "Fold %d/%d split sizes: Train=%d, Val=%d, Test=%d",
                fold_idx,
                n_folds,
                len(x_train),
                len(x_val),
                len(x_test),
            )
            logging.info(
                "Fold %d/%d achieved ratios: Train=%.1f%%, Val=%.1f%%, Test=%.1f%%",
                fold_idx,
                n_folds,
                train_pct,
                val_pct,
                test_pct,
            )

            self._validate_r2_prerequisites(y_train, y_val, y_test)
            yield fold_idx, n_folds, x_train, x_val, x_test, y_train, y_val, y_test

    def _validate_stratification_integrity(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        test_drug_identities: Set[str],
        test_cells: Set[str],
    ) -> None:
        """Validate strict test disjointness and pair-level split safety."""
        test_drug_identities = set(test_drug_identities)
        test_cells = set(test_cells)

        train_drug_ids = set(self._get_drug_identity_series(train_df))
        val_drug_ids = set(self._get_drug_identity_series(val_df))
        test_drug_ids = set(self._get_drug_identity_series(test_df))

        train_cells = set(train_df["cell_line_name"].astype(str))
        val_cells = set(val_df["cell_line_name"].astype(str))
        test_cells_actual = set(test_df["cell_line_name"].astype(str))

        if (train_drug_ids & test_drug_identities) or (
            val_drug_ids & test_drug_identities
        ):
            raise ValueError("Drug-identity leakage into strict test split detected.")
        if (train_cells & test_cells) or (val_cells & test_cells):
            raise ValueError("Cell-line leakage into strict test split detected.")

        if not test_drug_ids.issubset(test_drug_identities):
            raise ValueError(
                "Test set includes drug identities outside held-out identity set."
            )
        if not test_cells_actual.issubset(test_cells):
            raise ValueError("Test set includes cell lines outside held-out cell set.")

        # Pair-level leakage check between train and val.
        train_pairs = set(
            zip(
                self._get_drug_identity_series(train_df),
                train_df["cell_line_name"].astype(str),
            )
        )
        val_pairs = set(
            zip(
                self._get_drug_identity_series(val_df),
                val_df["cell_line_name"].astype(str),
            )
        )
        if train_pairs & val_pairs:
            raise ValueError("Pair leakage detected between train and validation sets.")

        logging.info("Stratification validation passed.")
        logging.info(
            "Strict test disjointness: held-out drug identities=%d, held-out cells=%d, strict test rows=%d",
            len(test_drug_identities),
            len(test_cells),
            len(test_df),
        )

    def prepare_dataset(
        self,
        dataset_dict: Dict[str, pd.DataFrame],
        _split_type: Optional[str],
        batch_size: int,
        random_state: Optional[int],
    ) -> Iterator[Tuple[Any, ...]]:
        """Prepare dataset iterator for drug-cell-stratified split(s)."""
        dataset_df = dataset_dict["dataset"]

        # Create a global lookup from the entire dataset for validation and test sets
        global_drug_smiles_lookup, global_cell_features_lookup = (
            self.create_drug_cell_dataset(dataset_df)
        )

        for (
            fold_idx,
            n_folds,
            x_train,
            x_val,
            x_test,
            y_train,
            y_val,
            y_test,
        ) in self._iter_drug_cell_kfold_splits(dataset_df, random_state):
            y_test_actual = y_test.copy()
            fold_metadata = {"fold_idx": fold_idx, "n_splits": n_folds}

            if self.residual_target:
                y_train, y_val, y_test, residual_metadata = (
                    self._apply_global_mean_residual_targets(
                        x_train,
                        y_train,
                        x_val,
                        y_val,
                        x_test,
                        y_test,
                    )
                )
                fold_metadata.update(residual_metadata)

            train_cell_ids, val_cell_ids, test_cell_ids = self._encode_cell_identities(
                dataset_df,
                x_train,
                x_val,
                x_test,
            )
            logging.info(
                "Ranking metadata (drug-cell stratified fold %d/%d): unique_cells=%d, train_pairs=%d",
                fold_idx,
                n_folds,
                dataset_df["cell_line_name"].nunique(),
                len(train_cell_ids),
            )

            # Create training-specific lookup tables to prevent data leakage
            train_df = dataset_df.loc[x_train.index]
            train_drug_smiles_lookup, train_cell_features_lookup = (
                self.create_drug_cell_dataset(train_df)
            )
            train_sample_weights = self._compute_train_sample_weights(train_df)

            target_col = y_test_actual.columns[0]

            mean, std = self._compute_cell_feature_stats(train_cell_features_lookup)
            train_cell_features_lookup = self._apply_cell_feature_transform(
                train_cell_features_lookup,
                mean,
                std,
            )
            fold_global_cell_features_lookup = self._apply_cell_feature_transform(
                self._clone_cell_feature_lookup(global_cell_features_lookup),
                mean,
                std,
            )

            (
                (drug_shape, cell_shape),
                train_dataset,
                valid_dataset,
                test_dataset,
                y_test_filtered_with_metadata,
            ) = self._create_parallel_data_loaders(
                x_train=x_train,
                y_train=y_train,
                x_val=x_val,
                y_val=y_val,
                x_test=x_test,
                y_test=y_test,
                batch_size=batch_size,
                train_cell_features=train_cell_features_lookup,
                train_drug_smiles=train_drug_smiles_lookup,
                eval_cell_features=fold_global_cell_features_lookup,
                eval_drug_smiles=global_drug_smiles_lookup,
                y_test_actual=y_test_actual,
                train_sample_weights=train_sample_weights,
                train_group_ids=train_cell_ids,
                val_group_ids=val_cell_ids,
                test_group_ids=test_cell_ids,
                val_cell_features=fold_global_cell_features_lookup,
                val_drug_smiles=global_drug_smiles_lookup,
            )
            y_test_filtered = y_test_filtered_with_metadata[[target_col]].copy()

            yield (
                drug_shape,
                cell_shape,
            ), train_dataset, valid_dataset, test_dataset, y_test_filtered, fold_metadata

    def _strict_disjoint_train_val_split(
        self, train_val_df: pd.DataFrame, random_state: Optional[int]
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split train/val so validation keeps unseen drug-cell combinations."""
        train_val_df = self._with_drug_identity(train_val_df)
        best = self._select_strict_train_val_masks(train_val_df, random_state)

        train_df = train_val_df[best["train_mask"]].copy()
        val_df = train_val_df[best["val_mask"]].copy()

        logging.info(
            "Strict disjoint train/val split: train=%d, val=%d (frac=%.3f)",
            len(train_df),
            len(val_df),
            best["frac"],
        )

        return train_df, val_df
