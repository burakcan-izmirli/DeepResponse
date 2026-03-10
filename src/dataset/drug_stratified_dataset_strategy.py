"""Drug stratified dataset strategy."""

import logging
from typing import Any, Dict, Iterator, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from config.constants import TEST_SPLIT_RATIO, VALIDATION_SPLIT_RATIO
from src.dataset.base_dataset_strategy import BaseDatasetStrategy

MIN_SPLIT_SAMPLES = 20
MIN_SAMPLES_PER_DRUG = 2
MIN_TRAIN_VAL_IDENTITIES = 2
MIN_HARD_VALIDATION_IDENTITIES = 3


class DrugStratifiedDatasetStrategy(BaseDatasetStrategy):
    """Drug stratified dataset strategy."""

    def _filter_rare_identities(self, dataset_df: pd.DataFrame) -> pd.DataFrame:
        """Filter out drug identities with too few samples."""
        identity_counts = dataset_df["drug_identity"].value_counts()
        valid_identities = identity_counts[
            identity_counts >= MIN_SAMPLES_PER_DRUG
        ].index

        logging.info(
            "Drug identity filtering: total=%d, kept(min_samples=%d)=%d, removed=%d",
            len(identity_counts),
            MIN_SAMPLES_PER_DRUG,
            len(valid_identities),
            len(identity_counts) - len(valid_identities),
        )

        return dataset_df[dataset_df["drug_identity"].isin(valid_identities)].copy()

    def _split_train_val_identities(
        self,
        train_val_identities: list[str],
        filtered_df: pd.DataFrame,
        val_count: int,
        rng: np.random.Generator,
    ) -> Tuple[set[str], set[str]]:
        """Split train/validation identities with optional hardness-aware validation."""
        if len(train_val_identities) < MIN_TRAIN_VAL_IDENTITIES:
            raise ValueError(
                "Train/validation identity pool must contain at least 2 identities."
            )

        val_count = min(max(1, int(val_count)), len(train_val_identities) - 1)
        perm_ids = rng.permutation(train_val_identities)

        if self.hard_validation and (
            len(train_val_identities) >= MIN_HARD_VALIDATION_IDENTITIES
        ):
            target_val_rows = max(1, int(len(filtered_df) * VALIDATION_SPLIT_RATIO))
            (
                train_identities,
                val_identities,
                val_rows,
                hardness_scores,
            ) = self._select_hard_validation_identities(
                filtered_df,
                train_val_identities,
                val_count,
                target_val_rows,
                rng,
            )
            if not val_identities or not train_identities:
                logging.warning(
                    "Hard validation selection failed; falling back to random identity split."
                )
                val_identities = set(perm_ids[:val_count])
                train_identities = set(perm_ids[val_count:])
            else:
                avg_val_hardness = float(
                    np.mean(
                        [
                            hardness_scores.get(str(identity), 0.0)
                            for identity in val_identities
                        ]
                    )
                )
                avg_train_hardness = float(
                    np.mean(
                        [
                            hardness_scores.get(str(identity), 0.0)
                            for identity in train_identities
                        ]
                    )
                )
                logging.info(
                    "Hard validation enabled (drug_stratified): val identities=%d, val rows=%d, "
                    "avg_val_hardness=%.3f, avg_train_hardness=%.3f",
                    len(val_identities),
                    val_rows,
                    avg_val_hardness,
                    avg_train_hardness,
                )
            return train_identities, val_identities

        val_identities = set(perm_ids[:val_count])
        train_identities = set(perm_ids[val_count:])
        return train_identities, val_identities

    def _validate_no_identity_overlap(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> None:
        """Validate that drug identities are disjoint across splits."""
        train_ids = set(train_df["drug_identity"].unique())
        val_ids = set(val_df["drug_identity"].unique())
        test_ids = set(test_df["drug_identity"].unique())

        overlap_train_val = train_ids & val_ids
        overlap_train_test = train_ids & test_ids
        overlap_val_test = val_ids & test_ids

        if overlap_train_val:
            raise ValueError(
                f"Drug-identity overlap between train and val: {len(overlap_train_val)}"
            )
        if overlap_train_test:
            raise ValueError(
                f"Drug-identity overlap between train and test: {len(overlap_train_test)}"
            )
        if overlap_val_test:
            raise ValueError(
                f"Drug-identity overlap between val and test: {len(overlap_val_test)}"
            )

    def split_dataset(
        self, dataset_df: pd.DataFrame, random_state: Optional[int]
    ) -> Tuple[pd.DataFrame, ...]:
        """Split dataset by disjoint drug identities."""
        logging.info(
            "Splitting dataset into train, validation, and test sets by drug identities."
        )
        dataset_df = self._with_drug_identity(dataset_df)
        filtered_df = self._filter_rare_identities(dataset_df)

        if len(filtered_df) < 100:
            raise ValueError(
                f"Dataset too small after filtering: {len(filtered_df)} samples"
            )

        # Create disjoint identity splits for proper generalization testing
        # Each molecular identity appears in only one split (train, val, or test)
        unique_identities = filtered_df["drug_identity"].unique()
        rng = np.random.default_rng(random_state)
        shuffled_identities = rng.permutation(unique_identities)

        n_identities = len(shuffled_identities)
        test_split = max(1, int(n_identities * TEST_SPLIT_RATIO))
        val_split = max(1, int(n_identities * VALIDATION_SPLIT_RATIO))
        train_split = n_identities - test_split - val_split

        if train_split < 1:
            raise ValueError(
                f"Too few drug identities ({n_identities}) for stratified splitting"
            )

        test_identities = set(shuffled_identities[:test_split])
        train_val_identities = list(shuffled_identities[test_split:])
        train_identities, val_identities = self._split_train_val_identities(
            train_val_identities,
            filtered_df,
            val_split,
            rng,
        )

        train_df = filtered_df[
            filtered_df["drug_identity"].isin(train_identities)
        ].copy()
        val_df = filtered_df[filtered_df["drug_identity"].isin(val_identities)].copy()
        test_df = filtered_df[filtered_df["drug_identity"].isin(test_identities)].copy()
        self._validate_no_identity_overlap(train_df, val_df, test_df)

        if len(val_df) < MIN_SPLIT_SAMPLES:
            raise ValueError(f"Validation set too small: {len(val_df)} samples")
        if len(test_df) < MIN_SPLIT_SAMPLES:
            raise ValueError(f"Test set too small: {len(test_df)} samples")

        x_train = self._select_model_inputs(train_df)
        y_train = train_df[["pic50"]]
        x_val = self._select_model_inputs(val_df)
        y_val = val_df[["pic50"]]
        x_test = self._select_model_inputs(test_df)
        y_test = test_df[["pic50"]]
        train_identity_count = int(train_df["drug_identity"].nunique())
        val_identity_count = int(val_df["drug_identity"].nunique())
        test_identity_count = int(test_df["drug_identity"].nunique())

        logging.info(
            "Drug-stratified split summary - samples: Train=%d, Val=%d, Test=%d | identities: Train=%d, Val=%d, Test=%d",
            len(x_train),
            len(x_val),
            len(x_test),
            train_identity_count,
            val_identity_count,
            test_identity_count,
        )

        self._validate_r2_prerequisites(y_train, y_val, y_test)

        return x_train, x_val, x_test, y_train, y_val, y_test

    def _iter_drug_kfold_splits(
        self, dataset_df: pd.DataFrame, random_state: Optional[int]
    ) -> Iterator[Tuple[Any, ...]]:
        dataset_df = self._with_drug_identity(dataset_df)
        filtered_df = self._filter_rare_identities(dataset_df)

        unique_identities = np.asarray(filtered_df["drug_identity"].unique())
        requested = max(1, int(self.n_splits))
        n_folds = min(requested, len(unique_identities))
        if n_folds < 2:
            x_train, x_val, x_test, y_train, y_val, y_test = self.split_dataset(
                dataset_df, random_state
            )
            yield 1, 1, x_train, x_val, x_test, y_train, y_val, y_test
            return

        if n_folds != requested:
            logging.warning(
                "Requested n_splits=%d but only %d valid drug identities are available. Using n_splits=%d.",
                requested,
                len(unique_identities),
                n_folds,
            )

        relative_val_size = self._relative_val_size_for_kfold(n_folds)
        splitter = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        logging.info(
            "Native drug-stratified K-Fold enabled: n_splits=%d over %d valid drug identities.",
            n_folds,
            len(unique_identities),
        )

        for fold_idx, (train_val_idx, test_idx) in enumerate(
            splitter.split(unique_identities), start=1
        ):
            fold_seed = self._fold_seed(random_state, fold_idx)
            rng = np.random.default_rng(fold_seed)
            test_identities = set(unique_identities[test_idx].tolist())
            train_val_identities = list(unique_identities[train_val_idx].tolist())

            val_count = max(1, int(round(len(train_val_identities) * relative_val_size)))
            val_count = min(val_count, len(train_val_identities) - 1)
            train_identities, val_identities = self._split_train_val_identities(
                train_val_identities,
                filtered_df,
                val_count,
                rng,
            )

            train_df = filtered_df[
                filtered_df["drug_identity"].isin(train_identities)
            ].copy()
            val_df = filtered_df[
                filtered_df["drug_identity"].isin(val_identities)
            ].copy()
            test_df = filtered_df[
                filtered_df["drug_identity"].isin(test_identities)
            ].copy()
            self._validate_no_identity_overlap(train_df, val_df, test_df)

            if len(val_df) < MIN_SPLIT_SAMPLES:
                raise ValueError(
                    f"Fold {fold_idx}/{n_folds} validation set too small: {len(val_df)} samples"
                )
            if len(test_df) < MIN_SPLIT_SAMPLES:
                raise ValueError(
                    f"Fold {fold_idx}/{n_folds} test set too small: {len(test_df)} samples"
                )

            x_train = self._select_model_inputs(train_df)
            y_train = train_df[["pic50"]]
            x_val = self._select_model_inputs(val_df)
            y_val = val_df[["pic50"]]
            x_test = self._select_model_inputs(test_df)
            y_test = test_df[["pic50"]]
            train_identity_count = int(train_df["drug_identity"].nunique())
            val_identity_count = int(val_df["drug_identity"].nunique())
            test_identity_count = int(test_df["drug_identity"].nunique())
            self._validate_r2_prerequisites(y_train, y_val, y_test)

            logging.info(
                "Fold %d/%d split summary - samples: Train=%d, Val=%d, Test=%d | identities: Train=%d, Val=%d, Test=%d",
                fold_idx,
                n_folds,
                len(x_train),
                len(x_val),
                len(x_test),
                train_identity_count,
                val_identity_count,
                test_identity_count,
            )

            yield fold_idx, n_folds, x_train, x_val, x_test, y_train, y_val, y_test

    def prepare_dataset(
        self,
        dataset_dict: Dict[str, pd.DataFrame],
        _split_type: Optional[str],
        batch_size: int,
        random_state: Optional[int],
    ) -> Iterator[Tuple[Any, ...]]:
        """Prepare train/val/test loaders for drug-stratified folds."""
        dataset_df = dataset_dict["dataset"]

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
        ) in self._iter_drug_kfold_splits(dataset_df, random_state):
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
                "Ranking metadata (drug-stratified fold %d/%d): unique_cells=%d, train_pairs=%d",
                fold_idx,
                n_folds,
                dataset_df["cell_line_name"].nunique(),
                len(train_cell_ids),
            )

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
