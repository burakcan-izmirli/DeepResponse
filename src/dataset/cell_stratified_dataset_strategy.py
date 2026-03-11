"""Cell stratified dataset strategy."""

import logging
from typing import Any, Dict, Iterator, Optional, Tuple

import pandas as pd
from sklearn.model_selection import KFold, train_test_split

from config.constants import TEST_SPLIT_RATIO, VALIDATION_SPLIT_RATIO
from src.dataset.base_dataset_strategy import BaseDatasetStrategy

MIN_UNIQUE_CELLS_FOR_SPLIT = 3
MIN_TRAIN_VAL_CELLS = 2


class CellStratifiedDatasetStrategy(BaseDatasetStrategy):
    """Cell stratified dataset strategy."""

    @staticmethod
    def _build_cell_split_frames(
        dataset_df: pd.DataFrame,
        train_cells,
        val_cells,
        test_cells,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        train_df = dataset_df[dataset_df["cell_line_name"].isin(train_cells)]
        val_df = dataset_df[dataset_df["cell_line_name"].isin(val_cells)]
        test_df = dataset_df[dataset_df["cell_line_name"].isin(test_cells)]
        return train_df, val_df, test_df

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
        self,
        dataset_df: pd.DataFrame,
        random_state: Optional[int],
    ) -> Tuple[pd.DataFrame, ...]:
        """Split dataset into train/validation/test sets by cell line."""
        unique_cells = dataset_df["cell_line_name"].unique()

        logging.info(
            "Cell-stratified split input - total unique cell lines=%d",
            len(unique_cells),
        )
        if len(unique_cells) < MIN_UNIQUE_CELLS_FOR_SPLIT:
            raise ValueError(
                "Cell-stratified split requires at least %d unique cell lines (found %d)."
                % (MIN_UNIQUE_CELLS_FOR_SPLIT, len(unique_cells))
            )

        train_cells, test_cells = train_test_split(
            unique_cells,
            test_size=TEST_SPLIT_RATIO,
            random_state=random_state,
        )
        train_cells, val_cells = train_test_split(
            train_cells,
            test_size=VALIDATION_SPLIT_RATIO / (1 - TEST_SPLIT_RATIO),
            random_state=random_state,
        )

        train_df, val_df, test_df = self._build_cell_split_frames(
            dataset_df,
            train_cells,
            val_cells,
            test_cells,
        )
        x_train, x_val, x_test, y_train, y_val, y_test = self._build_xy_triplet(
            train_df,
            val_df,
            test_df,
        )

        logging.info(
            "Cell-stratified sample counts - Train=%d, Validation=%d, Test=%d",
            len(x_train),
            len(x_val),
            len(x_test),
        )
        logging.info(
            "Cell-stratified unique cell-line counts - Train=%d, Validation=%d, Test=%d",
            len(train_cells),
            len(val_cells),
            len(test_cells),
        )
        self._validate_r2_prerequisites(y_train, y_val, y_test)
        return x_train, x_val, x_test, y_train, y_val, y_test

    def _iter_cell_kfold_splits(
        self,
        dataset_df: pd.DataFrame,
        random_state: Optional[int],
    ) -> Iterator[Tuple[Any, ...]]:
        unique_cells = dataset_df["cell_line_name"].unique()
        requested = max(1, int(self.n_splits))
        n_folds = min(requested, len(unique_cells))
        if n_folds < 2:
            x_train, x_val, x_test, y_train, y_val, y_test = self.split_dataset(
                dataset_df, random_state
            )
            yield 1, 1, x_train, x_val, x_test, y_train, y_val, y_test
            return

        if n_folds != requested:
            logging.warning(
                "Requested n_splits=%d but only %d unique cells are available. Using n_splits=%d.",
                requested,
                len(unique_cells),
                n_folds,
            )

        relative_val_size = self._relative_val_size_for_kfold(n_folds)
        splitter = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        logging.info(
            "Native cell-stratified K-Fold enabled: n_splits=%d over %d unique cells.",
            n_folds,
            len(unique_cells),
        )

        for fold_idx, (train_val_cell_idx, test_cell_idx) in enumerate(
            splitter.split(unique_cells), start=1
        ):
            train_val_cells = unique_cells[train_val_cell_idx]
            test_cells = unique_cells[test_cell_idx]

            if len(train_val_cells) < MIN_TRAIN_VAL_CELLS:
                raise ValueError(
                    f"Fold {fold_idx}/{n_folds} has insufficient train/val cells ({len(train_val_cells)})."
                )

            train_cells, val_cells = train_test_split(
                train_val_cells,
                test_size=relative_val_size,
                random_state=self._fold_seed(random_state, fold_idx),
            )

            train_df, val_df, test_df = self._build_cell_split_frames(
                dataset_df,
                train_cells,
                val_cells,
                test_cells,
            )
            x_train, x_val, x_test, y_train, y_val, y_test = self._build_xy_triplet(
                train_df,
                val_df,
                test_df,
            )

            logging.info(
                "Cell-stratified fold %d/%d sample counts - Train=%d, Validation=%d, Test=%d",
                fold_idx,
                n_folds,
                len(x_train),
                len(x_val),
                len(x_test),
            )
            logging.info(
                "Cell-stratified fold %d/%d unique cell-line counts - Train=%d, Validation=%d, Test=%d",
                fold_idx,
                n_folds,
                len(train_cells),
                len(val_cells),
                len(test_cells),
            )

            self._validate_r2_prerequisites(y_train, y_val, y_test)
            yield fold_idx, n_folds, x_train, x_val, x_test, y_train, y_val, y_test

    def prepare_dataset(
        self,
        dataset_dict: Dict[str, pd.DataFrame],
        _split_type: Optional[str],
        batch_size: int,
        random_state: Optional[int],
    ) -> Iterator[Tuple[Any, ...]]:
        """Prepare train/val/test loaders for cell-stratified folds."""
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
        ) in self._iter_cell_kfold_splits(dataset_df, random_state):
            y_test_actual = y_test.copy()
            target_col = y_test_actual.columns[0]
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
                "Ranking metadata (cell-stratified fold %d/%d): unique_cells=%d, train_pairs=%d",
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
