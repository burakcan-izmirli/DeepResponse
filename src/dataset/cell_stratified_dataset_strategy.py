"""Cell stratified dataset strategy."""

import logging
from typing import Any, Dict, Iterator, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split

from config.constants import TEST_SPLIT_RATIO, VALIDATION_SPLIT_RATIO
from src.dataset.base_dataset_strategy import BaseDatasetStrategy


class CellStratifiedDatasetStrategy(BaseDatasetStrategy):
    """Cell stratified dataset strategy."""

    def read_and_shuffle_dataset(self, random_state: Optional[int]) -> Dict[str, Any]:
        dataset_raw = self._read_dataset(self.data_path)

        dataset_raw = dataset_raw.sample(frac=1, random_state=random_state).reset_index(
            drop=True
        )
        logging.info("Dataset loaded with %d samples.", len(dataset_raw))
        return {"dataset": dataset_raw, "evaluation_dataset": None}

    def create_splitter(
        self, _dataset: pd.DataFrame, _random_state: Optional[int] = None
    ) -> None:
        return None

    def split_dataset(
        self,
        dataset_df: pd.DataFrame,
        random_state: Optional[int],
    ) -> Tuple[pd.DataFrame, ...]:
        unique_cells = dataset_df["cell_line_name"].unique()

        logging.info(
            "Cell-stratified split input - total unique cell lines=%d",
            len(unique_cells),
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

        train_df = dataset_df[dataset_df["cell_line_name"].isin(train_cells)]
        val_df = dataset_df[dataset_df["cell_line_name"].isin(val_cells)]
        test_df = dataset_df[dataset_df["cell_line_name"].isin(test_cells)]

        x_train = self._select_model_inputs(train_df)
        y_train = train_df[["pic50"]]
        x_val = self._select_model_inputs(val_df)
        y_val = val_df[["pic50"]]
        x_test = self._select_model_inputs(test_df)
        y_test = test_df[["pic50"]]

        logging.info(
            "Cell-stratified sample counts - Train=%d, Validation=%d, Test=%d",
            len(x_train),
            len(x_val),
            len(x_test),
        )
        logging.info(
            "Cell lines in sets - Train: %d, Val: %d, Test: %d",
            len(train_cells),
            len(val_cells),
            len(test_cells),
        )
        return x_train, x_val, x_test, y_train, y_val, y_test

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

            if len(train_val_cells) < 2:
                raise ValueError(
                    f"Fold {fold_idx}/{n_folds} has insufficient train/val cells ({len(train_val_cells)})."
                )

            train_cells, val_cells = train_test_split(
                train_val_cells,
                test_size=relative_val_size,
                random_state=self._fold_seed(random_state, fold_idx),
            )

            train_df = dataset_df[dataset_df["cell_line_name"].isin(train_cells)]
            val_df = dataset_df[dataset_df["cell_line_name"].isin(val_cells)]
            test_df = dataset_df[dataset_df["cell_line_name"].isin(test_cells)]

            x_train = self._select_model_inputs(train_df)
            y_train = train_df[["pic50"]]
            x_val = self._select_model_inputs(val_df)
            y_val = val_df[["pic50"]]
            x_test = self._select_model_inputs(test_df)
            y_test = test_df[["pic50"]]

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

            yield fold_idx, n_folds, x_train, x_val, x_test, y_train, y_val, y_test

    def prepare_dataset(
        self,
        dataset_dict: Dict[str, Any],
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

            train_df = dataset_df.loc[x_train.index]
            train_drug_smiles_lookup, train_cell_features_lookup = (
                self.create_drug_cell_dataset(train_df)
            )

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

            drug_shape, cell_shape, train_dataset = self.create_data_loader(
                x_train,
                y_train,
                batch_size,
                train_cell_features_lookup,
                train_drug_smiles_lookup,
                is_training=True,
            )
            _, _, valid_dataset = self.create_data_loader(
                x_val,
                y_val,
                batch_size,
                fold_global_cell_features_lookup,
                global_drug_smiles_lookup,
                is_training=False,
            )
            _, _, test_dataset, _, y_test_filtered = self.create_data_loader(
                x_test,
                y_test,
                batch_size,
                fold_global_cell_features_lookup,
                global_drug_smiles_lookup,
                is_training=False,
                return_filtered_data=True,
                original_y_data_df=y_test_actual,
            )

            yield (
                drug_shape,
                cell_shape,
            ), train_dataset, valid_dataset, test_dataset, y_test_filtered, {
                "fold_idx": fold_idx,
                "n_splits": n_folds,
            }
