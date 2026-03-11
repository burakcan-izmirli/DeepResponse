"""Random split dataset strategy."""

import logging
from typing import Any, Dict, Iterator, Optional, Tuple

import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from config.constants import TEST_SPLIT_RATIO, VALIDATION_SPLIT_RATIO
from src.dataset.base_dataset_strategy import BaseDatasetStrategy


class RandomSplitDatasetStrategy(BaseDatasetStrategy):
    """Random split dataset strategy."""

    def __init__(
        self,
        data_path: str,
        evaluation_data_path: Optional[str] = None,
        hard_validation: bool = True,
        ood_weighting: bool = True,
        residual_target: bool = False,
        n_splits: int = 1,
    ) -> None:
        super().__init__(
            data_path=data_path,
            evaluation_data_path=evaluation_data_path,
            hard_validation=hard_validation,
            ood_weighting=ood_weighting,
            residual_target=residual_target,
            n_splits=n_splits,
        )

    def _resolve_kfold_count(self, n_items: int) -> int:
        requested = max(1, int(self.n_splits))
        if requested <= 1:
            return 1
        if n_items < 2:
            raise ValueError(
                f"Cannot run K-Fold random split: only {n_items} items available."
            )
        if requested > n_items:
            logging.warning(
                "Requested n_splits=%d but only %d items are available. Using n_splits=%d.",
                requested,
                n_items,
                n_items,
            )
            return n_items
        return requested

    def split_dataset(
        self,
        dataset_to_split: pd.DataFrame,
        random_state: Optional[int],
    ) -> Tuple[pd.DataFrame, ...]:
        """Split dataset into train/validation/test sets."""
        logging.info(
            "Splitting dataset into train, validation, and test sets randomly."
        )

        dataset_with_identity = self._with_drug_identity(dataset_to_split)
        groups = self._pair_group_series(dataset_with_identity)

        x_all = self._select_model_inputs(dataset_with_identity)
        y_all = dataset_with_identity[["pic50"]]

        test_splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=TEST_SPLIT_RATIO,
            random_state=random_state,
        )
        train_val_idx, test_idx = next(
            test_splitter.split(x_all, y_all, groups=groups)
        )

        x_train_val = x_all.iloc[train_val_idx]
        y_train_val = y_all.iloc[train_val_idx]
        x_test = x_all.iloc[test_idx]
        y_test = y_all.iloc[test_idx]

        relative_val_size = VALIDATION_SPLIT_RATIO / (1.0 - TEST_SPLIT_RATIO)
        train_val_groups = groups.iloc[train_val_idx]
        val_splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=relative_val_size,
            random_state=random_state,
        )
        train_idx_local, val_idx_local = next(
            val_splitter.split(x_train_val, y_train_val, groups=train_val_groups)
        )
        x_train = x_train_val.iloc[train_idx_local]
        y_train = y_train_val.iloc[train_idx_local]
        x_val = x_train_val.iloc[val_idx_local]
        y_val = y_train_val.iloc[val_idx_local]

        logging.info(
            "Random split with pair grouping enabled: %d unique (drug identity, cell) pairs.",
            groups.nunique(),
        )

        logging.info(
            f"Split sizes: Train={len(x_train)}, Val={len(x_val)}, Test={len(x_test)}"
        )
        self._validate_r2_prerequisites(y_train, y_val, y_test)
        return x_train, x_val, x_test, y_train, y_val, y_test

    def _iter_kfold_splits(
        self, dataset_to_split: pd.DataFrame, random_state: Optional[int]
    ) -> Iterator[Tuple[Any, ...]]:
        dataset_with_identity = self._with_drug_identity(dataset_to_split)
        x_all = self._select_model_inputs(dataset_with_identity)
        y_all = dataset_with_identity[["pic50"]]
        groups = self._pair_group_series(dataset_with_identity)
        n_folds = self._resolve_kfold_count(groups.nunique())

        logging.info(
            "Random K-Fold enabled: n_splits=%d grouped by %d unique (drug identity, cell) pairs.",
            n_folds,
            groups.nunique(),
        )
        val_size = self._relative_val_size_for_kfold(n_folds)
        outer_splitter = GroupKFold(n_splits=n_folds)
        for fold_idx, (train_val_idx, test_idx) in enumerate(
            outer_splitter.split(x_all, y_all, groups=groups),
            start=1,
        ):
            x_train_val = x_all.iloc[train_val_idx]
            y_train_val = y_all.iloc[train_val_idx]
            x_test = x_all.iloc[test_idx]
            y_test = y_all.iloc[test_idx]
            train_val_groups = groups.iloc[train_val_idx]

            val_splitter = GroupShuffleSplit(
                n_splits=1,
                test_size=val_size,
                random_state=self._fold_seed(random_state, fold_idx),
            )
            train_idx_local, val_idx_local = next(
                val_splitter.split(x_train_val, y_train_val, groups=train_val_groups)
            )
            x_train = x_train_val.iloc[train_idx_local]
            y_train = y_train_val.iloc[train_idx_local]
            x_val = x_train_val.iloc[val_idx_local]
            y_val = y_train_val.iloc[val_idx_local]
            self._validate_r2_prerequisites(y_train, y_val, y_test)
            yield fold_idx, n_folds, x_train, x_val, x_test, y_train, y_val, y_test

    def _build_fold_output(
        self,
        dataset_df: pd.DataFrame,
        x_train: pd.DataFrame,
        x_val: pd.DataFrame,
        x_test: pd.DataFrame,
        y_train: pd.DataFrame,
        y_val: pd.DataFrame,
        y_test: pd.DataFrame,
        batch_size: int,
        fold_idx: Optional[int] = None,
        n_folds: Optional[int] = None,
        base_drug_smiles_lookup: Optional[pd.Series] = None,
        base_cell_features_lookup: Optional[pd.Series] = None,
    ) -> Tuple[Any, ...]:
        y_test_actual = y_test.copy()
        fold_metadata = {"fold_idx": int(fold_idx or 1), "n_splits": int(n_folds or 1)}

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
            "Ranking metadata (random split fold %d/%d): unique_cells=%d, train_pairs=%d",
            int(fold_idx or 1),
            int(n_folds or 1),
            dataset_df["cell_line_name"].nunique(),
            len(train_cell_ids),
        )

        train_df_full = dataset_df.loc[x_train.index]
        train_sample_weights = self._compute_train_sample_weights(train_df_full)

        cell_features_lookup = self._clone_cell_feature_lookup(base_cell_features_lookup)
        mean, std = self._compute_cell_feature_stats(
            cell_features_lookup, x_train["cell_line_name"]
        )
        cell_features_lookup = self._apply_cell_feature_transform(
            cell_features_lookup, mean, std
        )

        (
            loader_shapes,
            train_dataset,
            valid_dataset,
            test_dataset,
            y_test_filtered,
        ) = self._create_parallel_data_loaders(
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            x_test=x_test,
            y_test=y_test,
            batch_size=batch_size,
            train_cell_features=cell_features_lookup,
            train_drug_smiles=base_drug_smiles_lookup,
            eval_cell_features=cell_features_lookup,
            eval_drug_smiles=base_drug_smiles_lookup,
            y_test_actual=y_test_actual,
            train_sample_weights=train_sample_weights,
            train_group_ids=train_cell_ids,
            val_group_ids=val_cell_ids,
            test_group_ids=test_cell_ids,
        )

        return (
            loader_shapes,
            train_dataset,
            valid_dataset,
            test_dataset,
            y_test_filtered,
            fold_metadata,
        )

    def prepare_dataset(
        self,
        dataset_dict: Dict[str, pd.DataFrame],
        _split_type: Optional[str],
        batch_size: int,
        random_state: Optional[int],
    ) -> Iterator[Tuple[Any, ...]]:
        """Prepare dataset loaders for random split."""
        dataset_df = dataset_dict["dataset"]
        base_drug_smiles_lookup, base_cell_features_lookup = (
            self.create_drug_cell_dataset(dataset_df)
        )

        if int(self.n_splits) <= 1:
            x_train, x_val, x_test, y_train, y_val, y_test = self.split_dataset(
                dataset_df, random_state
            )
            fold_output = self._build_fold_output(
                dataset_df=dataset_df,
                x_train=x_train,
                x_val=x_val,
                x_test=x_test,
                y_train=y_train,
                y_val=y_val,
                y_test=y_test,
                batch_size=batch_size,
                fold_idx=1,
                n_folds=1,
                base_drug_smiles_lookup=base_drug_smiles_lookup,
                base_cell_features_lookup=base_cell_features_lookup,
            )
            yield fold_output
            return

        for (
            fold_idx,
            n_folds,
            x_train,
            x_val,
            x_test,
            y_train,
            y_val,
            y_test,
        ) in self._iter_kfold_splits(dataset_df, random_state):
            logging.info(
                "Fold %d/%d split sizes: Train=%d, Val=%d, Test=%d",
                fold_idx,
                n_folds,
                len(x_train),
                len(x_val),
                len(x_test),
            )
            yield self._build_fold_output(
                dataset_df=dataset_df,
                x_train=x_train,
                x_val=x_val,
                x_test=x_test,
                y_train=y_train,
                y_val=y_val,
                y_test=y_test,
                batch_size=batch_size,
                fold_idx=fold_idx,
                n_folds=n_folds,
                base_drug_smiles_lookup=base_drug_smiles_lookup,
                base_cell_features_lookup=base_cell_features_lookup,
            )
