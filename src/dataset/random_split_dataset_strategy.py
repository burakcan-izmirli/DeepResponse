"""Random split dataset strategy."""

import logging
from typing import Any, Dict, Iterator, Optional, Tuple, Union

import pandas as pd
from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,
    KFold,
    train_test_split,
)

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
        random_split_group_by_pair: bool = True,
    ) -> None:
        super().__init__(
            data_path=data_path,
            evaluation_data_path=evaluation_data_path,
            hard_validation=hard_validation,
            ood_weighting=ood_weighting,
            residual_target=residual_target,
            n_splits=n_splits,
        )
        self.random_split_group_by_pair = bool(random_split_group_by_pair)

    def read_and_shuffle_dataset(self, random_state: Optional[int]) -> Dict[str, Any]:
        """Read and shuffle dataset."""
        try:
            dataset_raw = self._read_dataset(self.data_path)
        except FileNotFoundError:
            logging.error(f"Dataset file not found at: {self.data_path}")
            raise

        dataset_raw = dataset_raw.sample(frac=1, random_state=random_state).reset_index(
            drop=True
        )
        logging.info("Dataset loaded with %d samples.", len(dataset_raw))
        return {"dataset": dataset_raw, "evaluation_dataset": None}

    def _resolve_kfold_count(self, n_groups: int) -> int:
        requested = max(1, int(self.n_splits))
        if requested <= 1:
            return 1
        if n_groups < 2:
            raise ValueError(
                f"Cannot run K-Fold random split: only {n_groups} groups available."
            )
        if requested > n_groups:
            logging.warning(
                "Requested n_splits=%d but only %d groups are available. Using n_splits=%d.",
                requested,
                n_groups,
                n_groups,
            )
            return n_groups
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

        if self.random_split_group_by_pair:
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

            n_groups = groups.nunique()
            logging.info(
                "Random split with pair grouping enabled: %d unique (drug identity, cell) pairs.",
                n_groups,
            )
        else:
            x_all = self._select_model_inputs(dataset_to_split)
            y_all = dataset_to_split[["pic50"]]

            x_train_val, x_test, y_train_val, y_test = train_test_split(
                x_all, y_all, test_size=TEST_SPLIT_RATIO, random_state=random_state
            )

            relative_val_size = VALIDATION_SPLIT_RATIO / (1.0 - TEST_SPLIT_RATIO)
            x_train, x_val, y_train, y_val = train_test_split(
                x_train_val,
                y_train_val,
                test_size=relative_val_size,
                random_state=random_state,
            )

        logging.info(
            f"Split sizes: Train={len(x_train)}, Val={len(x_val)}, Test={len(x_test)}"
        )
        return x_train, x_val, x_test, y_train, y_val, y_test

    def _iter_kfold_splits(
        self, dataset_to_split: pd.DataFrame, random_state: Optional[int]
    ) -> Iterator[Tuple[Any, ...]]:
        if self.random_split_group_by_pair:
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
                    val_splitter.split(
                        x_train_val, y_train_val, groups=train_val_groups
                    )
                )
                x_train = x_train_val.iloc[train_idx_local]
                y_train = y_train_val.iloc[train_idx_local]
                x_val = x_train_val.iloc[val_idx_local]
                y_val = y_train_val.iloc[val_idx_local]
                yield fold_idx, n_folds, x_train, x_val, x_test, y_train, y_val, y_test
            return

        x_all = self._select_model_inputs(dataset_to_split)
        y_all = dataset_to_split[["pic50"]]
        n_folds = self._resolve_kfold_count(len(x_all))
        logging.info(
            "Random K-Fold enabled: n_splits=%d over %d samples.",
            n_folds,
            len(x_all),
        )
        val_size = self._relative_val_size_for_kfold(n_folds)
        outer_splitter = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        for fold_idx, (train_val_idx, test_idx) in enumerate(
            outer_splitter.split(x_all), start=1
        ):
            x_train_val = x_all.iloc[train_val_idx]
            y_train_val = y_all.iloc[train_val_idx]
            x_test = x_all.iloc[test_idx]
            y_test = y_all.iloc[test_idx]

            x_train, x_val, y_train, y_val = train_test_split(
                x_train_val,
                y_train_val,
                test_size=val_size,
                random_state=self._fold_seed(random_state, fold_idx),
            )
            yield fold_idx, n_folds, x_train, x_val, x_test, y_train, y_val, y_test

    def _build_fold_output(
        self,
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
        )

        fold_metadata = {}
        if fold_idx is not None and n_folds is not None:
            fold_metadata = {"fold_idx": int(fold_idx), "n_splits": int(n_folds)}

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
    ) -> Union[Tuple[Any, ...], Iterator[Tuple[Any, ...]]]:
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
                x_train=x_train,
                x_val=x_val,
                x_test=x_test,
                y_train=y_train,
                y_val=y_val,
                y_test=y_test,
                batch_size=batch_size,
                base_drug_smiles_lookup=base_drug_smiles_lookup,
                base_cell_features_lookup=base_cell_features_lookup,
            )
            return fold_output[:5]

        def _fold_iterator():
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

        return _fold_iterator()
