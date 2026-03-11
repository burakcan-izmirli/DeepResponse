"""Cross-domain dataset strategy."""

import logging
from typing import Any, Dict, Iterator, Optional, Tuple

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from config.constants import VALIDATION_SPLIT_RATIO
from src.dataset.base_dataset_strategy import BaseDatasetStrategy


class CrossDomainDatasetStrategy(BaseDatasetStrategy):
    """Cross-domain dataset strategy."""

    def read_and_shuffle_dataset(self, random_state: Optional[int]) -> Dict[str, Any]:
        """Read and shuffle dataset."""
        dataset_dict = super().read_and_shuffle_dataset(random_state)

        evaluation_dataset_raw = self._read_dataset(self.evaluation_data_path)
        evaluation_dataset_raw = evaluation_dataset_raw.sample(
            frac=1, random_state=random_state
        ).reset_index(drop=True)
        logging.info(
            "Evaluation dataset loaded with %d samples.", len(evaluation_dataset_raw)
        )
        dataset_dict["evaluation_dataset"] = evaluation_dataset_raw
        return dataset_dict

    def split_dataset(
        self,
        dataset: pd.DataFrame,
        evaluation_dataset: pd.DataFrame,
        random_state: Optional[int],
    ) -> Tuple[pd.DataFrame, ...]:
        """Split source data into train/val and use target data as test."""

        dataset_with_identity = self._with_drug_identity(dataset)
        groups = self._pair_group_series(dataset_with_identity)
        x_all = self._select_model_inputs(dataset_with_identity)
        y_all = dataset_with_identity[["pic50"]]

        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=VALIDATION_SPLIT_RATIO,
            random_state=random_state,
        )
        train_idx, val_idx = next(splitter.split(x_all, y_all, groups=groups))
        x_train_df = x_all.iloc[train_idx]
        y_train_df = y_all.iloc[train_idx]
        x_val_df = x_all.iloc[val_idx]
        y_val_df = y_all.iloc[val_idx]

        x_test_df = self._select_model_inputs(evaluation_dataset)
        y_test_df = evaluation_dataset[["pic50"]]

        self._validate_r2_prerequisites(y_train_df, y_val_df, y_test_df)
        return x_train_df, x_val_df, x_test_df, y_train_df, y_val_df, y_test_df

    def prepare_dataset(
        self,
        dataset_dict: Dict[str, pd.DataFrame],
        _split_type: Optional[str],
        batch_size: int,
        random_state: Optional[int],
    ) -> Iterator[Tuple[Any, ...]]:
        """Prepare train/val/test loaders for cross-domain validation."""

        dataset, evaluation_dataset = (
            dataset_dict["dataset"],
            dataset_dict["evaluation_dataset"],
        )

        drug_smiles_lookup, cell_features_lookup = self.create_drug_cell_dataset(
            dataset
        )
        eval_drug_smiles_lookup, eval_cell_features_lookup = (
            self.create_drug_cell_dataset(evaluation_dataset)
        )

        train_axis = self._load_gene_axis(self.data_path)
        eval_axis = self._load_gene_axis(self.evaluation_data_path)
        eval_set = set(eval_axis)
        intersection = [g for g in train_axis if g in eval_set]
        if not intersection:
            raise ValueError(
                "No overlapping genes between training and evaluation axes."
            )
        cell_features_lookup = self._align_cell_feature_lookup(
            cell_features_lookup, train_axis, intersection
        )
        eval_cell_features_lookup = self._align_cell_feature_lookup(
            eval_cell_features_lookup, eval_axis, intersection
        )

        allowed_cols = {"drug_name", "cell_line_name", "pic50", "smiles"}
        dataset = dataset[[col for col in dataset.columns if col in allowed_cols]]
        evaluation_dataset = evaluation_dataset[
            [col for col in evaluation_dataset.columns if col in allowed_cols]
        ]

        # Splitting dataset into train, validation, and test
        x_train, x_val, x_test, y_train, y_val, y_test = self.split_dataset(
            dataset, evaluation_dataset, random_state
        )
        y_test_actual = y_test.copy()
        fold_metadata = {"fold_idx": 1, "n_splits": 1}

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

        combined_df = pd.concat([dataset, evaluation_dataset], ignore_index=True)
        train_cell_ids, val_cell_ids, test_cell_ids = self._encode_cell_identities(
            combined_df,
            x_train,
            x_val,
            x_test,
        )
        train_sample_weights = self._compute_train_sample_weights(dataset.loc[x_train.index])

        mean, std = self._compute_cell_feature_stats(
            cell_features_lookup, x_train["cell_line_name"]
        )
        cell_features_lookup = self._apply_cell_feature_transform(
            cell_features_lookup, mean, std
        )
        eval_cell_features_lookup = self._apply_cell_feature_transform(
            eval_cell_features_lookup,
            mean,
            std,
        )

        result = self._create_parallel_data_loaders(
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            x_test=x_test,
            y_test=y_test,
            batch_size=batch_size,
            train_cell_features=cell_features_lookup,
            train_drug_smiles=drug_smiles_lookup,
            eval_cell_features=eval_cell_features_lookup,
            eval_drug_smiles=eval_drug_smiles_lookup,
            y_test_actual=y_test_actual,
            train_sample_weights=train_sample_weights,
            train_group_ids=train_cell_ids,
            val_group_ids=val_cell_ids,
            test_group_ids=test_cell_ids,
        )
        yield (*result, fold_metadata)
