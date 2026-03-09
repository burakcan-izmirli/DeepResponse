"""Cross domain dataset strategy."""

from typing import Any, Dict, Optional, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from config.constants import VALIDATION_SPLIT_RATIO
from src.dataset.base_dataset_strategy import BaseDatasetStrategy


class CrossDomainDatasetStrategy(BaseDatasetStrategy):
    """Cross domain dataset strategy."""

    def read_and_shuffle_dataset(self, random_state: Optional[int]) -> Dict[str, Any]:
        """Read and shuffle dataset."""

        dataset_raw = self._read_dataset(self.data_path)
        dataset_raw = dataset_raw.sample(frac=1, random_state=random_state).reset_index(
            drop=True
        )

        evaluation_dataset_raw = self._read_dataset(self.evaluation_data_path)
        evaluation_dataset_raw = evaluation_dataset_raw.sample(
            frac=1, random_state=random_state
        ).reset_index(drop=True)

        return {"dataset": dataset_raw, "evaluation_dataset": evaluation_dataset_raw}

    def create_splitter(
        self, _dataset: pd.DataFrame, _random_state: Optional[int] = None
    ) -> None:
        """Unused for this strategy."""
        return None

    def split_dataset(
        self,
        dataset: pd.DataFrame,
        evaluation_dataset: pd.DataFrame,
        random_state: Optional[int],
    ) -> Tuple[pd.DataFrame, ...]:
        """Split source data into train/val and use target data as test."""

        x_train_df, x_val_df, y_train_df, y_val_df = train_test_split(
            self._select_model_inputs(dataset),
            dataset[["pic50"]],
            test_size=VALIDATION_SPLIT_RATIO,
            random_state=random_state,
        )

        x_test_df = self._select_model_inputs(evaluation_dataset)
        y_test_df = evaluation_dataset[["pic50"]]

        return x_train_df, x_val_df, x_test_df, y_train_df, y_val_df, y_test_df

    def prepare_dataset(
        self,
        dataset_dict: Dict[str, pd.DataFrame],
        _split_type: Optional[str],
        batch_size: int,
        random_state: Optional[int],
    ) -> Tuple[Any, ...]:
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

        allowed_cols = {"drug_name", "cell_line_name", "pic50", "drug_id", "smiles"}
        dataset = dataset[[col for col in dataset.columns if col in allowed_cols]]
        evaluation_dataset = evaluation_dataset[
            [col for col in evaluation_dataset.columns if col in allowed_cols]
        ]

        # Splitting dataset into train, validation, and test
        x_train, x_val, x_test, y_train, y_val, y_test = self.split_dataset(
            dataset, evaluation_dataset, random_state
        )
        y_test_actual = y_test.copy()

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

        return self._create_parallel_data_loaders(
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
        )
