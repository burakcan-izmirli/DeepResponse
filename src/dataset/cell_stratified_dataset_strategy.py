""" Cell stratified dataset strategy """
import math
import pandas as pd
import numpy as np
import concurrent.futures
import logging
from sklearn.model_selection import GroupKFold, train_test_split

from helper.enum.dataset.n_split import NSplit
from helper.enum.dataset.split_ratio import SplitRatio
from src.dataset.base_dataset_strategy import BaseDatasetStrategy


class CellStratifiedDatasetStrategy(BaseDatasetStrategy):
    """ Cell stratified dataset strategy """

    def read_and_shuffle_dataset(self, random_state):
        """ Read and shuffle dataset """
        logging.info("Reading and shuffling dataset for cell-stratified split...")
        try:
            dataset_raw = pd.read_pickle(self.data_path)
        except FileNotFoundError:
            logging.error(f"Dataset file not found at: {self.data_path}")
            raise

        dataset_raw = dataset_raw.sample(frac=1, random_state=random_state).reset_index(drop=True)
        logging.info(f"Dataset loaded with {len(dataset_raw)} samples.")
        return {'dataset': dataset_raw, 'evaluation_dataset': None}

    def create_splitter(self, dataset):
        """ Creates a GroupKFold splitter based on cell lines """
        n_splits = NSplit.stratified.value
        logging.info(f"Creating GroupKFold splitter with {n_splits} folds, grouped by cell_line_name.")
        return GroupKFold(n_splits=n_splits)

    def split_dataset(self, train_val_df, random_state):
        """ Split a fold's training data into training and validation sets by cell line """
        n_splits = NSplit.stratified.value
        train_ratio = (n_splits - 1) / n_splits if n_splits > 1 else 1
        relative_val_size = SplitRatio.validation_ratio.value / train_ratio

        unique_cells = train_val_df['cell_line_name'].unique()

        if len(unique_cells) < 2:
            logging.warning("Fewer than 2 unique cell lines in training fold. Falling back to random split for validation.")
            X = train_val_df[['drug_name', 'cell_line_name']]
            y = train_val_df[['pic50']]
            return train_test_split(X, y, test_size=relative_val_size, random_state=random_state)

        train_cells, val_cells = train_test_split(unique_cells, test_size=relative_val_size, random_state=random_state)

        train_mask = train_val_df['cell_line_name'].isin(train_cells)
        val_mask = train_val_df['cell_line_name'].isin(val_cells)

        x_train = train_val_df[train_mask][['drug_name', 'cell_line_name']]
        y_train = train_val_df[train_mask][['pic50']]
        x_val = train_val_df[val_mask][['drug_name', 'cell_line_name']]
        y_val = train_val_df[val_mask][['pic50']]

        logging.info(f"Fold split sizes: Train={len(x_train)}, Val={len(x_val)}")
        return x_train, x_val, y_train, y_val

    def prepare_dataset(self, dataset_dict, split_type, batch_size, random_state, learning_task_strategy):
        """
        Prepare dataset iterator for cell-stratified cross-validation.
        Yields ((smiles_shape, cell_line_shape), train_ds, val_ds, test_ds, y_test_fold_actual) for each fold.
        """
        dataset_df = dataset_dict['dataset']
        drug_smiles_lookup, cell_features_lookup = self.create_drug_and_conv_dataset(dataset_df)

        required_cols = ['drug_name', 'cell_line_name', 'pic50']
        if not all(col in dataset_df.columns for col in required_cols):
            raise ValueError(f"Dataset is missing one of the required columns: {required_cols}")

        X = dataset_df[required_cols]
        groups = dataset_df['cell_line_name']
        splitter = self.create_splitter(dataset_df)

        for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(X, groups=groups)):
            logging.info(f"----- Preparing CV Fold {fold_idx + 1} -----")

            train_val_df = X.iloc[train_idx]
            x_test = X.iloc[test_idx][['drug_name', 'cell_line_name']]
            y_test = X.iloc[test_idx][['pic50']]

            x_train, x_val, y_train, y_val = self.split_dataset(train_val_df, random_state)

            y_train = learning_task_strategy.process_targets(y_train)
            y_val = learning_task_strategy.process_targets(y_val)
            y_test = learning_task_strategy.process_targets(y_test)

            drug_shape, cell_shape, train_dataset = self.tf_dataset_creator(
                x_train, y_train, batch_size, cell_features_lookup, drug_smiles_lookup, learning_task_strategy, is_training=True
            )
            _, _, valid_dataset = self.tf_dataset_creator(
                x_val, y_val, batch_size, cell_features_lookup, drug_smiles_lookup, learning_task_strategy, is_training=False
            )
            _, _, test_dataset = self.tf_dataset_creator(
                x_test, y_test, batch_size, cell_features_lookup, drug_smiles_lookup, learning_task_strategy, is_training=False
            )

            yield (drug_shape, cell_shape), train_dataset, valid_dataset, test_dataset, y_test