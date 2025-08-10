""" Cell stratified dataset strategy """
import logging
import pandas as pd
from sklearn.model_selection import train_test_split

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
        """ Returns None as splitter is not used in single-fold strategy. """
        return None

    def split_dataset(self, dataset_df, random_state):
        """
        Splits the dataset into training, validation, and test sets based on cell lines for a single fold.
        """
        logging.info("Splitting dataset into train, validation, and test sets based on cell lines.")
        
        # Count samples per cell line - include all cell lines in the split
        cell_counts = dataset_df['cell_line_name'].value_counts()
        unique_cells = dataset_df['cell_line_name'].unique()
        
        logging.info(f"Total cells: {len(cell_counts)}")
        logging.info(f"Targeting 80/10/10 split ratios for train/val/test datasets")

        # Split all cell lines into train, validation, and test sets to achieve ~10% val and ~10% test
        train_cells, test_cells = train_test_split(
            unique_cells, test_size=SplitRatio.test_ratio.value, random_state=random_state
        )
        train_cells, val_cells = train_test_split(
            train_cells, test_size=SplitRatio.validation_ratio.value / (1 - SplitRatio.test_ratio.value),
            random_state=random_state
        )

        # Create dataframes based on cell line splits
        train_df = dataset_df[dataset_df['cell_line_name'].isin(train_cells)]
        val_df = dataset_df[dataset_df['cell_line_name'].isin(val_cells)]
        test_df = dataset_df[dataset_df['cell_line_name'].isin(test_cells)]

        x_train = train_df[['drug_name', 'cell_line_name']]
        y_train = train_df[['pic50']]
        x_val = val_df[['drug_name', 'cell_line_name']]
        y_val = val_df[['pic50']]
        x_test = test_df[['drug_name', 'cell_line_name']]
        y_test = test_df[['pic50']]

        logging.info(f"Split sizes: Train={len(x_train)}, Val={len(x_val)}, Test={len(x_test)}")
        logging.info(f"Cell lines in sets - Train: {len(train_cells)}, Val: {len(val_cells)}, Test: {len(test_cells)}")
        return x_train, x_val, x_test, y_train, y_val, y_test

    def prepare_dataset(self, dataset_dict, split_type, batch_size, random_state, learning_task_strategy):
        """
        Prepare dataset iterator for a single cell-stratified split.
        Yields ((smiles_shape, cell_line_shape), train_ds, val_ds, test_ds, y_test_fold_actual).
        """
        dataset_df = dataset_dict['dataset']

        required_cols = ['drug_name', 'cell_line_name', 'pic50']
        if not all(col in dataset_df.columns for col in required_cols):
            raise ValueError(f"Dataset is missing one of the required columns: {required_cols}")

        # Create a global lookup from the entire dataset for validation and test sets
        global_drug_smiles_lookup, global_cell_features_lookup = self.create_drug_and_conv_dataset(dataset_df)

        x_train, x_val, x_test, y_train, y_val, y_test = self.split_dataset(dataset_df, random_state)

        # Create training-specific lookup tables to prevent data leakage
        train_df = dataset_df.loc[x_train.index]
        train_drug_smiles_lookup, train_cell_features_lookup = self.create_drug_and_conv_dataset(train_df)

        y_train = learning_task_strategy.process_targets(y_train)
        y_val = learning_task_strategy.process_targets(y_val)
        y_test_processed = learning_task_strategy.process_targets(y_test)

        drug_shape, cell_shape, train_dataset = self.tf_dataset_creator(
            x_train, y_train, batch_size, train_cell_features_lookup, train_drug_smiles_lookup, learning_task_strategy, is_training=True
        )
        _, _, valid_dataset = self.tf_dataset_creator(
            x_val, y_val, batch_size, global_cell_features_lookup, global_drug_smiles_lookup, learning_task_strategy, is_training=False
        )
        _, _, test_dataset = self.tf_dataset_creator(
            x_test, y_test_processed, batch_size, global_cell_features_lookup, global_drug_smiles_lookup, learning_task_strategy, is_training=False
        )

        yield (drug_shape, cell_shape), train_dataset, valid_dataset, test_dataset, y_test