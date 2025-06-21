import logging
import pandas as pd
from sklearn.model_selection import train_test_split

from helper.enum.dataset.split_ratio import SplitRatio
from src.dataset.base_dataset_strategy import BaseDatasetStrategy


class RandomSplitDatasetStrategy(BaseDatasetStrategy):
    """Random split dataset strategy."""

    def read_and_shuffle_dataset(self, random_state):
        """Reads and shuffles the dataset from a pickle file."""
        logging.info("Reading and shuffling dataset for random split...")
        try:
            dataset_raw = pd.read_pickle(self.data_path)
        except FileNotFoundError:
            logging.error(f"Dataset file not found at: {self.data_path}")
            raise

        dataset_raw = dataset_raw.sample(frac=1, random_state=random_state).reset_index(drop=True)
        logging.info(f"Dataset loaded with {len(dataset_raw)} samples.")
        return {'dataset': dataset_raw, 'evaluation_dataset': None}

    def create_splitter(self, dataset, random_state):
        """Not needed for simple random split."""
        ...

    def split_dataset(self, dataset_to_split, random_state):
        """Splits dataset into train, validation, and test sets."""
        logging.info("Splitting dataset into train, validation, and test sets randomly.")

        required_cols = ['drug_name', 'cell_line_name', 'pic50']
        if not all(col in dataset_to_split.columns for col in required_cols):
            raise ValueError(f"Dataset for splitting is missing one of the required columns: {required_cols}")

        X = dataset_to_split[['drug_name', 'cell_line_name']]
        y = dataset_to_split[['pic50']]

        x_train_val, x_test, y_train_val, y_test = train_test_split(
            X, y,
            test_size=SplitRatio.test_ratio.value,
            random_state=random_state
        )

        # Calculate validation set size relative to the training set
        relative_val_size = SplitRatio.validation_ratio.value / (1.0 - SplitRatio.test_ratio.value)

        x_train, x_val, y_train, y_val = train_test_split(
            x_train_val, y_train_val,
            test_size=relative_val_size,
            random_state=random_state
        )

        logging.info(f"Split sizes: Train={len(x_train)}, Val={len(x_val)}, Test={len(x_test)}")
        return x_train, x_val, x_test, y_train, y_val, y_test

    def prepare_dataset(self, dataset_dict, split_type, batch_size, random_state, learning_task_strategy):
        """
        Prepares dataset for random split.
        Returns a tuple: ((drug_input_shape, cell_input_shape), train_ds, valid_ds, test_ds, y_test_df)
        """
        dataset_df = dataset_dict['dataset']

        drug_smiles_lookup, cell_features_lookup = self.create_drug_and_conv_dataset(dataset_df)

        x_train, x_val, x_test, y_train, y_val, y_test = self.split_dataset(
            dataset_df, random_state
        )

        # Process targets
        y_train = learning_task_strategy.process_targets(y_train)
        y_val = learning_task_strategy.process_targets(y_val)
        y_test = learning_task_strategy.process_targets(y_test)

        drug_shape, cell_shape, train_dataset = self.tf_dataset_creator(
            x_train, y_train, batch_size, cell_features_lookup, drug_smiles_lookup, learning_task_strategy,
            is_training=True
        )

        _, _, valid_dataset = self.tf_dataset_creator(
            x_val, y_val, batch_size, cell_features_lookup, drug_smiles_lookup, learning_task_strategy,
            is_training=False
        )

        _, _, test_dataset = self.tf_dataset_creator(
            x_test, y_test, batch_size, cell_features_lookup, drug_smiles_lookup, learning_task_strategy,
            is_training=False
        )

        return (drug_shape, cell_shape), train_dataset, valid_dataset, test_dataset, y_test
