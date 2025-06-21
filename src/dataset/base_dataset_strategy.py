import logging
import numpy as np
import tensorflow as tf
from abc import ABC, abstractmethod


class BaseDatasetStrategy(ABC):
    def __init__(self, data_path, evaluation_data_path=None):
        self.data_path = data_path
        self.evaluation_data_path = evaluation_data_path

    @abstractmethod
    def split_dataset(self, dataset, *args, **kwargs):
        ...

    @abstractmethod
    def create_splitter(self, dataset, random_state):
        ...

    @abstractmethod
    def read_and_shuffle_dataset(self, random_state):
        ...

    @abstractmethod
    def prepare_dataset(self, dataset_dict, split_type, batch_size, random_state,
                        learning_task_strategy):
        ...

    def create_drug_and_conv_dataset(self, dataset_raw):
        if 'drug_name' not in dataset_raw.columns or 'smiles' not in dataset_raw.columns:
            raise ValueError("Dataset must contain 'drug_name' and 'smiles' columns.")
        drug_smiles_lookup = \
        dataset_raw[['drug_name', 'smiles']].drop_duplicates(subset='drug_name').set_index('drug_name')['smiles']

        if 'cell_line_name' not in dataset_raw.columns or 'cell_line_features' not in dataset_raw.columns:
            raise ValueError("Dataset must contain 'cell_line_name' and 'cell_line_features' columns.")
        cell_features_lookup = \
        dataset_raw[['cell_line_name', 'cell_line_features']].drop_duplicates(subset='cell_line_name').set_index(
            'cell_line_name')['cell_line_features']
        return drug_smiles_lookup, cell_features_lookup

    def tf_dataset_creator(self, x_data_df, y_data_df, batch_size,
                           cell_features_lookup,
                           drug_smiles_lookup,
                           learning_task_strategy,
                           is_training=True):

        logging.info(f"Creating TensorFlow dataset. Training: {is_training}, Samples: {len(x_data_df)}")

        if len(x_data_df) != len(y_data_df):
            raise ValueError(f"Mismatch in lengths of x_data_df ({len(x_data_df)}) and y_data_df ({len(y_data_df)}).")

        required_cols = ['cell_line_name', 'drug_name']
        if not all(col in x_data_df.columns for col in required_cols):
            raise ValueError(f"x_data_df missing required columns: {required_cols}")

        # Prepare data using vectorized operations for efficiency
        x_data_df_reset = x_data_df.reset_index(drop=True)
        y_data_df_reset = y_data_df.reset_index(drop=True)

        valid_indices = x_data_df_reset['cell_line_name'].isin(cell_features_lookup.index) & \
                        x_data_df_reset['drug_name'].isin(drug_smiles_lookup.index)

        x_data_filtered = x_data_df_reset[valid_indices]
        y_data_filtered = y_data_df_reset[valid_indices]

        num_skipped = len(x_data_df_reset) - len(x_data_filtered)
        if num_skipped > 0:
            logging.warning(f"Skipped {num_skipped} entries due to missing cell features or drug SMILES.")

        if x_data_filtered.empty:
            raise ValueError("No valid data pairs found after filtering. Cannot proceed with an empty dataset.")

        cell_features = np.array(x_data_filtered['cell_line_name'].map(cell_features_lookup).tolist())
        drug_smiles = np.array(x_data_filtered['drug_name'].map(drug_smiles_lookup).tolist())
        targets = y_data_filtered.values.astype(np.float32)

        # Create TensorFlow dataset
        dataset = tf.data.Dataset.from_tensor_slices((
            (drug_smiles, cell_features),
            targets
        ))

        if is_training:
            dataset = dataset.shuffle(buffer_size=len(x_data_filtered),
                                      seed=self.random_state if hasattr(self, 'random_state') else None)

        dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

        drug_shape = ()
        cell_shape = cell_features.shape[1:]

        return drug_shape, cell_shape, dataset
