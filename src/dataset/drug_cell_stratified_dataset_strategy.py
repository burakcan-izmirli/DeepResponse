""" Drug cell stratified dataset strategy """
import logging
import pandas as pd
import math
import concurrent.futures
import tensorflow as tf
from sklearn.model_selection import train_test_split

from helper.enum.dataset.n_split import NSplit
from helper.enum.dataset.split_ratio import SplitRatio
from src.dataset.base_dataset_strategy import BaseDatasetStrategy

class DrugCellStratifiedDatasetStrategy(BaseDatasetStrategy):
    """ Drug cell stratified dataset strategy """

    def read_and_shuffle_dataset(self, random_state):
        """ Read and shuffle dataset """
        logging.info("Reading and shuffling dataset for drug-cell-stratified split...")
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
        Creates truly disjoint train/val/test sets for proper generalization testing.
        """
        logging.info("Splitting dataset for drug-cell stratified split.")
        
        unique_drugs = dataset_df['drug_name'].unique()
        unique_cells = dataset_df['cell_line_name'].unique()
        
        logging.info(f"Total unique drugs: {len(unique_drugs)}, Total unique cells: {len(unique_cells)}")
        
        # Split drugs into completely disjoint sets
        train_drugs, temp_drugs = train_test_split(
            unique_drugs, test_size=0.3, random_state=random_state
        )
        val_drugs, test_drugs = train_test_split(
            temp_drugs, test_size=0.5, random_state=random_state + 1
        )
        
        # Split cells into completely disjoint sets
        train_cells, temp_cells = train_test_split(
            unique_cells, test_size=0.3, random_state=random_state + 2
        )
        val_cells, test_cells = train_test_split(
            temp_cells, test_size=0.5, random_state=random_state + 3
        )

        logging.info(f"Drug splits - Train: {len(train_drugs)}, Val: {len(val_drugs)}, Test: {len(test_drugs)}")
        logging.info(f"Cell splits - Train: {len(train_cells)}, Val: {len(val_cells)}, Test: {len(test_cells)}")

        train_df = dataset_df[
            dataset_df['drug_name'].isin(train_drugs) & 
            dataset_df['cell_line_name'].isin(train_cells)
        ]
        
        val_df = dataset_df[
            dataset_df['drug_name'].isin(val_drugs) & 
            dataset_df['cell_line_name'].isin(val_cells)
        ]
        
        test_df = dataset_df[
            dataset_df['drug_name'].isin(test_drugs) & 
            dataset_df['cell_line_name'].isin(test_cells)
        ]

        # Validation checks for minimum dataset sizes
        min_samples = 50  # Minimum samples for reliable evaluation
        
        if len(val_df) < min_samples:
            raise ValueError(f"Validation set too small: {len(val_df)} samples (minimum: {min_samples})")
        if len(test_df) < min_samples:
            raise ValueError(f"Test set too small: {len(test_df)} samples (minimum: {min_samples})")
        if len(train_df) < min_samples * 5:
            raise ValueError(f"Training set too small: {len(train_df)} samples (minimum: {min_samples * 5})")

        # Validate no data leakage
        self._validate_stratification_integrity(
            train_drugs, val_drugs, test_drugs,
            train_cells, val_cells, test_cells
        )

        x_train = train_df[['drug_name', 'cell_line_name']]
        y_train = train_df[['pic50']]
        x_val = val_df[['drug_name', 'cell_line_name']]
        y_val = val_df[['pic50']]
        x_test = test_df[['drug_name', 'cell_line_name']]
        y_test = test_df[['pic50']]

        logging.info(f"Final Split sizes: Train={len(x_train)}, Val={len(x_val)}, Test={len(x_test)}")
        
        # Calculate actual percentages achieved
        total_samples = len(x_train) + len(x_val) + len(x_test)
        train_pct = len(x_train) / total_samples * 100
        val_pct = len(x_val) / total_samples * 100
        test_pct = len(x_test) / total_samples * 100
        
        logging.info(f"Achieved ratios: Train={train_pct:.1f}%, Val={val_pct:.1f}%, Test={test_pct:.1f}%")
        
        return x_train, x_val, x_test, y_train, y_val, y_test

    def _validate_stratification_integrity(self, train_drugs, val_drugs, test_drugs, 
                                         train_cells, val_cells, test_cells):
        """Validate that stratification maintains scientific integrity with no data leakage."""
        
        # Convert to sets for overlap checking
        train_drugs_set = set(train_drugs)
        val_drugs_set = set(val_drugs)
        test_drugs_set = set(test_drugs)
        
        train_cells_set = set(train_cells)
        val_cells_set = set(val_cells)
        test_cells_set = set(test_cells)
        
        # Check for drug overlaps (should be empty for proper stratification)
        drug_overlap_train_val = train_drugs_set & val_drugs_set
        drug_overlap_train_test = train_drugs_set & test_drugs_set
        drug_overlap_val_test = val_drugs_set & test_drugs_set
        
        # Check for cell overlaps (should be empty for proper stratification)  
        cell_overlap_train_val = train_cells_set & val_cells_set
        cell_overlap_train_test = train_cells_set & test_cells_set
        cell_overlap_val_test = val_cells_set & test_cells_set
        
        # Raise errors if any overlaps found
        if drug_overlap_train_val:
            raise ValueError(f"Drug leakage between train and val: {len(drug_overlap_train_val)} drugs overlap")
        if drug_overlap_train_test:
            raise ValueError(f"Drug leakage between train and test: {len(drug_overlap_train_test)} drugs overlap")
        if drug_overlap_val_test:
            raise ValueError(f"Drug leakage between val and test: {len(drug_overlap_val_test)} drugs overlap")
            
        if cell_overlap_train_val:
            raise ValueError(f"Cell leakage between train and val: {len(cell_overlap_train_val)} cells overlap")
        if cell_overlap_train_test:
            raise ValueError(f"Cell leakage between train and test: {len(cell_overlap_train_test)} cells overlap")
        if cell_overlap_val_test:
            raise ValueError(f"Cell leakage between val and test: {len(cell_overlap_val_test)} cells overlap")
            
        logging.info("✅ Stratification validation passed - no data leakage detected")
        logging.info(f"✅ Disjoint drug sets: Train={len(train_drugs_set)}, Val={len(val_drugs_set)}, Test={len(test_drugs_set)}")
        logging.info(f"✅ Disjoint cell sets: Train={len(train_cells_set)}, Val={len(val_cells_set)}, Test={len(test_cells_set)}")

    def prepare_dataset(self, dataset_dict, split_type, batch_size, random_state, learning_task_strategy):
        """
        Prepare dataset iterator for a single drug-cell-stratified split.       
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