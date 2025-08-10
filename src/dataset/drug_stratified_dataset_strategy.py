""" Drug stratified dataset strategy """
import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

from helper.enum.dataset.n_split import NSplit
from helper.enum.dataset.split_ratio import SplitRatio
from src.dataset.base_dataset_strategy import BaseDatasetStrategy


class DrugStratifiedDatasetStrategy(BaseDatasetStrategy):
    """ Drug stratified dataset strategy """

    def read_and_shuffle_dataset(self, random_state):
        """ Read and shuffle dataset """
        logging.info("Reading and shuffling dataset for drug-stratified split...")
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
        Uses stratified splitting to maintain drug distribution across splits.
        """
        logging.info("Splitting dataset based on drug names with validation safeguards.")
        
        # Filter out drugs with insufficient samples
        min_samples_per_drug = 5  # Minimum samples per drug for reliable statistics
        drug_counts = dataset_df['drug_name'].value_counts()
        valid_drugs = drug_counts[drug_counts >= min_samples_per_drug].index
        
        logging.info(f"Total drugs: {len(drug_counts)}")
        logging.info(f"Drugs with ≥{min_samples_per_drug} samples: {len(valid_drugs)}")
        logging.info(f"Filtered out {len(drug_counts) - len(valid_drugs)} drugs with insufficient samples")
        
        # Filter dataset to only include drugs with sufficient samples
        filtered_df = dataset_df[dataset_df['drug_name'].isin(valid_drugs)].copy()
        logging.info(f"Dataset size after filtering: {len(filtered_df)} samples")
        
        if len(filtered_df) < 100:
            raise ValueError(f"Dataset too small after filtering: {len(filtered_df)} samples")
        
        # Create disjoint drug splits for proper generalization testing
        # Each drug appears in only one split (train, val, or test)
        unique_drugs = filtered_df['drug_name'].unique()
        np.random.seed(random_state)
        shuffled_drugs = np.random.permutation(unique_drugs)
        
        # Calculate split sizes based on drug counts
        n_drugs = len(shuffled_drugs)
        test_split = max(1, int(n_drugs * SplitRatio.test_ratio.value))
        val_split = max(1, int(n_drugs * SplitRatio.validation_ratio.value))
        train_split = n_drugs - test_split - val_split
        
        if train_split < 1:
            raise ValueError(f"Too few drugs ({n_drugs}) for stratified splitting")
        
        # Assign drugs to splits ensuring disjoint sets
        test_drugs = set(shuffled_drugs[:test_split])
        val_drugs = set(shuffled_drugs[test_split:test_split + val_split])
        train_drugs = set(shuffled_drugs[test_split + val_split:])
        
        # Split dataframe by drug assignments
        train_df = filtered_df[filtered_df['drug_name'].isin(train_drugs)].copy()
        val_df = filtered_df[filtered_df['drug_name'].isin(val_drugs)].copy()
        test_df = filtered_df[filtered_df['drug_name'].isin(test_drugs)].copy()
        
        # Log split information
        logging.info(f"Drug splits - Train: {len(train_drugs)}, Val: {len(val_drugs)}, Test: {len(test_drugs)}")
        logging.info(f"Sample splits - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        
        # Validate disjoint drug sets
        train_val_overlap = train_drugs.intersection(val_drugs)
        train_test_overlap = train_drugs.intersection(test_drugs)
        val_test_overlap = val_drugs.intersection(test_drugs)
        
        if train_val_overlap or train_test_overlap or val_test_overlap:
            raise ValueError(f"Configuration error: Drug overlap between train and val: {len(train_val_overlap)} drugs")
        
        logging.info("✅ Drug stratification validation passed - no drug overlap detected")
        
        # Validate minimum samples per split
        if len(val_df) < 20:
            raise ValueError(f"Validation set too small: {len(val_df)} samples")
        if len(test_df) < 20:
            raise ValueError(f"Test set too small: {len(test_df)} samples")
        
        # Check drug distribution across splits
        train_drugs = set(train_df['drug_name'].unique())
        val_drugs = set(val_df['drug_name'].unique())
        test_drugs = set(test_df['drug_name'].unique())
        
        # Ensure disjoint drug sets (no drug appears in multiple splits)
        drug_overlap_train_val = train_drugs & val_drugs
        drug_overlap_train_test = train_drugs & test_drugs
        drug_overlap_val_test = val_drugs & test_drugs
        
        if drug_overlap_train_val:
            raise ValueError(f"Drug overlap between train and val: {len(drug_overlap_train_val)} drugs")
        if drug_overlap_train_test:
            raise ValueError(f"Drug overlap between train and test: {len(drug_overlap_train_test)} drugs")  
        if drug_overlap_val_test:
            raise ValueError(f"Drug overlap between val and test: {len(drug_overlap_val_test)} drugs")

        x_train = train_df[['drug_name', 'cell_line_name']]
        y_train = train_df[['pic50']]
        x_val = val_df[['drug_name', 'cell_line_name']]
        y_val = val_df[['pic50']]
        x_test = test_df[['drug_name', 'cell_line_name']]
        y_test = test_df[['pic50']]

        logging.info(f"Split sizes: Train={len(x_train)}, Val={len(x_val)}, Test={len(x_test)}")
        logging.info(f"Drugs in sets - Train: {len(train_drugs)}, Val: {len(val_drugs)}, Test: {len(test_drugs)}")
        
        # Validate R² calculation prerequisites
        self._validate_r2_prerequisites(y_train, y_val, y_test)
        
        return x_train, x_val, x_test, y_train, y_val, y_test

    def _validate_r2_prerequisites(self, y_train, y_val, y_test):
        """Validate that R² calculation will work properly."""
        
        datasets = [("training", y_train), ("validation", y_val), ("test", y_test)]
        
        for name, y_data in datasets:
            y_values = y_data['pic50'].values
            
            # Check minimum sample size
            if len(y_values) < 2:
                raise ValueError(f"{name} set has < 2 samples: cannot calculate R²")
            
            # Check for variance (avoid division by zero)
            if np.var(y_values) == 0:
                logging.warning(f"{name} set has zero variance in targets")
            
            # Check for reasonable value range
            if np.isnan(y_values).any():
                raise ValueError(f"{name} set contains NaN values")
            
            logging.info(f"✅ {name} set: {len(y_values)} samples, "
                        f"mean={np.mean(y_values):.3f}, "
                        f"std={np.std(y_values):.3f}, "
                        f"range=[{np.min(y_values):.3f}, {np.max(y_values):.3f}]")

    def prepare_dataset(self, dataset_dict, split_type, batch_size, random_state, learning_task_strategy):
        """
        Prepare dataset iterator for a single drug-stratified split.
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