""" Drug cell stratified dataset strategy """
import logging
import pandas as pd
import concurrent.futures
import tensorflow as tf
from sklearn.model_selection import GroupKFold, train_test_split

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
        """ Creates a GroupKFold splitter based on drug-cell pairs """
        n_splits = NSplit.stratified.value
        logging.info(f"Creating GroupKFold splitter with {n_splits} folds, grouped by drug_name and cell_line_name.")
        dataset['drug_cell_group'] = dataset['drug_name'] + "_" + dataset['cell_line_name']
        return GroupKFold(n_splits=n_splits)

    def split_dataset(self, train_val_df, random_state):
        """ Splits a fold's training data into training and validation sets randomly """
        n_splits = NSplit.stratified.value
        train_ratio = (n_splits - 1) / n_splits if n_splits > 1 else 1
        relative_val_size = SplitRatio.validation_ratio.value / train_ratio

        X = train_val_df[['drug_name', 'cell_line_name']]
        y = train_val_df[['pic50']]

        if len(train_val_df) < 2:
            return X, pd.DataFrame(columns=X.columns), y, pd.DataFrame(columns=y.columns)

        x_train, x_val, y_train, y_val = train_test_split(
            X, y, test_size=relative_val_size, random_state=random_state
        )
        logging.info(f"Fold split sizes: Train={len(x_train)}, Val={len(x_val)}")
        return x_train, x_val, y_train, y_val

    def prepare_dataset(self, dataset_dict, split_type, batch_size, random_state, learning_task_strategy):
        """
        Prepare dataset iterator for stratified cross-validation.
        Yields ((smiles_shape, cell_line_shape), train_dataset, val_dataset, test_dataset, y_test_fold_actual) for each fold.
        """
        dataset_df = dataset_dict['dataset']
        drug_smiles_lookup, cell_features_lookup = self.create_drug_and_conv_dataset(dataset_df)

        required_cols = ['drug_name', 'cell_line_name', 'pic50']
        if not all(col in dataset_df.columns for col in required_cols):
            raise ValueError(f"Dataset is missing one of the required columns: {required_cols}")

        X = dataset_df[required_cols].copy()
        splitter = self.create_splitter(X)
        groups = X['drug_cell_group']

        fold_count = 0
        for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(X, groups=groups)):
            fold_count += 1
            logging.info(f"----- Preparing CV Fold {fold_count} -----")

            train_val_df = X.iloc[train_idx]
            x_test = X.iloc[test_idx][['drug_name', 'cell_line_name']]
            y_test = X.iloc[test_idx][['pic50']]

            x_train, x_val, y_train, y_val = self.split_dataset(train_val_df, random_state)

            y_train = learning_task_strategy.process_targets(y_train)
            y_val = learning_task_strategy.process_targets(y_val)
            y_test = learning_task_strategy.process_targets(y_test)

            train_args = (x_train, y_train, batch_size, cell_features_lookup, drug_smiles_lookup, learning_task_strategy, True)
            val_args = (x_val, y_val, batch_size, cell_features_lookup, drug_smiles_lookup, learning_task_strategy, False)
            test_args = (x_test, y_test, batch_size, cell_features_lookup, drug_smiles_lookup, learning_task_strategy, False)

            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                future_train = executor.submit(self.tf_dataset_creator, *train_args)
                future_val = executor.submit(self.tf_dataset_creator, *val_args)
                future_test = executor.submit(self.tf_dataset_creator, *test_args)

                drug_shape_train, cell_shape_train, train_tf_dataset = future_train.result()
                _, _, val_tf_dataset = future_val.result() # Shapes should be consistent
                _, _, test_tf_dataset = future_test.result() # Shapes should be consistent

            train_cardinality = tf.data.experimental.cardinality(train_tf_dataset).numpy()
            val_cardinality = tf.data.experimental.cardinality(val_tf_dataset).numpy()
            test_cardinality = tf.data.experimental.cardinality(test_tf_dataset).numpy()

            if train_cardinality == 0 or val_cardinality == 0 or test_cardinality == 0 :
                logging.warning(f"Skipping Fold {fold_count} due to empty TF dataset created (TrainB: {train_cardinality}, ValB: {val_cardinality}, TestB: {test_cardinality}).")
                continue

            prefetch_buffer_size = tf.data.AUTOTUNE
            train_tf_dataset = train_tf_dataset.prefetch(prefetch_buffer_size)
            val_tf_dataset = val_tf_dataset.prefetch(prefetch_buffer_size)
            test_tf_dataset = test_tf_dataset.prefetch(prefetch_buffer_size)

            logging.info(f"Fold {fold_count}: Prepared Train TF ({train_cardinality}b), Val TF ({val_cardinality}b), Test TF ({test_cardinality}b).")
            yield (drug_shape_train, cell_shape_train), train_tf_dataset, val_tf_dataset, test_tf_dataset, y_test