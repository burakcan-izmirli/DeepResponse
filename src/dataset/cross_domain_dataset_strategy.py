""" Cross domain dataset strategy """
import pandas as pd
import concurrent.futures

from sklearn.model_selection import train_test_split

from helper.enum.dataset.split_ratio import SplitRatio
from src.dataset.base_dataset_strategy import BaseDatasetStrategy


class CrossDomainDatasetStrategy(BaseDatasetStrategy):
    """ Cross domain dataset strategy """

    def read_and_shuffle_dataset(self, random_state):
        """ Read and shuffle dataset """

        dataset_raw = pd.read_pickle(self.data_path)
        # Shuffling dataset
        dataset_raw = dataset_raw.sample(frac=1, random_state=random_state).reset_index(drop=True)

        evaluation_dataset_raw = pd.read_pickle(self.evaluation_data_path)
        evaluation_dataset_raw = evaluation_dataset_raw.sample(frac=1, random_state=random_state). \
            reset_index(drop=True)

        return {'dataset': dataset_raw, 'evaluation_dataset': evaluation_dataset_raw}

    def create_splitter(self, dataset, random_state):
        """ Create splitter """
        pass

    def split_dataset(self, dataset, evaluation_dataset, random_state):
        """ Splitting dataset as train, validation and test """

        x_train_df, x_val_df, y_train_df, y_val_df = train_test_split(
            dataset[['drug_name', 'cell_line_name']],
            dataset[['pic50']],
            test_size=SplitRatio.validation_ratio.value,
            random_state=random_state)

        x_test_df = evaluation_dataset[['drug_name', 'cell_line_name']]
        y_test_df = evaluation_dataset[['pic50']]

        return x_train_df, x_val_df, x_test_df, y_train_df, y_val_df, y_test_df

    def prepare_dataset(self, dataset_dict, split_type, batch_size, random_state, learning_task_strategy):
        """
        Main function for preparing dataset for cross-domain validation.
        :param dataset_dict: Dictionary containing 'dataset' and 'evaluation_dataset'.
        :param split_type: Split type (not used directly, but for signature consistency).
        :param batch_size: Batch size.
        :param random_state: Random state.
        :param learning_task_strategy: The learning task strategy instance.
        :return: (drug_shape, cell_shape), train_dataset, valid_dataset, test_dataset, y_test
        """

        dataset, evaluation_dataset = dataset_dict['dataset'], dataset_dict['evaluation_dataset']

        drug_smiles_lookup, cell_features_lookup = self.create_drug_and_conv_dataset(dataset)
        eval_drug_smiles_lookup, eval_cell_features_lookup = self.create_drug_and_conv_dataset(evaluation_dataset)

        dataset = dataset[['drug_name', 'cell_line_name', 'pic50']]
        evaluation_dataset = evaluation_dataset[['drug_name', 'cell_line_name', 'pic50']]

        # Splitting dataset into train, validation, and test
        x_train, x_val, x_test, y_train, y_val, y_test = self.split_dataset(dataset, evaluation_dataset, random_state)

        # Process targets
        y_train = learning_task_strategy.process_targets(y_train)
        y_val = learning_task_strategy.process_targets(y_val)
        y_test = learning_task_strategy.process_targets(y_test)

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            # Creating Tensorflow datasets in parallel
            future_train = executor.submit(self.tf_dataset_creator, x_train, y_train, batch_size, cell_features_lookup, drug_smiles_lookup, learning_task_strategy, is_training=True)
            future_val = executor.submit(self.tf_dataset_creator, x_val, y_val, batch_size, cell_features_lookup, drug_smiles_lookup, learning_task_strategy, is_training=False)
            future_test = executor.submit(self.tf_dataset_creator, x_test, y_test, batch_size, eval_cell_features_lookup, eval_drug_smiles_lookup, learning_task_strategy, is_training=False)

            drug_shape, cell_shape, train_dataset = future_train.result()
            _, _, valid_dataset = future_val.result()
            _, _, test_dataset = future_test.result()


        return (drug_shape, cell_shape), train_dataset, valid_dataset, test_dataset, y_test
