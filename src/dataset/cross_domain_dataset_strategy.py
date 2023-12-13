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

    def prepare_dataset(self, dataset, split_type, batch_size, random_state):
        """
        Main function for preparing dataset
        :param dataset: Dataset
        :param split_type: Split type [random, cell_stratified, drug_stratified, cell_drug_stratified]
        :param batch_size: Batch size
        :param random_state: Random state
        :return: atom_dim, bond_dim, train_dataset, valid_dataset, test_dataset
        """

        dataset, evaluation_dataset = dataset['dataset'], dataset['evaluation_dataset']

        mpnn_dataset, conv_dataset = self.create_mpnn_and_conv_dataset(dataset)
        mpnn_evaluation_dataset, conv_evaluation_dataset = self.create_mpnn_and_conv_dataset(evaluation_dataset)

        dataset = dataset[['drug_name', 'cell_line_name', 'pic50']]
        evaluation_dataset = evaluation_dataset[['drug_name', 'cell_line_name', 'pic50']]

        # Splitting dataset into train, validation, and test
        x_train, x_val, x_test, y_train, y_val, y_test = self.split_dataset(dataset, evaluation_dataset, random_state)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Creating Tensorflow datasets in parallel
            futures = [
                executor.submit(self.tf_dataset_creator, x_train, y_train, batch_size, mpnn_dataset, conv_dataset),
                executor.submit(self.tf_dataset_creator, x_val, y_val, batch_size, mpnn_dataset, conv_dataset),
                executor.submit(self.tf_dataset_creator, x_test, y_test, batch_size, mpnn_evaluation_dataset,
                                conv_evaluation_dataset)
            ]

            # Use as_completed to iterate over completed futures
            results = [future.result() for future in concurrent.futures.as_completed(futures)]

        # Unpack the results
        atom_dim, bond_dim, cell_line_dim = results[0][:3]
        train_dataset, valid_dataset, test_dataset = [result[3] for result in results]

        return (atom_dim, bond_dim, cell_line_dim), train_dataset, valid_dataset, test_dataset, y_test
