""" Random split dataset strategy """
from sklearn.model_selection import train_test_split

from helper.enum.dataset.split_ratio import SplitRatio
from src.dataset.base_dataset_strategy import BaseDatasetStrategy


class RandomSplitDatasetStrategy(BaseDatasetStrategy):
    """ Random split dataset strategy """

    def create_splitter(self, dataset, random_state):
        """ Create splitter """
        pass

    def split_dataset(self, dataset, random_state):
        """ Splitting dataset as train, validation and test """

        x_train_df, x_test_df, y_train_df, y_test_df = train_test_split(dataset[['drug_name', 'cell_line_name']],
                                                                        dataset[['pic50']],
                                                                        test_size=SplitRatio.test_ratio.value,
                                                                        random_state=random_state)

        x_train_df, x_val_df, y_train_df, y_val_df = train_test_split(x_train_df,
                                                                      y_train_df,
                                                                      test_size=SplitRatio.validation_ratio.value,
                                                                      random_state=random_state)
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
        mpnn_dataset, conv_dataset = self.create_mpnn_and_conv_dataset(dataset)

        dataset = dataset[['drug_name', 'cell_line_name', 'pic50']]
        print(len(dataset))
        # Splitting dataset into train, validation and test
        x_train, x_val, x_test, y_train, y_val, y_test = self.split_dataset(dataset, random_state)
        # Creating Tensorflow datasets
        atom_dim, bond_dim, cell_line_dim, train_dataset = self.tf_dataset_creator(x_train, y_train, batch_size,
                                                                                   mpnn_dataset, conv_dataset)
        atom_dim_valid, bond_dim_valid, cell_line_dim_valid, valid_dataset = self.tf_dataset_creator(x_val, y_val,
                                                                                                     batch_size,
                                                                                                     mpnn_dataset,
                                                                                                     conv_dataset)
        atom_dim_test, bond_dim_test, cell_line_dim_test, test_dataset = self.tf_dataset_creator(x_test, y_test,
                                                                                                 len(x_test),
                                                                                                 mpnn_dataset,
                                                                                                 conv_dataset)
        return (atom_dim, bond_dim, cell_line_dim), train_dataset, valid_dataset, test_dataset, y_test
