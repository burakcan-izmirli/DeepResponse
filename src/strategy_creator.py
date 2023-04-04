""" Strategy creator """
import logging
from helper.enum.dataset.data_type import DataType

from src.comet.use_comet_strategy import UseCometStrategy
from src.comet.skip_comet_strategy import SkipCometStrategy

from src.dataset.random_split_dataset_strategy import RandomSplitDatasetStrategy
from src.dataset.cell_stratified_dataset_strategy import CellStratifiedDatasetStrategy
from src.dataset.drug_stratified_dataset_strategy import DrugStratifiedDatasetStrategy
from src.dataset.drug_cell_stratified_dataset_strategy import DrugCellStratifiedDatasetStrategy

from src.model.model_creation.merged_model_creation_strategy import MergedModelStrategy

from src.model.training.stratified_split_model_training_strategy import StratifiedSplitTrainingStrategy
from src.model.training.random_split_training_strategy import RandomSplitTrainingStrategy


class StrategyCreator:
    """ Strategy creator """

    def __init__(self, use_comet, data_type, split_type, random_state, batch_size, epoch, learning_rate):
        self.use_comet = use_comet
        self.data_type = data_type
        self.split_type = split_type
        self.random_state = random_state
        self.batch_size = batch_size
        self.epoch = epoch
        self.learning_rate = learning_rate
        logging.basicConfig(level=logging.INFO)

    def get_comet_strategy(self):
        """ Get comet strategy """
        strategies = {
            True: UseCometStrategy(),
            False: SkipCometStrategy()
        }
        return strategies[self.use_comet]

    def get_dataset_path_by_data_type(self):
        """ Get dataset path by data type """
        paths = {
            DataType.normal.label: DataType.normal.path,
            DataType.l1000.label: DataType.l1000.path,
            DataType.pathway.label: DataType.pathway.path,
            DataType.pathway_reduced.label: DataType.pathway_reduced.path,
            DataType.digestive.label: DataType.digestive.path
        }
        return paths[self.data_type]

    def get_split_strategy(self):
        """ Get split strategy """
        strategies = {
            'random': {'dataset': RandomSplitDatasetStrategy(self.get_dataset_path_by_data_type()),
                       'training': RandomSplitTrainingStrategy()},
            'cell_stratified': {'dataset': CellStratifiedDatasetStrategy(self.get_dataset_path_by_data_type()),
                                'training': StratifiedSplitTrainingStrategy()},
            'drug_stratified': {'dataset': DrugStratifiedDatasetStrategy(self.get_dataset_path_by_data_type()),
                                'training': StratifiedSplitTrainingStrategy()},
            'drug_cell_stratified': {'dataset': DrugCellStratifiedDatasetStrategy(self.get_dataset_path_by_data_type()),
                                     'training': StratifiedSplitTrainingStrategy()}
        }
        return strategies[self.split_type]

    def get_model_strategy(self):
        """ Get model strategy """
        strategies = {
            'merged': MergedModelStrategy()
        }
        return strategies['merged']
