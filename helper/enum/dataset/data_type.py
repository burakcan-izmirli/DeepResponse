from enum import Enum


class DataType(Enum):
    def __init__(self, label, path):
        self.label = label
        self.path = path

    prefix = 'prefix', 'dataset/'
    normal = 'normal', '/processed/dataset.pkl'
    l1000 = 'l1000', '/processed/dataset_l1000.pkl'
    pathway = 'pathway', '/processed/dataset_pathway_sorted.pkl'
    pathway_reduced = 'pathway_reduced', '/processed/dataset_pathway_sorted_reduced.pkl'
    digestive = 'digestive', '/processed/dataset_tissue_digestive_system.pkl'
