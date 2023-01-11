from enum import Enum


class DataType(Enum):
    def __init__(self, label, path):
        self.label = label
        self.path = path

    normal = 'normal', 'data/processed/dataset.pkl'
    l1000 = 'l1000', 'data/processed/dataset_l1000.pkl'
    pathway = 'pathway', 'data/processed/dataset_pathway_sorted.pkl'
    pathway_reduced = 'pathway_reduced', 'data/processed/dataset_pathway_sorted_reduced.pkl'
    digestive = 'digestive', 'data/processed/dataset_tissue_digestive_system.pkl'
