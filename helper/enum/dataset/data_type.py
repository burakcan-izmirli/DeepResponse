from enum import Enum


class DataType(Enum):
    def __init__(self, label, path):
        self.label = label
        self.path = path

    normal = 'normal', 'dataset/processed/dataset.pkl'
    l1000 = 'l1000', 'dataset/processed/dataset_l1000.pkl'
    pathway = 'pathway', 'dataset/processed/dataset_pathway_sorted.pkl'
    pathway_reduced = 'pathway_reduced', 'dataset/processed/dataset_pathway_sorted_reduced.pkl'
    digestive = 'digestive', 'dataset/processed/dataset_tissue_digestive_system.pkl'
