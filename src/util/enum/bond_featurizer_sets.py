from enum import Enum


class BondFeaturizerSets(Enum):
    bond_type: dict = {'single', 'double', 'triple', 'aromatic'}
    conjugated: dict = {True, False}
