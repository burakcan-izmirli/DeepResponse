from enum import Enum


class AtomFeaturizerSets(Enum):
    symbol: dict = {'B', 'Br', 'C', 'Ca', 'Cl', 'F', 'H', 'I', 'N', 'Na', 'O', 'P', 'S'}
    n_valence: dict = {0, 1, 2, 3, 4, 5, 6}
    n_hydrogens: dict = {0, 1, 2, 3, 4}
    hybridization: float = {'s', 'sp', 'sp2', 'sp3'}