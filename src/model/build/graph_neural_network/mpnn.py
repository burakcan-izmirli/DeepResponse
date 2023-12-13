"""
Message-passing neural networks
This code was derived from Keras' Message-passing neural network (MPNN) for the molecular property prediction tutorial
and extended based on the needs.
"""
import numpy as np
import tensorflow as tf
import logging
import warnings
import concurrent.futures
from rdkit import Chem

from helper.enum.model.atom_featurizer_sets import AtomFeaturizerSets
from helper.enum.model.bond_featurizer_sets import BondFeaturizerSets

warnings.filterwarnings('ignore')


class Featurizer:
    def __init__(self, allowable_sets):
        self.dim = 0
        self.features_mapping = {}
        for k, s in allowable_sets.items():
            s = sorted(list(s))
            self.features_mapping[k] = dict(zip(s, range(self.dim, len(s) + self.dim)))
            self.dim += len(s)

    def encode(self, inputs):
        output = np.zeros((self.dim,))
        for name_feature, feature_mapping in self.features_mapping.items():
            feature = getattr(self, name_feature)(inputs)
            if feature not in feature_mapping:
                continue
            output[feature_mapping[feature]] = 1.0
        return output


class AtomFeaturizer(Featurizer):
    def __init__(self, allowable_sets):
        super().__init__(allowable_sets)

    def symbol(self, atom):
        return atom.GetSymbol()

    def n_valence(self, atom):
        return atom.GetTotalValence()

    def n_hydrogens(self, atom):
        return atom.GetTotalNumHs()

    def hybridization(self, atom):
        return atom.GetHybridization().name.lower()


class BondFeaturizer(Featurizer):
    def __init__(self, allowable_sets):
        super().__init__(allowable_sets)
        self.dim += 1

    def encode(self, bond):
        output = np.zeros((self.dim,))
        if bond is None:
            output[-1] = 1.0
            return output
        output = super().encode(bond)
        return output

    def bond_type(self, bond):
        return bond.GetBondType().name.lower()

    def conjugated(self, bond):
        return bond.GetIsConjugated()


atom_featurizer = AtomFeaturizer(
    allowable_sets={
        "symbol": AtomFeaturizerSets.symbol.value,
        "n_valence": AtomFeaturizerSets.n_valence.value,
        "n_hydrogens": AtomFeaturizerSets.n_hydrogens.value,
        "hybridization": AtomFeaturizerSets.hybridization.value,
    }
)

bond_featurizer = BondFeaturizer(
    allowable_sets={
        "bond_type": BondFeaturizerSets.bond_type.value,
        "conjugated": BondFeaturizerSets.conjugated.value,
    }
)


def molecule_from_smiles(smiles):
    # MolFromSmiles(m, sanitize=True) should be equivalent to
    # MolFromSmiles(m, sanitize=False) -> SanitizeMol(m) -> AssignStereochemistry(m, ...)
    molecule = Chem.MolFromSmiles(smiles, sanitize=False)

    # If sanitization is unsuccessful, catch the error, and try again without
    # the sanitization step that caused the error
    flag = Chem.SanitizeMol(molecule, catchErrors=True)
    if flag != Chem.SanitizeFlags.SANITIZE_NONE:
        Chem.SanitizeMol(molecule, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ flag)

    Chem.AssignStereochemistry(molecule, cleanIt=True, force=True)
    return molecule


def graph_from_molecule(molecule):
    # Initialize graph
    atom_features = []
    bond_features = []
    pair_indices = []

    for atom in molecule.GetAtoms():
        atom_features.append(atom_featurizer.encode(atom))

        # Add self-loops
        pair_indices.append([atom.GetIdx(), atom.GetIdx()])
        bond_features.append(bond_featurizer.encode(None))

        for neighbor in atom.GetNeighbors():
            bond = molecule.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx())
            pair_indices.append([atom.GetIdx(), neighbor.GetIdx()])
            bond_features.append(bond_featurizer.encode(bond))

    return np.array(atom_features), np.array(bond_features), np.array(pair_indices)


def process_smiles(smiles):
    """
    Process a SMILES string to generate molecule graph components.

    Parameters:
        smiles (str): The SMILES string representing a molecule.

    Returns:
        tuple: A tuple containing atom features, bond features, and pair indices.
    """
    molecule = molecule_from_smiles(smiles)
    atom_features, bond_features, pair_indices = graph_from_molecule(molecule)
    return atom_features, bond_features, pair_indices


def convert_smiles_to_graph(smiles_list):
    """
    Generate graphs from a list of SMILES strings.

    Parameters:
        smiles_list (list): List of SMILES strings.
    Returns:
        tuple: A tuple containing ragged tensors for atom features, bond features,
               and pair indices suitable for tf.dataset.Dataset.
    """
    logging.info("Graphs from smiles is started.")

    # Use parallel processing to generate graphs
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Submit processing tasks for each the SMILES string
        futures = [executor.submit(process_smiles, smiles) for smiles in smiles_list]

    # Wait for processing to complete and collect results
    results = [future.result() for future in concurrent.futures.as_completed(futures)]

    # Unpack results
    atom_features_list, bond_features_list, pair_indices_list = zip(*results)

    # Convert lists to ragged tensors for tf.dataset.Dataset later on
    return (
        tf.ragged.constant(atom_features_list, dtype=tf.float32),
        tf.ragged.constant(bond_features_list, dtype=tf.float32),
        tf.ragged.constant(pair_indices_list, dtype=tf.int64),
    )


# %%

def prepare_batch(x_batch_conv, x_batch_mpnn, y_batch):
    """
    Merges (sub)graphs of batch into a single global (disconnected) graph
    """

    atom_features, bond_features, pair_indices = x_batch_mpnn

    # Obtain number of atoms and bonds for each graph (molecule)
    num_atoms = atom_features.row_lengths()
    num_bonds = bond_features.row_lengths()

    # Obtain partition indices (molecule_indicator), which will be used to
    # gather (sub)graphs from global graph in model later on
    molecule_indices = tf.range(len(num_atoms))
    molecule_indicator = tf.repeat(molecule_indices, num_atoms)

    # Merge (sub)graphs into a global (disconnected) graph. Adding 'increment' to
    # 'pair_indices' (and merging ragged tensors) actualizes the global graph
    gather_indices = tf.repeat(molecule_indices[:-1], num_bonds[1:])
    increment = tf.cumsum(num_atoms[:-1])
    increment = tf.pad(tf.gather(increment, gather_indices), [(num_bonds[0], 0)])
    pair_indices = pair_indices.merge_dims(outer_axis=0, inner_axis=1).to_tensor()
    pair_indices = pair_indices + increment[:, tf.newaxis]
    atom_features = atom_features.merge_dims(outer_axis=0, inner_axis=1).to_tensor()
    bond_features = bond_features.merge_dims(outer_axis=0, inner_axis=1).to_tensor()

    return (x_batch_conv, atom_features, bond_features, pair_indices, molecule_indicator), y_batch
