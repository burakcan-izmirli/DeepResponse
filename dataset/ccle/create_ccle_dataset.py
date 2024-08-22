""" Create CCLE dataset """
import warnings
import numpy as np
import pandas as pd
from os import walk
from tqdm import tqdm

# Disabling redundant warnings of pandas
warnings.simplefilter(action='ignore', category=FutureWarning)

ROW_WISED_PROPORTION = 0.5
COLUMN_WISED_PROPORTION = 0.9


# %%
def create_unique_gene_name_df(l1000=False,
                               gdsc_cell_lines_path='../GDSC/raw/full/',
                               l1000_genes_path='../GDSC/raw/L1000_gene_list.txt'):
    """ Create unique gene name df """
    if l1000:
        l1000_gene_list_raw = pd.read_csv(l1000_genes_path, sep='\t')
        unique_gene_name_list = l1000_gene_list_raw['pr_gene_symbol'].tolist()
        unique_gene_name_list.sort()
    else:
        gdsc_cell_line_files = next(walk(gdsc_cell_lines_path), (None, None, []))[2]
        gdsc_cell_line_df = pd.read_csv(gdsc_cell_lines_path + gdsc_cell_line_files[0], sep='\t')
        unique_gene_name_list = gdsc_cell_line_df['gene_name'].unique().tolist()

    return pd.DataFrame(unique_gene_name_list, columns=['gene_name'])


def preprocess_gene_expression_data(gene_exp_path='raw/CCLE_gene_exp_extended_genes_cell_lines_v1_selected.txt',
                                    export_path='raw/CCLE_gene_exp_cell_lines_preprocessed.csv'):
    """ Preprocess gene expression data """
    gene_name_df = create_unique_gene_name_df()
    gene_exp_df_raw = pd.read_csv(gene_exp_path, sep='\t')
    gene_exp_df_raw['gene_name'] = gene_exp_df_raw['gene_name'].apply(str.upper)
    gene_exp_df_filtered = pd.merge(gene_name_df, gene_exp_df_raw, how='left')
    gene_exp_df_filtered.columns = [x.upper() if x != 'gene_name' else x for x in gene_exp_df_filtered.columns]
    gene_exp_df_filtered.set_index('gene_name', inplace=True)

    # Imputation
    row_wise_na_list = gene_exp_df_filtered.isna().sum(axis=1)
    row_wise_na_list = row_wise_na_list[
        row_wise_na_list < gene_exp_df_filtered.shape[1] * ROW_WISED_PROPORTION].index.to_list()

    column_wise_na_list = gene_exp_df_filtered.isna().sum(axis=0)
    column_wise_na_list = column_wise_na_list[
        column_wise_na_list < gene_exp_df_filtered.shape[0] * COLUMN_WISED_PROPORTION].index.to_list()

    gene_exp_df_filtered = gene_exp_df_filtered.where(
        gene_exp_df_filtered.loc[row_wise_na_list].notna(),
        gene_exp_df_filtered.loc[row_wise_na_list].mean(axis=1), axis=0)

    gene_exp_df_filtered = gene_exp_df_filtered.where(
        gene_exp_df_filtered[column_wise_na_list].notna(),
        gene_exp_df_filtered[column_wise_na_list].mean(axis=0), axis=1)

    gene_exp_df_filtered.reset_index(inplace=True)
    gene_exp_df_filtered.to_csv(export_path, index=False)


def preprocess_mutation_data(mutation_path='raw/CCLE_mutation_extended_genes_cell_lines_v1_selected.txt',
                             export_path='raw/CCLE_mutation_cell_lines_preprocessed.csv'):
    """ Preprocess gene expression data """
    gene_name_df = create_unique_gene_name_df()
    mutation_raw_df = pd.read_csv(mutation_path, sep='\t')
    mutation_raw_df = mutation_raw_df.iloc[:, 1:]
    mutation_raw_df['gene_name'] = mutation_raw_df['gene_name'].apply(str.upper)
    mutation_df_filtered = pd.merge(gene_name_df, mutation_raw_df, how='left')
    mutation_df_filtered.columns = [x.upper() if x != 'gene_name' else x for x in mutation_df_filtered.columns]

    # Imputation
    mutation_df_filtered.fillna(0, inplace=True)

    mutation_df_filtered.to_csv(export_path, index=False)


def preprocess_methylation_data(methylation_path='raw/CCLE_methylation_extended_genes_cell_lines_v1_selected.txt',
                                export_path='raw/CCLE_methylation_cell_lines_preprocessed.csv'):
    """ Preprocess gene expression data """
    gene_name_df = create_unique_gene_name_df()
    methylation_raw_df = pd.read_csv(methylation_path, sep='\t')
    methylation_raw_df['gene_name'] = methylation_raw_df['gene_name'].apply(str.upper)
    methylation_df_filtered = pd.merge(gene_name_df, methylation_raw_df, how='left')
    methylation_df_filtered.columns = [x.upper() if x != 'gene_name' else x for x in methylation_df_filtered.columns]
    methylation_df_filtered.set_index('gene_name', inplace=True)

    # Imputation
    row_wise_na_list = methylation_df_filtered.isna().sum(axis=1)
    row_wise_na_list = row_wise_na_list[
        row_wise_na_list < methylation_df_filtered.shape[1] * ROW_WISED_PROPORTION].index.to_list()

    column_wise_na_list = methylation_df_filtered.isna().sum(axis=0)
    column_wise_na_list = column_wise_na_list[
        column_wise_na_list < methylation_df_filtered.shape[0] * COLUMN_WISED_PROPORTION].index.to_list()

    methylation_df_filtered = methylation_df_filtered.where(
        methylation_df_filtered.loc[row_wise_na_list].notna(),
        methylation_df_filtered.loc[row_wise_na_list].mean(axis=1), axis=0)

    methylation_df_filtered = methylation_df_filtered.where(
        methylation_df_filtered[column_wise_na_list].notna(),
        methylation_df_filtered[column_wise_na_list].mean(axis=0), axis=1)

    methylation_df_filtered.reset_index(inplace=True)
    methylation_df_filtered.to_csv(export_path, index=False)


def preprocess_cnv_data(cnv_path='raw/CCLE_cnv_df_extended_genes_cell_lines_v1_selected.txt',
                        export_path='raw/CCLE_cnv_cell_lines_preprocessed.csv'):
    """ Preprocess gene expression data """
    gene_name_df = create_unique_gene_name_df()
    cnv_raw_df = pd.read_csv(cnv_path, sep='\t')
    cnv_raw_df['gene_name'] = cnv_raw_df['gene_name'].apply(str.upper)
    cnv_df_filtered = pd.merge(gene_name_df, cnv_raw_df, how='left')
    cnv_df_filtered.columns = [x.upper() if x != 'gene_name' else x for x in cnv_df_filtered.columns]
    cnv_df_filtered.set_index('gene_name', inplace=True)

    # Imputation
    row_wise_na_list = cnv_df_filtered.isna().sum(axis=1)
    row_wise_na_list = row_wise_na_list[
        row_wise_na_list < cnv_df_filtered.shape[1] * ROW_WISED_PROPORTION].index.to_list()

    column_wise_na_list = cnv_df_filtered.isna().sum(axis=0)
    column_wise_na_list = column_wise_na_list[
        column_wise_na_list < cnv_df_filtered.shape[0] * COLUMN_WISED_PROPORTION].index.to_list()

    cnv_df_filtered = cnv_df_filtered.where(
        cnv_df_filtered.loc[row_wise_na_list].notna(),
        cnv_df_filtered.loc[row_wise_na_list].median(axis=1), axis=0)

    cnv_df_filtered = cnv_df_filtered.where(
        cnv_df_filtered[column_wise_na_list].notna(),
        cnv_df_filtered[column_wise_na_list].median(axis=0), axis=1)

    cnv_df_filtered.reset_index(inplace=True)
    cnv_df_filtered.to_csv(export_path, index=False)


# %%
def create_drug_cell_dataframe(drug_cell_path=
                               'raw/CCLE_drug_response_drug_name_cell_line_name_pIC50_v1_edits_and_fps.txt',
                               drug_smiles_path='raw/CCLE_drug_name_SMILES_and_ECFP4_v1.txt'):
    """
    Creating dataframe contains drug_name, cell_line_name, smiles and pic50
    :param drug_cell_path: Path of drug-cell pairs dataset
    :param drug_smiles_path: Path of drug-smiles pairs dataset
    """
    drug_cell_pairs_raw = pd.read_csv(drug_cell_path, sep='\t')
    drug_cell_pairs_raw.columns = drug_cell_pairs_raw.columns.str.lower()

    drug_smiles_raw = pd.read_csv(drug_smiles_path, sep='\t')
    drug_smiles_raw.columns = drug_smiles_raw.columns.str.lower()

    return pd.merge(drug_cell_pairs_raw, drug_smiles_raw, how='outer')


# %%

def create_dataset(gene_exp_path='raw/CCLE_gene_exp_cell_lines_preprocessed.csv',
                   mutation_path='raw/CCLE_mutation_cell_lines_preprocessed.csv',
                   methylation_path='raw/CCLE_methylation_cell_lines_preprocessed.csv',
                   cnv_path='raw/CCLE_cnv_cell_lines_preprocessed.csv',
                   export_path='processed/dataset.pkl'):
    """
    Creating dataset using all genes and without any pathway or tissue information
    :param gene_exp_path: Path of gene expression dataset
    :param mutation_path: Path of mutation_path dataset
    :param methylation_path: Path of methylation dataset
    :param cnv_path: Path of CNV dataset
    :param export_path: Export path
    :return: dataset
    """

    drug_cell_smiles_df = create_drug_cell_dataframe()
    gene_exp_raw_df = pd.read_csv(gene_exp_path)
    unique_cell_line_names_list = gene_exp_raw_df.iloc[:, 1:].columns.values.tolist()
    unique_cell_line_names_list = [x.upper() if x != 'gene_name' else x for x in unique_cell_line_names_list]

    gene_exp_df = pd.read_csv(gene_exp_path)
    mutation_df = pd.read_csv(mutation_path)
    methylation_df = pd.read_csv(methylation_path)
    cnv_df = pd.read_csv(cnv_path)

    cell_lines_list = []
    for cell_line in tqdm(unique_cell_line_names_list):
        cell_line_features_last = create_unique_gene_name_df()
        for cell_line_features_dump in [gene_exp_df, mutation_df, methylation_df, cnv_df]:
            cell_line_features_dump = cell_line_features_dump[['gene_name', cell_line]]
            cell_line_features_last = pd.merge(cell_line_features_last, cell_line_features_dump, how='left',
                                               on='gene_name')
        cell_line_features_last.columns = ['gene_name', 'Exp', 'Mut', 'Met', 'Cnv']
        if len(cell_line_features_last.columns[cell_line_features_last.isna().any()].tolist()) > 0:
            continue
        cell_line_features_last.drop(columns=['gene_name'], inplace=True)
        cell_lines_list.append({'cell_line_name': cell_line,
                                'cell_line_features': cell_line_features_last.to_numpy(dtype=np.float16)})

    cell_line_features_df = pd.DataFrame(cell_lines_list)
    dataset = pd.merge(drug_cell_smiles_df, cell_line_features_df, how='outer')
    dataset = dataset[['drug_name', 'cell_line_name', 'pic50', 'smiles', 'cell_line_features']]
    dataset.dropna(inplace=True)

    dataset.to_pickle(export_path)

    return dataset


# %%

def create_l1000_dataset(cross_domain='False',
                         gene_exp_path='raw/CCLE_gene_exp_cell_lines_preprocessed.csv',
                         mutation_path='raw/CCLE_mutation_cell_lines_preprocessed.csv',
                         methylation_path='raw/CCLE_methylation_cell_lines_preprocessed.csv',
                         cnv_path='raw/CCLE_cnv_cell_lines_preprocessed.csv',
                         export_path='processed/dataset_l1000'):
    """
    Creating dataset using all genes and without any pathway or tissue information
    :param cross_domain: Whether dataset to be used for cross domain analysis
    :param gene_exp_path: Path of gene expression dataset
    :param mutation_path: Path of mutation_path dataset
    :param methylation_path: Path of methylation dataset
    :param cnv_path: Path of CNV dataset
    :param export_path: Export path
    :return: dataset
    """

    drug_cell_smiles_df = create_drug_cell_dataframe()
    gene_exp_raw_df = pd.read_csv(gene_exp_path)
    unique_cell_line_names_list = gene_exp_raw_df.iloc[:, 1:].columns.values.tolist()
    unique_cell_line_names_list = [x.upper() if x != 'gene_name' else x for x in unique_cell_line_names_list]

    gene_exp_df = pd.read_csv(gene_exp_path)
    mutation_df = pd.read_csv(mutation_path)
    methylation_df = pd.read_csv(methylation_path)
    cnv_df = pd.read_csv(cnv_path)

    cell_lines_list = []
    for cell_line in tqdm(unique_cell_line_names_list):
        cell_line_features_last = create_unique_gene_name_df(l1000=True)
        for cell_line_features_dump in [gene_exp_df, mutation_df, methylation_df, cnv_df]:
            cell_line_features_dump = cell_line_features_dump[['gene_name', cell_line]]
            join_type = 'inner' if cross_domain else 'left'
            cell_line_features_last = pd.merge(cell_line_features_last, cell_line_features_dump, how=join_type,
                                               on='gene_name')
        cell_line_features_last.columns = ['gene_name', 'Exp', 'Mut', 'Met', 'Cnv']
        if len(cell_line_features_last.columns[cell_line_features_last.isna().all()].tolist()) > 0:
            continue
        cell_line_features_last.drop(columns=['gene_name'], inplace=True)
        cell_lines_list.append({'cell_line_name': cell_line,
                                'cell_line_features': cell_line_features_last.to_numpy(dtype=np.float16)})

    cell_line_features_df = pd.DataFrame(cell_lines_list)
    dataset = pd.merge(drug_cell_smiles_df, cell_line_features_df, how='outer')
    dataset = dataset[['drug_name', 'cell_line_name', 'pic50', 'smiles', 'cell_line_features']]
    dataset.dropna(inplace=True)

    export_path = export_path + "_cross_domain.pkl" if cross_domain else export_path + ".pkl"
    dataset.to_pickle(export_path)

    return dataset

