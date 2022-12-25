import pandas as pd
from os import walk
from tqdm import tqdm


# %%
def create_pathway_gene_dataframe(kegg_gene_path='dataset/KEGG_Identifiers_gene_name_pathway_fixed_v1.txt',
                                  kegg_pathway_path='dataset/kegg_pathway_protein_associations_1.tsv',
                                  export_path='dataset/pathway_gene_merged.pkl'):
    """
    Creating pathway sorted gene dataframe
    :param kegg_gene_path: Path of data containing gene and kegg_id
    :param kegg_pathway_path: Path of data containing pathway and kegg_id
    :param export_path: Export path
    :return: Pathway sorted gene dataframe
    """
    kegg_gene_df_raw = pd.read_csv(kegg_gene_path, sep = '\t')
    kegg_gene_df_raw.rename(columns = {'KEGG_ID': 'kegg_id'}, inplace = True)

    kegg_pathway_df_raw = pd.read_csv(kegg_pathway_path, sep = '\t')
    kegg_pathway_df_raw.rename(columns = {'KEGG_GeneID': 'kegg_id'}, inplace = True)

    pathway_gene_merged = pd.merge(kegg_pathway_df_raw, kegg_gene_df_raw)
    pathway_gene_merged = pathway_gene_merged[
        ['kegg_id', 'KEGG_PathwayID', 'KEGG_PathwayName', 'hgnc_id', 'hgnc_symbol']]

    pathway_gene_merged.rename(columns = {'KEGG_PathwayID': 'kegg_pathway_id', 'KEGG_PathwayName': 'kegg_pathway_name',
                                          'hgnc_id': 'gene_id', 'hgnc_symbol': 'gene_name'},
                               inplace = True)

    pathway_gene_merged.drop_duplicates(inplace = True)
    pathway_gene_merged.sort_values('kegg_pathway_id', inplace = True)

    pathway_gene_merged.to_pickle(export_path)

    return pathway_gene_merged


# %%
def create_drug_cell_dataframe(drug_cell_path=
                               'dataset/GDSC_338K_drug_response_and_cansyl_drug_cell_line_pairs_concatenated.txt',
                               drug_smiles_path='dataset/GDSC_drug_name_SMILES_and_ECFP4_v2.txt'):
    """
    Creating dataframe contains drug_name, cell_line_name, smiles and pic50
    :param drug_cell_path: Path of drug-cell pairs dataset
    :param drug_smiles_path: Path of drug-smiles pairs dataset
    """
    drug_cell_pairs_raw = pd.read_csv(drug_cell_path, sep = '\t')
    drug_cell_pairs_raw.rename(columns = {'DRUG_NAME': 'drug_name', 'CELL_LINE_NAME': 'cell_line_name',
                                          'pIC50': 'pic50'}, inplace = True)

    drug_smiles_raw = pd.read_csv(drug_smiles_path, sep = '\t')
    drug_smiles_raw.rename(columns = {'DRUG_NAME': 'drug_name', 'SMILES': 'smiles', 'ECFP4': 'ecfp4'}, inplace = True)

    return pd.merge(drug_cell_pairs_raw, drug_smiles_raw, how = 'outer')


# %%
def create_dataset(cell_lines_path='dataset/full/',
                   export_path='dataset.pkl'):
    """
    Creating dataset using all genes and without any pathway or tissue information
    :param cell_lines_path: Path of cell line features dataset
    :param export_path: Export path
    :return: dataset
    """

    drug_cell_smiles_df = create_drug_cell_dataframe()

    filenames = next(walk(cell_lines_path), (None, None, []))[2]
    cell_lines_list = []
    for i in tqdm(filenames):
        name = i.split("_")[0]
        cell_line_features_raw = pd.read_csv(cell_lines_path + i, sep = '\t')
        cell_line_features_raw.drop(columns = ['gene_name'], inplace = True)
        cell_lines_list.append({'cell_line_name': name,
                                'cell_line_features': cell_line_features_raw})

    cell_line_features_df = pd.DataFrame(cell_lines_list)

    dataset = pd.merge(drug_cell_smiles_df, cell_line_features_df, how = 'outer')
    dataset.drop("ecfp4", axis = 1, inplace = True)
    dataset.dropna(inplace = True)

    dataset.to_pickle(export_path)

    return dataset


# %%
def create_l1000_dataset(cell_lines_path='dataset/full/',
                         l1000_genes_path='dataset/L1000_gene_list.txt',
                         export_path='dataset_l1000.pkl'):
    """
    Creating dataset only contains L1000 genes.
    :param cell_lines_path: Path of cell line features
    :param l1000_genes_path: L1000 genes path
    :param export_path: Export path
    :return: L1000 Dataset
    """
    drug_cell_smiles_df = create_drug_cell_dataframe()

    l1000_gene_list_raw = pd.read_csv(l1000_genes_path, sep = '\t')
    l1000_gene_list = l1000_gene_list_raw['pr_gene_symbol'].to_list()

    filenames = next(walk(cell_lines_path), (None, None, []))[2]
    cell_lines_list = []
    for i in tqdm(filenames):
        name = i.split("_")[0]
        cell_line_features_raw = pd.read_csv(cell_lines_path + i, sep = '\t')
        cell_line_features = cell_line_features_raw.query("gene_name in @l1000_gene_list").copy()
        cell_line_features.reset_index(inplace = True, drop = True)
        cell_line_features.drop(columns = ['gene_name'], inplace = True)
        cell_lines_list.append({'cell_line_name': name,
                                'cell_line_features': cell_line_features})

    cell_line_features_l1000 = pd.DataFrame(cell_lines_list)

    dataset_l1000 = pd.merge(drug_cell_smiles_df, cell_line_features_l1000, how = 'outer')
    dataset_l1000.drop("ecfp4", axis = 1, inplace = True)
    dataset_l1000.dropna(inplace = True)

    dataset_l1000.to_pickle(export_path)

    return dataset_l1000


# %%
def create_pathway_sorted_dataset(cell_lines_path='dataset/full/',
                                  export_path='dataset_pathway_sorted.pkl'):
    """
    Creating pathway sorted dataset
    :param cell_lines_path: Path of cell line features
    :param export_path: Export path
    :return: Pathway Sorted Dataset
    """
    drug_cell_smiles_df = create_drug_cell_dataframe()
    pathway_gene_merged = create_pathway_gene_dataframe()

    filenames = next(walk(cell_lines_path), (None, None, []))[2]
    cell_lines_list = []
    for i in tqdm(filenames):
        name = i.split("_")[0]
        cell_line_features_raw = pd.read_csv(cell_lines_path + i, sep = '\t')
        cell_line_features_df = cell_line_features_raw.set_index('gene_name')
        cell_line_features_df = cell_line_features_df.reindex(index = pathway_gene_merged['gene_name'])
        cell_line_features_df.reset_index(inplace = True)
        cell_line_features_df.dropna(inplace = True)
        gene_list = cell_line_features_df['gene_name'].to_list()
        genes_without_pathway = cell_line_features_raw.query('gene_name not in @gene_list')
        cell_line_features_df = pd.concat([cell_line_features_df, genes_without_pathway])
        cell_line_features_df.drop(columns = ['gene_name'], inplace = True)
        cell_lines_list.append({'cell_line_name': name,
                                'cell_line_features': cell_line_features_df})

    cell_line_features_pathway_sorted = pd.DataFrame(cell_lines_list)

    dataset_pathway_sorted = pd.merge(drug_cell_smiles_df, cell_line_features_pathway_sorted, how = 'outer')
    dataset_pathway_sorted.drop("ecfp4", axis = 1, inplace = True)
    dataset_pathway_sorted.dropna(inplace = True)
    dataset_pathway_sorted.to_pickle(export_path)
    return dataset_pathway_sorted


# %%
def create_pathway_sorted_reduced_dataset(cell_lines_path='dataset/full/',
                                          export_path='dataset_pathway_sorted_reduced.pkl'):
    """
    Creating pathway sorted dataset only contains genes with pathway information.
    :param cell_lines_path: Path of cell line features
    :param export_path: Export path
    :return: Pathway Sorted - Reduced Dataset
    """
    drug_cell_smiles_df = create_drug_cell_dataframe()
    pathway_gene_merged = create_pathway_gene_dataframe()
    # Pathway Sorted only genes with pathway
    filenames = next(walk(cell_lines_path), (None, None, []))[2]
    cell_lines_list = []
    for i in tqdm(filenames):
        name = i.split("_")[0]
        cell_line_features_raw = pd.read_csv(cell_lines_path + i, sep = '\t')
        cell_line_features_df = cell_line_features_raw.set_index('gene_name')
        cell_line_features_df = cell_line_features_df.reindex(index = pathway_gene_merged['gene_name'])
        cell_line_features_df.reset_index(inplace = True)
        cell_line_features_df.dropna(inplace = True)
        cell_line_features_df.drop(columns = ['gene_name'], inplace = True)
        cell_lines_list.append({'cell_line_name': name,
                                'cell_line_features': cell_line_features_df})

    cell_line_features_pathway_sorted_reduced = pd.DataFrame(cell_lines_list)

    dataset_pathway_sorted_reduced = pd.merge(drug_cell_smiles_df, cell_line_features_pathway_sorted_reduced,
                                              how = 'outer')
    dataset_pathway_sorted_reduced.drop("ecfp4", axis = 1, inplace = True)
    dataset_pathway_sorted_reduced.dropna(inplace = True)

    dataset_pathway_sorted_reduced.to_pickle(export_path)

    return dataset_pathway_sorted_reduced


# %%
def create_tissue_dataset(tissue_name):
    """
    Creating tissue dataset
    :param tissue_name: Tissue name e.g digestive_system
    :return Tissue dataset
    """
    cell_line_tissue_paired = pd.read_csv('dataset/GDSC_986_cell_lines_matched_with_TCGA_tissue_names.txt', sep = '\t')

    cell_line_tissue_paired_filtered = cell_line_tissue_paired.query("Tissue_name == @tissue_name")
    tissue_list = cell_line_tissue_paired_filtered['Name'].to_list()

    dataset = create_dataset()
    tissue_filtered_dataset = dataset.query('cell_line_name in @tissue_list')

    tissue_filtered_dataset.to_pickle(f"dataset_tissue_{tissue_name}.pkl")

    return tissue_filtered_dataset
# %%
