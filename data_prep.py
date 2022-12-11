import pandas as pd
from os import walk
from tqdm import tqdm
# %%

# kegg_gene_df_raw = pd.read_csv('dataset/KEGG_Identifiers_gene_name_pathway_fixed_v1.txt', sep = '\t')
#
# kegg_gene_df_raw.rename(columns = {'KEGG_ID': 'kegg_id'}, inplace = True)
# kegg_pathway_df_raw = pd.read_csv('dataset/kegg_pathway_protein_associations_1.tsv', sep = '\t')
# kegg_pathway_df_raw.rename(columns = {'KEGG_GeneID': 'kegg_id'}, inplace = True)
#
# pathway_gene_merged = pd.merge(kegg_pathway_df_raw, kegg_gene_df_raw)
#
# pathway_gene_merged = pathway_gene_merged[['kegg_id', 'KEGG_PathwayID', 'KEGG_PathwayName', 'hgnc_id', 'hgnc_symbol']]
#
# pathway_gene_merged.rename(columns = {'KEGG_PathwayID': 'kegg_pathway_id', 'KEGG_PathwayName': 'kegg_pathway_name',
#                                       'hgnc_id': 'gene_id', 'hgnc_symbol': 'gene_name'},
#                            inplace = True)
#
# pathway_gene_merged.drop_duplicates(inplace = True)
#
# pathway_gene_merged.sort_values('kegg_pathway_id', inplace = True)
#
# pathway_gene_merged.to_pickle("dataset/pathway_gene_merged.pkl")
# %%
pathway_gene_merged = pd.read_pickle("dataset/pathway_gene_merged.pkl")
# %%
drug_cell_pairs_raw = pd.read_csv("dataset/GDSC_338K_drug_response_and_cansyl_drug_cell_line_pairs_concatenated.txt",
                                  sep = '\t')
drug_cell_pairs_raw.rename(columns = {'DRUG_NAME': 'drug_name', 'CELL_LINE_NAME': 'cell_line_name', 'pIC50': 'pic50'},
                           inplace = True)
# drug_cell_pairs_raw["smiles"] = np.nan
# %%
drug_smiles_raw = pd.read_csv("dataset/GDSC_drug_name_SMILES_and_ECFP4_v2.txt", sep = '\t')
drug_smiles_raw.rename(columns = {'DRUG_NAME': 'drug_name', 'SMILES': 'smiles', 'ECFP4': 'ecfp4'}, inplace = True)
# %%
l1000_gene_list_raw = pd.read_csv("dataset/L1000_gene_list.txt", sep = '\t')
l1000_gene_list = l1000_gene_list_raw['pr_gene_symbol'].to_list()
# %%
first_table = pd.merge(drug_cell_pairs_raw, drug_smiles_raw, how = 'outer')

# %%
filenames = next(walk("dataset/full"), (None, None, []))[2]
cell_lines_list = []
for i in tqdm(filenames):
    name = i.split("_")[0]
    cell_line_features_raw = pd.read_csv("dataset/full/" + i, sep = '\t')
    cell_line_features_raw.drop(columns = ['gene_name'], inplace = True)
    cell_lines_list.append({'cell_line_name': name,
                            'cell_line_features': cell_line_features_raw})

cell_line_features = pd.DataFrame(cell_lines_list)
# %%
# L1000
filenames = next(walk("dataset/full"), (None, None, []))[2]
cell_lines_list = []
for i in tqdm(filenames):
    name = i.split("_")[0]
    cell_line_features_raw = pd.read_csv("dataset/full/" + i, sep = '\t')
    cell_line_features_raw = cell_line_features_raw.query("gene_name in @l1000_gene_list")
    cell_line_features_raw.reset_index(inplace=True, drop= True)
    cell_line_features_raw.drop(columns = ['gene_name'], inplace = True)
    cell_lines_list.append({'cell_line_name': name,
                            'cell_line_features': cell_line_features_raw})

cell_line_features_l1000 = pd.DataFrame(cell_lines_list)
# %%
# Pathway Sorted
filenames = next(walk("dataset/full"), (None, None, []))[2]
cell_lines_list = []
for i in tqdm(filenames):
    name = i.split("_")[0]
    cell_line_features_raw = pd.read_csv("dataset/full/" + i, sep = '\t')
    cell_line_features_df = cell_line_features_raw.set_index('gene_name')
    cell_line_features_df = cell_line_features_df.reindex(index=pathway_gene_merged['gene_name'])
    cell_line_features_df.reset_index(inplace = True)
    cell_line_features_df.dropna(inplace = True)
    gene_list = cell_line_features_df['gene_name'].to_list()
    genes_without_pathway = cell_line_features_raw.query('gene_name not in @gene_list')
    cell_line_features_df = pd.concat([cell_line_features_df, genes_without_pathway])
    cell_line_features_df.drop(columns = ['gene_name'], inplace = True)
    cell_lines_list.append({'cell_line_name': name,
                            'cell_line_features': cell_line_features_df})

cell_line_features_pathway_sorted = pd.DataFrame(cell_lines_list)

# %%
last_table_raw = pd.merge(first_table, cell_line_features, how = 'outer')
last_table = last_table_raw.drop("ecfp4", axis = 1)

last_table_raw_l1000 = pd.merge(first_table, cell_line_features_l1000, how = 'outer')
last_table_l1000 = last_table_raw_l1000.drop("ecfp4", axis = 1)

last_table_raw_pathway_sorted = pd.merge(first_table, cell_line_features_pathway_sorted, how = 'outer')
last_table_pathway_sorted = last_table_raw_pathway_sorted.drop("ecfp4", axis = 1)
# %%
last_table = last_table.dropna()
last_table_l1000 = last_table_l1000.dropna()
last_table_pathway_sorted = last_table_pathway_sorted.dropna()


# %%
last_table.to_pickle('burakcan_dataset.pkl')
last_table_l1000.to_pickle('burakcan_dataset_l1000.pkl')
last_table_pathway_sorted.to_pickle('burakcan_dataset_pathway_sorted.pkl')
