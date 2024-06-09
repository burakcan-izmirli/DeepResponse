import pandas as pd


def rename_columns(columns):
    """Function to rename columns by removing parentheses and number."""
    return [col.split(' ')[0] for col in columns]


gene_expression = pd.read_csv("raw/OmicsExpressionProteinCodingGenesTPMLogp1.csv")
gene_expression = gene_expression.rename(columns={'Unnamed: 0': 'cell_line_name'})

gene_expression.columns = rename_columns(gene_expression.columns)
# %%
# mutation = pd.read_csv("raw/OmicsSomaticMutations.csv")
# mutation.rename(columns={'Unnamed: 0': 'cell_line_name'}, inplace=True)
#
# mutation.columns = rename_columns(gene_expression.columns)
# %%
crispr = pd.read_csv("raw/CRISPRGeneDependency.csv")
crispr.rename(columns={'Unnamed: 0': 'cell_line_name'}, inplace=True)

crispr.columns = rename_columns(crispr.columns)
# %%
copy_number_variation = pd.read_csv("raw/CRISPRGeneDependency.csv")
copy_number_variation.rename(columns={'Unnamed: 0': 'cell_line_name'}, inplace=True)

copy_number_variation.columns = rename_columns(copy_number_variation.columns)
# %%
# Merging DataFrames
merged_df = gene_expression[['cell_line_name']].copy()
merged_df['cell_line_features'] = merged_df['cell_line_name'].apply(
    lambda x: pd.DataFrame(columns=['gene_name', 'gene_expression', 'crispr', 'copy_number_variation'])
)

for index, row in merged_df.iterrows():
    cell_line = row['cell_line_name']

    # Extracting the row for each cell line from all three dataframes
    row_gene_expression = gene_expression[gene_expression['cell_line_name'] == cell_line].drop(columns=['cell_line_name'])
    row_crispr = crispr[crispr['cell_line_name'] == cell_line].drop(columns=['cell_line_name'])
    row_copy_number_variation = copy_number_variation[copy_number_variation['cell_line_name'] == cell_line].drop(columns=['cell_line_name'])

    # Aligning the gene names using outer join
    combined = pd.concat([row_gene_expression, row_crispr, row_copy_number_variation], keys=['gene_expression', 'crispr', 'copy_number_variation'], axis=1)
    combined = combined.fillna(float('nan'))

    # Creating a new dataframe for cell line features and flattening the arrays
    cell_line_features = pd.DataFrame({
        'gene_name': combined.index,
        'gene_expression': combined['gene_expression'].values.flatten(),
        'crispr': combined['crispr'].values.flatten(),
        'copy_number_variation': combined['copy_number_variation'].values.flatten()
    })

    # Assigning the features dataframe to the corresponding cell line
    merged_df.at[index, 'cell_line_features'] = cell_line_features



#%%
