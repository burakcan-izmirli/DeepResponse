import numpy as np
import pandas as pd
import re

def rename_columns(columns):
    """Function to rename columns by removing parentheses and number."""
    return [col.split(' ')[0] for col in columns]


# Function to extract the first part before the first underscore
def extract_first_part(column_name):
    return column_name.split('_')[0]

def clean_string(s):
    """
    Clean the input string by removing non-alphanumeric characters,
    converting to lowercase, and stripping all whitespace.
    """
    s = str(s)
    s = re.sub(r'\W+', ' ', s)  # Replace non-alphanumeric characters with space
    s = s.lower()
    s = re.sub(r'\s+', '', s)  # Remove all whitespace
    return s

# %%
def load_and_process_data():
    # Load data
    gene_expression = pd.read_csv("raw/OmicsExpressionProteinCodingGenesTPMLogp1.csv")
    gene_expression = gene_expression.rename(columns={'Unnamed: 0': 'cell_line_name'})
    gene_expression.columns = rename_columns(gene_expression.columns)

    crispr = pd.read_csv("raw/CRISPRGeneDependency.csv")
    crispr.rename(columns={'Unnamed: 0': 'cell_line_name'}, inplace=True)
    crispr.columns = rename_columns(crispr.columns)

    copy_number_variation = pd.read_csv("raw/OmicsCNGene.csv")
    copy_number_variation.rename(columns={'Unnamed: 0': 'cell_line_name'}, inplace=True)
    copy_number_variation.columns = rename_columns(copy_number_variation.columns)

    mutation = pd.read_csv("raw/OmicsSomaticMutations.csv", low_memory=False)
    mutation = mutation.rename(columns={'ModelID': 'cell_line_name', 'LofGeneName': 'gene_name'})
    mutation = mutation[['cell_line_name', 'gene_name']]

    methylation = pd.read_csv('raw/Methylation_(1kb_upstream_TSS)_subsetted.csv', low_memory=False)
    methylation = methylation.drop(
        columns=['cell_line_display_name', 'lineage_1', 'lineage_2', 'lineage_3', 'lineage_5',
                 'lineage_6', 'lineage_4'])
    methylation.rename(columns={'depmap_id': 'cell_line_name'}, inplace=True)

    methylation.columns = ['cell_line_name'] + [extract_first_part(col) for col in methylation.columns[1:]]
    methylation = methylation.loc[:, ~methylation.columns.duplicated()]

    def merge_cell_line_data(cell_line_name):
        # Extract rows for the cell line from each dataframe
        row_gene_expression = gene_expression[gene_expression['cell_line_name'] == cell_line_name].drop(
            columns=['cell_line_name']).T
        row_crispr = crispr[crispr['cell_line_name'] == cell_line_name].drop(columns=['cell_line_name']).T
        row_copy_number_variation = copy_number_variation[
            copy_number_variation['cell_line_name'] == cell_line_name].drop(columns=['cell_line_name']).T
        row_mutation = mutation[mutation['cell_line_name'] == cell_line_name].dropna()
        row_methylation = methylation[methylation['cell_line_name'] == cell_line_name].drop(
            columns=['cell_line_name']).T

        if not row_gene_expression.empty:
            row_gene_expression.columns = ['gene_expression']
        else:
            row_gene_expression['gene_expression'] = pd.Series(index=row_gene_expression.index,
                                                               data=[np.nan] * len(row_gene_expression))

        if not row_crispr.empty:
            row_crispr.columns = ['crispr']
        else:
            row_crispr['crispr'] = pd.Series(index=row_crispr.index, data=[np.nan] * len(row_crispr))

        if not row_copy_number_variation.empty:
            row_copy_number_variation.columns = ['copy_number_variation']
        else:
            row_copy_number_variation['copy_number_variation'] = pd.Series(index=row_copy_number_variation.index,
                                                                           data=[np.nan] * len(
                                                                               row_copy_number_variation))
        if not row_methylation.empty:
            row_methylation.columns = ['methylation']
        else:
            row_methylation['methylation'] = pd.Series(index=row_methylation.index,
                                                       data=[np.nan] * len(row_methylation))

        # Merge the dataframes on the index
        merged = row_gene_expression.join(row_crispr, how='outer').join(row_copy_number_variation, how='outer'). \
            join(row_methylation, how='outer')

        # Create mutation column
        mutation_col = pd.DataFrame(index=merged.index, data={'mutation': [0] * len(merged)})
        if not row_mutation.empty:
            mutated_genes = row_mutation['gene_name'].tolist()
            mutation_col[merged.index.isin(mutated_genes)] = 1
        merged = merged.join(mutation_col, how='outer')
        merged = merged.reset_index().rename(columns={'index': 'gene_name'})

        return merged

    # Get a set of all cell line names from all datasets
    all_cell_lines = set(gene_expression['cell_line_name']).union(set(crispr['cell_line_name']),
                                                                  set(copy_number_variation['cell_line_name']),
                                                                  set(mutation['cell_line_name']),
                                                                  set(methylation['cell_line_name']))

    # Merge data for all cell lines
    merged_df = pd.DataFrame({'cell_line_name': list(all_cell_lines)})
    merged_df['cell_line_features'] = merged_df['cell_line_name'].apply(
        lambda cell_line: merge_cell_line_data(cell_line))

    return merged_df


# Example usage
cell_line_features_raw = load_and_process_data()
# %%
# Define the function to filter genes
def filter_genes(df):
    gene_dataframes = [cell_line_features for cell_line_features in df['cell_line_features'] if
                       isinstance(cell_line_features, pd.DataFrame)]
    genes_to_drop = pd.Index([])
    omics_categories = ['gene_expression', 'mutation', 'crispr', 'copy_number_variation', 'methylation']

    combined_genes_raw = pd.concat(gene_dataframes, axis=1)
    combined_genes_raw = combined_genes_raw.set_index('gene_name')
    combined_genes_raw.index = [tup[0] if len(set(tup)) == 1 else tup for tup in combined_genes_raw.index]
    combined_genes_raw = combined_genes_raw.reset_index()
    for omic in omics_categories:
        combined_genes = combined_genes_raw[omic]
        overall_missing_percentage = combined_genes.isna().mean(axis=1)
        genes_to_drop = genes_to_drop.union(overall_missing_percentage[overall_missing_percentage > 0.6].index)

    def drop_genes(nested_df):
        if isinstance(nested_df, pd.DataFrame):
            return nested_df.drop(index=genes_to_drop, errors='ignore').reset_index(drop=True)
        return nested_df.reset_index(drop=True)

    filtered_df = df.copy()
    filtered_df['cell_line_features'] = df['cell_line_features'].apply(drop_genes)
    return filtered_df


cell_line_features = filter_genes(cell_line_features_raw)


# %%

def filter_dataframe(df):
    def is_valid_nested_df(nested_df):
        missing_percentage = nested_df.isna().mean()
        if (missing_percentage > 0.5).any():
            return False
        return True

    filtered_df = df[df['cell_line_features'].apply(is_valid_nested_df)].reset_index(drop=True)
    return filtered_df


cell_line_features_f = filter_dataframe(cell_line_features)


def remove_cell_line_name_column(nested_df):
    if isinstance(nested_df, pd.DataFrame):
        return nested_df.loc[:, ~nested_df.columns.str.contains('cell_line_name')]
    return nested_df


cell_line_features_f['cell_line_features'] = cell_line_features_f['cell_line_features'].apply(
    remove_cell_line_name_column)


# Impute missing values with the median of the corresponding gene across all cell lines
def impute_missing_values(df):
    omics_categories = ['gene_expression', 'mutation', 'crispr', 'copy_number_variation', 'methylation']

    # Calculate medians separately for each omics category
    medians = {}
    for omics in omics_categories:
        all_genes = pd.concat(
            [nested_df[omics] for nested_df in df['cell_line_features'] if isinstance(nested_df, pd.DataFrame)], axis=1)
        gene_medians = all_genes.median(axis=1)
        medians[omics] = gene_medians


def impute_missing_values(df):
    omics_categories = ['gene_expression', 'mutation', 'crispr', 'copy_number_variation', 'methylation']

    medians_df = pd.DataFrame()

    # Calculate medians separately for each omics category and store in the DataFrame
    for omics in omics_categories:
        all_genes = pd.concat(
            [nested_df.set_index('gene_name')[omics] for nested_df in df['cell_line_features'] if
             isinstance(nested_df, pd.DataFrame)],
            axis=1
        )
        gene_medians = all_genes.median(axis=1)
        medians_df[omics] = gene_medians

    def fill_missing(nested_df):
        if isinstance(nested_df, pd.DataFrame):
            nested_df = nested_df.set_index('gene_name')
            for gene_name in nested_df.index:
                for omics in omics_categories:
                    if pd.isna(nested_df.at[gene_name, omics]):
                        if gene_name in medians_df[omics]:
                            nested_df.at[gene_name, omics] = medians_df[omics][gene_name]
        return nested_df.reset_index()

    df['cell_line_features'] = df['cell_line_features'].apply(fill_missing)
    return df


# Apply the imputation
cell_line_features_f_imputed = impute_missing_values(cell_line_features_f)

# %%
def check_na_in_nested_df(nested_df):
    return nested_df.isna().sum()


# Apply the function to the 'cell_line_features' column
na_counts = cell_line_features_f_imputed['cell_line_features'].apply(check_na_in_nested_df)
# %%

for index, row in cell_line_features_f_imputed.iterrows():
    features_df = row['cell_line_features']
    features_df = features_df.drop(columns=['gene_name'])
    features_array = features_df.to_numpy(dtype=np.float16)
    cell_line_features_f_imputed.at[index, 'cell_line_features'] = features_array



#%%
# cell_line_features_f_imputed.to_pickle('processed/cell_line_features_f_imputed.pkl')
cell_line_features = pd.read_pickle('processed/cell_line_features_f_imputed.pkl')


ic50_df_raw = pd.read_csv('raw/sanger-dose-response.csv')

ic50_df_raw = ic50_df_raw[['DATASET', 'DRUG_ID', 'DRUG_NAME', 'ARXSPAN_ID', 'IC50_PUBLISHED']]

ic50_df_raw = ic50_df_raw.rename(columns={'DATASET': 'dataset',
                                          'DRUG_ID': 'drug_id',
                                          'DRUG_NAME': 'drug_name',
                                          'ARXSPAN_ID': 'cell_line_name',
                                          'IC50_PUBLISHED': 'ic50'})
# %%
# Create a custom sort order
ic50_df_raw['sort_order'] = ic50_df_raw['dataset'].apply(lambda x: 0 if x == 'GDSC2' else 1)

# Sort the dataframe
ic50_df_raw = ic50_df_raw.sort_values(by=['drug_name', 'cell_line_name', 'sort_order'])

# Drop duplicates
ic50_df = ic50_df_raw.drop_duplicates(subset=['drug_name', 'cell_line_name'], keep='first').drop(columns=['sort_order'])

ic50_df = ic50_df.drop(columns=['dataset', 'drug_id'])

ic50_df = ic50_df.dropna()

# Perform the transformation with unique ID assignment
expanded_ic50_df = (
    ic50_df.assign(unique_id=ic50_df.index)
    .set_index('unique_id')
    .apply(lambda row: pd.Series(row['drug_name'].split(','), name=row.name), axis=1)
    .stack()
    .reset_index(level=1, drop=True)
    .reset_index(name='drug_name')
    .merge(ic50_df, left_on='unique_id', right_index=True)
    .drop(columns=['drug_name_y'])
    .rename(columns={'drug_name_x': 'drug_name'})
)

expanded_ic50_df['drug_name'] = expanded_ic50_df['drug_name'].apply(clean_string)
# %%
# Read vocabulary and clean the drug names and synonyms
vocabulary = pd.read_csv('raw/drugbank_vocabulary.csv')[['Common name', 'Synonyms']]
vocabulary['Common name'] = vocabulary['Common name'].apply(clean_string)
vocabulary['Synonyms'] = vocabulary['Synonyms'].apply(
    lambda x: [clean_string(syn) for syn in x.split('|')] if pd.notna(x) else [])

# Create a dictionary for quick lookup of common names and their synonyms
combined_dict = {}
for common_name, synonyms in zip(vocabulary['Common name'], vocabulary['Synonyms']):
    combined_dict[common_name] = common_name
    for synonym in synonyms:
        combined_dict[synonym] = common_name


# Generate unique IDs
def generate_unique_id():
    generate_unique_id.counter += 1
    return generate_unique_id.counter


generate_unique_id.counter = 0

# Create a dictionary to hold the unique IDs for each common name
unique_id_v_dict = {}

# Create a list to store new rows
new_rows = []

for _, row in expanded_ic50_df.iterrows():
    drug_name = clean_string(row['drug_name'])
    cell_line_name = row['cell_line_name']
    ic50 = row['ic50']
    unique_id = row['unique_id']

    # Find the common name for the drug
    found_common_name = combined_dict.get(drug_name, drug_name)

    # If the common name does not already have a unique ID, generate one
    if found_common_name not in unique_id_v_dict:
        unique_id_v = generate_unique_id()
        unique_id_v_dict[found_common_name] = unique_id_v

    # Get the unique ID for the current drug
    unique_id_v = unique_id_v_dict[found_common_name]

    # Add the original row with the unique ID to the new_rows list
    new_rows.append({'cell_line_name': cell_line_name, 'drug_name': drug_name, 'ic50': ic50, 'unique_id': unique_id,
                     'unique_id_v': unique_id_v})

    # Add new rows for each synonym with the same unique ID
    if found_common_name in combined_dict:
        for synonym in vocabulary[vocabulary['Common name'] == found_common_name]['Synonyms'].values[0]:
            new_rows.append(
                {'cell_line_name': cell_line_name, 'drug_name': synonym, 'ic50': ic50, 'unique_id': unique_id,
                 'unique_id_v': unique_id_v})

# Create a new DataFrame from the new_rows list
expanded_ic50_df_w_synonyms = pd.DataFrame(new_rows)


# %%
# Define a function to merge synonyms based on different separators
def merge_columns_to_synonyms(row):
    synonyms = []

    if pd.notna(row['cmpdsynonym']):
        synonyms.extend(row['cmpdsynonym'].split('|'))

    if pd.notna(row['inchi']):
        synonyms.extend(row['inchi'].split('/'))

    if pd.notna(row['iupacname']):
        synonyms.append(row['iupacname'])

    if pd.notna(row['meshheadings']):
        synonyms.append(row['meshheadings'])

    return '|'.join(set(synonyms))


vocabulary_extend = pd.read_csv('raw/vocab_extended.csv')[
    ['cmpdname', 'cmpdsynonym', 'inchi', 'iupacname', 'meshheadings']]

# Apply the function to create the synonyms column
vocabulary_extend['Synonyms'] = vocabulary_extend.apply(merge_columns_to_synonyms, axis=1)

# Create the final vocabulary DataFrame with 'Common name' and 'Synonyms'
vocabulary_extend = vocabulary_extend[['cmpdname', 'Synonyms']].rename(columns={'cmpdname': 'Common name'})

vocabulary_extend['Common name'] = vocabulary_extend['Common name'].apply(clean_string)
vocabulary_extend['Synonyms'] = vocabulary_extend['Synonyms'].apply(
    lambda x: [clean_string(syn) for syn in x.split('|')] if pd.notna(x) else [])

# Create a dictionary for quick lookup of common names and their synonyms
combined_dict = {}
for common_name, synonyms in zip(vocabulary_extend['Common name'], vocabulary_extend['Synonyms']):
    combined_dict[common_name] = common_name
    for synonym in synonyms:
        combined_dict[synonym] = common_name


# Generate unique IDs
def generate_unique_id():
    generate_unique_id.counter += 1
    return generate_unique_id.counter


generate_unique_id.counter = 0

# Create a dictionary to hold the unique IDs for each common name
unique_id_v_e_dict = {}

# Create a list to store new rows
new_rows = []

for _, row in expanded_ic50_df_w_synonyms.iterrows():
    drug_name = clean_string(row['drug_name'])
    cell_line_name = row['cell_line_name']
    ic50 = row['ic50']
    unique_id = row['unique_id']
    unique_id_v = row['unique_id_v']

    # Find the common name for the drug
    found_common_name = combined_dict.get(drug_name, drug_name)

    # If the common name does not already have a unique ID, generate one
    if found_common_name not in unique_id_v_e_dict:
        unique_id_v_e = generate_unique_id()
        unique_id_v_e_dict[found_common_name] = unique_id_v_e

    # Get the unique ID for the current drug
    unique_id_v_e = unique_id_v_e_dict[found_common_name]

    # Add the original row with the unique ID to the new_rows list
    new_rows.append({'cell_line_name': cell_line_name, 'drug_name': drug_name, 'ic50': ic50, 'unique_id': unique_id,
                     'unique_id_v': unique_id_v, 'unique_id_v_e': unique_id_v_e})

    # Add new rows for each synonym with the same unique ID
    if found_common_name in combined_dict:
        for synonym in vocabulary_extend[vocabulary_extend['Common name'] == found_common_name]['Synonyms'].values[0]:
            new_rows.append(
                {'cell_line_name': cell_line_name, 'drug_name': synonym, 'ic50': ic50, 'unique_id': unique_id,
                 'unique_id_v': unique_id_v, 'unique_id_v_e': unique_id_v_e})

# Create a new DataFrame from the new_rows list
expanded_ic50_df_w_synonyms_extended = pd.DataFrame(new_rows)

# %%
last_dataset = pd.merge(cell_line_features, expanded_ic50_df_w_synonyms_extended, how='outer')

smiles_extended = pd.read_csv('raw/vocab_extended.csv')[['cmpdname', 'canonicalsmiles']]
smiles_extended = smiles_extended.rename(columns={'cmpdname': 'drug_name', 'canonicalsmiles': 'smiles'})
smiles_extended['drug_name'] = smiles_extended['drug_name'].apply(clean_string)
# %%
last_dataset = pd.merge(last_dataset, smiles_extended, how='left')
last_dataset = last_dataset.dropna()
# %%
last_dataset = last_dataset.drop_duplicates(subset=['cell_line_name', 'unique_id', 'unique_id_v', 'unique_id_v_e'])
# %%
last_dataset.drop(columns=['unique_id', 'unique_id_v', 'unique_id_v_e'], inplace=True)

last_dataset['pic50'] = -np.log10(last_dataset['ic50'] * 1e-6)

last_dataset = last_dataset.drop(columns=['ic50'])
#%%


#%%
