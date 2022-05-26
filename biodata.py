import pandas as pd
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
#%%
a = pd.read_csv('full/8-MG-BA_specific_16501_genes_4_features_df.txt', sep='\t')
#%%
b = pd.read_csv('full/22RV1_specific_16501_genes_4_features_df.txt', sep='\t')
#%%
# c = pd.read_csv('GDSC_cell_line_feature_vectors_986_extracted_cell_lines_66K_rows.txt', sep='\t')
#%%
d = pd.read_csv('GDSC_338K_drug_response_and_cansyl_drug_cell_line_pairs_concatenated_for_Navid.txt', sep='\t')
# #%%
e = pd.read_csv('GDSC_drug_name_SMILES_and_ECFP4_v2.txt', sep='\t')
#%%


model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.summary()


# 1,4 filtreler
#
# 5-6 hidden layer
#
# mlp için 2-3 hidden layer
#
# output regression layer
#
# relu-leakyrelu
#
# gen ifadesi exp
#
# batch size = 64/128/256
#
#
# drop

#%%
