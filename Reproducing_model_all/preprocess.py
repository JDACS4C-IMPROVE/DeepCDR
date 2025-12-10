import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
print(physical_devices)
# try:
#     tf.config.experimental.set_memory_growth(physical_devices[0], True)
#     print(physical_devices[0], flush = True)
    
# except:
#   # Invalid device or cannot modify virtual devices once initialized.
#   pass

import pickle
import numpy as np
import matplotlib.pyplot as plt
import improve_utils as iu
import pandas as pd
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
import scipy.sparse as sp
from rdkit import Chem
import deepchem as dc



# import the data - gene expression and methylation are easy
df_ge = iu.load_gene_expression_data(gene_system_identifier="Gene_Symbol")
df_mut = iu.load_mutation_count_data(gene_system_identifier="Gene_Symbol")
df_methy = iu.load_dna_methylation_data(gene_system_identifier="Gene_Symbol")

print(df_ge.shape)
print(df_mut.shape)
print(df_methy.shape)

# impute methylation missing values - missing values only in methylation
df_methy = df_methy.replace('     NA', np.nan)
df_methy = df_methy.astype("float64")

df_methy = df_methy.fillna(df_methy.mean())



def get_emb_models(dataset, norm = False):
    std = StandardScaler()
    unique_ids = dataset.index
    text_vec_layer = tf.keras.layers.TextVectorization(max_tokens = dataset.shape[0] + 2, 
                                                  standardize=None, split = None, 
                                                  output_mode = "int", 
                                                  vocabulary = unique_ids.tolist())
    # weights = dataset.drop(id_col, 1).values
    weights = dataset.values
    padding_zeros = np.zeros((2, weights.shape[1]))
    weights = np.vstack((padding_zeros, weights))
    if norm == True:
        std.fit(weights)
        weights = std.transform(weights)
    emb_layer = tf.keras.layers.Embedding(dataset.shape[0] + 2, 
                                     weights.shape[1], 
                                     weights = [weights], 
                                     trainable = False)
    input_layer = tf.keras.layers.Input(shape = (1,), dtype = tf.string)
    vec_out = text_vec_layer(input_layer)
    emb_out = emb_layer(vec_out)
    flat_out = tf.keras.layers.Flatten()(emb_out)
    emb_model = tf.keras.models.Model(input_layer, flat_out)
    return emb_model

# get the embedding models
cancer_gen_expr_model = get_emb_models(df_ge, norm = True)
cancer_gen_mut_model = get_emb_models(df_mut, norm = True)
cancer_dna_methy_model = get_emb_models(df_methy, norm = True)

# Save the models -  in a folder named Models
cancer_gen_expr_model.save("Models//cancer_gen_expr_model")
cancer_gen_mut_model.save("Models//cancer_gen_mut_model")
cancer_dna_methy_model.save("Models//cancer_dna_methy_model")

all_smiles = iu.load_smiles_data()

def NormalizeAdj(adj):
    adj = adj + np.eye(adj.shape[0])
    d = sp.diags(np.power(np.array(adj.sum(1)), -0.5).flatten(), 0).toarray()
    a_norm = adj.dot(d).transpose().dot(d)
    return a_norm
def random_adjacency_matrix(n):   
    matrix = [[random.randint(0, 1) for i in range(n)] for j in range(n)]
    # No vertex connects to itself
    for i in range(n):
        matrix[i][i] = 0
    # If i is connected to j, j is connected to i
    for i in range(n):
        for j in range(n):
            matrix[j][i] = matrix[i][j]
    return matrix
def CalculateGraphFeat(feat_mat,adj_list, Max_atoms):
    assert feat_mat.shape[0] == len(adj_list)
    feat = np.zeros((Max_atoms,feat_mat.shape[-1]),dtype='float32')
    adj_mat = np.zeros((Max_atoms,Max_atoms),dtype='float32')
    if israndom:
        feat = np.random.rand(Max_atoms,feat_mat.shape[-1])
        adj_mat[feat_mat.shape[0]:,feat_mat.shape[0]:] = random_adjacency_matrix(Max_atoms-feat_mat.shape[0])        
    feat[:feat_mat.shape[0],:] = feat_mat
    for i in range(len(adj_list)):
        nodes = adj_list[i]
        for each in nodes:
            adj_mat[i,int(each)] = 1
    assert np.allclose(adj_mat,adj_mat.T)
    adj_ = adj_mat[:len(adj_list),:len(adj_list)]
    adj_2 = adj_mat[len(adj_list):,len(adj_list):]
    norm_adj_ = NormalizeAdj(adj_)
    norm_adj_2 = NormalizeAdj(adj_2)
    adj_mat[:len(adj_list),:len(adj_list)] = norm_adj_
    adj_mat[len(adj_list):,len(adj_list):] = norm_adj_2    
    return [feat,adj_mat]

atom_list = []
for i, smiles in enumerate(all_smiles["smiles"].values):
    # print(each)
    molecules=[]
    molecules.append(Chem.MolFromSmiles(smiles))
    featurizer = dc.feat.graph_features.ConvMolFeaturizer()
    mol_object = featurizer.featurize(molecules)
    features = mol_object[0].atom_features
    atom_list.append(features.shape[0])

Max_atoms = np.max(atom_list)

israndom = False

dict_features = {}
# dict_num_atoms = {}
dict_adj_mat = {}
for i, smiles in enumerate(all_smiles["smiles"].values):
    # print(each)
    molecules=[]
    molecules.append(Chem.MolFromSmiles(smiles))
    featurizer = dc.feat.graph_features.ConvMolFeaturizer()
    mol_object = featurizer.featurize(molecules)
    features = mol_object[0].atom_features
    drug_id_cur = all_smiles.iloc[i,:]["improve_chem_id"]
    adj_list = mol_object[0].canon_adj_list
    l = CalculateGraphFeat(features,adj_list, Max_atoms)
    dict_features[str(drug_id_cur)] = l[0]
    dict_adj_mat[str(drug_id_cur)] = l[1]

with open("csa_data//drug_features.pickle", "wb") as f:
    pickle.dump(dict_features, f)

with open("csa_data//norm_adj_mat.pickle", "wb") as f:
    pickle.dump(dict_adj_mat, f)


