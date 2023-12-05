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
# import matplotlib.pyplot as plt
import improve_utils as iu
import pandas as pd
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
import scipy.sparse as sp
from rdkit import Chem
import deepchem as dc
import os
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler, MaxAbsScaler, MinMaxScaler, RobustScaler

# IMPROVE imports
from improve import framework as frm
from improve import drug_resp_pred as drp

filepath = Path(__file__).resolve().parent
print(filepath)
print("ran!")


# [Req] App-specific params (App: monotherapy drug response prediction)
# TODO: consider moving this list to drug_resp_pred.py module
app_preproc_params = [
    # These arg should be specified in the [modelname]_default_model.txt:
    # y_data_files, x_data_canc_files, x_data_drug_files
    {"name": "y_data_files", # default
     "type": str,
     "help": "List of files that contain the y (prediction variable) data. \
             Example: [['response.tsv']]",
    },
    {"name": "x_data_canc_files", # [Req]
     "type": str,
     "help": "List of feature files including gene_system_identifer. Examples: \n\
             1) [['cancer_gene_expression.tsv', ['Gene_Symbol']]] \n\
             2) [['cancer_copy_number.tsv', ['Ensembl', 'Entrez']]].",
    },
    {"name": "x_data_drug_files", # [Req]
     "type": str,
     "help": "List of feature files. Examples: \n\
             1) [['drug_SMILES.tsv']] \n\
             2) [['drug_SMILES.tsv'], ['drug_ecfp4_nbits512.tsv']]",
    },
    {"name": "canc_col_name",
     "default": "improve_sample_id", # default
     "type": str,
     "help": "Column name in the y (response) data file that contains the cancer sample ids.",
    },
    {"name": "drug_col_name", # default
     "default": "improve_chem_id",
     "type": str,
     "help": "Column name in the y (response) data file that contains the drug ids.",
    },

]

preprocess_params = app_preproc_params
req_preprocess_args = [ll["name"] for ll in preprocess_params]

def run(params):
    """ Execute data pre-processing for GraphDRP model.

    :params: Dict params: A dictionary of CANDLE/IMPROVE keywords and parsed values.
    """

    # ------------------------------------------------------
    # [Req] Build paths and create ML data dir
    # ----------------------------------------
    # Build paths for raw_data, x_data, y_data, splits
    params = frm.build_paths(params)  

    # Create outdir for ML data (to save preprocessed data)
    processed_outdir = frm.create_outdir(outdir=params["ml_data_outdir"])
    # ----------------------------------------
    # [Req] Load omics data - and set index
    # ---------------------
    print("\nLoading omics data ...")
    oo = drp.OmicsLoader(params)
    ge = oo.dfs['cancer_gene_expression.tsv'] 
    ge = ge.set_index('improve_sample_id')
    mut = oo.dfs['cancer_mutation_count.tsv'] 
    mut = mut.set_index('improve_sample_id')
    methyl = oo.dfs['cancer_DNA_methylation.tsv']
    methyl = methyl.set_index('improve_sample_id')

    # impute missing values in methylation
    methyl = methyl.replace('     NA', np.nan)
    methyl = methyl.astype("float64")
    methyl = methyl.fillna(methyl.mean())
    # ------------------------------------------------------
    # [Req] Load drug data
    # --------------------
    print("\nLoading drugs data...")
    dd = drp.DrugsLoader(params)
    smi = dd.dfs['drug_SMILES.tsv']  # get only the SMILES data
    # --------------------

    # -------------------------------------------
    # Construct ML data for every stage (train, val, test)
    # [Req] All models must load response data (y data) using DrugResponseLoader().
    # -------------------------------------------
    stages = {"train": params["train_split_file"],
              "val": params["val_split_file"],
              "test": params["test_split_file"]}
    scaler = None

    for stage, split_file in stages.items():

        # ------------------------
        # [Req] Load response data
        # ------------------------
        
        rr = drp.DrugResponseLoader(params, split_file=split_file, verbose=True)
        df_response = rr.dfs["response.tsv"]

        # keep only the required columns in the dataframe
        df_response = df_response[['improve_sample_id', 'improve_chem_id', 'auc']]

        # print dataset shape and head
        print(df_response.shape)
        print(df_response.head())
        # ------------------------
        # -----------------------
        # [Req] Save ML data files in params["ml_data_outdir"]
        # The implementation of this step, depends on the model.
        # -----------------------
        # Give a name to the response file
        data_fname = frm.build_ml_data_name(params, stage)

        # # [Req] Save y dataframe for the current stage
        frm.save_stage_ydf(df_response, params, stage)

    return ge, mut, methyl, smi, processed_outdir

params = frm.initialize_parameters(
        filepath,
        default_model="deepcdr_default_model.txt",
        additional_definitions=preprocess_params,
        required=req_preprocess_args,
    )

df_ge, df_mut, df_methy, all_smiles, directory= run(params)

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
cancer_gen_expr_model.save(os.path.join(directory, "cancer_gen_expr_model"))
cancer_gen_mut_model.save(os.path.join(directory,"cancer_gen_mut_model"))
cancer_dna_methy_model.save(os.path.join(directory, "cancer_dna_methy_model"))

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


with open(os.path.join(directory, "drug_features.pickle"), "wb") as f:
    pickle.dump(dict_features, f)

with open(os.path.join(directory, "norm_adj_mat.pickle"), "wb") as f:
    pickle.dump(dict_adj_mat, f)