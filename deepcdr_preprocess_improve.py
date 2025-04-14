import tensorflow as tf
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import scipy.sparse as sp
from rdkit import Chem
import deepchem as dc
import os
import sys
from pathlib import Path
from typing import Dict
import joblib
from sklearn.preprocessing import StandardScaler
#from pympler import asizeof
#import h5py
from numpy.lib.format import open_memmap
import time
import gc
import psutil

# # device ID
# os.environ["CUDA_VISIBLE_DEVICES"] = "7"

# [Req] IMPROVE imports
# Core improvelib imports
from improvelib.applications.drug_response_prediction.config import DRPPreprocessConfig
from improvelib.utils import str2bool
import improvelib.utils as frm
# Application-specific (DRP) imports
import improvelib.applications.drug_response_prediction.drug_utils as drugs_utils
import improvelib.applications.drug_response_prediction.omics_utils as omics_utils
import improvelib.applications.drug_response_prediction.drp_utils as drp

# Model-specific imports
from model_params_def import preprocess_params # [Req]

filepath = Path(__file__).resolve().parent

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
print(f"Total CPUs available: {os.cpu_count()}")

def process_chunk(drug_ids, dict_features, dict_adj_mat):
    # Preallocate arrays
    num_drugs = len(drug_ids)
    feature_shape = next(iter(dict_features.values())).shape
    adj_shape = next(iter(dict_adj_mat.values())).shape

    gcn_feats_chunk = np.zeros((num_drugs, *feature_shape), dtype=np.float16)
    adj_list_chunk = np.zeros((num_drugs, *adj_shape), dtype=np.float16)

    # Fill preallocated arrays
    for i, drug_id in enumerate(drug_ids):
        gcn_feats_chunk[i] = dict_features[drug_id]
        adj_list_chunk[i] = dict_adj_mat[drug_id]

    return gcn_feats_chunk, adj_list_chunk


# Function to monitor memory usage
def log_memory_usage():
    process = psutil.Process()
    memory_info = process.memory_info()
    print(f"Memory usage: {memory_info.rss / (1024 ** 3):.2f} GB")  # Resident memory in GB

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

def CalculateGraphFeat(feat_mat,adj_list, Max_atoms, israndom = False):
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

# [Req]
def run(params: Dict):
    """ Execute data pre-processing for GraphDRP model.

    :params: Dict params: A dictionary of CANDLE/IMPROVE keywords and parsed values.
    """
    # ----------------------------------------
    # [Req] Load omics data - and set index
    # ---------------------
    start_time = time.time()

    print("\nLoad omics data ...")
    omics_start = time.time()

    omics_obj = omics_utils.OmicsLoader(params)
    ge = omics_obj.dfs['cancer_gene_expression.tsv'] 
    ge = ge.set_index('improve_sample_id')
    mut = omics_obj.dfs['cancer_mutation_count.tsv'] 
    mut = mut.set_index('improve_sample_id')
    methyl = omics_obj.dfs['cancer_DNA_methylation.tsv']
    methyl = methyl.set_index('improve_sample_id')

    # impute missing values in methylation
    methyl = methyl.replace('     NA', np.nan)
    methyl = methyl.astype("float64")
    methyl = methyl.fillna(methyl.mean())

    # get the embedding models
    cancer_gen_expr_model = get_emb_models(ge, norm = True)
    cancer_gen_mut_model = get_emb_models(mut, norm = True)
    cancer_dna_methy_model = get_emb_models(methyl, norm = True)

    # Save the models -  these will get saved in the exp_result folder
    cancer_gen_expr_model.save(os.path.join(params["output_dir"], "cancer_gen_expr_model"))
    cancer_gen_mut_model.save(os.path.join(params["output_dir"],"cancer_gen_mut_model"))
    cancer_dna_methy_model.save(os.path.join(params["output_dir"], "cancer_dna_methy_model"))
    print(f"Loaded and processed omics data in {(time.time() - omics_start) / 60:.2f} minutes")
    # ------------------------------------------------------
    # [Req] Load drug data
    # ------------------------------------------------------
    print("\nLoad drugs data...")
    drugs_start = time.time()

    drugs_obj = drugs_utils.DrugsLoader(params)
    smi = drugs_obj.dfs['drug_SMILES.tsv']  # get only the SMILES data
    # --------------------

    # reset index of the smiles file
    all_smiles = smi.reset_index()

    # rename the columns
    all_smiles.columns = ['improve_chem_id', 'canSMILES']

    # get the maximum number of atoms
    atom_list = []
    for i, smiles in enumerate(all_smiles["canSMILES"].values):
        molecules=[]
        molecules.append(Chem.MolFromSmiles(smiles))
        featurizer = dc.feat.graph_features.ConvMolFeaturizer()
        mol_object = featurizer.featurize(molecules)
        features = mol_object[0].atom_features
        atom_list.append(features.shape[0])

    Max_atoms = np.max(atom_list)

    dict_features = {}
    dict_adj_mat = {}
    for i, smiles in enumerate(all_smiles["canSMILES"].values):
    # print(each)
        molecules=[]
        molecules.append(Chem.MolFromSmiles(smiles))
        featurizer = dc.feat.graph_features.ConvMolFeaturizer()
        mol_object = featurizer.featurize(molecules)
        features = mol_object[0].atom_features
        drug_id_cur = all_smiles.iloc[i,:]["improve_chem_id"]
        adj_list = mol_object[0].canon_adj_list
        l = CalculateGraphFeat(features,adj_list, Max_atoms, israndom = False)
        dict_features[str(drug_id_cur)] = l[0].astype("float16")
        dict_adj_mat[str(drug_id_cur)] = l[1].astype("float16")

    # save the features and adjacency matrices
    with open(os.path.join(params["output_dir"], "drug_features.pickle"), "wb") as f:
        pickle.dump(dict_features, f)

    with open(os.path.join(params["output_dir"], "norm_adj_mat.pickle"), "wb") as f:
        pickle.dump(dict_adj_mat, f)
    
    print(f"Loaded drug data and prepared features and adjacency matrics in {(time.time() - drugs_start) / 60:.2f} minutes")

    # -------------------------------------------
    # Construct ML data for every stage (train, val, test)
    # [Req] All models must load response data (y data) using DrugResponseLoader().
    # -------------------------------------------
    construct_start = time.time()

    stages = {"train": params["train_split_file"],
              "val": params["val_split_file"],
              "test": params["test_split_file"]}

    for stage, split_file in stages.items():
        
        stage_start = time.time()
        
        # ------------------------
        # [Req] Load response data
        # ------------------------
        
        rsp = drp.DrugResponseLoader(params,
                                     split_file=split_file,
                                     verbose=False).dfs["response.tsv"]


        # keep only the required columns in the dataframe
        rsp = rsp[['improve_sample_id', 'improve_chem_id', 'auc']]
        # ------------------------
        # -----------------------
        # [Req] Save ML data files in params["ml_data_outdir"]
        # The implementation of this step, depends on the model.
        # -----------------------
        # Give a name to the response file
        data_fname = frm.build_ml_data_file_name(data_format=params["data_format"], stage=stage)

        # [Req] Save y dataframe for the current stage
        frm.save_stage_ydf(ydf=rsp, stage=stage, output_dir=params["output_dir"])

        # Initialize memmap files
        num_drugs = len(rsp["improve_chem_id"].values)
        gcn_feat_shape = dict_features[next(iter(dict_features))].shape
        adj_mat_shape = dict_adj_mat[next(iter(dict_adj_mat))].shape

        # Explicitly set dtype for memmap
        gcn_feats_dtype = dict_features[next(iter(dict_features))].dtype  # Get the dtype of the features
        adj_mat_dtype = dict_adj_mat[next(iter(dict_adj_mat))].dtype      # Get the dtype of the adjacency matrix

        print("Number of drugs:", num_drugs)
        print("GCN Feature Shape:", gcn_feat_shape)
        print("GCN Feature Dtype:", gcn_feats_dtype)
        print("Adj Matrix Shape:", adj_mat_shape)
        print("Adj Matrix Dtype:", adj_mat_dtype)

        gcn_feats_memmap = open_memmap(
            os.path.join(params["output_dir"], f'{stage}_drug_features.npy'),
            dtype=gcn_feats_dtype, mode='w+', shape=(num_drugs, *gcn_feat_shape)
        )

        adj_list_memmap = open_memmap(
            os.path.join(params["output_dir"], f'{stage}_norm_adj_mat.npy'),
            dtype=adj_mat_dtype, mode='w+', shape=(num_drugs, *adj_mat_shape)
        )

        # Parallel processing using ProcessPoolExecutor
        chunk_size = params["chunk_size"]  # Adjust chunk size based on memory capacity
        drug_ids = rsp["improve_chem_id"].values
        chunks = [drug_ids[i:i + chunk_size] for i in range(0, num_drugs, chunk_size)]
        total_chunks = (num_drugs + chunk_size - 1) // chunk_size
        print(f"Total chunks: {total_chunks}")

        # Monitor memory usage at the start
        log_memory_usage()

        completed_chunks = 0
        flush_batch_size = 10  # Reopen the memmap every x chunks

        for idx, chunk in enumerate(chunks):
            start_idx = idx * chunk_size
            end_idx = min(start_idx + chunk_size, num_drugs)

            # Process the chunk
            gcn_chunk, adj_chunk = process_chunk(chunk, dict_features, dict_adj_mat)

            # Write processed data to memmap
            gcn_feats_memmap[start_idx:end_idx] = gcn_chunk
            adj_list_memmap[start_idx:end_idx] = adj_chunk

            # Update and print progress
            completed_chunks += 1
            progress = (completed_chunks / total_chunks) * 100
            print(f"Progress: {progress:.2f}% ({completed_chunks}/{total_chunks} chunks completed)")
            log_memory_usage()

            # Reopen memmap files every `flush_batch_size` chunks to release memory
            if completed_chunks % flush_batch_size == 0:
                print(f"Flushing and reopening memmap after {completed_chunks} chunks.")
                gcn_feats_memmap.flush()
                adj_list_memmap.flush()

                # Explicitly close and reopen to release memory
                del gcn_feats_memmap
                del adj_list_memmap
                gc.collect()

                gcn_feats_memmap = open_memmap(
                    os.path.join(params["output_dir"], f'{stage}_drug_features.npy'),
                    dtype=gcn_feats_dtype, mode='r+'
                )
                adj_list_memmap = open_memmap(
                    os.path.join(params["output_dir"], f'{stage}_norm_adj_mat.npy'),
                    dtype=adj_mat_dtype, mode='r+'
                )

            # Clear intermediate data to free memory
            del gcn_chunk, adj_chunk
            gc.collect()

        # Final flush after all chunks are processed
        gcn_feats_memmap.flush()
        adj_list_memmap.flush()
        print("Final data flush completed.")

        # Clean up memmap objects
        del gcn_feats_memmap
        del adj_list_memmap
        gc.collect()

        print(f"Processed stage '{stage}' in {(time.time() - stage_start) / 60:.2f} minutes")
        print("\nMemmap-based processing completed.")
        log_memory_usage()
    
    print(f"Constructed response and memmap files in {(time.time() - construct_start) / 60:.2f} minutes")
    print(f"Total execution time: {(time.time() - start_time) / 60:.2f} minutes")

    return params["output_dir"]

# [Req]
def main(args):
    # [Req]
    additional_definitions = preprocess_params
    cfg = DRPPreprocessConfig()
    params = cfg.initialize_parameters(
        pathToModelDir=filepath,
        default_config="deepcdr_params.txt",
        additional_definitions=additional_definitions
    )
    ml_data_outdir = run(params)
    print("\nFinished data preprocessing.")

# [Req]
if __name__ == "__main__":
    main(sys.argv[1:])