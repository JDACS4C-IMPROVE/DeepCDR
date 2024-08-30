from candle import CandleCkptPyTorch
from improve import framework as frm
from improve.metrics import compute_metrics
import os
import json
import sys
import warnings
from pathlib import Path
from pprint import pformat
from typing import Dict, Union
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
import pickle
import pandas as pd
import numpy as np
from tensorflow.keras import backend as K

filepath = Path(__file__).resolve().parent
print(filepath)

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

# [Req] List of metrics names to be compute performance scores
metrics_list = ["mse", "rmse", "pcc", "scc", "r2"]  

# [Req] App-specific params (App: monotherapy drug response prediction)
# Currently, there are no app-specific args for the train script.
app_train_params = []

def run(params):
    """ Execute specified model training.

    :params: Dict params: A dictionary of CANDLE/IMPROVE keywords and parsed values.

    :return: List of floats evaluating model predictions according to
             specified metrics_list.
    :rtype: float list
    """
    # import pdb; pdb.set_trace()

    # ------------------------------------------------------
    # [Req] Create output dir for the model. 
    # ------------------------------------------------------
    # import pdb; pdb.set_trace()
    # modelpath = frm.create_model_outpath(params)
    modelpath = frm.create_outdir(outdir=params["model_outdir"])
    # modelpath = frm.build_model_path(params, model_dir=params["model_outdir"])
    print(modelpath)

    # ------------------------------------------------------
    # [Req] Create data names for train and val
    # ------------------------------------------------------

    train_data_fname = frm.build_ml_data_name(params, stage="train")
    val_data_fname = frm.build_ml_data_name(params, stage="val")
    print(train_data_fname)
    print(val_data_fname)

    return modelpath


params = frm.initialize_parameters(
        filepath,
        default_model="deepcdr_default_model.txt",
        additional_definitions=preprocess_params,
        required=req_preprocess_args,
    )
model_path = run(params)

# import the preprocessed data
# specify the directory where preprocessed data is stored
data_dir = frm.build_model_path(params, model_dir=params["ml_data_outdir"])
data_dir = os.path.dirname(data_dir)
print(data_dir)

# load the models
cancer_gen_expr_model = tf.keras.models.load_model(os.path.join(data_dir,"cancer_gen_expr_model"))
cancer_gen_mut_model = tf.keras.models.load_model(os.path.join(data_dir, "cancer_gen_mut_model"))
cancer_dna_methy_model = tf.keras.models.load_model(os.path.join(data_dir, "cancer_dna_methy_model"))

cancer_gen_expr_model.trainable = False
cancer_gen_mut_model.trainable = False
cancer_dna_methy_model.trainable = False

# load the drug data
with open(os.path.join(data_dir, "drug_features.pickle"),"rb") as f:
    dict_features = pickle.load(f)

with open(os.path.join(data_dir, "norm_adj_mat.pickle"),"rb") as f:
    dict_adj_mat = pickle.load(f)

train_keep = pd.read_csv(os.path.join(data_dir, "train_y_data.csv"))
valid_keep = pd.read_csv(os.path.join(data_dir, "val_y_data.csv"))

train_keep.columns = ["Cell_Line", "Drug_ID", "AUC"]
valid_keep.columns = ["Cell_Line", "Drug_ID", "AUC"]

print(train_keep.head())
samp_drug = valid_keep["Drug_ID"].unique()[-1]
samp_ach = np.array(valid_keep["Cell_Line"].unique()[-1])

train_gcn_feats = []
train_adj_list = []
for drug_id in train_keep["Drug_ID"].values:
    train_gcn_feats.append(dict_features[drug_id])
    train_adj_list.append(dict_adj_mat[drug_id])

valid_gcn_feats = []
valid_adj_list = []
for drug_id in valid_keep["Drug_ID"].values:
    valid_gcn_feats.append(dict_features[drug_id])
    valid_adj_list.append(dict_adj_mat[drug_id])

train_gcn_feats = np.array(train_gcn_feats).astype("float16")
valid_gcn_feats = np.array(valid_gcn_feats).astype("float16")

train_adj_list = np.array(train_adj_list).astype("float16")
valid_adj_list = np.array(valid_adj_list).astype("float16")

training = False
dropout1 = 0.10
dropout2 = 0.20

## get the model architecture
def deepcdrgcn(training = training, dropout1 = dropout1, dropout2 = dropout2):
    
    input_gcn_features = tf.keras.layers.Input(shape = (dict_features[samp_drug].shape[0], 75))
    input_norm_adj_mat = tf.keras.layers.Input(shape = (dict_adj_mat[samp_drug].shape[0], dict_adj_mat[samp_drug].shape[0]))
    mult_1 = tf.keras.layers.Dot(1)([input_norm_adj_mat, input_gcn_features])
    dense_layer_gcn = tf.keras.layers.Dense(256, activation = "relu")
    dense_out = dense_layer_gcn(mult_1)
    dense_out = tf.keras.layers.BatchNormalization()(dense_out)
    dense_out = tf.keras.layers.Dropout(dropout1)(dense_out, training = training)
    mult_2 = tf.keras.layers.Dot(1)([input_norm_adj_mat, dense_out])
    dense_layer_gcn = tf.keras.layers.Dense(256, activation = "relu")
    dense_out = dense_layer_gcn(mult_2)
    dense_out = tf.keras.layers.BatchNormalization()(dense_out)
    dense_out = tf.keras.layers.Dropout(dropout1)(dense_out, training = training)

    dense_layer_gcn = tf.keras.layers.Dense(100, activation = "relu")
    mult_3 = tf.keras.layers.Dot(1)([input_norm_adj_mat, dense_out])
    dense_out = dense_layer_gcn(mult_3)
    dense_out = tf.keras.layers.BatchNormalization()(dense_out)
    dense_out = tf.keras.layers.Dropout(dropout1)(dense_out, training = training)

    dense_out = tf.keras.layers.GlobalAvgPool1D()(dense_out)
    # All above code is for GCN for drugs

    # methylation data
    input_gen_methy1 = tf.keras.layers.Input(shape = (1,), dtype = tf.string)
    input_gen_methy = cancer_dna_methy_model(input_gen_methy1)
    input_gen_methy.trainable = False
    gen_methy_layer = tf.keras.layers.Dense(256, activation = "tanh")
    
    gen_methy_emb = gen_methy_layer(input_gen_methy)
    gen_methy_emb = tf.keras.layers.BatchNormalization()(gen_methy_emb)
    gen_methy_emb = tf.keras.layers.Dropout(dropout1)(gen_methy_emb, training = training)
    gen_methy_layer = tf.keras.layers.Dense(100, activation = "relu")
    gen_methy_emb = gen_methy_layer(gen_methy_emb)

    # gene expression data
    input_gen_expr1 = tf.keras.layers.Input(shape = (1,), dtype = tf.string)
    input_gen_expr = cancer_gen_expr_model(input_gen_expr1)
    input_gen_expr.trainable = False
    gen_expr_layer = tf.keras.layers.Dense(256, activation = "tanh")
    
    gen_expr_emb = gen_expr_layer(input_gen_expr)
    gen_expr_emb = tf.keras.layers.BatchNormalization()(gen_expr_emb)
    gen_expr_emb = tf.keras.layers.Dropout(dropout1)(gen_expr_emb, training = training)
    gen_expr_layer = tf.keras.layers.Dense(100, activation = "relu")
    gen_expr_emb = gen_expr_layer(gen_expr_emb)
    
    
    input_gen_mut1 = tf.keras.layers.Input(shape = (1,), dtype = tf.string)
    input_gen_mut = cancer_gen_mut_model(input_gen_mut1)
    input_gen_mut.trainable = False
    
    reshape_gen_mut = tf.keras.layers.Reshape((1, cancer_gen_mut_model(samp_ach).numpy().shape[0], 1))
    reshape_gen_mut = reshape_gen_mut(input_gen_mut)
    gen_mut_layer = tf.keras.layers.Conv2D(50, (1, 700), strides=5, activation = "tanh")
    gen_mut_emb = gen_mut_layer(reshape_gen_mut)
    pool_layer = tf.keras.layers.MaxPooling2D((1,5))
    pool_out = pool_layer(gen_mut_emb)
    gen_mut_layer = tf.keras.layers.Conv2D(30, (1, 5), strides=2, activation = "relu")
    gen_mut_emb = gen_mut_layer(pool_out)
    pool_layer = tf.keras.layers.MaxPooling2D((1,10))
    pool_out = pool_layer(gen_mut_emb)
    flatten_layer = tf.keras.layers.Flatten()
    flatten_out = flatten_layer(pool_out)
    x_mut = tf.keras.layers.Dense(100,activation = 'relu')(flatten_out)
    x_mut = tf.keras.layers.Dropout(dropout1)(x_mut)
    
    all_omics = tf.keras.layers.Concatenate()([dense_out, gen_methy_emb, gen_expr_emb, x_mut])
    x = tf.keras.layers.Dense(300,activation = 'tanh')(all_omics)
    x = tf.keras.layers.Dropout(dropout1)(x, training = training)
    x = tf.keras.layers.Lambda(lambda x: K.expand_dims(x,axis=-1))(x)
    x = tf.keras.layers.Lambda(lambda x: K.expand_dims(x,axis=1))(x)
    x = tf.keras.layers.Conv2D(filters=30, kernel_size=(1,150),strides=(1, 1), activation = 'relu',padding='valid')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(1,2))(x)
    x = tf.keras.layers.Conv2D(filters=10, kernel_size=(1,5),strides=(1, 1), activation = 'relu',padding='valid')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(1,3))(x)
    x = tf.keras.layers.Conv2D(filters=5, kernel_size=(1,5),strides=(1, 1), activation = 'relu',padding='valid')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(1,3))(x)
    x = tf.keras.layers.Dropout(dropout1)(x, training = training)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dropout(dropout2)(x, training = training)
    final_out_layer = tf.keras.layers.Dense(1, activation = "linear")
    final_out = final_out_layer(x)
    simplecdr = tf.keras.models.Model([input_gcn_features, input_norm_adj_mat, input_gen_expr1,
                                   input_gen_methy1, input_gen_mut1], final_out)
    simplecdr.compile(loss = tf.keras.losses.MeanSquaredError(), 
                      # optimizer = tf.keras.optimizers.Adam(lr=1e-3),
                    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False), 
                    metrics = [tf.keras.metrics.RootMeanSquaredError()])
    
    return simplecdr

check = deepcdrgcn(training = training, dropout1 = dropout1, dropout2 = dropout2)
print(check.summary())

check.fit([train_gcn_feats, train_adj_list, train_keep["Cell_Line"].values.reshape(-1,1), train_keep["Cell_Line"].values.reshape(-1,1), train_keep["Cell_Line"].values.reshape(-1,1)],
          train_keep["AUC"].values.reshape(-1,1),
         validation_data = ([valid_gcn_feats, valid_adj_list, valid_keep["Cell_Line"].values.reshape(-1,1), valid_keep["Cell_Line"].values.reshape(-1,1), valid_keep["Cell_Line"].values.reshape(-1,1)],
          valid_keep["AUC"].values.reshape(-1,1)), 
         batch_size = 32, epochs = 200, 
         callbacks = tf.keras.callbacks.EarlyStopping(monitor = "val_loss", patience = 20, restore_best_weights=True, 
                                                     mode = "min"), shuffle = True, verbose = 1, 
         validation_batch_size = 32)

# make predictions with the model for validation data, and save the validation prediction files
# get the predictions on the test set
y_val_preds = check.predict([valid_gcn_feats, valid_adj_list, valid_keep["Cell_Line"].values.reshape(-1,1), valid_keep["Cell_Line"].values.reshape(-1,1), valid_keep["Cell_Line"].values.reshape(-1,1)])
# get the responses corresponding to the preds in the test set
y_val_true = valid_keep["AUC"].values.reshape(-1,1) 

# [Req] Save raw predictions in dataframe
# -----------------------------
frm.store_predictions_df(params, y_true=y_val_true, y_pred=y_val_preds, stage="val", outdir=model_path)

# [Req] Compute performance scores
val_scores = frm.compute_performace_scores( params, y_true=y_val_true, y_pred=y_val_preds, stage="val",outdir=model_path, metrics=metrics_list)

# save the model in the created model directory
check.save(os.path.join(model_path, "DeepCDR_on_gcsi"))