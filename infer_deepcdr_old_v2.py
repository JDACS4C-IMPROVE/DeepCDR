import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
from tensorflow.keras import backend as K
import pickle
import numpy as np
import pandas as pd
import os
import json
import sys
import warnings
from pathlib import Path
from pprint import pformat
from typing import Dict, Union
from improve import framework as frm
from improve.metrics import compute_metrics

from preprocess import preprocess_params
from train import metrics_list, train_params

filepath = Path(__file__).resolve().parent
print(filepath)

# [Req] List of metrics names to be compute performance scores
metrics_list = ["mse", "rmse", "pcc", "scc", "r2"]  

# [Req] App-specific params (App: monotherapy drug response prediction)
# Currently, there are no app-specific args for the train script.
app_infer_params = []
model_infer_params = []

infer_params = app_infer_params + model_infer_params

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
    infer_dir = frm.create_outdir(outdir=params["infer_outdir"])
    print(infer_dir)

    # ------------------------------------------------------
    # [Req] Create data names for train and val
    # ------------------------------------------------------

    test_data_fname = frm.build_ml_data_name(params, stage="test")
    print(test_data_fname)

    return infer_dir

additional_definitions = preprocess_params + train_params + infer_params
params = frm.initialize_parameters(
        filepath,
        default_model="deepcdr_default_model.txt",
        additional_definitions=additional_definitions,
        required=None,
    )
infer_dir_path = run(params)

# import the preprocessed data
# specify the directory where preprocessed data is stored
data_dir = frm.build_model_path(params, model_dir=params["ml_data_outdir"])
data_dir = os.path.dirname(data_dir)

# load models for preprocessed data
cancer_gen_expr_model = tf.keras.models.load_model(os.path.join(data_dir,"cancer_gen_expr_model"))
cancer_gen_mut_model = tf.keras.models.load_model(os.path.join(data_dir, "cancer_gen_mut_model"))
cancer_dna_methy_model = tf.keras.models.load_model(os.path.join(data_dir, "cancer_dna_methy_model"))

cancer_gen_expr_model.trainable = False
cancer_gen_mut_model.trainable = False
cancer_dna_methy_model.trainable = False

with open(os.path.join(data_dir, "drug_features.pickle"),"rb") as f:
    dict_features = pickle.load(f)

with open(os.path.join(data_dir, "norm_adj_mat.pickle"),"rb") as f:
    dict_adj_mat = pickle.load(f)

test_keep = pd.read_csv(os.path.join(data_dir, "test_y_data.csv"))
test_keep.columns = ["Cell_Line", "Drug_ID", "AUC"]

test_gcn_feats = []
test_adj_list = []
for drug_id in test_keep["Drug_ID"].values:
    test_gcn_feats.append(dict_features[drug_id])
    test_adj_list.append(dict_adj_mat[drug_id])

test_gcn_feats = np.array(test_gcn_feats).astype("float32")
test_adj_list = np.array(test_adj_list).astype("float32")


# load the model
model_path = frm.build_model_path(params, model_dir=params["model_outdir"])
model_path = os.path.dirname(model_path)
model_path = os.path.join(model_path, "DeepCDR_on_gcsi")
check = tf.keras.models.load_model(model_path)

# get the predictions on the test set
preds_test = check.predict([test_gcn_feats, test_adj_list, test_keep["Cell_Line"].values.reshape(-1,1), test_keep["Cell_Line"].values.reshape(-1,1), test_keep["Cell_Line"].values.reshape(-1,1)])
preds_test = preds_test.flatten()
# get the responses corresponding to the preds in the test set
target_test = test_keep["AUC"].values

# [Req] Save raw predictions in dataframe
# -----------------------------
frm.store_predictions_df(params, y_true=target_test, y_pred=preds_test, stage="test", outdir=infer_dir_path)

# [Req] Compute performance scores
test_scores = frm.compute_performace_scores(params, y_true=target_test, y_pred=preds_test, stage="test",outdir=infer_dir_path, metrics=metrics_list)