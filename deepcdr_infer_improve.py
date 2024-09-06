import tensorflow as tf
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
from create_data_generator import data_generator, batch_predict

os.environ["CUDA_VISIBLE_DEVICES"] = "7"

# [Req] Imports from preprocess script
from deepcdr_preprocess_improve import preprocess_params
from deepcdr_train_improve import train_params

filepath = Path(__file__).resolve().parent

# [Req] List of metrics names to be compute performance scores
metrics_list = ["mse", "rmse", "pcc", "scc", "r2"]  

# [Req] App-specific params (App: monotherapy drug response prediction)
# Currently, there are no app-specific args for the train script.
app_infer_params = []
model_infer_params = [
    {"name": "test_batch",
     "type": int,
     "default": 256,
     "help": "Infer batch size."
    },
]

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
    # print path
    print("Infer directory path: ", infer_dir)
    
    # import the preprocessed data
    # specify the directory where preprocessed data is stored
    data_dir = frm.build_model_path(params, model_dir=params["test_ml_data_dir"])
    data_dir = os.path.dirname(data_dir)
    print("Data directory path: ", data_dir)

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
    model_path = frm.build_model_path(params, model_dir=params["model_dir"])
    model_path = os.path.dirname(model_path)
    print("Model path", model_path)
    model_path = os.path.join(model_path, "DeepCDR_model")
    check = tf.keras.models.load_model(model_path)

    # # get the predictions on the test set
    generator_batch_size = params['test_batch']
    test_steps = int(np.ceil(len(test_gcn_feats) / generator_batch_size))
    preds_test, target_test = batch_predict(check, data_generator(test_gcn_feats, test_adj_list, test_keep["Cell_Line"].values.reshape(-1,1), test_keep["Cell_Line"].values.reshape(-1,1), test_keep["Cell_Line"].values.reshape(-1,1), test_keep["AUC"].values.reshape(-1,1), generator_batch_size, shuffle = False), test_steps)
    print(preds_test.shape, target_test.shape)

    # [Req] Save raw predictions in dataframe
    # -----------------------------
    frm.store_predictions_df(params, y_true=target_test, y_pred=preds_test, stage="test", outdir=infer_dir)

    # [Req] Compute performance scores
    test_scores = frm.compute_performace_scores(params, y_true=target_test, y_pred=preds_test, stage="test",outdir=infer_dir, metrics=metrics_list)


    # ------------------------------------------------------
    # [Req] Create data names for train and val
    # ------------------------------------------------------
    test_data_fname = frm.build_ml_data_name(params, stage="test")

    return test_scores

# [Req]
def main(args):
    # [Req]
    additional_definitions = preprocess_params + train_params + infer_params
    params = frm.initialize_parameters(
        filepath,
        default_model="deepcdr_params.txt",
        additional_definitions=additional_definitions,
        required=None,
    )
    test_scores = run(params)
    print("\nFinished model inference.")


# [Req]
if __name__ == "__main__":
    main(sys.argv[1:])
