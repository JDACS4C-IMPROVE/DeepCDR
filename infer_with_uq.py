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
from create_data_generator import data_generator, batch_predict

# load models for preprocessed data
cancer_gen_expr_model = tf.keras.models.load_model(os.path.join('exp_result',"cancer_gen_expr_model"))
cancer_gen_mut_model = tf.keras.models.load_model(os.path.join('exp_result', "cancer_gen_mut_model"))
cancer_dna_methy_model = tf.keras.models.load_model(os.path.join('exp_result', "cancer_dna_methy_model"))

# set model weight trainig to false
cancer_gen_expr_model.trainable = False
cancer_gen_mut_model.trainable = False
cancer_dna_methy_model.trainable = False

with open(os.path.join('exp_result', "drug_features.pickle"),"rb") as f:
    dict_features = pickle.load(f)

with open(os.path.join('exp_result', "norm_adj_mat.pickle"),"rb") as f:
    dict_adj_mat = pickle.load(f)

test_keep = pd.read_csv(os.path.join('exp_result', "test_y_data.csv"))
test_keep.columns = ["Cell_Line", "Drug_ID", "AUC"]

test_gcn_feats = []
test_adj_list = []
for drug_id in test_keep["Drug_ID"].values:
    test_gcn_feats.append(dict_features[drug_id])
    test_adj_list.append(dict_adj_mat[drug_id])

test_gcn_feats = np.array(test_gcn_feats).astype("float32")
test_adj_list = np.array(test_adj_list).astype("float32")

# load trained model here
CCLE_model = tf.keras.models.load_model('exp_result/DeepCDR_model/DeepCDR_model')

samp_drug = test_keep["Drug_ID"].unique()[-1]
samp_ach = np.array(test_keep["Cell_Line"].unique()[-1])

# redefine the model
training = True
dropout1 = 0.10
dropout2 = 0.20

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
x_mut = tf.keras.layers.Dropout(dropout1)(x_mut, training = training)
    
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

weights_train = CCLE_model.get_weights()
simplecdr.set_weights(weights_train)
weights_new = simplecdr.get_weights()

generator_batch_size = 32
test_steps = int(np.ceil(len(test_gcn_feats) / generator_batch_size))

# get predictions 25 times
all_predicted_values = []
for i in range(25):
    preds_test, target_test = batch_predict(simplecdr, data_generator(test_gcn_feats, test_adj_list, test_keep["Cell_Line"].values.reshape(-1,1), test_keep["Cell_Line"].values.reshape(-1,1), test_keep["Cell_Line"].values.reshape(-1,1), test_keep["AUC"].values.reshape(-1,1), generator_batch_size, shuffle = False), test_steps)
    all_predicted_values.append(preds_test)

preds_df = pd.DataFrame(all_predicted_values)
preds_df_final = preds_df.T
preds_df_final.columns = ['predicted_vals_' + str(i + 1) for i in range(preds_df_final.shape[1])]

combined_df = pd.concat((preds_df_final, pd.DataFrame(target_test, columns = ['target_auc'])), axis = 1)

# width
li_test = np.percentile(preds_df_final, axis = 1, q = (2.5, 97.5))[0,:].reshape(-1,1)     
ui_test = np.percentile(preds_df_final, axis = 1, q = (2.5, 97.5))[1,:].reshape(-1,1)   

width_test = ui_test - li_test
avg_width_test = width_test.mean(0)[0]
print("width: ", avg_width_test)

# coverage
ind_test = (target_test >= li_test) & (target_test <= ui_test)
coverage_test= ind_test.mean(0)[0]
print("Coverage: ", coverage_test)