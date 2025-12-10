import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
import improve_utils as iu
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from tensorflow.keras import backend as K

# load model
cancer_gen_expr_model = tf.keras.models.load_model("Models//cancer_gen_expr_model")
cancer_gen_mut_model = tf.keras.models.load_model("Models//cancer_gen_mut_model")
cancer_dna_methy_model = tf.keras.models.load_model("Models//cancer_dna_methy_model")

cancer_gen_expr_model.trainable = False
cancer_gen_mut_model.trainable = False
cancer_dna_methy_model.trainable = False

with open("csa_data//drug_features.pickle", "rb") as f:
    dict_features = pickle.load(f)

with open("csa_data//norm_adj_mat.pickle", "rb") as f:
    dict_adj_mat = pickle.load(f)

train_keep = iu.load_single_drug_response_data_v2(source = 'CTRPv2', split_file_name='CTRPv2_split_0_train.txt', y_col_name='auc')[['improve_sample_id', 'improve_chem_id', 'auc']]
valid_keep = iu.load_single_drug_response_data_v2(source = 'CTRPv2', split_file_name='CTRPv2_split_0_val.txt', y_col_name='auc')[['improve_sample_id', 'improve_chem_id', 'auc']]
train_keep.head()
train_keep.columns = ["Cell_Line", "Drug_ID", "AUC"]
valid_keep.columns = ["Cell_Line", "Drug_ID", "AUC"]
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

print("now here")
check = deepcdrgcn(training = training, dropout1 = dropout1, dropout2 = dropout2)
print(check.summary())

train_gcn_feats = train_gcn_feats
train_adj_list = train_adj_list
valid_gcn_feats = valid_gcn_feats
valid_adj_list = valid_adj_list
train_gcn_feats = np.array(train_gcn_feats).astype("float16")
valid_gcn_feats = np.array(valid_gcn_feats).astype("float16")

train_adj_list = np.array(train_adj_list).astype("float16")
valid_adj_list = np.array(valid_adj_list).astype("float16")

train_keep = train_keep.iloc
valid_keep = valid_keep.iloc

print(train_gcn_feats.shape)
print(train_adj_list.shape)
print(valid_gcn_feats.shape)
print(valid_adj_list.shape)

check.fit([train_gcn_feats, train_adj_list, train_keep["Cell_Line"].values.reshape(-1,1), train_keep["Cell_Line"].values.reshape(-1,1), train_keep["Cell_Line"].values.reshape(-1,1)],
          train_keep["AUC"].values.reshape(-1,1),
         validation_data = ([valid_gcn_feats, valid_adj_list, valid_keep["Cell_Line"].values.reshape(-1,1), valid_keep["Cell_Line"].values.reshape(-1,1), valid_keep["Cell_Line"].values.reshape(-1,1)],
          valid_keep["AUC"].values.reshape(-1,1)), 
         batch_size = 32, epochs = 1000, 
         callbacks = tf.keras.callbacks.EarlyStopping(monitor = "val_loss", patience = 20, restore_best_weights=True, 
                                                     mode = "min"), shuffle = True, verbose = 1, 
         validation_batch_size = 32)

check.save("Models/DeepCDR_on_ctrpv2")