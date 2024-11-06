import os
import json
import sys
import warnings
from pathlib import Path
from pprint import pformat
from typing import Dict, Union
import tensorflow as tf
import pickle
import pandas as pd
import numpy as np
from tensorflow.keras import backend as K
from create_data_generator import data_generator, batch_predict

# Setup mixed precision if supported and beneficial for your hardware
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy("mixed_float16")  # Adjust this depending on hardware

# [Req] IMPROVE imports
from improvelib.applications.drug_response_prediction.config import DRPTrainConfig
from improvelib.utils import str2bool
import improvelib.utils as frm
from improvelib.metrics import compute_metrics

# Model-specific imports
from model_params_def import train_params # [Req]

# # device ID
# os.environ["CUDA_VISIBLE_DEVICES"] = "7"

filepath = Path(__file__).resolve().parent # [Req]

# wrap the generator function with tf.data.Dataset
def create_tf_dataset(generator_func, output_signature, *generator_args):
    dataset = tf.data.Dataset.from_generator(
        lambda: generator_func(*generator_args),  # no batch_size here
        output_signature=output_signature
    )

    # Apply selective conversion, avoiding string tensors
    def convert_structure(x, y):
        # Convert each element in x as per dtype
        converted_x = [
            tf.convert_to_tensor(item) if item.dtype.is_floating else item for item in x
        ]
        # Convert y as needed
        converted_y = tf.convert_to_tensor(y) if y.dtype.is_floating else y
        return tuple(converted_x), converted_y
    
    dataset = dataset.map(convert_structure, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Apply prefetching for performance
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    dataset = dataset.map(convert_structure, num_parallel_calls=tf.data.AUTOTUNE)

    return dataset

def load_memmap_unknown_shape(directory, filename, mode='r'):
    """
    Loads a .npy file as a memory-mapped numpy array, even if shape is unknown.

    Parameters:
    - directory (str): Path to the directory containing the file.
    - filename (str): Name of the .npy file.
    - mode (str): Mode in which to open the memory map (default is 'r' for read-only access).

    Returns:
    - np.memmap: Memory-mapped numpy array.
    """
    # combine directory and filename
    file_path = os.path.join(directory, filename)
    
    # use np.lib.format.open_memmap to load the file as a memory-mapped array
    memmap_array = np.lib.format.open_memmap(file_path, mode=mode)
    
    return memmap_array

## get the model architecture
def deepcdrgcn(dict_features, dict_adj_mat, samp_drug, samp_ach, cancer_dna_methy_model, cancer_gen_expr_model, cancer_gen_mut_model, training = False, dropout1 = 0.1, dropout2 = 0.2):
    
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
    # all above code is for GCN for drugs

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
    
    return simplecdr

# [Req]
def run(params: Dict):
    """ Run model training.

    Args:
        params (dict): dict of IMPROVE parameters and parsed values.

    Returns:
        dict: prediction performance scores computed on validation data
            according to the metrics_list.
    """
    # ------------------------------------------------------
    # [Req] Build model path 
    # ------------------------------------------------------
    modelpath = frm.build_model_path(model_file_name=params["model_file_name"], model_file_format=params["model_file_format"], model_dir=params["output_dir"])

    # ------------------------------------------------------
    # [Req] Create data names for train and val
    # ------------------------------------------------------

    #train_data_fname = frm.build_ml_data_file_name(data_format=params["data_format"], stage="train")  # [Req]
    #val_data_fname = frm.build_ml_data_file_name(data_format=params["data_format"], stage="val")  # [Req]

    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():

        # import the preprocessed data
        # specify the directory where preprocessed data is stored
        data_dir = params['input_dir']

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

        # load the true values
        train_keep = pd.read_csv(os.path.join(data_dir, "train_y_data.csv"))
        valid_keep = pd.read_csv(os.path.join(data_dir, "val_y_data.csv"))
        # rename the columns
        train_keep.columns = ["Cell_Line", "Drug_ID", "AUC"]
        valid_keep.columns = ["Cell_Line", "Drug_ID", "AUC"]
        # get unique values
        samp_drug = valid_keep["Drug_ID"].unique()[-1]
        samp_ach = np.array(valid_keep["Cell_Line"].unique()[-1])

        # load the drug data as memmap objects
        train_gcn_feats = load_memmap_unknown_shape(data_dir, 'train_drug_features.npy')
        train_adj_list = load_memmap_unknown_shape(data_dir, 'train_norm_adj_mat.npy')
        valid_gcn_feats = load_memmap_unknown_shape(data_dir, 'val_drug_features.npy')
        valid_adj_list = load_memmap_unknown_shape(data_dir, 'val_norm_adj_mat.npy')

        # create a data generator for the train data
        batch_size = params['batch_size']
        generator_batch_size = params['val_batch']
        
        # prepare the train data generator
        #train_gen =  data_generator(train_gcn_feats, train_adj_list, train_keep["Cell_Line"].values.reshape(-1,1), train_keep["Cell_Line"].values.reshape(-1,1), 
        #    train_keep["Cell_Line"].values.reshape(-1,1), train_keep["AUC"].values.reshape(-1,1), batch_size, shuffle=True, peek=True, verbose=False)

        # prepare the validation data generator
        #val_gen =  data_generator(valid_gcn_feats, valid_adj_list, valid_keep["Cell_Line"].values.reshape(-1,1), valid_keep["Cell_Line"].values.reshape(-1,1), 
        #    valid_keep["Cell_Line"].values.reshape(-1,1), valid_keep["AUC"].values.reshape(-1,1), generator_batch_size, peek=True, verbose=False)
        
        # define the output signature for the tf.data.Dataset
        train_output_signature = (
            (
                tf.TensorSpec(shape=(None, *train_gcn_feats.shape[1:]), dtype=tf.float16),  
                tf.TensorSpec(shape=(None, *train_adj_list.shape[1:]), dtype=tf.float16),
                tf.TensorSpec(shape=(None, 1), dtype=tf.string),
                tf.TensorSpec(shape=(None, 1), dtype=tf.string),
                tf.TensorSpec(shape=(None, 1), dtype=tf.string)
            ),
            tf.TensorSpec(shape=(None, 1), dtype=tf.float64)
        )

        # define the output signature for the tf.data.Dataset
        val_output_signature = (
            (
                tf.TensorSpec(shape=(None, *valid_gcn_feats.shape[1:]), dtype=tf.float16),  
                tf.TensorSpec(shape=(None, *valid_adj_list.shape[1:]), dtype=tf.float16),
                tf.TensorSpec(shape=(None, 1), dtype=tf.string),
                tf.TensorSpec(shape=(None, 1), dtype=tf.string),
                tf.TensorSpec(shape=(None, 1), dtype=tf.string)
            ),
            tf.TensorSpec(shape=(None, 1), dtype=tf.float64)
        )

        train_cell_line = np.array(train_keep["Cell_Line"].values.reshape(-1, 1), dtype=np.str_).reshape(-1, 1)
        train_response = np.array(train_keep["AUC"].values.reshape(-1, 1), dtype=np.str_).reshape(-1, 1)
        val_cell_line = np.array(valid_keep["Cell_Line"].values.reshape(-1, 1), dtype=np.str_).reshape(-1, 1)
        val_response = np.array(valid_keep["AUC"].values.reshape(-1, 1), dtype=np.str_).reshape(-1, 1)

        # Load datasets using the optimized function
        train_dataset = create_tf_dataset(
            data_generator, train_output_signature, 
            train_gcn_feats, train_adj_list,
            train_cell_line,
            train_cell_line,
            train_cell_line,
            train_response,
            batch_size,
            True, # shuffle
            True, # peek
            False # verbose
        )

        # Validation dataset without shuffling
        val_dataset = create_tf_dataset(
            data_generator, val_output_signature, 
            valid_gcn_feats, valid_adj_list,
            val_cell_line,
            val_cell_line,
            val_cell_line,
            val_response,
            generator_batch_size, 
            True, # shuffle
            True, # peek
            False # verbose
        )

        #train_dataset = train_dataset.map(lambda x, y: (tf.convert_to_tensor(x), tf.convert_to_tensor(y)), num_parallel_calls=tf.data.experimental.AUTOTUNE)

        #val_dataset = val_dataset.map(lambda x, y: (tf.convert_to_tensor(x), tf.convert_to_tensor(y)), num_parallel_calls=tf.data.experimental.AUTOTUNE)

        # apply options for better distributed training
        #options = tf.data.Options()
        #options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
        #train_dataset = train_dataset.with_options(options)
        #val_dataset = val_dataset.with_options(options)

        steps_per_epoch = int(np.ceil(len(train_gcn_feats) / batch_size))
        #train_steps = int(np.ceil(len(train_gcn_feats) / generator_batch_size))
        validation_steps = int(np.ceil(len(valid_gcn_feats) / generator_batch_size))

        training = False
        dropout1 = 0.10
        dropout2 = 0.20

        # initialize model
        check = deepcdrgcn(dict_features, dict_adj_mat, samp_drug, samp_ach, cancer_dna_methy_model, cancer_gen_expr_model, cancer_gen_mut_model, training = training, dropout1 = dropout1, dropout2 = dropout2)
        
        # compile the model
        lr = params['learning_rate']
        check.compile(loss = tf.keras.losses.MeanSquaredError(), 
                            # optimizer = tf.keras.optimizers.Adam(lr=1e-3),
                            optimizer = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999, amsgrad=False), 
                            metrics = [tf.keras.metrics.RootMeanSquaredError()])
        
        # fit the model              
        epoch_num = params['epochs']
        patience_val = params['patience']
        check.fit(train_dataset,
                validation_data = val_dataset, 
                epochs = epoch_num, steps_per_epoch=steps_per_epoch, validation_steps=validation_steps,
                callbacks = tf.keras.callbacks.EarlyStopping(monitor = "val_loss", patience = patience_val, restore_best_weights=True, 
                                                            mode = "min") ,validation_batch_size = generator_batch_size)
    
        # predict on validation dataset
        y_val_preds, y_val_true = batch_predict(check, data_generator(valid_gcn_feats, valid_adj_list, valid_keep["Cell_Line"].values.reshape(-1,1), valid_keep["Cell_Line"].values.reshape(-1,1), valid_keep["Cell_Line"].values.reshape(-1,1), valid_keep["AUC"].values.reshape(-1,1), generator_batch_size, verbose = False, shuffle = False), validation_steps)
    

    # ------------------------------------------------------
    # [Req] Save raw predictions in dataframe
    # ------------------------------------------------------
    frm.store_predictions_df(
        y_true=y_val_true, 
        y_pred=y_val_preds, 
        stage="val",
        y_col_name=params["y_col_name"],
        output_dir=params["output_dir"],
        input_dir=params["input_dir"]
    )
    # ------------------------------------------------------
    # [Req] Compute performance scores
    # ------------------------------------------------------
    val_scores = frm.compute_performance_scores(
        y_true=y_val_true, 
        y_pred=y_val_preds, 
        stage="val",
        metric_type=params["metric_type"],
        output_dir=params["output_dir"]
    )

    # # save the model in the created model directory
    check.save(os.path.join(modelpath, "DeepCDR_model"))

    return val_scores

# [Req]
def initialize_parameters():
    """This initialize_parameters() is define this way to support Supervisor
    workflows such as HPO.

    Returns:
        dict: dict of IMPROVE/CANDLE parameters and parsed values.
    """
    # [Req] Initialize parameters
    additional_definitions = train_params
    cfg = DRPTrainConfig()
    params = cfg.initialize_parameters(
          pathToModelDir=filepath,
        default_config="deepcdr_params.txt",
        additional_definitions=additional_definitions)
    return params


# [Req]
def main(args):
    # [Req]
    params = initialize_parameters()
    val_scores = run(params)
    print("\nFinished training model.")


# [Req]
if __name__ == "__main__":
    main(sys.argv[1:])
