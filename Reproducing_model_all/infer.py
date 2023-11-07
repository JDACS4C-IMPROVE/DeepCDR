import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
# try:
#     tf.config.experimental.set_memory_growth(physical_devices[0], True)
#     print(physical_devices[0], flush = True)
    
# except:
#   # Invalid device or cannot modify virtual devices once initialized.
#   pass

from tensorflow.keras import backend as K
import improve_utils as iu
import pickle
import numpy as np
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr

# prepare the test data
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

test_keep = iu.load_single_drug_response_data_v2(source = 'CTRPv2', split_file_name='CTRPv2_split_0_test.txt', y_col_name='auc')[['improve_sample_id', 'improve_chem_id', 'auc']]
test_keep.columns = ["Cell_Line", "Drug_ID", "AUC"]

test_gcn_feats = []
test_adj_list = []
for drug_id in test_keep["Drug_ID"].values:
    test_gcn_feats.append(dict_features[drug_id])
    test_adj_list.append(dict_adj_mat[drug_id])

test_gcn_feats = np.array(test_gcn_feats).astype("float32")
test_adj_list = np.array(test_adj_list).astype("float32")


# load the model
check = tf.keras.models.load_model('Models/DeepCDR_on_ctrpv2')

# get the predictions on the test set
preds = check.predict([test_gcn_feats, test_adj_list, test_keep["Cell_Line"].values.reshape(-1,1), test_keep["Cell_Line"].values.reshape(-1,1), test_keep["Cell_Line"].values.reshape(-1,1)])
# get the responses corresponding to the preds in the test set
target = test_keep["AUC"].values.reshape(-1,1)

# compute the rmse for the test set
rmse = np.sqrt(mean_squared_error(preds, target))
print("RMSE ", rmse)

# compute the correlation
res = pearsonr(preds[:,0], target[:,0])
pearson = res[0]
print("pearson - r", pearson)

# compute the coefficient of determination
r_square = np.square(pearson)
print("R - square", r_square)
