
# DeepCDR

This repository demonstrates how to use the [IMPROVE library v0.1.0-alpha](https://jdacs4c-improve.github.io/docs/v0.1.0-alpha/) for building a drug response prediction (DRP) model using LightGBM (LGBM), and provides examples with the benchmark [cross-study analysis (CSA) dataset](https://web.cels.anl.gov/projects/IMPROVE_FTP/candle/public/improve/benchmarks/single_drug_drp/benchmark-data-pilot1/csa_data/).

This version, tagged as `v0.1.0-alpha`, introduces a new API which is designed to encourage broader adoption of IMPROVE and its curated models by the research community.

## Dependencies
Installation instuctions are detialed below in [Step-by-step instructions](#step-by-step-instructions).


ML framework:
+ [Tensorflow](https://www.tensorflow.org/) -- deep learning framework for building the prediction model

IMPROVE dependencies:
+ [IMPROVE v0.1.0-alpha](https://jdacs4c-improve.github.io/docs/v0.1.0-alpha/)


## Dataset
Benchmark data for cross-study analysis (CSA) can be downloaded from this [site](https://web.cels.anl.gov/projects/IMPROVE_FTP/candle/public/improve/benchmarks/single_drug_drp/benchmark-data-pilot1/csa_data/).

The data tree is shown below:
```
csa_data/raw_data/
├── splits
│   ├── CCLE_all.txt
│   ├── CCLE_split_0_test.txt
│   ├── CCLE_split_0_train.txt
│   ├── CCLE_split_0_val.txt
│   ├── CCLE_split_1_test.txt
│   ├── CCLE_split_1_train.txt
│   ├── CCLE_split_1_val.txt
│   ├── ...
│   ├── GDSCv2_split_9_test.txt
│   ├── GDSCv2_split_9_train.txt
│   └── GDSCv2_split_9_val.txt
├── x_data
│   ├── cancer_copy_number.tsv
│   ├── cancer_discretized_copy_number.tsv
│   ├── cancer_DNA_methylation.tsv
│   ├── cancer_gene_expression.tsv
│   ├── cancer_miRNA_expression.tsv
│   ├── cancer_mutation_count.tsv
│   ├── cancer_mutation_long_format.tsv
│   ├── cancer_mutation.parquet
│   ├── cancer_RPPA.tsv
│   ├── drug_ecfp4_nbits512.tsv
│   ├── drug_info.tsv
│   ├── drug_mordred_descriptor.tsv
│   └── drug_SMILES.tsv
└── y_data
    └── response.tsv
```

Note that `original_work` folder contains data files and scripts used to train and evaluate the DeepCDR for the original paper.


## Model scripts and parameter file
+ `deepcdr_preprocess_improve.py` - takes benchmark data files and transforms into files for trianing and inference
+ `deepcdr_train_improve.py` - trains a deepcdr DRP model
+ `deepcdr_infer_improve.py` - runs inference with the trained deepcdr model
+ `model_params_def.py` - definitions of parameters that are specific to the model
+ `deepcdr_params.txt` - default parameter file (parameter values specified in this file override the defaults)



# Step-by-step instructions

### 1. Clone the model repository
```
git clone https://github.com/JDACS4C-IMPROVE/DeepCDR.git
cd DeepCDR
git checkout develop
```


### 2. Set computational environment

Option 1: create the conda env using the yml file.

```
conda env create -f deepcdr_env.yml
```

Use the following commands to create the environment.
```
conda create --name DeepCDR_IMPROVE_env python=3.10
conda activate DeepCDR_IMPROVE_env
conda install tensorflow-gpu=2.10.0
pip install rdkit==2023.9.6
pip install deepchem==2.8.0
pip install PyYAML
```

### 3. Run `setup_improve.sh`.
```bash
source setup_improve.sh
```

This will:
1. Download cross-study analysis (CSA) benchmark data into `./csa_data/`.
2. Clone IMPROVE repo (checkout `develop`) outside the LGBM model repo.
3. Set up env variables: `IMPROVE_DATA_DIR` (to `./csa_data/`) and `PYTHONPATH` (adds IMPROVE repo).


### 4. Preprocess CSA benchmark data (_raw data_) to construct model input data (_ML data_)
```bash
python deepcdr_preprocess_improve.py --input_dir ./csa_data/raw_data --output_dir exp_result
```

Preprocesses the CSA data and creates train, validation (val), and test datasets.

Generates:
* five model input data files: `cancer_dna_methy_model`, `cancer_gen_expr_model`, `cancer_gen_mut_model`, `drug_features.pickle`, `norm_adj_mat.pickle`
* three tabular data files, each containing the drug response values (i.e. AUC) and corresponding metadata: `train_y_data.csv`, `val_y_data.csv`, `test_y_data.csv`

```
exp_result
 ├── param_log_file.txt
 ├── cancer_dna_methy_model
 ├── cancer_gen_expr_model
 ├── cancer_gen_mut_model
 ├── test_y_data.csv
 ├── train_y_data.csv
 ├── val_y_data.csv
 ├── drug_features.pickle
 └── norm_adj_mat.pickle
```

### 5. Train DeepCDR model
```bash
python deepcdr_train_improve.py --input_dir exp_result --output_dir exp_result
```

Trains DeepCDR using the model input data generated in the previous step.

Generates:
* trained model: `DeepCDR_model`
* predictions on val data (tabular data): `val_y_data_predicted.csv`
* prediction performance scores on val data: `val_scores.json`
```
exp_result
 ├── DeepCDR_model
 ├── val_scores.json
 └── val_y_data_predicted.csv
```


### 6. Run inference on test data with the trained model

```bash
python deepcdr_infer_improve.py --input_data_dir exp_result --input_model_dir exp_result --output_dir exp_result --calc_infer_score true
```

Evaluates the performance on a test dataset with the trained model.

Generates:
* predictions on test data (tabular data): `test_y_data_predicted.csv`
* prediction performance scores on test data: `test_scores.json`
```
exp_result
 ├── test_scores.json
 └── test_y_data_predicted.csv
```
