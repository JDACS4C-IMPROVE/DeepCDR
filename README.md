# DeepCDR
Cancer Drug Response Prediction via a Hybrid Graph Convolutional Network

A requirements_new.txt file is available that gives the dependencies for the environment. 

The preprocess, train, and infer scripts are in <code>preprocess_with_data_loader.py</code>, <code>train_with_data_loader.py</code>, and <code>infer_with_data_loader.py</code> files. 

The cross-study analysis is in the <code>CSA_all_splits.py</code> file. 

Steps to create the conda environment.

* conda create --name DeepCDR_IMPROVE_env python=3.10
* conda activate DeepCDR_IMPROVE_env
* conda install tensorflow-gpu=2.10.0
* pip install git+https://github.com/ECP-CANDLE/candle_lib@develop
* pip install protobuf==3.20.0
* pip install rdkit==2023.9.6
* pip install deepchem==2.8.0



