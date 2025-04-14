"""
Model-specific params (Model: DeepCDR)
If no params are required by the model, then it should be an empty list.
"""

from improvelib.utils import str2bool


preprocess_params = [
    {"name": "chunk_size",
    "type": int,
    "default": 100,
    "help": "Size of data chunks for processing drug features and adjacency matrices.",
    },
]


train_params = []


infer_params = []