"""
Model-specific params (Model: DeepCDR)
If no params are required by the model, then it should be an empty list.
"""

from improvelib.utils import str2bool


preprocess_params = []


train_params = [
    {"name": "epochs",
     "type": int,
     "default": 150,
     "help": "Number of epochs for training."
    },

    {"name": "learning_rate",
     "type": float,
     "default": 0.0001,
     "help": "Learning rate for the optimizer."
    },

    {"name": "batch_size",
     "type": int,
     "default": 256,
     "help": "Training batch size."
    },

    {"name": "val_batch",
     "type": int,
     "default": 256,
     "help": "Validation batch size."
    }
]


infer_params = [
    {"name": "infer_batch",
     "type": int,
     "default": 256,
     "help": "Inference batch size."
    }
]