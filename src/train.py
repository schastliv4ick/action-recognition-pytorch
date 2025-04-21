import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

import config
from models.__all_models import DensePoseCNN
import launcher.launched_trainer as launcher

if __name__ == "__main__":
    # Specify the model you want to train
    model_to_train = DensePoseCNN

    # Call the training function from launched_trainer.py
    launcher.train_model(config, model_to_train, num_classes=20)
