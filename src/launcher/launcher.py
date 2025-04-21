import sys
import os
from datetime import datetime

# Set project root as path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from launched_trainer import train_model
import models.__all_models as all_models

# Uploading configs of models to train
from configs import config1, config2, config3, config4, config5, config7, config8, config9

# Setting up models and configs
launch_list = [
    {"model": all_models.PoseCNNsc_stage111, "config": config4},
    {"model": all_models.PoseCNNsc_stage111, "config": config5},
]

if __name__ == "__main__":
    today = datetime.today().strftime("%d.%m")

    for i, entry in enumerate(launch_list):
        print("\n===================================================================================")
        print(f"    Launching the model: {entry['model'].__name__} with config {entry['config'].__name__}")
        print("===================================================================================\n")

        config = entry['config']
        model_class = entry['model']

        # train_model(config=config, model_class=model_class, rare_classes_threshold=0.025)
        train_model(config=config, model_class=model_class, rare_classes_threshold=0.02)

