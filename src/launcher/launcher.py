import sys
import os
from datetime import datetime

# Set project root as path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.launcher.launched_trainer import train_model
import src.launcher.models.__all_models as all_models

# Uploading configs of models to train
from src.launcher.configs import config1, config2, config3, config4, config5, config7, config8, config9, config_final_2, config_final_4, config_final_5

# Setting up models and configs

classes_to_exclude_1 = ['inactivity quiet/light', 'religious activities', 'running',
                        'self care', 'volunteer activities', 'transportation']  # 6 classes excluded, 14 classes left

classes_to_exclude_2 = ['inactivity quiet/light', 'religious activities', 'running',
                        'self care', 'volunteer activities', 'transportation',
                        'walking', 'dancing', 'music playing', 'bicycling']  # 10 classes excluded, 10 classes left

launch_list = [
    {"model": all_models.PoseCNNsc_stage111, "config": config_final_4, "classes_to_exclude": classes_to_exclude_1},
    {"model": all_models.PoseCNNsc_13_24_35_final2, "config": config_final_5, "classes_to_exclude": classes_to_exclude_1},
    {"model": all_models.PoseCNNsc_13_24_35_final2, "config": config_final_5, "classes_to_exclude": classes_to_exclude_2}
]


if __name__ == "__main__":
    today = datetime.today().strftime("%d.%m")

    for i, entry in enumerate(launch_list):
        print("\n===================================================================================")
        print(f"    Launching the model: {entry['model'].__name__} with config {entry['config'].__name__}")
        print("===================================================================================\n")

        config = entry['config']
        model_class = entry['model']
        classes_to_exclude = entry['classes_to_exclude']

        train_model(config=config, model_class=model_class, classes_to_exclude=classes_to_exclude)
        # train_model(config, model_class)
