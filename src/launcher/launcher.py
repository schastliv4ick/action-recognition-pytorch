import sys
import os
from datetime import datetime

# Установка project root в путь
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.launcher.launched_trainer import train_model  # импортируем функцию, которую выделили
import src.launcher.models.__all_models as all_models

# Загружаем нужные конфиги
from src.launcher.configs import config1, config2, config3

# Параметры запуска: список моделей и соответствующих конфигов
launch_list = [
    # {"model": all_models.PoseCNNsc_13_24_35_stage1, "config": config1},
    {"model": all_models.PoseCNNsc_stage2, "config": config2},
    {"model": all_models.PoseCNNsc_stage3, "config": config3},
]

if __name__ == "__main__":
    today = datetime.today().strftime("%d.%m")

    for i, entry in enumerate(launch_list):
        print("\n===============================================================================")
        print(f"       Запуск модели: {entry['model'].__name__} с конфигом {entry['config'].__name__}")
        print("===============================================================================\n")
        train_model(config=entry['config'], model_class=entry['model'])
