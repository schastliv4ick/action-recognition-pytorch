import subprocess
import itertools

# Пути к конфигам
configs = [
    "configs/config1.py",
    # "configs/config2.py",
    # "configs/config3.py"
]

# Названия моделей, допустим ты по ним переключаешь логику в launched_trainer.py
models = [
    "PoseCNNsc",
    # "PoseCNNv2_Lite",
    # "PoseCNNsc_13_35"
]

# Перебираем все комбинации
for model_name, config_path in itertools.product(models, configs):
    print(f"\nЗапуск модели {model_name} с конфигом {config_path}...\n")

    result = subprocess.run(
        ["python", "launched_trainer.py", "--model", model_name, "--config", config_path],
        capture_output=True,
        text=True
    )

    print(f"Готово: {model_name} + {config_path}")
