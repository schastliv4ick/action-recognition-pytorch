# Launcher
This script allows sequential training of multiple models
with their corresponding configurations.
## Overview

The script does the following:

1. Adds the project root to `sys.path` to allow relative imports.
2. Imports the required models and their configurations.
3. Defines a `launch_list` with model/config pairs.
4. Iterates through the list and trains each model using `train_model`.

## How to Set Up Models and Configurations

To use this launcher script for training, you need to properly organize your models and config files, and register them in the code.

### 1. Define Your Model Classes

Model classes should be placed in `launcher/models/`. Each model should be implemented as a Python class (e.g., subclassing `nn.Module` if using PyTorch).

Example:

```python
# launcher/models/posecnn_stage1.py

import torch.nn as nn

class PoseCNNsc_13_24_35_stage1(nn.Module):
    def __init__(self, config):
        super().__init__()
        # define layers based on config
```

### 2. Add Models to __all_models.py
Register your model classes by importing them
in __all_models.py:

```
from .posecnn_stage1 import PoseCNNsc_13_24_35_stage1
from .posecnn_stage2 import PoseCNNsc_stage2
from .posecnn_stage3 import PoseCNNsc_stage3
```

### 3. Create Configuration Files
Place config files in `launcher/configs`. Every config is a .py
file with variables and hyperparameters used in training. Config
file must contain the name of the model.

```python
PATH_TO_DATA = "V:\ML\yandex-ml-2025\data"

BATCH_SIZE = 32
LEARNING_RATE = 0.001
MOMENTUM = 0.9
NUM_EPOCHS = 60

WEIGHT_DECAY = 1e-2

TRAIN_AUGMENTATION_TYPE = "basic"
VALID_AUGMENTATION_TYPE = None

MODEL = "PoseCNNsc_13_24_35"
```

### 4. Import configs in `launcher.py`

```python
from launcher.configs import config1, config2, config3
```

### 5. Add Model and Config to the Launcher

In launch.py, add your model and config to the launch_list:

```python

launch_list = [
    {"model": all_models.PoseCNNsc_13_24_35_stage1, "config": config1},
    {"model": all_models.PoseCNNsc_stage2, "config": config2},
    {"model": all_models.PoseCNNsc_stage3, "config": config3},
]
```
### 6. Run training

Once everything is set up, simply run the launcher script:

```
python -m src.launcher.launcher
```