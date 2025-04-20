# configs/__init__.py

import importlib
import os
from pathlib import Path

__all__ = []

_current_dir = Path(__file__).parent
for file in os.listdir(_current_dir):
    if file.startswith("config") and file.endswith(".py"):
        module_name = file[:-3]
        importlib.import_module(f".{module_name}", package=__name__)
        __all__.append(module_name)
