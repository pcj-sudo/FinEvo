"""Resolve task data independently of code and model checkpoints."""
import os
from pathlib import Path

def dataset_root(root):
    value=Path(os.environ.get('RSI_DATASET_ROOT','examples/financialstatements1')).expanduser()
    return value if value.is_absolute() else Path(root)/value
