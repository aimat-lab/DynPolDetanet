"""Public package interface: `from detanet_model import *` (used by
trainer.py and the top-level scripts) exposes DetaNet plus everything from
spectra_simulator, model_loader, constant, and metrics directly, without
needing to import each submodule individually."""
from .detanet import DetaNet
from .spectra_simulator import *
from .model_loader import *
from .constant import *
from .metrics import *