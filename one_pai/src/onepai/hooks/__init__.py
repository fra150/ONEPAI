"""
onepai.hooks - Ganci per l'integrazione con framework ML.
"""
from .pytorch import register_pytorch_hook
from .tensorflow import OnePaiTensorflowCallback
from .gemma import hook_gemma_model

__all__ = [
    "register_pytorch_hook",
    "OnePaiTensorflowCallback",
    "hook_gemma_model",
]