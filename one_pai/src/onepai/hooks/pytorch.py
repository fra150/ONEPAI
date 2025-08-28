"""
Gancio (Hook) per modelli PyTorch.
"""
import torch
from torch import nn
from onepai.core.observer import Observer

def get_onepai_hook(observer: Observer):
    """Funzione interna che restituisce il callback da registrare."""
    def hook(model, input, output):
        # 'output' è il tensore di attivazioni del layer
        observer.capture_potentiality(output.detach().cpu().numpy())
    return hook

def register_pytorch_hook(model: nn.Module, layer_name: str, observer: Observer):
    """
    Registra un forward hook di ONEPAI su un layer specifico di un modello PyTorch.

    Args:
        model: Il modello PyTorch da osservare.
        layer_name: Il nome del layer a cui agganciarsi (es. 'layer4.conv1').
        observer: L'istanza dell'Osservatore di ONEPAI.
    """
    layer = dict(model.named_modules())[layer_name]
    layer.register_forward_hook(get_onepai_hook(observer))
    print(f"INFO: Hook ONEPAI registrato sul layer PyTorch: {layer_name}")