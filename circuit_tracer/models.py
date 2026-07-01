# -----------------------------------------------------------------------------
# File: circuit_tracer/utils/disk_offload.py
# -----------------------------------------------------------------------------

"""
Utility per l'offloading su disco dei parametri del modello.

Questo modulo fornisce:
- Offloading parametri su CPU o disco
- Gestione memoria per modelli grandi
- Context manager per operazioni temporanee
"""

import torch
import logging
from typing import Dict, Any, Optional, List
from contextlib import contextmanager
import tempfile
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)


def offload_modules(
    modules: Dict[str, torch.nn.Module],
    offload_type: str = "cpu",
    temp_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Offload moduli su CPU o disco per liberare memoria GPU.
    
    Args:
        modules: Dizionario di moduli da offload
        offload_type: Tipo di offload ('cpu' o 'disk')
        temp_dir: Directory temporanea per offload su disco
    
    Returns:
        Dizionario con informazioni di offload
    """
    
    logger.info(f"Avvio offload di {len(modules)} moduli su {offload_type}")
    
    offload_info = {
        "offload_type": offload_type,
        "modules": {},
        "temp_dir": temp_dir
    }
    
    for name, module in modules.items():
        if offload_type == "cpu":
            offload_info["modules"][name] = _offload_to_cpu(module, name)
        elif offload_type == "disk":
            offload_info["modules"][name] = _offload_to_disk(module, name, temp_dir)
        else:
            raise ValueError(f"Tipo di offload non supportato: {offload_type}")
    
    logger.info(f"Offload completato per {len(modules)} moduli")
    return offload_info


def _offload_to_cpu(module: torch.nn.Module, name: str) -> Dict[str, Any]:
    """
    Offload modulo su CPU.
    """
    
    original_device = next(module.parameters()).device
    
    # Sposta su CPU
    module.cpu()
    
    logger.debug(f"Modulo {name} spostato da {original_device} a CPU")
    
    return {
        "original_device": str(original_device),
        "current_device": "cpu",
        "method": "cpu"
    }


def _offload_to_disk(module: torch.nn.Module, name: str, temp_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Offload modulo su disco.
    """
    
    if temp_dir is None:
        temp_dir = tempfile.gettempdir()
    
    temp_path = Path(temp_dir) / f"offload_{name}.pt"
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    
    original_device = next(module.parameters()).device
    
    # Salva stato su disco
    torch.save(module.state_dict(), temp_path)
    
    # Rimuovi parametri dalla memoria
    for param in module.parameters():
        param.data = torch.empty(0)
    
    logger.debug(f"Modulo {name} salvato su disco: {temp_path}")
    
    return {
        "original_device": str(original_device),
        "temp_path": str(temp_path),
        "method": "disk"
    }


def restore_modules(
    modules: Dict[str, torch.nn.Module],
    offload_info: Dict[str, Any]
) -> None:
    """
    Ripristina moduli dall'offload.
    
    Args:
        modules: Dizionario di moduli da ripristinare
        offload_info: Informazioni di offload
    """
    
    offload_type = offload_info["offload_type"]
    module_info = offload_info["modules"]
    
    logger.info(f"Ripristino {len(modules)} moduli da {offload_type}")
    
    for name, module in modules.items():
        if name in module_info:
            info = module_info[name]
            
            if info["method"] == "cpu":
                _restore_from_cpu(module, info)
            elif info["method"] == "disk":
                _restore_from_disk(module, info)
    
    logger.info("Ripristino moduli completato")


def _restore_from_cpu(module: torch.nn.Module, info: Dict[str, Any]) -> None:
    """
    Ripristina modulo da CPU.
    """
    
    original_device = info["original_device"]
    
    # Sposta al device originale
    module.to(original_device)
    
    logger.debug(f"Modulo ripristinato da CPU a {original_device}")


def _restore_from_disk(module: torch.nn.Module, info: Dict[str, Any]) -> None:
    """
    Ripristina modulo da disco.
    """
    
    temp_path = info["temp_path"]
    original_device = info["original_device"]
    
    # Carica stato da disco
    state_dict = torch.load(temp_path, map_location=original_device)
    module.load_state_dict(state_dict)
    
    # Pulisci file temporaneo
    Path(temp_path).unlink(missing_ok=True)
    
    logger.debug(f"Modulo ripristinato da disco a {original_device}")


@contextmanager
def temporary_offload(
    modules: Dict[str, torch.nn.Module],
    offload_type: str = "cpu",
    temp_dir: Optional[str] = None
):
    """
    Context manager per offload temporaneo.
    
    Args:
        modules: Moduli da offload temporaneamente
        offload_type: Tipo di offload
        temp_dir: Directory temporanea
    
    Yields:
        Informazioni di offload
    """
    
    offload_info = offload_modules(modules, offload_type, temp_dir)
    
    try:
        yield offload_info
    finally:
        restore_modules(modules, offload_info)


class OffloadManager:
    """
    Manager per gestire offload di moduli in modo strutturato.
    """
    
    def __init__(self, temp_dir: Optional[str] = None):
        self.temp_dir = temp_dir or tempfile.gettempdir()
        self.offloaded_modules = {}
        self.offload_info = {}
    
    def offload(self, name: str, module: torch.nn.Module, offload_type: str = "cpu"):
        """
        Offload un singolo modulo.
        """
        
        if name in self.offloaded_modules:
            logger.warning(f"Modulo {name} già in offload")
            return
        
        self.offloaded_modules[name] = module
        
        info = offload_modules({name: module}, offload_type, self.temp_dir)
        self.offload_info[name] = info["modules"][name]
        
        logger.info(f"Modulo {name} offloaded su {offload_type}")
    
    def restore(self, name: str):
        """
        Ripristina un singolo modulo.
        """
        
        if name not in self.offloaded_modules:
            logger.warning(f"Modulo {name} non in offload")
            return
        
        module = self.offloaded_modules[name]
        info = self.offload_info[name]
        
        if info["method"] == "cpu":
            _restore_from_cpu(module, info)
        elif info["method"] == "disk":
            _restore_from_disk(module, info)
        
        del self.offloaded_modules[name]
        del self.offload_info[name]
        
        logger.info(f"Modulo {name} ripristinato")
    
    def restore_all(self):
        """
        Ripristina tutti i moduli in offload.
        """
        
        for name in list(self.offloaded_modules.keys()):
            self.restore(name)
    
    def cleanup(self):
        """
        Pulisce tutti i file temporanei.
        """
        
        for info in self.offload_info.values():
            if info["method"] == "disk" and "temp_path" in info:
                Path(info["temp_path"]).unlink(missing_ok=True)
        
        self.offloaded_modules.clear()
        self.offload_info.clear()
        
        logger.info("Cleanup offload completato")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()