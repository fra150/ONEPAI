# ----------------------------------------------------------------------------- 
 # File: onepai/core/registry.py 
 # ----------------------------------------------------------------------------- 
 
 """Indice in‑memory delle tracce shadow raccolte.""" 
 
 from __future__ import annotations 
 
 from collections import defaultdict 
 from typing import Dict, List 
 
 from .observer import Observation 
 
 
 class ShadowRegistry: 
     """Registrazione veloce di observation indicizzate per layer.""" 
 
     def __init__(self) -> None: 
         self._by_layer: Dict[str, List[Observation]] = defaultdict(list) 
 
     def add(self, obs: Observation) -> None: 
         layer = obs.name.split(":", 1)[0] 
         self._by_layer[layer].append(obs) 
 
     def get(self, layer: str) -> List[Observation]: 
         return self._by_layer.get(layer, []) 
 
     def summary(self) -> Dict[str, int]: 
         """Restituisce conteggio record per layer.""" 
 
         return {k: len(v) for k, v in self._by_layer.items()}