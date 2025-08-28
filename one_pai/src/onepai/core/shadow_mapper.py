# ----------------------------------------------------------------------------- 
 # File: onepai/core/shadow_mapper.py 
 # ----------------------------------------------------------------------------- 
 
 """Costruisce una mappa concettuale dell'invisibile (ombre/silenzi).""" 
 
 from __future__ import annotations 
 
 from typing import Dict, List, Tuple 
 
 import networkx as nx  # type: ignore[import] 
 
 from .registry import ShadowRegistry 
 from .observer import Observation 
 
 
 class ShadowMapper: 
     """Traduce indici shadow in un grafo di relazioni invisibili.""" 
 
     def __init__(self, registry: ShadowRegistry): 
         self.registry = registry 
         self.graph: nx.DiGraph = nx.DiGraph()