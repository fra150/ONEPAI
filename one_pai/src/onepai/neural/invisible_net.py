"""
Rete dell'invisibile - Componente per la rilevazione e analisi dei percorsi neurali
che non sono stati attivati durante il processo decisionale dell'IA.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class InvisiblePath:
    """Rappresenta un percorso neurale invisibile (non attivato)"""
    layer_id: str
    neuron_indices: List[int]
    potential_activation: float
    context: Dict

class InvisibleNet:
    """
    Rete specializzata nell'identificare e mappare i percorsi neurali
    che avrebbero potuto essere attivati ma non lo sono stati.
    """
    
    def __init__(self, model_architecture: Dict):
        """
        Inizializza la rete invisibile con l'architettura del modello target.
        
        Args:
            model_architecture: Dizionario che descrive l'architettura del modello
        """
        self.model_arch = model_architecture
        self.invisible_paths = {}
        self.activation_threshold = 0.1  # Soglia per considerare un'attivazione potenziale
        
    def scan_for_invisible_paths(self, activations: Dict[str, np.ndarray],
                                decisions: Dict[str, int]) -> List[InvisiblePath]:
        """
        Scansiona le attivazioni per identificare percorsi invisibili.
        
        Args:
            activations: Dizionario con le attivazioni per layer
            decisions: Dizionario con le decisioni prese per layer
            
        Returns:
            Lista di percorsi invisibili identificati
        """
        invisible_paths = []
        
        for layer_id, activation_tensor in activations.items():
            # Trova neuroni con attivazione potenziale ma non attivati
            sub_threshold_neurons = self._find_sub_threshold_neurons(activation_tensor)
            
            for neuron_idx in sub_threshold_neurons:
                potential_activation = float(activation_tensor.flatten()[neuron_idx])
                if potential_activation > self.activation_threshold:
                    path = InvisiblePath(
                        layer_id=layer_id,
                        neuron_indices=[neuron_idx],
                        potential_activation=potential_activation,
                        context={"decision": decisions.get(layer_id, None)}
                    )
                    invisible_paths.append(path)
        
        self.invisible_paths.update({path.layer_id: path for path in invisible_paths})
        return invisible_paths
    
    def _find_sub_threshold_neurons(self, activation_tensor: np.ndarray) -> List[int]:
        """
        Trova gli indici dei neuroni con attivazione sotto soglia ma potenzialmente rilevanti.
        
        Args:
            activation_tensor: Tensor delle attivazioni
            
        Returns:
            Lista di indici dei neuroni rilevanti ma sotto soglia
        """
        flattened = activation_tensor.flatten()
        # Trova neuroni con attivazione tra la soglia minima e massima
        return [i for i, val in enumerate(flattened)
                if self.activation_threshold < val < 0.5]
    
    def get_invisible_paths_by_layer(self, layer_id: str) -> Optional[List[InvisiblePath]]:
        """
        Recupera i percorsi invisibili per un layer specifico.
        
        Args:
            layer_id: ID del layer
            
        Returns:
            Lista di percorsi invisibili o None se non presenti
        """
        return [path for path in self.invisible_paths.values()
                if path.layer_id == layer_id]
    
    def analyze_invisible_patterns(self) -> Dict[str, List[Dict]]:
        """
        Analizza i pattern nei percorsi invisibili per identificare tendenze.
        
        Returns:
            Dizionario con i pattern identificati per layer
        """
        patterns = {}
        
        for layer_id, paths in self._group_paths_by_layer().items():
            layer_patterns = []
            
            # Analizza la distribuzione delle attivazioni potenziali
            activations = [path.potential_activation for path in paths]
            if activations:
                layer_patterns.append({
                    "type": "activation_distribution",
                    "mean": np.mean(activations),
                    "std": np.std(activations),
                    "max": np.max(activations),
                    "min": np.min(activations)
                })
            
            # Analizza la frequenza di neuroni invisibili
            neuron_counts = {}
            for path in paths:
                for neuron_idx in path.neuron_indices:
                    neuron_counts[neuron_idx] = neuron_counts.get(neuron_idx, 0) + 1
            
            if neuron_counts:
                layer_patterns.append({
                    "type": "frequent_invisible_neurons",
                    "neurons": sorted(neuron_counts.items(), key=lambda x: x[1], reverse=True)[:5]
                })
            
            patterns[layer_id] = layer_patterns
        
        return patterns
    
    def _group_paths_by_layer(self) -> Dict[str, List[InvisiblePath]]:
        """Raggruppa i percorsi invisibili per layer."""
        grouped = {}
        for path in self.invisible_paths.values():
            if path.layer_id not in grouped:
                grouped[path.layer_id] = []
            grouped[path.layer_id].append(path)
        return grouped