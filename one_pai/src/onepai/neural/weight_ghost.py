"""
Pesi fantasma - Componente per identificare e analizzare i pesi neurali
che sono presenti nella rete ma raramente o mai attivati durante l'inferenza.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class WeightGhost:
    """Rappresenta un peso (o gruppo di pesi) raramente attivato"""
    layer_id: str
    weight_indices: Tuple[int, ...]  # Indici multidimensionali del peso
    activation_count: int
    total_opportunities: int
    activation_ratio: float
    magnitude: float
    context: Dict

class WeightGhost:
    """
    Sistema per identificare e analizzare i pesi "fantasma" in una rete neurale,
    ovvero pesi che sono presenti ma raramente o mai utilizzati durante
    il processo decisionale.
    """
    
    def __init__(self, ghost_threshold: float = 0.05):
        """
        Inizializza il rilevatore di pesi fantasma.
        
        Args:
            ghost_threshold: Soglia per considerare un peso come fantasma
        """
        self.ghost_threshold = ghost_threshold
        self.weight_stats = defaultdict(lambda: {
            "activation_count": 0,
            "total_opportunities": 0,
            "magnitude": 0.0,
            "contexts": []
        })
        self.ghost_weights = []
    
    def register_layer_weights(self, layer_id: str, weights: np.ndarray):
        """
        Registra i pesi di un layer per il tracciamento.
        
        Args:
            layer_id: ID del layer
            weights: Tensor dei pesi del layer
        """
        # Itera su tutti i pesi del layer
        for indices in np.ndindex(weights.shape):
            weight_key = (layer_id,) + indices
            self.weight_stats[weight_key]["magnitude"] = float(weights[indices])
    
    def update_activation_stats(self, layer_id: str, activation_pattern: np.ndarray, 
                               context: Optional[Dict] = None):
        """
        Aggiorna le statistiche di attivazione per i pesi di un layer.
        
        Args:
            layer_id: ID del layer
            activation_pattern: Pattern di attivazione (booleano o float)
            context: Contesto aggiuntivo opzionale
        """
        # Itera su tutti i pesi del layer
        for indices in np.ndindex(activation_pattern.shape):
            weight_key = (layer_id,) + indices
            stats = self.weight_stats[weight_key]
            stats["total_opportunities"] += 1
            
            # Controlla se il peso è stato attivato
            if isinstance(activation_pattern[indices], bool):
                activated = activation_pattern[indices]
            else:
                activated = activation_pattern[indices] > 0.5
            
            if activated:
                stats["activation_count"] += 1
            
            # Aggiungi il contesto se fornito
            if context:
                stats["contexts"].append(context)
    
    def identify_ghost_weights(self) -> List[WeightGhost]:
        """
        Identifica i pesi fantasma basandosi sulle statistiche raccolte.
        
        Returns:
            Lista di pesi identificati come fantasma
        """
        self.ghost_weights = []
        
        for weight_key, stats in self.weight_stats.items():
            layer_id = weight_key[0]
            weight_indices = weight_key[1:]
            
            # Calcola il rapporto di attivazione
            if stats["total_opportunities"] > 0:
                activation_ratio = stats["activation_count"] / stats["total_opportunities"]
            else:
                activation_ratio = 0.0
            
            # Controlla se il peso è un fantasma
            if activation_ratio <= self.ghost_threshold and stats["total_opportunities"] > 10:
                ghost = WeightGhost(
                    layer_id=layer_id,
                    weight_indices=weight_indices,
                    activation_count=stats["activation_count"],
                    total_opportunities=stats["total_opportunities"],
                    activation_ratio=activation_ratio,
                    magnitude=stats["magnitude"],
                    context={"activation_contexts": stats["contexts"]}
                )
                self.ghost_weights.append(ghost)
        
        return self.ghost_weights
    
    def get_ghost_weights_by_layer(self, layer_id: str) -> List[WeightGhost]:
        """
        Recupera i pesi fantasma per un layer specifico.
        
        Args:
            layer_id: ID del layer
            
        Returns:
            Lista di pesi fantasma nel layer specificato
        """
        return [ghost for ghost in self.ghost_weights if ghost.layer_id == layer_id]
    
    def analyze_ghost_patterns(self) -> Dict[str, any]:
        """
        Analizza i pattern nei pesi fantasma per identificare tendenze.
        
        Returns:
            Dizionario con i risultati dell'analisi
        """
        if not self.ghost_weights:
            return {"message": "No ghost weights identified"}
        
        # Analisi per layer
        layer_analysis = {}
        for ghost in self.ghost_weights:
            if ghost.layer_id not in layer_analysis:
                layer_analysis[ghost.layer_id] = {
                    "count": 0,
                    "avg_activation_ratio": 0,
                    "avg_magnitude": 0,
                    "magnitude_distribution": {"low": 0, "medium": 0, "high": 0}
                }
            
            layer_data = layer_analysis[ghost.layer_id]
            layer_data["count"] += 1
            
            # Aggiorna rapporto di attivazione medio
            layer_data["avg_activation_ratio"] = (
                (layer_data["avg_activation_ratio"] * (layer_data["count"] - 1) + ghost.activation_ratio) / 
                layer_data["count"]
            )
            
            # Aggiorna magnitudine media
            layer_data["avg_magnitude"] = (
                (layer_data["avg_magnitude"] * (layer_data["count"] - 1) + ghost.magnitude) / 
                layer_data["count"]
            )
            
            # Aggiorna distribuzione magnitudine
            if abs(ghost.magnitude) < 0.1:
                layer_data["magnitude_distribution"]["low"] += 1
            elif abs(ghost.magnitude) < 0.5:
                layer_data["magnitude_distribution"]["medium"] += 1
            else:
                layer_data["magnitude_distribution"]["high"] += 1
        
        # Analisi generale
        general_analysis = {
            "total_ghost_weights": len(self.ghost_weights),
            "avg_activation_ratio": np.mean([ghost.activation_ratio for ghost in self.ghost_weights]),
            "avg_magnitude": np.mean([ghost.magnitude for ghost in self.ghost_weights]),
            "layers_with_most_ghosts": sorted(
                [(layer_id, data["count"]) for layer_id, data in layer_analysis.items()],
                key=lambda x: x[1], reverse=True
            )[:3]
        }
        
        return {
            "general": general_analysis,
            "by_layer": layer_analysis
        }
    
    def get_weight_clusters(self) -> Dict[str, List[List[WeightGhost]]]:
        """
        Identifica cluster di pesi fantasma adiacenti nella rete.
        
        Returns:
            Dizionario che mappa layer ID a liste di cluster di pesi fantasma
        """
        clusters_by_layer = {}
        
        # Raggruppa i pesi fantasma per layer
        ghosts_by_layer = defaultdict(list)
        for ghost in self.ghost_weights:
            ghosts_by_layer[ghost.layer_id].append(ghost)
        
        # Per ogni layer, identifica i cluster di pesi fantasma
        for layer_id, ghosts in ghosts_by_layer.items():
            clusters = []
            visited = set()
            
            for ghost in ghosts:
                if ghost in visited:
                    continue
                
                # Avvia un nuovo cluster
                cluster = [ghost]
                visited.add(ghost)
                
                # Cerca pesi adiacenti non visitati
                self._find_adjacent_ghosts(ghost, ghosts, cluster, visited)
                
                # Aggiungi il cluster se ha più di un peso
                if len(cluster) > 1:
                    clusters.append(cluster)
            
            clusters_by_layer[layer_id] = clusters
        
        return clusters_by_layer
    
    def _find_adjacent_ghosts(self, ghost: WeightGhost, all_ghosts: List[WeightGhost], 
                             cluster: List[WeightGhost], visited: Set[WeightGhost]):
        """
        Trova ricorsivamente pesi fantasma adiacenti a un dato peso.
        
        Args:
            ghost: Peso fantasma di partenza
            all_ghosts: Lista di tutti i pesi fantasma nel layer
            cluster: Cluster in costruzione
            visited: Insieme dei pesi già visitati
        """
        for other_ghost in all_ghosts:
            if other_ghost in visited:
                continue
            
            # Controlla se i pesi sono adiacenti
            if self._are_adjacent(ghost, other_ghost):
                cluster.append(other_ghost)
                visited.add(other_ghost)
                
                # Continua la ricerca ricorsiva
                self._find_adjacent_ghosts(other_ghost, all_ghosts, cluster, visited)
    
    def _are_adjacent(self, ghost1: WeightGhost, ghost2: WeightGhost) -> bool:
        """
        Determina se due pesi fantasma sono adiacenti nella rete.
        
        Args:
            ghost1: Primo peso fantasma
            ghost2: Secondo peso fantasma
            
        Returns:
            True se i pesi sono adiacenti, False altrimenti
        """
        # I pesi sono adiacenti se differiscono di al massimo 1 in ogni dimensione
        indices1 = ghost1.weight_indices
        indices2 = ghost2.weight_indices
        
        if len(indices1) != len(indices2):
            return False
        
        for i1, i2 in zip(indices1, indices2):
            if abs(i1 - i2) > 1:
                return False
        
        return True