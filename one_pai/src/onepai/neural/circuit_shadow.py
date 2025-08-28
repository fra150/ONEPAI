"""
Ombra dei circuiti - Componente per mappare e analizzare i circuiti neurali
che sono presenti nell'architettura ma raramente utilizzati durante l'inferenza.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from collections import defaultdict
import networkx as nx

@dataclass
class CircuitShadow:
    """Rappresenta un circuito neurale raramente utilizzato"""
    circuit_id: str
    layer_ids: List[str]
    neuron_indices: List[Tuple[int, ...]]
    activation_frequency: float
    avg_activation_strength: float
    context: Dict

class CircuitShadow:
    """
    Sistema per identificare e analizzare i circuiti neurali "ombra", ovvero
    percorsi attraverso la rete che sono presenti nell'architettura ma
    raramente attivati durante l'inferenza.
    """
    
    def __init__(self, shadow_threshold: float = 0.1):
        """
        Inizializza il rilevatore di circuiti ombra.
        
        Args:
            shadow_threshold: Soglia per considerare un circuito come ombra
        """
        self.shadow_threshold = shadow_threshold
        self.circuit_stats = defaultdict(lambda: {
            "activation_count": 0,
            "total_opportunities": 0,
            "activation_strengths": [],
            "contexts": []
        })
        self.circuit_shadows = []
        self.network_graph = nx.DiGraph()
    
    def build_network_graph(self, model_architecture: Dict):
        """
        Costruisce il grafo della rete neurale basandosi sull'architettura del modello.
        
        Args:
            model_architecture: Dizionario che descrive l'architettura del modello
        """
        # Svuota il grafo esistente
        self.network_graph.clear()
        
        # Aggiungi i nodi (layer)
        for layer_info in model_architecture.get("layers", []):
            layer_id = layer_info["id"]
            self.network_graph.add_node(layer_id, **layer_info)
        
        # Aggiungi gli archi (connessioni tra layer)
        for connection in model_architecture.get("connections", []):
            source = connection["source"]
            target = connection["target"]
            self.network_graph.add_edge(source, target, **connection.get("properties", {}))
    
    def register_circuit(self, circuit_id: str, layer_ids: List[str], 
                        neuron_indices: List[Tuple[int, ...]]):
        """
        Registra un circuito per il tracciamento.
        
        Args:
            circuit_id: ID univoco del circuito
            layer_ids: Lista di ID dei layer che compongono il circuito
            neuron_indices: Lista di indici dei neuroni per ogni layer
        """
        # Verifica che il circuito sia valido nel grafo
        if not self._is_valid_circuit(layer_ids):
            return False
        
        # Inizializza le statistiche per il circuito
        self.circuit_stats[circuit_id] = {
            "layer_ids": layer_ids,
            "neuron_indices": neuron_indices,
            "activation_count": 0,
            "total_opportunities": 0,
            "activation_strengths": [],
            "contexts": []
        }
        
        return True
    
    def _is_valid_circuit(self, layer_ids: List[str]) -> bool:
        """
        Verifica che una sequenza di layer formi un circuito valido nel grafo.
        
        Args:
            layer_ids: Lista di ID dei layer
            
        Returns:
            True se il circuito è valido, False altrimenti
        """
        # Verifica che tutti i layer esistano nel grafo
        for layer_id in layer_ids:
            if layer_id not in self.network_graph.nodes:
                return False
        
        # Verifica che i layer siano connessi in sequenza
        for i in range(len(layer_ids) - 1):
            if not self.network_graph.has_edge(layer_ids[i], layer_ids[i+1]):
                return False
        
        return True
    
    def update_circuit_activation(self, circuit_id: str, activation_strength: float,
                                 context: Optional[Dict] = None):
        """
        Aggiorna le statistiche di attivazione per un circuito.
        
        Args:
            circuit_id: ID del circuito
            activation_strength: Intensità dell'attivazione (0-1)
            context: Contesto aggiuntivo opzionale
        """
        if circuit_id not in self.circuit_stats:
            return False
        
        stats = self.circuit_stats[circuit_id]
        stats["total_opportunities"] += 1
        
        # Controlla se il circuito è stato attivato
        if activation_strength > 0.5:
            stats["activation_count"] += 1
            stats["activation_strengths"].append(activation_strength)
        
        # Aggiungi il contesto se fornito
        if context:
            stats["contexts"].append(context)
        
        return True
    
    def identify_circuit_shadows(self) -> List[CircuitShadow]:
        """
        Identifica i circuiti ombra basandosi sulle statistiche raccolte.
        
        Returns:
            Lista di circuiti identificati come ombra
        """
        self.circuit_shadows = []
        
        for circuit_id, stats in self.circuit_stats.items():
            # Calcola la frequenza di attivazione
            if stats["total_opportunities"] > 0:
                activation_frequency = stats["activation_count"] / stats["total_opportunities"]
            else:
                activation_frequency = 0.0
            
            # Calcola l'intensità media di attivazione
            if stats["activation_strengths"]:
                avg_activation_strength = np.mean(stats["activation_strengths"])
            else:
                avg_activation_strength = 0.0
            
            # Controlla se il circuito è un'ombra
            if activation_frequency <= self.shadow_threshold and stats["total_opportunities"] > 5:
                shadow = CircuitShadow(
                    circuit_id=circuit_id,
                    layer_ids=stats["layer_ids"],
                    neuron_indices=stats["neuron_indices"],
                    activation_frequency=activation_frequency,
                    avg_activation_strength=avg_activation_strength,
                    context={"activation_contexts": stats["contexts"]}
                )
                self.circuit_shadows.append(shadow)
        
        return self.circuit_shadows
    
    def get_circuit_shadows_by_layer(self, layer_id: str) -> List[CircuitShadow]:
        """
        Recupera i circuiti ombra che includono un layer specifico.
        
        Args:
            layer_id: ID del layer
            
        Returns:
            Lista di circuiti ombra che includono il layer specificato
        """
        return [shadow for shadow in self.circuit_shadows 
                if layer_id in shadow.layer_ids]
    
    def analyze_shadow_patterns(self) -> Dict[str, any]:
        """
        Analizza i pattern nei circuiti ombra per identificare tendenze.
        
        Returns:
            Dizionario con i risultati dell'analisi
        """
        if not self.circuit_shadows:
            return {"message": "No circuit shadows identified"}
        
        # Analisi per layer
        layer_analysis = {}
        for shadow in self.circuit_shadows:
            for layer_id in shadow.layer_ids:
                if layer_id not in layer_analysis:
                    layer_analysis[layer_id] = {
                        "count": 0,
                        "circuits": [],
                        "avg_activation_frequency": 0,
                        "avg_activation_strength": 0
                    }
                
                layer_data = layer_analysis[layer_id]
                layer_data["count"] += 1
                layer_data["circuits"].append(shadow.circuit_id)
        
        # Calcola le medie per ogni layer
        for layer_id, data in layer_analysis.items():
            shadows_in_layer = [s for s in self.circuit_shadows if layer_id in s.layer_ids]
            
            if shadows_in_layer:
                data["avg_activation_frequency"] = np.mean([s.activation_frequency for s in shadows_in_layer])
                data["avg_activation_strength"] = np.mean([s.avg_activation_strength for s in shadows_in_layer])
        
        # Analisi generale
        general_analysis = {
            "total_circuit_shadows": len(self.circuit_shadows),
            "avg_activation_frequency": np.mean([s.activation_frequency for s in self.circuit_shadows]),
            "avg_activation_strength": np.mean([s.avg_activation_strength for s in self.circuit_shadows]),
            "layers_with_most_shadows": sorted(
                [(layer_id, data["count"]) for layer_id, data in layer_analysis.items()],
                key=lambda x: x[1], reverse=True
            )[:3]
        }
        
        return {
            "general": general_analysis,
            "by_layer": layer_analysis
        }
    
    def find_shadow_clusters(self) -> List[List[CircuitShadow]]:
        """
        Identifica cluster di circuiti ombra che condividono layer o neuroni.
        
        Returns:
            Lista di cluster di circuiti ombra correlati
        """
        if not self.circuit_shadows:
            return []
        
        # Costruisci un grafo dei circuiti ombra
        shadow_graph = nx.Graph()
        
        # Aggiungi i nodi (circuiti ombra)
        for shadow in self.circuit_shadows:
            shadow_graph.add_node(shadow.circuit_id, shadow=shadow)
        
        # Aggiungi gli archi tra circuiti che condividono layer o neuroni
        for i, shadow1 in enumerate(self.circuit_shadows):
            for shadow2 in self.circuit_shadows[i+1:]:
                if self._are_circuits_related(shadow1, shadow2):
                    shadow_graph.add_edge(shadow1.circuit_id, shadow2.circuit_id)
        
        # Trova le componenti connesse (cluster)
        clusters = []
        for component in nx.connected_components(shadow_graph):
            cluster = [self.circuit_shadows_by_id[circuit_id] for circuit_id in component]
            clusters.append(cluster)
        
        return clusters
    
    @property
    def circuit_shadows_by_id(self) -> Dict[str, CircuitShadow]:
        """Restituisce un dizionario che mappa ID a circuiti ombra."""
        return {shadow.circuit_id: shadow for shadow in self.circuit_shadows}
    
    def _are_circuits_related(self, shadow1: CircuitShadow, shadow2: CircuitShadow) -> bool:
        """
        Determina se due circuiti ombra sono correlati.
        
        Args:
            shadow1: Primo circuito ombra
            shadow2: Secondo circuito ombra
            
        Returns:
            True se i circuiti sono correlati, False altrimenti
        """
        # Controlla se condividono almeno un layer
        shared_layers = set(shadow1.layer_ids) & set(shadow2.layer_ids)
        if shared_layers:
            return True
        
        # Controlla se i neuroni sono adiacenti in qualche layer
        # (Questa è una semplificazione; un'implementazione più completa
        # considererebbe la topologia della rete)
        return False