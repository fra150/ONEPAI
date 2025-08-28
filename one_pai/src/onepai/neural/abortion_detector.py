"""
Rilevatore di pensieri abortiti - Componente per identificare e analizzare
i percorsi decisionali che sono stati iniziati ma non completati.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime

@dataclass
class AbortedThought:
    """Rappresenta un pensiero abortito durante il processo decisionale"""
    thought_id: str
    layer_id: str
    start_time: datetime
    end_time: datetime
    activation_trace: List[float]
    decision_point: int
    abortion_reason: str
    context: Dict[str, Any]

class AbortionDetector:
    """
    Sistema per rilevare e analizzare i pensieri abortiti durante il processo
    decisionale di un'IA, identificando i punti in cui un percorso potenziale
    viene abbandonato.
    """
    
    def __init__(self, abortion_threshold: float = 0.3):
        """
        Inizializza il rilevatore di pensieri abortiti.
        
        Args:
            abortion_threshold: Soglia per considerare un pensiero come abortito
        """
        self.abortion_threshold = abortion_threshold
        self.aborted_thoughts = []
        self.active_thoughts = {}  # Pensieri attualmente in corso
        self.thought_counter = 0
    
    def start_thought_tracking(self, layer_id: str, initial_activation: float) -> str:
        """
        Inizia il tracciamento di un nuovo pensiero.
        
        Args:
            layer_id: ID del layer in cui inizia il pensiero
            initial_activation: Valore iniziale di attivazione
            
        Returns:
            ID del pensiero avviato
        """
        thought_id = f"thought_{self.thought_counter}"
        self.thought_counter += 1
        
        self.active_thoughts[thought_id] = {
            "layer_id": layer_id,
            "start_time": datetime.now(),
            "activation_trace": [initial_activation],
            "context": {}
        }
        
        return thought_id
    
    def update_thought_activation(self, thought_id: str, activation: float, 
                                 context: Optional[Dict] = None) -> bool:
        """
        Aggiorna l'attivazione di un pensiero in tracciamento.
        
        Args:
            thought_id: ID del pensiero
            activation: Nuovo valore di attivazione
            context: Contesto aggiuntivo opzionale
            
        Returns:
            True se il pensiero è ancora attivo, False se è stato abortito
        """
        if thought_id not in self.active_thoughts:
            return False
        
        # Aggiorna la traccia di attivazione
        self.active_thoughts[thought_id]["activation_trace"].append(activation)
        
        # Aggiorna il contesto se fornito
        if context:
            self.active_thoughts[thought_id]["context"].update(context)
        
        # Controlla se il pensiero è abortito
        if activation < self.abortion_threshold:
            self._finalize_aborted_thought(thought_id, "activation_below_threshold")
            return False
        
        return True
    
    def _finalize_aborted_thought(self, thought_id: str, reason: str):
        """
        Finalizza un pensiero abortito e lo sposta nella lista dei pensieri abortiti.
        
        Args:
            thought_id: ID del pensiero
            reason: Motivo dell'aborto
        """
        if thought_id not in self.active_thoughts:
            return
        
        thought_data = self.active_thoughts[thought_id]
        
        aborted_thought = AbortedThought(
            thought_id=thought_id,
            layer_id=thought_data["layer_id"],
            start_time=thought_data["start_time"],
            end_time=datetime.now(),
            activation_trace=thought_data["activation_trace"],
            decision_point=len(thought_data["activation_trace"]) - 1,
            abortion_reason=reason,
            context=thought_data["context"]
        )
        
        self.aborted_thoughts.append(aborted_thought)
        del self.active_thoughts[thought_id]
    
    def complete_thought(self, thought_id: str, context: Optional[Dict] = None):
        """
        Completa un pensiero senza abortirlo.
        
        Args:
            thought_id: ID del pensiero
            context: Contesto aggiuntivo opzionale
        """
        if thought_id not in self.active_thoughts:
            return
        
        # Aggiorna il contesto se fornito
        if context:
            self.active_thoughts[thought_id]["context"].update(context)
        
        # Rimuovi il pensiero da quelli attivi senza classificarlo come abortito
        del self.active_thoughts[thought_id]
    
    def get_aborted_thoughts_by_layer(self, layer_id: str) -> List[AbortedThought]:
        """
        Recupera i pensieri abortiti per un layer specifico.
        
        Args:
            layer_id: ID del layer
            
        Returns:
            Lista di pensieri abortiti nel layer specificato
        """
        return [thought for thought in self.aborted_thoughts 
                if thought.layer_id == layer_id]
    
    def analyze_abortion_patterns(self) -> Dict[str, Any]:
        """
        Analizza i pattern nei pensieri abortiti per identificare tendenze.
        
        Returns:
            Dizionario con i risultati dell'analisi
        """
        if not self.aborted_thoughts:
            return {"message": "No aborted thoughts to analyze"}
        
        # Analisi per layer
        layer_analysis = {}
        for thought in self.aborted_thoughts:
            if thought.layer_id not in layer_analysis:
                layer_analysis[thought.layer_id] = {
                    "count": 0,
                    "avg_duration": 0,
                    "avg_decision_point": 0,
                    "reasons": {}
                }
            
            layer_data = layer_analysis[thought.layer_id]
            layer_data["count"] += 1
            
            # Calcola durata in secondi
            duration = (thought.end_time - thought.start_time).total_seconds()
            layer_data["avg_duration"] = (
                (layer_data["avg_duration"] * (layer_data["count"] - 1) + duration) / 
                layer_data["count"]
            )
            
            # Aggiorna punto decisionale medio
            layer_data["avg_decision_point"] = (
                (layer_data["avg_decision_point"] * (layer_data["count"] - 1) + thought.decision_point) / 
                layer_data["count"]
            )
            
            # Aggiorna conteggio motivi
            reason = thought.abortion_reason
            layer_data["reasons"][reason] = layer_data["reasons"].get(reason, 0) + 1
        
        # Analisi generale
        general_analysis = {
            "total_aborted_thoughts": len(self.aborted_thoughts),
            "most_common_reason": self._get_most_common_abortion_reason(),
            "layers_with_most_aborted_thoughts": sorted(
                [(layer_id, data["count"]) for layer_id, data in layer_analysis.items()],
                key=lambda x: x[1], reverse=True
            )[:3]
        }
        
        return {
            "general": general_analysis,
            "by_layer": layer_analysis
        }
    
    def _get_most_common_abortion_reason(self) -> Tuple[str, int]:
        """
        Identifica il motivo più comune di aborto dei pensieri.
        
        Returns:
            Tupla (motivo, conteggio) del motivo più comune
        """
        reason_counts = {}
        for thought in self.aborted_thoughts:
            reason = thought.abortion_reason
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
        
        if not reason_counts:
            return ("none", 0)
        
        return max(reason_counts.items(), key=lambda x: x[1])