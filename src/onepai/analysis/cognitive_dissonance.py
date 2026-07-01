# -----------------------------------------------------------------------------
# File: onepai/analysis/cognitive_dissonance.py
# -----------------------------------------------------------------------------

"""
Analisi della Dissonanza Cognitiva
===================================

Modulo per rilevare e analizzare conflitti interni nei processi decisionali
dell'IA, identificando pensieri contraddittori e tensioni cognitive.

Author: Francesco Bulla (Brainverse)
"""

import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class ConflictingThought:
    """Rappresenta un pensiero in conflitto rilevato nell'IA."""
    thought_id: str
    layer_name: str
    conflict_type: str  # 'contradiction', 'ambivalence', 'paradox'
    confidence_score: float
    opposing_thoughts: List[str]
    timestamp: datetime
    context: Dict[str, Any]


@dataclass
class CognitiveDissonance:
    """Rappresenta una dissonanza cognitiva rilevata."""
    dissonance_id: str
    intensity: float  # 0.0 - 1.0
    conflicting_thoughts: List[ConflictingThought]
    resolution_attempts: List[str]
    duration: float  # in milliseconds
    impact_score: float


def detect_conflicting_thoughts(
    treasure_fragments: List[Dict[str, Any]], 
    conflict_threshold: float = 0.7
) -> List[ConflictingThought]:
    """
    Rileva pensieri in conflitto nei frammenti del tesoro.
    
    Args:
        treasure_fragments: Lista di frammenti di potenziale decifrati
        conflict_threshold: Soglia per considerare un conflitto significativo
        
    Returns:
        Lista di pensieri in conflitto rilevati
    """
    conflicting_thoughts = []
    
    for i, fragment in enumerate(treasure_fragments):
        if 'aborted_thoughts' not in fragment:
            continue
            
        aborted_thoughts = fragment['aborted_thoughts']
        
        # Analizza ogni pensiero abortito per conflitti
        for j, (thought, probability) in enumerate(aborted_thoughts):
            if probability < conflict_threshold:
                continue
                
            # Cerca pensieri opposti nello stesso frammento
            opposing_thoughts = []
            for k, (other_thought, other_prob) in enumerate(aborted_thoughts):
                if k != j and _are_conflicting(thought, other_thought):
                    opposing_thoughts.append(other_thought)
            
            if opposing_thoughts:
                conflict = ConflictingThought(
                    thought_id=f"conflict_{i}_{j}",
                    layer_name=fragment.get('layer_name', 'unknown'),
                    conflict_type=_classify_conflict_type(thought, opposing_thoughts),
                    confidence_score=probability,
                    opposing_thoughts=opposing_thoughts,
                    timestamp=datetime.now(),
                    context=fragment.get('context', {})
                )
                conflicting_thoughts.append(conflict)
    
    return conflicting_thoughts


def measure_internal_tension(
    treasure_fragments: List[Dict[str, Any]]
) -> Dict[str, float]:
    """
    Misura la tensione interna complessiva dell'IA.
    
    Args:
        treasure_fragments: Lista di frammenti di potenziale
        
    Returns:
        Dizionario con metriche di tensione interna
    """
    if not treasure_fragments:
        return {
            'overall_tension': 0.0,
            'decision_uncertainty': 0.0,
            'cognitive_load': 0.0,
            'resolution_efficiency': 0.0
        }
    
    tensions = []
    uncertainties = []
    cognitive_loads = []
    resolution_times = []
    
    for fragment in treasure_fragments:
        # Calcola tensione basata sui pensieri abortiti
        if 'aborted_thoughts' in fragment:
            aborted_count = len(fragment['aborted_thoughts'])
            total_thoughts = fragment.get('total_thoughts', aborted_count)
            tension = aborted_count / total_thoughts if total_thoughts > 0 else 0.0
            tensions.append(tension)
        
        # Calcola incertezza decisionale
        if 'decision_probabilities' in fragment:
            probs = fragment['decision_probabilities']
            uncertainty = _calculate_entropy(probs)
            uncertainties.append(uncertainty)
        
        # Calcola carico cognitivo
        if 'processing_time' in fragment and 'complexity_score' in fragment:
            cognitive_load = fragment['processing_time'] * fragment['complexity_score']
            cognitive_loads.append(cognitive_load)
        
        # Tempo di risoluzione dei conflitti
        if 'conflict_resolution_time' in fragment:
            resolution_times.append(fragment['conflict_resolution_time'])
    
    return {
        'overall_tension': np.mean(tensions) if tensions else 0.0,
        'decision_uncertainty': np.mean(uncertainties) if uncertainties else 0.0,
        'cognitive_load': np.mean(cognitive_loads) if cognitive_loads else 0.0,
        'resolution_efficiency': 1.0 / np.mean(resolution_times) if resolution_times else 0.0
    }


def analyze_decision_contradictions(
    treasure_fragments: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Analizza le contraddizioni nelle decisioni dell'IA.
    
    Args:
        treasure_fragments: Lista di frammenti di potenziale
        
    Returns:
        Analisi dettagliata delle contraddizioni decisionali
    """
    contradictions = []
    contradiction_patterns = {}
    temporal_contradictions = []
    
    for i, fragment in enumerate(treasure_fragments):
        if 'decisions' not in fragment:
            continue
            
        decisions = fragment['decisions']
        timestamp = fragment.get('timestamp', datetime.now())
        
        # Analizza contraddizioni interne al frammento
        for j, decision in enumerate(decisions):
            for k, other_decision in enumerate(decisions[j+1:], j+1):
                if _are_contradictory_decisions(decision, other_decision):
                    contradiction = {
                        'fragment_id': i,
                        'decision_1': decision,
                        'decision_2': other_decision,
                        'contradiction_type': 'internal',
                        'timestamp': timestamp,
                        'severity': _calculate_contradiction_severity(decision, other_decision)
                    }
                    contradictions.append(contradiction)
        
        # Cerca pattern di contraddizione
        decision_types = [d.get('type', 'unknown') for d in decisions]
        for decision_type in decision_types:
            if decision_type not in contradiction_patterns:
                contradiction_patterns[decision_type] = 0
            contradiction_patterns[decision_type] += 1
    
    # Analizza contraddizioni temporali (tra frammenti diversi)
    for i in range(len(treasure_fragments) - 1):
        current_fragment = treasure_fragments[i]
        next_fragment = treasure_fragments[i + 1]
        
        if 'final_decision' in current_fragment and 'final_decision' in next_fragment:
            if _are_contradictory_decisions(
                current_fragment['final_decision'], 
                next_fragment['final_decision']
            ):
                temporal_contradictions.append({
                    'fragment_1': i,
                    'fragment_2': i + 1,
                    'time_gap': _calculate_time_gap(current_fragment, next_fragment),
                    'contradiction_severity': _calculate_contradiction_severity(
                        current_fragment['final_decision'],
                        next_fragment['final_decision']
                    )
                })
    
    return {
        'total_contradictions': len(contradictions),
        'internal_contradictions': len([c for c in contradictions if c['contradiction_type'] == 'internal']),
        'temporal_contradictions': len(temporal_contradictions),
        'contradiction_patterns': contradiction_patterns,
        'average_severity': np.mean([c['severity'] for c in contradictions]) if contradictions else 0.0,
        'most_contradictory_types': sorted(
            contradiction_patterns.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:5]
    }


# Funzioni di supporto private

def _are_conflicting(thought1: str, thought2: str) -> bool:
    """
    Determina se due pensieri sono in conflitto.
    Implementazione semplificata basata su parole chiave opposte.
    """
    opposing_pairs = [
        ('yes', 'no'), ('true', 'false'), ('accept', 'reject'),
        ('positive', 'negative'), ('good', 'bad'), ('right', 'wrong'),
        ('should', 'should not'), ('can', 'cannot'), ('will', 'will not')
    ]
    
    thought1_lower = thought1.lower()
    thought2_lower = thought2.lower()
    
    for word1, word2 in opposing_pairs:
        if (word1 in thought1_lower and word2 in thought2_lower) or \
           (word2 in thought1_lower and word1 in thought2_lower):
            return True
    
    return False


def _classify_conflict_type(thought: str, opposing_thoughts: List[str]) -> str:
    """
    Classifica il tipo di conflitto basato sui pensieri opposti.
    """
    if len(opposing_thoughts) == 1:
        return 'contradiction'
    elif len(opposing_thoughts) > 1:
        return 'ambivalence'
    else:
        return 'paradox'


def _calculate_entropy(probabilities: List[float]) -> float:
    """
    Calcola l'entropia di Shannon per una distribuzione di probabilità.
    """
    probs = np.array(probabilities)
    probs = probs[probs > 0]  # Rimuovi probabilità zero
    if len(probs) == 0:
        return 0.0
    return -np.sum(probs * np.log2(probs))


def _are_contradictory_decisions(decision1: Dict[str, Any], decision2: Dict[str, Any]) -> bool:
    """
    Determina se due decisioni sono contraddittorie.
    """
    # Implementazione semplificata
    if decision1.get('action') == decision2.get('action'):
        return False
    
    # Controlla se le azioni sono opposte
    action1 = decision1.get('action', '').lower()
    action2 = decision2.get('action', '').lower()
    
    opposing_actions = [
        ('accept', 'reject'), ('approve', 'deny'), ('start', 'stop'),
        ('enable', 'disable'), ('allow', 'block'), ('create', 'delete')
    ]
    
    for act1, act2 in opposing_actions:
        if (act1 in action1 and act2 in action2) or \
           (act2 in action1 and act1 in action2):
            return True
    
    return False


def _calculate_contradiction_severity(decision1: Dict[str, Any], decision2: Dict[str, Any]) -> float:
    """
    Calcola la gravità di una contraddizione tra due decisioni.
    """
    # Implementazione semplificata basata sulla confidenza delle decisioni
    conf1 = decision1.get('confidence', 0.5)
    conf2 = decision2.get('confidence', 0.5)
    
    # Più alta è la confidenza in entrambe le decisioni contraddittorie,
    # più grave è la contraddizione
    return (conf1 + conf2) / 2.0


def _calculate_time_gap(fragment1: Dict[str, Any], fragment2: Dict[str, Any]) -> float:
    """
    Calcola il gap temporale tra due frammenti.
    """
    time1 = fragment1.get('timestamp', datetime.now())
    time2 = fragment2.get('timestamp', datetime.now())
    
    if isinstance(time1, str):
        time1 = datetime.fromisoformat(time1)
    if isinstance(time2, str):
        time2 = datetime.fromisoformat(time2)
    
    return abs((time2 - time1).total_seconds())