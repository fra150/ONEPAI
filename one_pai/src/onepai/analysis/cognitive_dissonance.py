# -----------------------------------------------------------------------------
# File: onepai/analysis/dream_logic.py
# -----------------------------------------------------------------------------

"""
Analisi della Logica Onirica
============================

Modulo per rilevare e analizzare pattern di pensiero non-lineari e logica
onirica nei processi dell'IA, identificando connessioni surreali e salti logici.

Author: Francesco Bulla (Brainverse)
"""

import numpy as np
from typing import Dict, List, Any, Tuple, Optional, Set
from dataclasses import dataclass
from datetime import datetime
import re
from collections import defaultdict


@dataclass
class DreamSequence:
    """Rappresenta una sequenza di pensiero onirico rilevata."""
    sequence_id: str
    thoughts: List[str]
    logic_jumps: List[Tuple[int, int, float]]  # (from_idx, to_idx, jump_score)
    surreal_score: float
    coherence_level: float
    timestamp: datetime
    context: Dict[str, Any]


@dataclass
class LogicJump:
    """Rappresenta un salto logico nel pensiero."""
    jump_id: str
    from_concept: str
    to_concept: str
    jump_type: str  # 'associative', 'metaphorical', 'surreal', 'random'
    strength: float
    semantic_distance: float
    context: Dict[str, Any]


def detect_dream_sequences(
    treasure_fragments: List[Dict[str, Any]], 
    surreal_threshold: float = 0.6
) -> List[DreamSequence]:
    """
    Rileva sequenze di pensiero onirico nei frammenti del tesoro.
    
    Args:
        treasure_fragments: Lista di frammenti di potenziale decifrati
        surreal_threshold: Soglia per considerare una sequenza surreale
        
    Returns:
        Lista di sequenze oniriche rilevate
    """
    dream_sequences = []
    
    for i, fragment in enumerate(treasure_fragments):
        if 'thought_sequence' not in fragment:
            continue
            
        thoughts = fragment['thought_sequence']
        if len(thoughts) < 3:  # Sequenze troppo corte per analisi onirica
            continue
        
        # Analizza salti logici nella sequenza
        logic_jumps = _analyze_logic_jumps(thoughts)
        
        # Calcola punteggio surreale
        surreal_score = _calculate_surreal_score(thoughts, logic_jumps)
        
        # Calcola livello di coerenza
        coherence_level = _calculate_coherence_level(thoughts)
        
        if surreal_score >= surreal_threshold:
            dream_seq = DreamSequence(
                sequence_id=f"dream_{i}",
                thoughts=thoughts,
                logic_jumps=logic_jumps,
                surreal_score=surreal_score,
                coherence_level=coherence_level,
                timestamp=datetime.now(),
                context=fragment.get('context', {})
            )
            dream_sequences.append(dream_seq)
    
    return dream_sequences


def analyze_non_linear_patterns(
    treasure_fragments: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Analizza pattern di pensiero non-lineari nell'IA.
    
    Args:
        treasure_fragments: Lista di frammenti di potenziale
        
    Returns:
        Analisi dettagliata dei pattern non-lineari
    """
    non_linear_patterns = {
        'circular_thoughts': [],
        'recursive_loops': [],
        'tangential_shifts': [],
        'associative_chains': [],
        'metaphorical_leaps': []
    }
    
    pattern_frequencies = defaultdict(int)
    complexity_scores = []
    
    for fragment in treasure_fragments:
        if 'thought_sequence' not in fragment:
            continue
            
        thoughts = fragment['thought_sequence']
        
        # Rileva pensieri circolari
        circular = _detect_circular_thoughts(thoughts)
        non_linear_patterns['circular_thoughts'].extend(circular)
        pattern_frequencies['circular'] += len(circular)
        
        # Rileva loop ricorsivi
        recursive = _detect_recursive_loops(thoughts)
        non_linear_patterns['recursive_loops'].extend(recursive)
        pattern_frequencies['recursive'] += len(recursive)
        
        # Rileva spostamenti tangenziali
        tangential = _detect_tangential_shifts(thoughts)
        non_linear_patterns['tangential_shifts'].extend(tangential)
        pattern_frequencies['tangential'] += len(tangential)
        
        # Rileva catene associative
        associative = _detect_associative_chains(thoughts)
        non_linear_patterns['associative_chains'].extend(associative)
        pattern_frequencies['associative'] += len(associative)
        
        # Rileva salti metaforici
        metaphorical = _detect_metaphorical_leaps(thoughts)
        non_linear_patterns['metaphorical_leaps'].extend(metaphorical)
        pattern_frequencies['metaphorical'] += len(metaphorical)
        
        # Calcola complessità del pattern
        complexity = _calculate_pattern_complexity(thoughts)
        complexity_scores.append(complexity)
    
    return {
        'patterns': non_linear_patterns,
        'pattern_frequencies': dict(pattern_frequencies),
        'average_complexity': np.mean(complexity_scores) if complexity_scores else 0.0,
        'total_non_linear_events': sum(pattern_frequencies.values()),
        'dominant_pattern': max(pattern_frequencies.items(), key=lambda x: x[1])[0] if pattern_frequencies else None
    }


def measure_surreal_connections(
    treasure_fragments: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Misura le connessioni surreali tra concetti nell'IA.
    
    Args:
        treasure_fragments: Lista di frammenti di potenziale
        
    Returns:
        Analisi delle connessioni surreali
    """
    surreal_connections = []
    concept_network = defaultdict(set)
    surreal_scores = []
    
    for fragment in treasure_fragments:
        if 'concepts' not in fragment:
            continue
            
        concepts = fragment['concepts']
        
        # Analizza tutte le coppie di concetti
        for i, concept1 in enumerate(concepts):
            for j, concept2 in enumerate(concepts[i+1:], i+1):
                # Calcola distanza semantica
                semantic_distance = _calculate_semantic_distance(concept1, concept2)
                
                # Se la distanza è alta ma i concetti sono connessi,
                # potrebbe essere una connessione surreale
                if semantic_distance > 0.7:  # Soglia per distanza semantica alta
                    connection_strength = _calculate_connection_strength(
                        concept1, concept2, fragment
                    )
                    
                    if connection_strength > 0.5:  # Connessione forte nonostante distanza
                        surreal_score = semantic_distance * connection_strength
                        surreal_connections.append({
                            'concept1': concept1,
                            'concept2': concept2,
                            'semantic_distance': semantic_distance,
                            'connection_strength': connection_strength,
                            'surreal_score': surreal_score,
                            'context': fragment.get('context', {})
                        })
                        surreal_scores.append(surreal_score)
                
                # Costruisci rete di concetti
                concept_network[concept1].add(concept2)
                concept_network[concept2].add(concept1)
    
    # Analizza la rete di concetti per pattern surreali
    network_analysis = _analyze_concept_network(concept_network)
    
    return {
        'surreal_connections': surreal_connections,
        'total_connections': len(surreal_connections),
        'average_surreal_score': np.mean(surreal_scores) if surreal_scores else 0.0,
        'max_surreal_score': max(surreal_scores) if surreal_scores else 0.0,
        'concept_network_analysis': network_analysis,
        'most_surreal_pairs': sorted(
            surreal_connections, 
            key=lambda x: x['surreal_score'], 
            reverse=True
        )[:10]
    }


# Funzioni di supporto private

def _analyze_logic_jumps(thoughts: List[str]) -> List[Tuple[int, int, float]]:
    """
    Analizza i salti logici in una sequenza di pensieri.
    """
    logic_jumps = []
    
    for i in range(len(thoughts) - 1):
        current_thought = thoughts[i]
        next_thought = thoughts[i + 1]
        
        # Calcola la distanza semantica tra pensieri consecutivi
        semantic_distance = _calculate_semantic_distance(current_thought, next_thought)
        
        # Se la distanza è alta, potrebbe essere un salto logico
        if semantic_distance > 0.6:
            jump_score = semantic_distance
            logic_jumps.append((i, i + 1, jump_score))
    
    return logic_jumps


def _calculate_surreal_score(thoughts: List[str], logic_jumps: List[Tuple[int, int, float]]) -> float:
    """
    Calcola il punteggio di surrealtà di una sequenza di pensieri.
    """
    if not logic_jumps:
        return 0.0
    
    # Media dei punteggi di salto logico
    jump_scores = [jump[2] for jump in logic_jumps]
    avg_jump_score = np.mean(jump_scores)
    
    # Frequenza dei salti (normalizzata)
    jump_frequency = len(logic_jumps) / (len(thoughts) - 1) if len(thoughts) > 1 else 0
    
    # Combina punteggio medio e frequenza
    surreal_score = (avg_jump_score * 0.7) + (jump_frequency * 0.3)
    
    return min(surreal_score, 1.0)  # Normalizza a [0, 1]


def _calculate_coherence_level(thoughts: List[str]) -> float:
    """
    Calcola il livello di coerenza di una sequenza di pensieri.
    """
    if len(thoughts) < 2:
        return 1.0
    
    coherence_scores = []
    
    for i in range(len(thoughts) - 1):
        # Calcola coerenza tra pensieri consecutivi
        coherence = 1.0 - _calculate_semantic_distance(thoughts[i], thoughts[i + 1])
        coherence_scores.append(coherence)
    
    return np.mean(coherence_scores)


def _detect_circular_thoughts(thoughts: List[str]) -> List[Dict[str, Any]]:
    """
    Rileva pensieri circolari in una sequenza.
    """
    circular_patterns = []
    
    # Cerca ripetizioni di concetti simili
    for i, thought in enumerate(thoughts):
        for j, other_thought in enumerate(thoughts[i+2:], i+2):
            similarity = 1.0 - _calculate_semantic_distance(thought, other_thought)
            if similarity > 0.8:  # Soglia per similarità alta
                circular_patterns.append({
                    'start_index': i,
                    'end_index': j,
                    'similarity': similarity,
                    'cycle_length': j - i,
                    'thoughts': thoughts[i:j+1]
                })
    
    return circular_patterns


def _detect_recursive_loops(thoughts: List[str]) -> List[Dict[str, Any]]:
    """
    Rileva loop ricorsivi nel pensiero.
    """
    recursive_loops = []
    
    # Cerca pattern che si ripetono
    for window_size in range(2, min(5, len(thoughts) // 2)):
        for i in range(len(thoughts) - window_size * 2):
            pattern1 = thoughts[i:i+window_size]
            
            # Cerca lo stesso pattern più avanti
            for j in range(i + window_size, len(thoughts) - window_size + 1):
                pattern2 = thoughts[j:j+window_size]
                
                # Calcola similarità tra pattern
                pattern_similarity = _calculate_pattern_similarity(pattern1, pattern2)
                
                if pattern_similarity > 0.7:
                    recursive_loops.append({
                        'pattern_start1': i,
                        'pattern_start2': j,
                        'pattern_length': window_size,
                        'similarity': pattern_similarity,
                        'pattern': pattern1
                    })
    
    return recursive_loops


def _detect_tangential_shifts(thoughts: List[str]) -> List[Dict[str, Any]]:
    """
    Rileva spostamenti tangenziali nel pensiero.
    """
    tangential_shifts = []
    
    for i in range(len(thoughts) - 2):
        # Analizza tre pensieri consecutivi
        thought1 = thoughts[i]
        thought2 = thoughts[i + 1]
        thought3 = thoughts[i + 2]
        
        # Calcola distanze semantiche
        dist_1_2 = _calculate_semantic_distance(thought1, thought2)
        dist_2_3 = _calculate_semantic_distance(thought2, thought3)
        dist_1_3 = _calculate_semantic_distance(thought1, thought3)
        
        # Se il pensiero intermedio è molto diverso da entrambi,
        # potrebbe essere uno spostamento tangenziale
        if dist_1_2 > 0.7 and dist_2_3 > 0.7 and dist_1_3 < 0.5:
            tangential_shifts.append({
                'shift_index': i + 1,
                'tangent_thought': thought2,
                'context_thoughts': [thought1, thought3],
                'tangent_score': (dist_1_2 + dist_2_3) / 2
            })
    
    return tangential_shifts


def _detect_associative_chains(thoughts: List[str]) -> List[Dict[str, Any]]:
    """
    Rileva catene associative nel pensiero.
    """
    associative_chains = []
    
    # Cerca sequenze di pensieri con associazioni graduali
    for start in range(len(thoughts)):
        chain = [thoughts[start]]
        chain_scores = []
        
        for i in range(start + 1, len(thoughts)):
            # Calcola associazione con l'ultimo pensiero nella catena
            association_score = _calculate_association_strength(
                chain[-1], thoughts[i]
            )
            
            if association_score > 0.4:  # Soglia per associazione
                chain.append(thoughts[i])
                chain_scores.append(association_score)
            else:
                break  # Interrompi la catena
        
        if len(chain) >= 3:  # Catene significative
            associative_chains.append({
                'start_index': start,
                'chain_length': len(chain),
                'thoughts': chain,
                'average_association': np.mean(chain_scores),
                'total_association': sum(chain_scores)
            })
    
    return associative_chains


def _detect_metaphorical_leaps(thoughts: List[str]) -> List[Dict[str, Any]]:
    """
    Rileva salti metaforici nel pensiero.
    """
    metaphorical_leaps = []
    
    # Parole chiave che indicano metafore
    metaphor_indicators = ['like', 'as', 'similar to', 'reminds me of', 'metaphor', 'analogy']
    
    for i, thought in enumerate(thoughts):
        # Cerca indicatori di metafora
        has_metaphor_indicator = any(indicator in thought.lower() for indicator in metaphor_indicators)
        
        if has_metaphor_indicator and i > 0:
            previous_thought = thoughts[i - 1]
            
            # Calcola la forza del salto metaforico
            metaphor_strength = _calculate_metaphor_strength(previous_thought, thought)
            
            if metaphor_strength > 0.5:
                metaphorical_leaps.append({
                    'leap_index': i,
                    'source_thought': previous_thought,
                    'metaphor_thought': thought,
                    'metaphor_strength': metaphor_strength,
                    'indicators_found': [ind for ind in metaphor_indicators if ind in thought.lower()]
                })
    
    return metaphorical_leaps


def _calculate_pattern_complexity(thoughts: List[str]) -> float:
    """
    Calcola la complessità di un pattern di pensiero.
    """
    if len(thoughts) < 2:
        return 0.0
    
    # Calcola varianza delle distanze semantiche
    distances = []
    for i in range(len(thoughts) - 1):
        dist = _calculate_semantic_distance(thoughts[i], thoughts[i + 1])
        distances.append(dist)
    
    # Complessità basata su varianza e lunghezza
    variance = np.var(distances) if distances else 0.0
    length_factor = min(len(thoughts) / 10.0, 1.0)  # Normalizza lunghezza
    
    return variance * length_factor


def _calculate_semantic_distance(text1: str, text2: str) -> float:
    """
    Calcola la distanza semantica tra due testi.
    Implementazione semplificata basata su sovrapposizione di parole.
    """
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    if not words1 or not words2:
        return 1.0
    
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    # Jaccard distance
    jaccard_similarity = len(intersection) / len(union) if union else 0.0
    return 1.0 - jaccard_similarity


def _calculate_connection_strength(concept1: str, concept2: str, fragment: Dict[str, Any]) -> float:
    """
    Calcola la forza di connessione tra due concetti in un frammento.
    """
    # Implementazione semplificata
    # In una versione più avanzata, si potrebbero usare embedding semantici
    
    # Cerca co-occorrenze nel contesto
    context_text = str(fragment.get('context', ''))
    concept1_count = context_text.lower().count(concept1.lower())
    concept2_count = context_text.lower().count(concept2.lower())
    
    # Normalizza in base alla lunghezza del contesto
    context_length = len(context_text.split()) if context_text else 1
    
    strength = (concept1_count + concept2_count) / context_length
    return min(strength, 1.0)


def _analyze_concept_network(concept_network: Dict[str, Set[str]]) -> Dict[str, Any]:
    """
    Analizza la rete di concetti per pattern surreali.
    """
    if not concept_network:
        return {'total_concepts': 0, 'total_connections': 0, 'network_density': 0.0}
    
    total_concepts = len(concept_network)
    total_connections = sum(len(connections) for connections in concept_network.values()) // 2
    
    # Densità della rete
    max_possible_connections = total_concepts * (total_concepts - 1) // 2
    network_density = total_connections / max_possible_connections if max_possible_connections > 0 else 0.0
    
    # Trova hub (concetti con molte connessioni)
    concept_degrees = {concept: len(connections) for concept, connections in concept_network.items()}
    top_hubs = sorted(concept_degrees.items(), key=lambda x: x[1], reverse=True)[:5]
    
    return {
        'total_concepts': total_concepts,
        'total_connections': total_connections,
        'network_density': network_density,
        'average_degree': np.mean(list(concept_degrees.values())),
        'top_concept_hubs': top_hubs
    }


def _calculate_pattern_similarity(pattern1: List[str], pattern2: List[str]) -> float:
    """
    Calcola la similarità tra due pattern di pensiero.
    """
    if len(pattern1) != len(pattern2):
        return 0.0
    
    similarities = []
    for thought1, thought2 in zip(pattern1, pattern2):
        similarity = 1.0 - _calculate_semantic_distance(thought1, thought2)
        similarities.append(similarity)
    
    return np.mean(similarities)


def _calculate_association_strength(thought1: str, thought2: str) -> float:
    """
    Calcola la forza di associazione tra due pensieri.
    """
    # Implementazione semplificata basata su sovrapposizione semantica
    semantic_similarity = 1.0 - _calculate_semantic_distance(thought1, thought2)
    
    # Bonus per associazioni comuni (parole chiave associative)
    associative_words = ['because', 'therefore', 'thus', 'hence', 'so', 'then', 'also', 'and']
    
    has_associative_link = any(word in thought2.lower() for word in associative_words)
    associative_bonus = 0.2 if has_associative_link else 0.0
    
    return min(semantic_similarity + associative_bonus, 1.0)


def _calculate_metaphor_strength(source_thought: str, metaphor_thought: str) -> float:
    """
    Calcola la forza di un salto metaforico.
    """
    # Distanza semantica alta + presenza di indicatori metaforici = metafora forte
    semantic_distance = _calculate_semantic_distance(source_thought, metaphor_thought)
    
    # Conta indicatori metaforici
    metaphor_indicators = ['like', 'as', 'similar to', 'reminds me of']
    indicator_count = sum(1 for indicator in metaphor_indicators if indicator in metaphor_thought.lower())
    
    # Combina distanza semantica e presenza di indicatori
    metaphor_strength = (semantic_distance * 0.7) + (min(indicator_count / 2.0, 1.0) * 0.3)
    
    return min(metaphor_strength, 1.0)