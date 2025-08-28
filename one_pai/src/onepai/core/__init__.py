"""
Metriche del Silenzio e del Vuoto.

Funzioni per calcolare indicatori quantitativi basati
sui dati non manifesti dell'IA.
"""
from typing import List, Dict, Any
import numpy as np

def calculate_silence_ratio(treasure_fragments: List[Dict[str, Any]], neuron_id: str) -> float:
    """
    Calcola la frequenza con cui un dato neurone è rimasto 'silente'.

    Args:
        treasure_fragments: Lista di frammenti di potenziale.
        neuron_id: L'identificativo del neurone da analizzare.

    Returns:
        Un valore float tra 0 e 1 che rappresenta il rapporto di silenzio.
    """
    total_events = len(treasure_fragments)
    silence_count = 0
    # ... Logica per contare quante volte il neurone è stato silente ...
    return silence_count / total_events if total_events > 0 else 0.0

def calculate_void_entropy(potential_distribution: List[float]) -> float:
    """
    Calcola l'entropia di una distribuzione di probabilità dei 'pensieri abortiti'.
    Un'alta entropia significa che le scelte scartate erano molto varie e incerte.

    Args:
        potential_distribution: Lista delle probabilità delle scelte non intraprese.

    Returns:
        Il valore di entropia di Shannon.
    """
    probabilities = np.array(potential_distribution)
    # Filtra probabilità nulle per evitare log(0)
    probabilities = probabilities[probabilities > 0]
    return -np.sum(probabilities * np.log2(probabilities))


def calculate_silence_entropy(treasure_fragments: List[Dict[str, Any]]) -> float:
    """
    Calcola l'entropia del silenzio attraverso tutti i frammenti del tesoro.
    
    Args:
        treasure_fragments: Lista di frammenti di potenziale decifrati
        
    Returns:
        Valore di entropia del silenzio
    """
    if not treasure_fragments:
        return 0.0
    
    silence_counts = {}
    total_observations = 0
    
    for fragment in treasure_fragments:
        if 'silence_patterns' in fragment:
            for pattern in fragment['silence_patterns']:
                silence_counts[pattern] = silence_counts.get(pattern, 0) + 1
                total_observations += 1
    
    if total_observations == 0:
        return 0.0
    
    # Calcola probabilità
    probabilities = [count / total_observations for count in silence_counts.values()]
    
    return calculate_void_entropy(probabilities)


def measure_void_depth(treasure_fragments: List[Dict[str, Any]], layer_name: str) -> float:
    """
    Misura la profondità del vuoto cognitivo in un layer specifico.
    
    Args:
        treasure_fragments: Lista di frammenti di potenziale
        layer_name: Nome del layer da analizzare
        
    Returns:
        Profondità del vuoto (0.0 = nessun vuoto, 1.0 = vuoto completo)
    """
    if not treasure_fragments:
        return 0.0
    
    void_measurements = []
    
    for fragment in treasure_fragments:
        if 'layer_voids' in fragment and layer_name in fragment['layer_voids']:
            void_data = fragment['layer_voids'][layer_name]
            if 'depth_score' in void_data:
                void_measurements.append(void_data['depth_score'])
    
    if not void_measurements:
        return 0.0
    
    # Calcola la profondità media del vuoto
    return np.mean(void_measurements)


def analyze_suppression_patterns(treasure_fragments: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analizza i pattern di soppressione nei frammenti del tesoro.
    
    Args:
        treasure_fragments: Lista di frammenti di potenziale
        
    Returns:
        Dizionario con statistiche sui pattern di soppressione
    """
    if not treasure_fragments:
        return {
            'total_suppressions': 0,
            'suppression_rate': 0.0,
            'most_suppressed_thoughts': [],
            'suppression_entropy': 0.0
        }
    
    suppression_counts = {}
    total_thoughts = 0
    total_suppressions = 0
    
    for fragment in treasure_fragments:
        if 'suppressed_thoughts' in fragment:
            for thought in fragment['suppressed_thoughts']:
                thought_type = thought.get('type', 'unknown')
                suppression_counts[thought_type] = suppression_counts.get(thought_type, 0) + 1
                total_suppressions += 1
        
        if 'total_thoughts' in fragment:
            total_thoughts += fragment['total_thoughts']
    
    # Calcola statistiche
    suppression_rate = total_suppressions / total_thoughts if total_thoughts > 0 else 0.0
    
    # Trova i pensieri più soppressi
    most_suppressed = sorted(suppression_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    
    # Calcola entropia della soppressione
    if suppression_counts:
        probabilities = [count / total_suppressions for count in suppression_counts.values()]
        suppression_entropy = calculate_void_entropy(probabilities)
    else:
        suppression_entropy = 0.0
    
    return {
        'total_suppressions': total_suppressions,
        'suppression_rate': suppression_rate,
        'most_suppressed_thoughts': most_suppressed,
        'suppression_entropy': suppression_entropy,
        'suppression_types': len(suppression_counts)
    }
