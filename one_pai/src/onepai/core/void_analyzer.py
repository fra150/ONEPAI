"""
Statistiche del Vuoto Cognitivo.

Funzioni per calcolare statistiche aggregate sul contenuto dello Scrigno.
"""
from typing import List, Dict, Any
import pandas as pd

def get_potentiality_summary(treasure_fragments: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Crea un sommario statistico dei frammenti del tesoro.

    Args:
        treasure_fragments: Lista di frammenti di potenziale.

    Returns:
        Un DataFrame di Pandas con le statistiche principali.
    """
    if not treasure_fragments:
        return pd.DataFrame()

    df = pd.DataFrame(treasure_fragments)
    
    # Esempio di analisi
    summary = {
        "total_fragments": len(df),
        "avg_aborted_thoughts_prob": df['aborted_thoughts'].apply(lambda x: np.mean([p for _, p in x])).mean(),
        "most_latent_neuron": df['latent_activations'].explode().mode()[0],
    }
    
    return pd.DataFrame([summary])