"""
Esempio di applicazione degli hook ONEPAI a un modello Gemma.
"""
from onepai.core.observer import Observer
from onepai.hooks.pytorch import register_pytorch_hook

def hook_gemma_model(gemma_model, observer: Observer):
    """
    Funzione di convenienza per applicare gli hook ai layer chiave di Gemma.

    Args:
        gemma_model: Un'istanza di un modello Gemma (es. da Hugging Face).
        observer: L'istanza dell'Osservatore ONEPAI.
    """
    # Esempio: agganciamoci all'ultimo blocco di attenzione
    # Il nome esatto del layer dipende dall'implementazione specifica del modello
    LAST_ATTENTION_BLOCK = "model.layers.27.self_attn" # Nome ipotetico
    
    try:
        register_pytorch_hook(gemma_model, LAST_ATTENTION_BLOCK, observer)
        print("INFO: Modello Gemma agganciato con successo a ONEPAI.")
    except KeyError:
        print(f"ERRORE: Layer '{LAST_ATTENTION_BLOCK}' non trovato nel modello Gemma.")
    
    return gemma_model