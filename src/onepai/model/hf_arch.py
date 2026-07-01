"""
Registro delle architetture HuggingFace — rende ONEPAI agnostico al modello.

Ogni famiglia di transformer decoder mappa i propri sottomoduli sui ruoli che
ONEPAI conosce (layer, MLP, attivazione, proiezione d'attenzione, norma finale,
unembed). Aggiungere un'architettura = aggiungere una voce qui, senza toccare
il resto del framework.

Percorsi a livello di modello:  layers, final_norm, unembed
Percorsi a livello di blocco:   attn, attn_out_proj, mlp, mlp_pre, mlp_act
"""

from __future__ import annotations

from functools import reduce
from typing import Dict, Optional

__all__ = ["ARCH_SPECS", "resolve_spec", "get_path"]


# model_type (config.model_type) -> mappa dei ruoli
ARCH_SPECS: Dict[str, Dict[str, Optional[str]]] = {
    # GPT-2 / DistilGPT-2 / DialoGPT
    "gpt2": {
        "layers": "transformer.h",
        "final_norm": "transformer.ln_f",
        "unembed": "lm_head",
        "attn": "attn",
        "attn_out_proj": "attn.c_proj",
        "mlp": "mlp",
        "mlp_pre": "mlp.c_fc",
        "mlp_act": "mlp.act",
    },
    # GPT-Neo
    "gpt_neo": {
        "layers": "transformer.h",
        "final_norm": "transformer.ln_f",
        "unembed": "lm_head",
        "attn": "attn.attention",
        "attn_out_proj": "attn.attention.out_proj",
        "mlp": "mlp",
        "mlp_pre": "mlp.c_fc",
        "mlp_act": "mlp.act",
    },
    # GPT-NeoX / Pythia
    "gpt_neox": {
        "layers": "gpt_neox.layers",
        "final_norm": "gpt_neox.final_layer_norm",
        "unembed": "embed_out",
        "attn": "attention",
        "attn_out_proj": "attention.dense",
        "mlp": "mlp",
        "mlp_pre": "mlp.dense_h_to_4h",
        "mlp_act": "mlp.act",
    },
    # OPT
    "opt": {
        "layers": "model.decoder.layers",
        "final_norm": "model.decoder.final_layer_norm",
        "unembed": "lm_head",
        "attn": "self_attn",
        "attn_out_proj": "self_attn.out_proj",
        "mlp": "fc2",            # la down-projection: il suo output è il contributo MLP
        "mlp_pre": "fc1",
        "mlp_act": "activation_fn",
    },
    # GPT-J
    "gptj": {
        "layers": "transformer.h",
        "final_norm": "transformer.ln_f",
        "unembed": "lm_head",
        "attn": "attn",
        "attn_out_proj": "attn.out_proj",
        "mlp": "mlp",
        "mlp_pre": "mlp.fc_in",
        "mlp_act": "mlp.act",
    },
    # LLaMA / Mistral (SwiGLU: down_proj è il contributo; act dentro mlp)
    "llama": {
        "layers": "model.layers",
        "final_norm": "model.norm",
        "unembed": "lm_head",
        "attn": "self_attn",
        "attn_out_proj": "self_attn.o_proj",
        "mlp": "mlp.down_proj",
        "mlp_pre": "mlp.up_proj",
        "mlp_act": "mlp.act_fn",
    },
}
# Alias: famiglie che condividono l'architettura
ARCH_SPECS["mistral"] = ARCH_SPECS["llama"]


def get_path(obj, path: Optional[str]):
    """Naviga `obj.a.b.c` da una stringa 'a.b.c'. None se path è None o assente."""
    if path is None:
        return None
    try:
        return reduce(getattr, path.split("."), obj)
    except AttributeError:
        return None


def resolve_spec(model_type: str) -> Dict[str, Optional[str]]:
    """
    Ritorna la mappa dei ruoli per `model_type`.
    Solleva KeyError con messaggio utile se l'architettura non è supportata.
    """
    if model_type in ARCH_SPECS:
        return ARCH_SPECS[model_type]
    raise KeyError(
        f"Architettura {model_type!r} non ancora supportata da ONEPAI. "
        f"Supportate: {sorted(ARCH_SPECS)}. Aggiungi una voce in hf_arch.ARCH_SPECS."
    )
