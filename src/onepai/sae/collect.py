"""
collect_activations — raccoglie attivazioni reali da un ModelAdapter.

Fa girare il modello su un batch di sequenze e impila le attivazioni di un
hook (es. resid_post, mlp.post) su tutte le posizioni e sequenze in una
matrice (N, d) pronta per addestrare una SparseAutoencoder.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

__all__ = ["collect_activations"]


def collect_activations(
    model,
    token_batch: Sequence[np.ndarray],
    hook_name: str,
    max_tokens: Optional[int] = None,
    skip_first: int = 0,
) -> np.ndarray:
    """
    Raccoglie le attivazioni di `hook_name` su tutto il batch.

    Args:
        model:       un ModelAdapter (TinyTransformer, HFAdapter, ...)
        token_batch: lista di array di token
        hook_name:   nome dell'hook (es. HookPoint.resid_post(6))
        max_tokens:  limite opzionale al numero di righe raccolte
        skip_first:  salta le prime N posizioni di ogni sequenza (es. il BOS)

    Returns:
        Matrice (N, d_model) di attivazioni.
    """
    rows: List[np.ndarray] = []
    total = 0
    for tokens in token_batch:
        _, cache = model.run_with_cache(tokens)
        act = cache.get(hook_name)
        if act is None:
            continue
        act = np.asarray(act, dtype=np.float32)
        if act.ndim == 1:
            act = act[None, :]
        if skip_first > 0:
            act = act[skip_first:]
        rows.append(act)
        total += act.shape[0]
        if max_tokens is not None and total >= max_tokens:
            break
    if not rows:
        return np.empty((0, 0), dtype=np.float32)
    out = np.concatenate(rows, axis=0)
    if max_tokens is not None and out.shape[0] > max_tokens:
        out = out[:max_tokens]
    return out
