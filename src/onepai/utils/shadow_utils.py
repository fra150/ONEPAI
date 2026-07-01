"""Utility specializzate per l'analisi delle ombre neurali."""

from __future__ import annotations

from typing import Dict, List, Tuple, Optional

import numpy as np

__all__ = [
    "compute_shadow_matrix",
    "find_shadow_clusters",
    "shadow_overlap",
    "activation_to_shadow_map",
    "merge_shadow_maps",
    "shadow_diff",
]


def compute_shadow_matrix(
    activations: np.ndarray,
    threshold: float = 1e-6,
) -> np.ndarray:
    """
    Dato un array (n_observations, n_neurons), ritorna una matrice booleana
    dove True = il neurone era silenzioso in quella osservazione.
    """
    a = np.asarray(activations, dtype=np.float32)
    return np.abs(a) < threshold


def find_shadow_clusters(
    shadow_matrix: np.ndarray,
    min_cluster_size: int = 2,
    correlation_threshold: float = 0.7,
) -> List[List[int]]:
    """
    Raggruppa i neuroni che tendono ad essere silenziosi assieme.
    shadow_matrix: (n_obs, n_neurons) booleano.
    Ritorna lista di cluster (liste di indici neurone).
    """
    mat = shadow_matrix.astype(np.float32)
    n_neurons = mat.shape[1]
    if n_neurons == 0:
        return []
    # Correlazione tra colonne (neuroni)
    # Uso il prodotto punto normalizzato come proxy veloce
    col_norms = np.linalg.norm(mat, axis=0) + 1e-9
    normed = mat / col_norms[np.newaxis, :]
    corr = normed.T @ normed       # (n_neurons, n_neurons)
    visited = set()
    clusters = []
    for i in range(n_neurons):
        if i in visited:
            continue
        cluster = [i]
        visited.add(i)
        for j in range(i + 1, n_neurons):
            if j not in visited and corr[i, j] >= correlation_threshold:
                cluster.append(j)
                visited.add(j)
        if len(cluster) >= min_cluster_size:
            clusters.append(cluster)
    return clusters


def shadow_overlap(
    shadow_a: Dict[str, float],
    shadow_b: Dict[str, float],
) -> float:
    """
    Jaccard similarity tra due mappe di silenzio (layer→silence_ratio).
    Tratta i layer con ratio > 0.5 come "silenziosi".
    """
    set_a = {k for k, v in shadow_a.items() if v > 0.5}
    set_b = {k for k, v in shadow_b.items() if v > 0.5}
    if not set_a and not set_b:
        return 1.0
    union = len(set_a | set_b)
    if union == 0:
        return 1.0
    return len(set_a & set_b) / union


def activation_to_shadow_map(
    layer_activations: Dict[str, List[float]],
    threshold: float = 1e-6,
) -> Dict[str, float]:
    """
    Converte {layer: [activation_values]} in {layer: silence_ratio}.
    """
    result = {}
    for layer, vals in layer_activations.items():
        if not vals:
            result[layer] = 0.0
            continue
        a = np.abs(np.asarray(vals, dtype=np.float32))
        result[layer] = float((a < threshold).mean())
    return result


def merge_shadow_maps(
    maps: List[Dict[str, float]],
    weights: Optional[List[float]] = None,
) -> Dict[str, float]:
    """
    Combina più mappe di silenzio in una sola (media pesata).
    """
    if not maps:
        return {}
    if weights is None:
        weights = [1.0] * len(maps)
    total_weight = sum(weights)
    result: Dict[str, float] = {}
    for w, m in zip(weights, maps):
        for layer, ratio in m.items():
            result[layer] = result.get(layer, 0.0) + w * ratio / total_weight
    return result


def shadow_diff(
    before: Dict[str, float],
    after: Dict[str, float],
) -> Dict[str, float]:
    """
    Differenza tra due shadow map: positivo = più silenzio, negativo = meno silenzio.
    Utile per monitorare come cambia il silenzio durante il training.
    """
    all_keys = set(before) | set(after)
    return {
        k: after.get(k, 0.0) - before.get(k, 0.0)
        for k in all_keys
    }
