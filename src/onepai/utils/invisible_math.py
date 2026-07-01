"""
Matematica per l'invisibile — entropia, divergenze, distanze usate nel framework.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

__all__ = [
    "entropy", "kl_divergence", "js_divergence",
    "cosine_similarity", "silence_score",
    "void_depth", "activation_sparsity",
    "gini_coefficient", "softmax", "normalize",
]


def entropy(probs: Sequence[float], base: float = 2.0) -> float:
    """
    Entropia di Shannon: H = -sum(p * log_b(p)).
    Valori vicini a 0 → distribuzione concentrata; vicini a log_b(n) → uniforme.
    """
    p = np.asarray(probs, dtype=np.float64)
    p = p[p > 0]
    if len(p) == 0:
        return 0.0
    log_fn = np.log2 if base == 2.0 else (np.log if base == math.e else
                                            lambda x: np.log(x) / np.log(base))
    return float(-np.sum(p * log_fn(p)))


def kl_divergence(p: Sequence[float], q: Sequence[float],
                   eps: float = 1e-9) -> float:
    """
    KL(P||Q) = sum(p * log(p/q)).
    Misura quanto Q approssima male P.  Non simmetrica.
    """
    p_arr = np.asarray(p, dtype=np.float64) + eps
    q_arr = np.asarray(q, dtype=np.float64) + eps
    p_arr = p_arr / p_arr.sum()
    q_arr = q_arr / q_arr.sum()
    return float(np.sum(p_arr * np.log(p_arr / q_arr)))


def js_divergence(p: Sequence[float], q: Sequence[float]) -> float:
    """
    Jensen-Shannon Divergence: simmetrica, sempre in [0, 1] (in bits).
    JSD(P,Q) = 0.5*KL(P||M) + 0.5*KL(Q||M) dove M=(P+Q)/2.
    """
    p_arr = np.asarray(p, dtype=np.float64)
    q_arr = np.asarray(q, dtype=np.float64)
    p_arr = p_arr / (p_arr.sum() + 1e-9)
    q_arr = q_arr / (q_arr.sum() + 1e-9)
    m = 0.5 * (p_arr + q_arr)
    return 0.5 * kl_divergence(p_arr, m) + 0.5 * kl_divergence(q_arr, m)


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity in [-1, 1]. 1=paralleli, 0=ortogonali, -1=antiparalleli."""
    a_arr = np.asarray(a, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)
    norm_a = np.linalg.norm(a_arr)
    norm_b = np.linalg.norm(b_arr)
    if norm_a < 1e-9 or norm_b < 1e-9:
        return 0.0
    return float(np.dot(a_arr, b_arr) / (norm_a * norm_b))


def silence_score(activations: Sequence[float], threshold: float = 1e-6) -> float:
    """
    Frazione di attivazioni sotto threshold (silence ratio).
    0.0 = tutte attive, 1.0 = tutte silenziose.
    """
    if not activations:
        return 0.0
    a = np.abs(np.asarray(activations, dtype=np.float64))
    return float((a < threshold).sum() / len(a))


def void_depth(activation_sequence: Sequence[float],
               window: int = 5) -> float:
    """
    Durata media dei periodi di silenzio in una sequenza temporale.
    """
    a = np.abs(np.asarray(activation_sequence, dtype=np.float64))
    threshold = a.mean() * 0.1 + 1e-9
    silent = (a < threshold).astype(int)
    if silent.sum() == 0:
        return 0.0
    runs = []
    count = 0
    for v in silent:
        if v:
            count += 1
        elif count > 0:
            runs.append(count)
            count = 0
    if count > 0:
        runs.append(count)
    return float(np.mean(runs)) if runs else 0.0


def activation_sparsity(weight_matrix: np.ndarray,
                         threshold: float = 1e-5) -> float:
    """Frazione di elementi della matrice con |w| < threshold."""
    w = np.asarray(weight_matrix, dtype=np.float64)
    return float((np.abs(w) < threshold).sum() / w.size)


def gini_coefficient(values: Sequence[float]) -> float:
    """
    Coefficiente di Gini per misurare la diseguaglianza delle attivazioni.
    0 = tutte uguali, 1 = una sola attiva.
    """
    v = np.abs(np.asarray(values, dtype=np.float64))
    v = np.sort(v)
    n = len(v)
    if n == 0 or v.sum() == 0:
        return 0.0
    indices = np.arange(1, n + 1, dtype=np.float64)
    return float((2 * np.sum(indices * v) / (n * v.sum())) - (n + 1) / n)


def softmax(x: Sequence[float], temperature: float = 1.0) -> np.ndarray:
    """Softmax numericamente stabile con temperatura."""
    a = np.asarray(x, dtype=np.float64) / max(temperature, 1e-9)
    a = a - a.max()
    e = np.exp(a)
    return (e / (e.sum() + 1e-9)).astype(np.float32)


def normalize(x: Sequence[float], method: str = "l2") -> np.ndarray:
    """
    Normalizza un vettore.
    method: "l2" (unit norm), "minmax" (0-1), "zscore" (media 0, std 1).
    """
    a = np.asarray(x, dtype=np.float64)
    if method == "l2":
        norm = np.linalg.norm(a)
        return (a / norm if norm > 1e-9 else a).astype(np.float32)
    if method == "minmax":
        lo, hi = a.min(), a.max()
        span = hi - lo
        return ((a - lo) / span if span > 1e-9 else np.zeros_like(a)).astype(np.float32)
    if method == "zscore":
        mu, sigma = a.mean(), a.std()
        return ((a - mu) / sigma if sigma > 1e-9 else np.zeros_like(a)).astype(np.float32)
    raise ValueError(f"Unknown normalization method: {method!r}")
