"""
TokenInfluencer — Quale token di input influenza di più l'output?

Tre metodi di attribuzione:
1. Attention Rollout — propaga i pesi di attenzione strato per strato
2. Gradient × Input  — grad del loss rispetto all'embedding × valore dell'embedding
3. Perturbation      — misura empirica: rimuovi un token e vedi quanto cambia l'output
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

__all__ = ["TokenInfluenceMap", "TokenInfluencer"]


@dataclass
class TokenInfluenceMap:
    """Attribution map per un singolo target position."""
    target_position: int
    source_scores: Dict[int, float]    # token_idx → influence score
    method: str
    step: int = 0
    metadata: Dict = field(default_factory=dict)

    @property
    def top_k(self) -> List[Tuple[int, float]]:
        """Restituisce i token più influenti in ordine decrescente."""
        return sorted(self.source_scores.items(), key=lambda x: x[1], reverse=True)

    def top(self, k: int = 5) -> List[Tuple[int, float]]:
        return self.top_k[:k]


class TokenInfluencer:
    """
    Calcola l'influenza di ogni token di input sul token di output selezionato.

    Uso::

        influencer = TokenInfluencer()
        # Metodo 1: Attention Rollout (serve stack di attention matrices)
        attn_stack = [layer0_attn, layer1_attn, ...]   # list di (seq, seq) arrays
        imap = influencer.attention_rollout(attn_stack, target_pos=7)

        # Metodo 2: Gradient×Input (serve embeddings e gradients)
        imap = influencer.gradient_input(embeddings, gradients, target_pos=7)

        # Metodo 3: Perturbazione (serve funzione modello)
        imap = influencer.perturbation_influence(model_fn, tokens, target_pos=7)

        top5 = imap.top(5)
    """

    def __init__(self):
        self._maps: List[TokenInfluenceMap] = []
        self._step = 0

    # ------------------------------------------------------------------
    # Metodo 1 — Attention Rollout
    # ------------------------------------------------------------------

    def attention_rollout(
        self,
        attention_matrices: List[np.ndarray],
        target_pos: int,
        add_residual: bool = True,
    ) -> TokenInfluenceMap:
        """
        Calcola l'influenza aggregando i pesi di attenzione su tutti i layer.
        attention_matrices: lista di (seq_len × seq_len) arrays, uno per layer.
        """
        if not attention_matrices:
            return TokenInfluenceMap(target_pos, {}, "attention_rollout")
        mats = [np.asarray(m, dtype=np.float32) for m in attention_matrices]
        seq_len = mats[0].shape[0]
        # Rollout: moltiplica iterativamente le matrici di attenzione
        rollout = np.eye(seq_len, dtype=np.float32)
        for attn in mats:
            a = attn.copy()
            if add_residual:
                # Residual connection: aggiunge identità prima della normalizzazione
                a = 0.5 * a + 0.5 * np.eye(seq_len)
            # Normalizza ogni riga
            row_sums = a.sum(axis=-1, keepdims=True)
            row_sums = np.where(row_sums == 0, 1.0, row_sums)
            a = a / row_sums
            rollout = a @ rollout
        if target_pos >= seq_len:
            target_pos = seq_len - 1
        scores = {i: float(rollout[target_pos, i]) for i in range(seq_len)}
        imap = TokenInfluenceMap(target_pos, scores, "attention_rollout", step=self._step)
        self._step += 1
        self._maps.append(imap)
        return imap

    # ------------------------------------------------------------------
    # Metodo 2 — Gradient × Input
    # ------------------------------------------------------------------

    def gradient_input(
        self,
        embeddings: np.ndarray,
        gradients: np.ndarray,
        target_pos: int,
    ) -> TokenInfluenceMap:
        """
        Attribution = element-wise product di embeddings e gradients,
        aggregato sulla dimensione del modello con L2-norm.

        embeddings:  (seq_len, embed_dim)
        gradients:   (seq_len, embed_dim) — ∂loss/∂embedding
        """
        emb = np.asarray(embeddings, dtype=np.float32)
        grd = np.asarray(gradients, dtype=np.float32)
        product = emb * grd                           # (seq_len, embed_dim)
        # Norma L2 per ogni token = importanza del token
        importance = np.linalg.norm(product, axis=-1) # (seq_len,)
        # Normalizza a somma 1
        total = importance.sum()
        if total > 0:
            importance = importance / total
        scores = {i: float(importance[i]) for i in range(len(importance))}
        imap = TokenInfluenceMap(target_pos, scores, "gradient_input", step=self._step)
        self._step += 1
        self._maps.append(imap)
        return imap

    # ------------------------------------------------------------------
    # Metodo 3 — Perturbazione empirica
    # ------------------------------------------------------------------

    def perturbation_influence(
        self,
        model_fn: Callable[[np.ndarray], float],
        tokens: np.ndarray,
        target_pos: int,
        mask_value: float = 0.0,
        normalize: bool = True,
    ) -> TokenInfluenceMap:
        """
        Misura empirica: sostituisce ogni token con `mask_value` e misura
        |output_with_mask - output_baseline|.

        model_fn: funzione che prende (seq_len, ...) e ritorna uno scalare (score).
        tokens: (seq_len,) o (seq_len, embed_dim).
        """
        tokens = np.asarray(tokens, dtype=np.float32)
        seq_len = tokens.shape[0]
        baseline_score = model_fn(tokens)
        scores = {}
        for i in range(seq_len):
            perturbed = tokens.copy()
            if tokens.ndim == 1:
                perturbed[i] = mask_value
            else:
                perturbed[i, :] = mask_value
            try:
                score = model_fn(perturbed)
                scores[i] = abs(float(score) - float(baseline_score))
            except Exception:
                scores[i] = 0.0
        if normalize:
            total = sum(scores.values()) + 1e-9
            scores = {k: v / total for k, v in scores.items()}
        imap = TokenInfluenceMap(target_pos, scores, "perturbation", step=self._step)
        self._step += 1
        self._maps.append(imap)
        return imap

    # ------------------------------------------------------------------
    # Matrice di influenza completa
    # ------------------------------------------------------------------

    def build_full_influence_matrix(
        self,
        attention_matrices: List[np.ndarray],
    ) -> np.ndarray:
        """
        Matrice (seq_len × seq_len): M[i, j] = influenza di j sull'output in posizione i.
        Usa attention rollout per ogni target position.
        """
        if not attention_matrices:
            return np.array([[]])
        seq_len = attention_matrices[0].shape[0]
        mat = np.zeros((seq_len, seq_len), dtype=np.float32)
        for target in range(seq_len):
            imap = self.attention_rollout(attention_matrices, target_pos=target)
            for src, score in imap.source_scores.items():
                if src < seq_len:
                    mat[target, src] = score
        return mat

    # ------------------------------------------------------------------
    # Aggregazione
    # ------------------------------------------------------------------

    def aggregate_across_positions(self) -> Dict[int, float]:
        """
        Per ogni token sorgente, somma la sua influenza su tutti i target.
        Utile per trovare i token più "globalmente influenti" nella sequenza.
        """
        total: Dict[int, float] = {}
        for imap in self._maps:
            for src, score in imap.source_scores.items():
                total[src] = total.get(src, 0.0) + score
        if total:
            max_v = max(total.values())
            if max_v > 0:
                total = {k: v / max_v for k, v in total.items()}
        return dict(sorted(total.items()))

    def compare_methods(self, position: int) -> Dict[str, Dict[int, float]]:
        """
        Confronta le mappe di influenza prodotte da metodi diversi per la stessa position.
        """
        result = {}
        for imap in self._maps:
            if imap.target_position == position:
                result[imap.method] = imap.source_scores
        return result

    def __repr__(self) -> str:
        return f"TokenInfluencer(maps={len(self._maps)})"
