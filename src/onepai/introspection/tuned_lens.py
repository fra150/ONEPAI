"""
Tuned Lens — un logit lens che non mente sui layer bassi.

Il logit lens grezzo (`apply_unembed(resid_i)`) proietta lo stato intermedio con
la matrice di unembedding finale, ma i layer intermedi vivono in una geometria
diversa: il risultato è rumoroso, soprattutto sui layer bassi (token spazzatura
tipo `.,'` o `退`). Il tuned lens impara, per ogni layer, una mappa affine

    resid_finale ≈ resid_layer @ W_l + b_l

e poi applica la norma finale + unembedding a quello stato *tradotto*.

Qui la mappa è stimata con **regressione ridge** su un corpus di attivazioni
resid_layer → resid_finale (la variante economica del tuned lens canonico, che
invece minimizza la KL verso la distribuzione finale con SGD). Il risultato
pratico è lo stesso: un lens pulito, in numpy, che riusa `collect_activations`.

Riferimento: Belrose et al., "Eliciting Latent Predictions from Transformers
with the Tuned Lens" (2023).

Uso::

    lens = RidgeTunedLens(model).fit(corpus_batches)
    probs = lens.probs(resid_layer, layer=7)[-1]      # distribuzione pulita
    profile = lens.kl_profile(tokens)                  # tuned vs raw, layer per layer
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..model.adapter import HookPoint, ModelAdapter

__all__ = ["LayerLensFit", "RidgeTunedLens"]


@dataclass
class LayerLensFit:
    """Mappa affine appresa per un layer."""
    layer: int
    W: np.ndarray            # (d_model, d_model)
    b: np.ndarray            # (d_model,)
    n_rows: int
    mse: float


class RidgeTunedLens:
    """
    Tuned lens per regressione ridge: resid_layer → resid_finale.

    Esempio::

        lens = RidgeTunedLens(model, alpha=1e-3).fit(reference_batches)
        for layer, m in lens.kl_profile(tokens).items():
            print(layer, round(m["raw_kl"], 3), "->", round(m["tuned_kl"], 3))
    """

    def __init__(self, model: ModelAdapter, alpha: float = 1e-3):
        self.model = model
        self.alpha = alpha
        self.fits: Dict[int, LayerLensFit] = {}

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(
        self,
        token_batches: Sequence[np.ndarray],
        layers: Optional[Sequence[int]] = None,
        max_rows: int = 6000,
        verbose: bool = False,
    ) -> "RidgeTunedLens":
        """
        Stima W_l, b_l per ogni layer su attivazioni reali del corpus.

        `max_rows` limita le righe per layer (memoria/CPU); le righe vengono
        campionate uniformemente da tutte le posizioni raccolte.
        """
        model = self.model
        n = model.n_layers
        layers = list(layers) if layers is not None else list(range(n))

        X: Dict[int, List[np.ndarray]] = {l: [] for l in layers}
        Y: List[np.ndarray] = []
        total = 0
        for tokens in token_batches:
            _, cache = model.run_with_cache(tokens)
            final = cache.get(HookPoint.resid_post(n - 1))
            if final is None:
                continue
            Y.append(np.asarray(final, dtype=np.float32))
            for l in layers:
                r = cache.get(HookPoint.resid_post(l))
                if r is not None:
                    X[l].append(np.asarray(r, dtype=np.float32))
            total += final.shape[0]
            if total >= max_rows:
                break
        if not Y:
            raise RuntimeError("Nessuna attivazione raccolta: fit impossibile.")

        Y_all = np.concatenate(Y, axis=0)
        if Y_all.shape[0] > max_rows:
            idx = np.linspace(0, Y_all.shape[0] - 1, max_rows).astype(int)
            Y_all = Y_all[idx]

        d = Y_all.shape[1]
        for l in layers:
            if not X[l]:
                continue
            X_all = np.concatenate(X[l], axis=0)
            if X_all.shape[0] > max_rows:
                idx = np.linspace(0, X_all.shape[0] - 1, max_rows).astype(int)
                X_all = X_all[idx]
            m = min(X_all.shape[0], Y_all.shape[0])
            if l == n - 1:
                # L'ultimo layer E' il residual finale: la traduzione corretta
                # e' l'identita'. Stimarla con la ridge la farebbe solo
                # peggiorare (shrinkage) e romperebbe il sanity check KL→0.
                self.fits[l] = LayerLensFit(
                    layer=l, W=np.eye(d, dtype=np.float32),
                    b=np.zeros(d, dtype=np.float32), n_rows=m, mse=0.0)
            else:
                self.fits[l] = self._solve(X_all[:m], Y_all[:m], l)
            if verbose:
                print(f"  layer {l:2d}: mse {self.fits[l].mse:.4f} "
                      f"({self.fits[l].n_rows} righe)")

        if not self.fits:
            raise RuntimeError("Fit vuoto: nessun layer risolto.")
        return self

    @staticmethod
    def _solve(X: np.ndarray, Y: np.ndarray, layer: int) -> LayerLensFit:
        """Ridge chiusa con centratura (b = media(Y) - media(X) W)."""
        x_mean = X.mean(axis=0)
        y_mean = Y.mean(axis=0)
        Xc = X - x_mean
        Yc = Y - y_mean
        XtX = Xc.T @ Xc
        d = XtX.shape[0]
        lam = 1e-3 * (np.trace(XtX) / max(d, 1))
        W = np.linalg.solve(XtX + lam * np.eye(d, dtype=np.float32), Xc.T @ Yc)
        b = y_mean - x_mean @ W
        resid = Xc @ W - Yc
        return LayerLensFit(layer=layer, W=W.astype(np.float32),
                            b=b.astype(np.float32), n_rows=int(X.shape[0]),
                            mse=float(np.mean(resid ** 2)))

    # ------------------------------------------------------------------
    # Applicazione
    # ------------------------------------------------------------------

    @property
    def fitted(self) -> bool:
        return bool(self.fits)

    def translate(self, resid: np.ndarray, layer: int) -> np.ndarray:
        """Porta lo stato del layer nello spazio del residual finale."""
        if layer not in self.fits:
            raise KeyError(f"layer {layer} non addestrato.")
        fit = self.fits[layer]
        x = np.atleast_2d(np.asarray(resid, dtype=np.float32))
        return (x @ fit.W + fit.b).astype(np.float32)

    def logits(self, resid: np.ndarray, layer: int) -> np.ndarray:
        """Logit predetti dal tuned lens per lo stato di quel layer."""
        return self.model.apply_unembed(self.translate(resid, layer))

    def probs(self, resid: np.ndarray, layer: int) -> np.ndarray:
        logits = self.logits(resid, layer)
        return _softmax(logits)

    # ------------------------------------------------------------------
    # Valutazione: il lens è davvero migliore di quello grezzo?
    # ------------------------------------------------------------------

    def kl_profile(self, tokens, position: int = -1) -> Dict[int, Dict[str, float]]:
        """
        Per ogni layer: KL(final || tuned) e KL(final || raw), più se il top-1
        coincide con quello finale. Il tuned deve scendere monotonicamente.
        """
        model = self.model
        _, cache = model.run_with_cache(tokens)
        n = model.n_layers
        final_resid = cache.get(HookPoint.resid_post(n - 1))
        if final_resid is None:
            raise RuntimeError("resid_post finale assente.")
        final_p = _softmax(model.apply_unembed(final_resid)[position])
        final_top = int(np.argmax(final_p))

        out: Dict[int, Dict[str, float]] = {}
        for l in sorted(self.fits):
            resid = cache.get(HookPoint.resid_post(l))
            if resid is None:
                continue
            raw_p = _softmax(model.apply_unembed(resid)[position])
            tuned_p = _softmax(self.logits(resid, l)[position])
            out[l] = {
                "raw_kl": float(_kl(final_p, raw_p)),
                "tuned_kl": float(_kl(final_p, tuned_p)),
                "raw_top_match": float(int(np.argmax(raw_p)) == final_top),
                "tuned_top_match": float(int(np.argmax(tuned_p)) == final_top),
            }
        return out

    def summary(self) -> Dict:
        return {
            "layers_fitted": len(self.fits),
            "d_model": self.model.d_model,
            "alpha": self.alpha,
            "mean_mse": round(float(np.mean([f.mse for f in self.fits.values()])), 5),
            "rows": max((f.n_rows for f in self.fits.values()), default=0),
        }


def _softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / (np.sum(e, axis=-1, keepdims=True) + 1e-9)


def _kl(p: np.ndarray, q: np.ndarray) -> float:
    """KL(p || q) con q rinormalizzata e clamp numerico."""
    q = np.clip(q, 1e-12, None)
    q = q / q.sum()
    return float(np.sum(p * np.log(np.clip(p, 1e-12, None) / q)))
