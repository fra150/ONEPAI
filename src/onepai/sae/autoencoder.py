"""
SparseAutoencoder — autoencoder sparso in numpy puro, con backprop esplicito.

Architettura (convenzione tipo Anthropic):
    f      = ReLU((x - b_dec) @ W_enc + b_enc)      # feature sparse (d_hidden)
    x_hat  = f @ W_dec + b_dec                       # ricostruzione

Loss = MSE(x, x_hat) + l1_coef * mean(|f|)
Le righe di W_dec (direzioni delle feature) sono rinormalizzate a norma 1 dopo
ogni passo (trucco standard per evitare di "barare" sulla sparsità).

L'input viene standardizzato internamente (z-score per dimensione) per stabilità.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

__all__ = ["SparseAutoencoder", "FeatureReport"]


@dataclass
class FeatureReport:
    """Analisi delle feature apprese dalla SAE su un set di attivazioni."""
    n_features: int
    n_live: int
    n_dead: int
    mean_l0: float                 # numero medio di feature attive per campione (sparsità)
    variance_explained: float      # 1 - var(residuo)/var(segnale)
    frequencies: np.ndarray        # (d_hidden,) frequenza di attivazione per feature
    top_features: List[Tuple[int, float]] = field(default_factory=list)  # (idx, freq)

    def summary(self) -> Dict:
        return {
            "features": self.n_features,
            "live": self.n_live,
            "dead": self.n_dead,
            "mean_l0": round(self.mean_l0, 2),
            "variance_explained": round(self.variance_explained, 3),
        }

    def __repr__(self) -> str:
        return (f"FeatureReport(live={self.n_live}/{self.n_features}, "
                f"dead={self.n_dead}, L0={self.mean_l0:.1f}, "
                f"var_expl={self.variance_explained:.2f})")


class SparseAutoencoder:
    """
    SAE addestrabile su attivazioni reali.

    Esempio::

        sae = SparseAutoencoder(d_in=768, d_hidden=2048, l1_coef=2e-3, seed=0)
        sae.fit(activations, epochs=8, batch_size=256, lr=1e-3)
        report = sae.analyze(activations)
        print(report)                       # live/dead features, varianza spiegata
        feats = sae.encode(activations)     # (N, d_hidden) attivazioni delle feature
    """

    def __init__(self, d_in: int, d_hidden: int, l1_coef: float = 1e-3, seed: int = 0):
        self.d_in = d_in
        self.d_hidden = d_hidden
        self.l1_coef = l1_coef
        rng = np.random.RandomState(seed)
        # Decoder: direzioni casuali normalizzate
        W_dec = rng.randn(d_hidden, d_in).astype(np.float32)
        W_dec /= (np.linalg.norm(W_dec, axis=1, keepdims=True) + 1e-8)
        self.W_dec = W_dec
        self.W_enc = W_dec.T.copy() * 0.1          # encoder ~ trasposta del decoder
        self.b_enc = np.zeros(d_hidden, dtype=np.float32)
        self.b_dec = np.zeros(d_in, dtype=np.float32)
        # Standardizzazione (impostata in fit)
        self._mean = np.zeros(d_in, dtype=np.float32)
        self._std = np.ones(d_in, dtype=np.float32)
        self.history: List[float] = []

    # ------------------------------------------------------------------
    # Standardizzazione
    # ------------------------------------------------------------------

    def _standardize(self, x: np.ndarray) -> np.ndarray:
        return (np.asarray(x, dtype=np.float32) - self._mean) / self._std

    def _fit_scaler(self, x: np.ndarray) -> None:
        self._mean = x.mean(axis=0).astype(np.float32)
        self._std = (x.std(axis=0) + 1e-3).astype(np.float32)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def encode(self, x: np.ndarray, standardize: bool = True) -> np.ndarray:
        xs = self._standardize(x) if standardize else np.asarray(x, dtype=np.float32)
        pre = (xs - self.b_dec) @ self.W_enc + self.b_enc
        return np.maximum(pre, 0.0)

    def decode(self, f: np.ndarray) -> np.ndarray:
        return f @ self.W_dec + self.b_dec

    def reconstruct(self, x: np.ndarray, standardize: bool = True) -> np.ndarray:
        return self.decode(self.encode(x, standardize=standardize))

    # ------------------------------------------------------------------
    # Training (Adam, backprop esplicito)
    # ------------------------------------------------------------------

    def fit(
        self,
        activations: np.ndarray,
        epochs: int = 8,
        batch_size: int = 256,
        lr: float = 1e-3,
        verbose: bool = False,
        seed: int = 0,
    ) -> "SparseAutoencoder":
        X = np.asarray(activations, dtype=np.float32)
        self._fit_scaler(X)
        Xs = self._standardize(X)
        n = Xs.shape[0]
        rng = np.random.RandomState(seed)

        params = {"W_enc": self.W_enc, "b_enc": self.b_enc,
                  "W_dec": self.W_dec, "b_dec": self.b_dec}
        m = {k: np.zeros_like(v) for k, v in params.items()}
        v = {k: np.zeros_like(v) for k, v in params.items()}
        b1, b2, eps = 0.9, 0.999, 1e-8
        t = 0

        for epoch in range(epochs):
            perm = rng.permutation(n)
            epoch_loss = 0.0
            n_batches = 0
            for start in range(0, n, batch_size):
                idx = perm[start:start + batch_size]
                xb = Xs[idx]
                loss, grads = self._loss_and_grads(xb, params)
                t += 1
                for k in params:
                    g = grads[k]
                    m[k] = b1 * m[k] + (1 - b1) * g
                    v[k] = b2 * v[k] + (1 - b2) * (g * g)
                    mhat = m[k] / (1 - b1 ** t)
                    vhat = v[k] / (1 - b2 ** t)
                    params[k] -= lr * mhat / (np.sqrt(vhat) + eps)
                # Rinormalizza le direzioni del decoder a norma 1
                norms = np.linalg.norm(params["W_dec"], axis=1, keepdims=True) + 1e-8
                params["W_dec"] /= norms
                epoch_loss += loss
                n_batches += 1
            avg = epoch_loss / max(n_batches, 1)
            self.history.append(avg)
            if verbose:
                print(f"  epoch {epoch+1}/{epochs}  loss={avg:.4f}")

        self.W_enc, self.b_enc = params["W_enc"], params["b_enc"]
        self.W_dec, self.b_dec = params["W_dec"], params["b_dec"]
        return self

    def _loss_and_grads(self, xb: np.ndarray, params: Dict[str, np.ndarray]):
        W_enc, b_enc = params["W_enc"], params["b_enc"]
        W_dec, b_dec = params["W_dec"], params["b_dec"]
        B, d_in = xb.shape

        centered = xb - b_dec
        pre = centered @ W_enc + b_enc          # (B, h)
        f = np.maximum(pre, 0.0)                 # (B, h)
        x_hat = f @ W_dec + b_dec                # (B, d_in)
        err = x_hat - xb                         # (B, d_in)

        recon = float(np.mean(err ** 2))
        l1 = float(self.l1_coef * np.mean(np.sum(np.abs(f), axis=1)))
        loss = recon + l1

        # Backprop
        g_xhat = (2.0 / (B * d_in)) * err        # d recon / d x_hat
        grad_W_dec = f.T @ g_xhat                # (h, d_in)
        g_f = g_xhat @ W_dec.T                    # (B, h)
        g_f += (self.l1_coef / B) * np.sign(f)    # gradiente L1
        g_pre = g_f * (pre > 0)                   # ReLU
        grad_W_enc = centered.T @ g_pre           # (d_in, h)
        grad_b_enc = g_pre.sum(axis=0)            # (h,)
        grad_b_dec = g_xhat.sum(axis=0) - (g_pre @ W_enc.T).sum(axis=0)

        return loss, {"W_enc": grad_W_enc, "b_enc": grad_b_enc,
                      "W_dec": grad_W_dec, "b_dec": grad_b_dec}

    # ------------------------------------------------------------------
    # Analisi
    # ------------------------------------------------------------------

    def feature_frequencies(self, activations: np.ndarray,
                            threshold: float = 1e-6) -> np.ndarray:
        f = self.encode(activations)
        return (f > threshold).mean(axis=0)

    def variance_explained(self, activations: np.ndarray) -> float:
        Xs = self._standardize(activations)
        x_hat = self.reconstruct(activations)
        resid_var = float(np.var(Xs - x_hat))
        total_var = float(np.var(Xs)) + 1e-9
        return 1.0 - resid_var / total_var

    def analyze(self, activations: np.ndarray,
                dead_threshold: float = 1e-6) -> FeatureReport:
        f = self.encode(activations)
        freqs = (f > dead_threshold).mean(axis=0)
        live = int((freqs > 0).sum())
        dead = int((freqs == 0).sum())
        mean_l0 = float((f > dead_threshold).sum(axis=1).mean())
        top = sorted(enumerate(freqs.tolist()), key=lambda kv: kv[1], reverse=True)[:20]
        return FeatureReport(
            n_features=self.d_hidden, n_live=live, n_dead=dead,
            mean_l0=mean_l0, variance_explained=self.variance_explained(activations),
            frequencies=freqs, top_features=[(i, float(v)) for i, v in top],
        )

    def feature_direction(self, feature_idx: int) -> np.ndarray:
        """Direzione del decoder per una feature (in spazio standardizzato)."""
        return self.W_dec[feature_idx].copy()

    def __repr__(self) -> str:
        return (f"SparseAutoencoder(d_in={self.d_in}, d_hidden={self.d_hidden}, "
                f"l1={self.l1_coef})")
