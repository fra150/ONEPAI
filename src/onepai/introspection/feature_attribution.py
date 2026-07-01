"""
FeatureAttributor — A quale caratteristica dell'input si deve la predizione?

Tre metodi:
1. Integrated Gradients (Sundararajan et al. 2017) — interpolazione baseline→input
2. Gradient × Input — veloce, buona approssimazione
3. SmoothGrad (Smilkov et al. 2017) — media su versioni con rumore
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

__all__ = ["Attribution", "FeatureAttributor"]


@dataclass
class Attribution:
    """Mappa di attribuzione per un singolo esempio."""
    input_shape: Tuple
    scores: np.ndarray         # stessa forma dell'input — importanza di ogni feature
    method: str
    baseline: Optional[np.ndarray] = None
    metadata: Dict = field(default_factory=dict)

    @property
    def top_features(self) -> List[Tuple[int, float]]:
        """Indici piatti delle feature più importanti (ordine decrescente)."""
        flat = self.scores.ravel()
        order = np.argsort(flat)[::-1]
        return [(int(i), float(flat[i])) for i in order]

    def top(self, k: int = 10) -> List[Tuple[int, float]]:
        return self.top_features[:k]

    def positive_mask(self, threshold: float = 0.0) -> np.ndarray:
        """Maschera booleana delle feature con attribuzione > threshold."""
        return self.scores > threshold

    def normalize(self) -> "Attribution":
        """Ritorna una copia con scores normalizzati in [-1, 1]."""
        max_abs = np.abs(self.scores).max() + 1e-9
        return Attribution(
            input_shape=self.input_shape,
            scores=self.scores / max_abs,
            method=self.method,
            baseline=self.baseline,
            metadata=self.metadata,
        )


class FeatureAttributor:
    """
    Calcola mappe di attribuzione feature per modelli differenziabili.

    Uso (con modello differenziabile, es. PyTorch via numpy API)::

        def model_fn(x: np.ndarray) -> float:
            # chiama il tuo modello, ritorna uno scalare (es. log-prob)
            return float(model(torch.tensor(x)).item())

        attributor = FeatureAttributor()
        ig  = attributor.integrated_gradients(model_fn, input_vec, baseline)
        gxi = attributor.gradient_input(grad_vec, input_vec)
        sg  = attributor.smooth_grad(model_fn, input_vec, n_samples=50)

    Se non hai accesso ai gradienti usa perturbation_attribution (model-agnostic):
        pa = attributor.perturbation_attribution(model_fn, input_vec)
    """

    def __init__(self):
        self._attributions: List[Attribution] = []

    # ------------------------------------------------------------------
    # Integrated Gradients
    # ------------------------------------------------------------------

    def integrated_gradients(
        self,
        model_fn: Callable[[np.ndarray], float],
        input_x: np.ndarray,
        baseline: Optional[np.ndarray] = None,
        n_steps: int = 50,
        grad_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ) -> Attribution:
        """
        IG(x) = (x - baseline) ⊙ ∫[α=0→1] ∇f(baseline + α*(x-baseline)) dα

        Se grad_fn è None, calcola i gradienti numericamente (più lento ma
        funziona senza autograd).
        """
        x = np.asarray(input_x, dtype=np.float32)
        if baseline is None:
            baseline = np.zeros_like(x)
        bl = np.asarray(baseline, dtype=np.float32)
        delta = x - bl
        # Punti di interpolazione
        alphas = np.linspace(0.0, 1.0, n_steps + 1, dtype=np.float32)
        interpolated = np.stack([bl + a * delta for a in alphas], axis=0)  # (steps+1, *shape)
        # Calcola i gradienti a ogni punto
        if grad_fn is not None:
            grads = np.stack([grad_fn(interpolated[i]) for i in range(len(alphas))], axis=0)
        else:
            grads = np.stack([
                self._numerical_gradient(model_fn, interpolated[i])
                for i in range(len(alphas))
            ], axis=0)
        # Integrazione tramite trapezoidal rule
        avg_grads = (grads[:-1] + grads[1:]).mean(axis=0) / 2.0
        ig_scores = delta * avg_grads
        attr = Attribution(
            input_shape=x.shape,
            scores=ig_scores,
            method="integrated_gradients",
            baseline=bl,
            metadata={"n_steps": n_steps, "baseline_type": "zeros" if baseline is None else "custom"},
        )
        self._attributions.append(attr)
        return attr

    # ------------------------------------------------------------------
    # Gradient × Input
    # ------------------------------------------------------------------

    def gradient_input(
        self,
        gradient: np.ndarray,
        input_x: np.ndarray,
    ) -> Attribution:
        """
        Attribuzione = gradient ⊙ input (approssimazione veloce di IG con 1 step).
        gradient: ∂loss/∂x alla valutazione corrente.
        """
        g = np.asarray(gradient, dtype=np.float32)
        x = np.asarray(input_x, dtype=np.float32)
        scores = g * x
        attr = Attribution(
            input_shape=x.shape,
            scores=scores,
            method="gradient_input",
            metadata={"type": "first_order_approximation"},
        )
        self._attributions.append(attr)
        return attr

    # ------------------------------------------------------------------
    # SmoothGrad
    # ------------------------------------------------------------------

    def smooth_grad(
        self,
        model_fn: Callable[[np.ndarray], float],
        input_x: np.ndarray,
        n_samples: int = 25,
        noise_level: float = 0.1,
        grad_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ) -> Attribution:
        """
        SmoothGrad: media dei gradienti su versioni dell'input con rumore gaussiano.
        Riduce l'artefatto di gradiente "puntato" presente in Gradient×Input.
        """
        x = np.asarray(input_x, dtype=np.float32)
        sigma = noise_level * (x.max() - x.min() + 1e-9)
        smooth_scores = np.zeros_like(x)
        for _ in range(n_samples):
            noise = np.random.normal(0, sigma, x.shape).astype(np.float32)
            x_noisy = x + noise
            if grad_fn is not None:
                g = np.asarray(grad_fn(x_noisy), dtype=np.float32)
            else:
                g = self._numerical_gradient(model_fn, x_noisy)
            smooth_scores += g * x_noisy
        smooth_scores /= n_samples
        attr = Attribution(
            input_shape=x.shape,
            scores=smooth_scores,
            method="smooth_grad",
            metadata={"n_samples": n_samples, "noise_level": noise_level},
        )
        self._attributions.append(attr)
        return attr

    # ------------------------------------------------------------------
    # Perturbation (model-agnostic, senza gradienti)
    # ------------------------------------------------------------------

    def perturbation_attribution(
        self,
        model_fn: Callable[[np.ndarray], float],
        input_x: np.ndarray,
        mask_value: float = 0.0,
        normalize: bool = True,
    ) -> Attribution:
        """
        Rimuove ogni feature una alla volta e misura il drop nell'output.
        Funziona con qualsiasi modello (black-box).
        """
        x = np.asarray(input_x, dtype=np.float32)
        flat = x.ravel()
        baseline_score = model_fn(x)
        scores_flat = np.zeros(len(flat), dtype=np.float32)
        for i in range(len(flat)):
            perturbed_flat = flat.copy()
            perturbed_flat[i] = mask_value
            perturbed = perturbed_flat.reshape(x.shape)
            try:
                score = model_fn(perturbed)
                scores_flat[i] = abs(float(score) - float(baseline_score))
            except Exception:
                scores_flat[i] = 0.0
        scores = scores_flat.reshape(x.shape)
        if normalize:
            max_v = scores.max() + 1e-9
            scores = scores / max_v
        attr = Attribution(
            input_shape=x.shape,
            scores=scores,
            method="perturbation",
            metadata={"mask_value": mask_value, "baseline_score": float(baseline_score)},
        )
        self._attributions.append(attr)
        return attr

    # ------------------------------------------------------------------
    # Confronto e aggregazione
    # ------------------------------------------------------------------

    def compare_methods(self) -> Dict[str, np.ndarray]:
        """Restituisce un dizionario method → scores per tutti i metodi registrati."""
        return {a.method: a.scores for a in self._attributions}

    def consensus_attribution(self, normalize: bool = True) -> Optional[Attribution]:
        """
        Media delle attribuzioni di tutti i metodi registrati.
        Utile per ridurre il bias di un singolo metodo.
        """
        if not self._attributions:
            return None
        # Tenta di allineare le forme
        shapes = {a.scores.shape for a in self._attributions}
        if len(shapes) > 1:
            return None  # forme diverse — impossibile mediare
        stacked = np.stack([a.scores for a in self._attributions], axis=0)
        avg = stacked.mean(axis=0)
        if normalize:
            max_v = np.abs(avg).max() + 1e-9
            avg = avg / max_v
        return Attribution(
            input_shape=avg.shape,
            scores=avg,
            method="consensus",
            metadata={"n_methods": len(self._attributions),
                      "methods": [a.method for a in self._attributions]},
        )

    def find_critical_features(self, threshold: float = 0.5,
                                method: Optional[str] = None) -> List[Tuple[int, float]]:
        """
        Feature con |attribution| > threshold nell'ultimo metodo (o in `method`).
        """
        if not self._attributions:
            return []
        if method:
            candidates = [a for a in self._attributions if a.method == method]
            attr = candidates[-1] if candidates else self._attributions[-1]
        else:
            attr = self._attributions[-1]
        flat = attr.scores.ravel()
        return [(int(i), float(flat[i])) for i in range(len(flat))
                if abs(flat[i]) > threshold]

    # ------------------------------------------------------------------
    # Interno
    # ------------------------------------------------------------------

    @staticmethod
    def _numerical_gradient(
        fn: Callable[[np.ndarray], float],
        x: np.ndarray,
        eps: float = 1e-4,
    ) -> np.ndarray:
        """Gradiente numerico (differenze finite centrali). Lento ma universale."""
        flat = x.ravel().copy()
        grad_flat = np.zeros_like(flat)
        for i in range(len(flat)):
            flat[i] += eps
            f_plus = fn(flat.reshape(x.shape))
            flat[i] -= 2 * eps
            f_minus = fn(flat.reshape(x.shape))
            flat[i] += eps
            grad_flat[i] = (f_plus - f_minus) / (2 * eps)
        return grad_flat.reshape(x.shape)

    def __repr__(self) -> str:
        methods = [a.method for a in self._attributions]
        return f"FeatureAttributor(methods={methods})"
