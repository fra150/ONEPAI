"""
Direct Logit Attribution (DLA) — quanto scrive ogni componente *contro* il token.

Le ablation dicono "se tolgo questo componente, il token risale": un effetto
causale, ma costoso e (con zero-ablation) fuori distribuzione. La DLA dà lo
stesso tipo di informazione in **un solo forward**:

    contributo(componente c) = (c - mean(c)) · scale_finale · W_U[target]

cioè la parte del logit finale del target scritta direttamente da c, sotto la
linearizzazione della LayerNorm/RMSNorm finale.

Come si usa:
  - DLA dice *chi scrive* contro il target (contributo negativo);
  - l'ablation dice *chi, se rimosso, lo ripristina*;
  - dove i due **concordano** c'è un effetto diretto;
  - dove **discordano** c'è un effetto indiretto (il componente influenza altri
    componenti a valle, ed è lì che stanno i circuiti interessanti).

Riferimento: nostalgebraist (2020), Elhage et al., "A Mathematical Framework for
Transformer Circuits" (2021).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..model.adapter import HookPoint, ModelAdapter

__all__ = ["DLAContribution", "DLAResult", "DirectLogitAttributor"]


@dataclass
class DLAContribution:
    """Contributo diretto di un componente al logit di un token target."""
    component: str            # "embed" | "blocks.11.mlp" | "blocks.11.head.3"
    layer: int
    kind: str                 # "embed" | "attn" | "mlp" | "head"
    value: float

    @property
    def abs_value(self) -> float:
        return abs(self.value)


@dataclass
class DLAResult:
    """Attribuzione completa a una posizione, con verifica di ricostruzione."""
    model_name: str
    target_token: int
    position: int
    actual_logit: float
    total: float                      # somma dei contributi attribuiti
    contributions: List[DLAContribution] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def constant_offset(self) -> float:
        """Parte del logit non attribuibile (bias della norma finale, ecc.)."""
        return self.actual_logit - self.total

    def sorted(self) -> List[DLAContribution]:
        return sorted(self.contributions, key=lambda c: c.value)

    def against(self, k: int = 5) -> List[DLAContribution]:
        """I componenti che scrivono di più CONTRO il target (contributo negativo)."""
        return [c for c in self.sorted() if c.value < 0][:k]

    def for_(self, k: int = 5) -> List[DLAContribution]:
        """I componenti che scrivono di più A FAVORE del target."""
        return sorted(self.contributions, key=lambda c: c.value, reverse=True)[:k]

    def by_component(self) -> Dict[str, float]:
        return {c.component: c.value for c in self.contributions}

    def summary(self) -> Dict:
        return {
            "model": self.model_name,
            "target_token": self.target_token,
            "position": self.position,
            "actual_logit": round(self.actual_logit, 3),
            "attributed_total": round(self.total, 3),
            "constant_offset": round(self.constant_offset, 3),
            "components": len(self.contributions),
            "top_against": [c.component for c in self.against(3)],
        }

    def narrate(self, token_str: Optional[str] = None) -> str:
        tok = repr(token_str) if token_str else f"token {self.target_token}"
        against = self.against(1)
        s = (f"logit({tok}) = {self.actual_logit:+.2f}; attribuito {self.total:+.2f} "
             f"(offset costante {self.constant_offset:+.2f})")
        if against:
            c = against[0]
            s += f"; chi scrive più contro: {c.component} ({c.value:+.2f})"
        return s


class DirectLogitAttributor:
    """
    Calcola la DLA per tutti i componenti di un forward.

    Esempio::

        dla = DirectLogitAttributor(model)
        res = dla.attribute(adapter.encode("The capital of France is"), target_token)
        for c in res.against(5):
            print(c.component, round(c.value, 3))
    """

    def __init__(self, model: ModelAdapter, include_heads: bool = True):
        self.model = model
        self.W_U = model.unembed_matrix()
        if self.W_U is None:
            raise ValueError(
                "Il modello non espone la matrice di unembedding: DLA impossibile.")
        self.include_heads = include_heads
        self._head_W: Dict[int, Optional[np.ndarray]] = {}

    # ------------------------------------------------------------------

    def _head_matrix(self, layer: int) -> Optional[np.ndarray]:
        if layer not in self._head_W:
            self._head_W[layer] = self.model.head_unembed(layer)
        return self._head_W[layer]

    def attribute(
        self,
        tokens,
        target_token: int,
        position: int = -1,
        include_heads: Optional[bool] = None,
    ) -> DLAResult:
        """Contributi di embed, attention, MLP e (opz.) singole teste al target."""
        model = self.model
        include_heads = self.include_heads if include_heads is None else include_heads
        logits, cache = model.run_with_cache(tokens)

        n = model.n_layers
        resid = cache.get(HookPoint.resid_post(n - 1))
        if resid is None:
            raise RuntimeError("resid_post finale assente dalla cache.")
        scale, subtract_mean = model.final_norm_scale(resid)
        if scale is None:
            scale = np.ones_like(resid)
        target_dir = np.asarray(self.W_U[:, target_token], dtype=np.float32)

        def contrib(vector: np.ndarray) -> float:
            v = np.asarray(vector, dtype=np.float32)
            if v.ndim > 1:
                v = v[position]
            if subtract_mean:
                v = v - v.mean()
            return float((v * scale[position]) @ target_dir)

        out: List[DLAContribution] = []
        embed = cache.get(HookPoint.EMBED)
        if embed is not None:
            out.append(DLAContribution("embed", -1, "embed", contrib(embed)))

        for i in range(n):
            attn = cache.get(HookPoint.attn_out(i))
            mlp = cache.get(HookPoint.mlp_out(i))

            # Con le teste accese, l'attention aggregata viene sostituita dalle
            # singole teste (+ eventuale residuo non attribuibile, es. il bias),
            # così i contributi restano una partizione esatta del residual.
            z = cache.get(HookPoint.attn_z(i)) if include_heads else None
            W_O = self._head_matrix(i) if include_heads else None
            if z is not None and W_O is not None and z.ndim == 3:
                z_pos = np.asarray(z[position], dtype=np.float32)         # (nh, dh)
                heads_out = np.einsum("he,hed->hd", z_pos, W_O)           # (nh, d)
                for h in range(heads_out.shape[0]):
                    out.append(DLAContribution(
                        f"blocks.{i}.head.{h}", i, "head", contrib(heads_out[h])))
                if attn is not None:
                    remainder = np.asarray(attn[position], dtype=np.float32) - heads_out.sum(axis=0)
                    if float(np.abs(remainder).max()) > 1e-6:
                        out.append(DLAContribution(
                            f"blocks.{i}.attn.bias", i, "attn", contrib(remainder)))
            elif attn is not None:
                out.append(DLAContribution(f"blocks.{i}.attn", i, "attn", contrib(attn)))

            if mlp is not None:
                out.append(DLAContribution(f"blocks.{i}.mlp", i, "mlp", contrib(mlp)))

        result = DLAResult(
            model_name=model.model_name,
            target_token=int(target_token),
            position=position,
            actual_logit=float(logits[position, target_token]),
            total=float(sum(c.value for c in out)),
            contributions=out,
        )
        # Verifica: la somma deve ricostruire il logit a meno di una costante
        # (bias della norma finale). Un errore grande = DLA inaffidabile.
        if abs(result.constant_offset) > 5.0:
            result.warnings.append(
                f"offset costante alto ({result.constant_offset:+.2f}): "
                "possibile norma finale non lineare o pesi non allineati.")
        return result
