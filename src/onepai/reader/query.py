"""
Query DSL per cercare e filtrare Observation negli archivi .onepai.

Uso::

    q = Query().where_layer("mlp.0").where_silence(min_ratio=0.5).limit(100)
    results = q.execute(archive)

    # Oppure con sintassi fluente:
    results = (Query()
        .where(lambda obs: obs.activation_value > 0.1)
        .order_by("activation_value", descending=True)
        .limit(20)
        .execute(archive))
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, List, Optional


def _meta(obs) -> dict:
    """Legge il dizionario meta da obs.meta o obs.metadata (tollerante a entrambi)."""
    m = getattr(obs, "meta", None) or getattr(obs, "metadata", None)
    return m if isinstance(m, dict) else {}


def _field_value(obs, field: str):
    """Legge un campo da attributo diretto, poi da meta, poi da data."""
    v = getattr(obs, field, None)
    if v is None:
        v = _meta(obs).get(field)
    if v is None and field in ("activation_value", "data"):
        v = getattr(obs, "data", None)
    return v

__all__ = ["Query", "QueryResult"]


@dataclass
class QueryResult:
    observations: list
    total_scanned: int
    total_matched: int
    query_repr: str

    def __len__(self) -> int:
        return len(self.observations)

    def __iter__(self):
        return iter(self.observations)


class Query:
    """
    Costruttore di query fluente per archivi ONEPAI.
    """

    def __init__(self):
        self._filters: List[Callable] = []
        self._order_by_field: Optional[str] = None
        self._descending: bool = False
        self._limit_n: Optional[int] = None
        self._offset_n: int = 0

    # ------------------------------------------------------------------
    # Filtri predefiniti
    # ------------------------------------------------------------------

    def where(self, predicate: Callable) -> "Query":
        """Aggiunge un filtro lambda generico."""
        self._filters.append(predicate)
        return self

    def where_layer(self, layer_name: str) -> "Query":
        """Filtra osservazioni il cui nome inizia con layer_name."""
        self._filters.append(
            lambda obs, ln=layer_name: obs.name.split(":")[0] == ln
        )
        return self

    def where_name_contains(self, substring: str) -> "Query":
        """Filtra osservazioni il cui nome contiene substring (case-insensitive)."""
        self._filters.append(
            lambda obs, s=substring.lower(): s in obs.name.lower()
        )
        return self

    def where_silence(
        self,
        min_ratio: float = 0.0,
        max_ratio: float = 1.0,
    ) -> "Query":
        """Filtra per silence_ratio — controlla attributo diretto o obs.meta['silence_ratio']."""
        def _f(obs, lo=min_ratio, hi=max_ratio):
            v = getattr(obs, "silence_ratio", None)
            if v is None:
                v = (_meta(obs)).get("silence_ratio")
            if v is None:
                return False
            return lo <= float(v) <= hi
        self._filters.append(_f)
        return self

    def where_activation(
        self,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
    ) -> "Query":
        """Filtra per activation_value — controlla attributo diretto, meta o data."""
        def _f(obs, lo=min_value, hi=max_value):
            v = getattr(obs, "activation_value", None)
            if v is None:
                v = _meta(obs).get("activation_value")
            if v is None:
                raw = getattr(obs, "data", None)
                if raw is not None:
                    try:
                        v = float(raw)
                    except (TypeError, ValueError):
                        pass
            if v is None:
                return False
            v = float(v)
            if lo is not None and v < lo:
                return False
            if hi is not None and v > hi:
                return False
            return True
        self._filters.append(_f)
        return self

    def where_metadata(self, key: str, value: Any = None) -> "Query":
        """Filtra per presenza (e opzionalmente valore) di una chiave in meta/metadata."""
        def _f(obs, k=key, v=value):
            meta = _meta(obs)
            if k not in meta:
                return False
            return v is None or meta[k] == v
        self._filters.append(_f)
        return self

    # ------------------------------------------------------------------
    # Ordinamento e paginazione
    # ------------------------------------------------------------------

    def order_by(self, field: str, descending: bool = False) -> "Query":
        """Ordina per un campo dell'Observation."""
        self._order_by_field = field
        self._descending = descending
        return self

    def limit(self, n: int) -> "Query":
        """Ritorna al massimo n risultati."""
        self._limit_n = n
        return self

    def offset(self, n: int) -> "Query":
        """Salta i primi n risultati."""
        self._offset_n = n
        return self

    # ------------------------------------------------------------------
    # Esecuzione
    # ------------------------------------------------------------------

    def execute(self, archive) -> QueryResult:
        """
        Esegue la query su un Archive o TreasureArchive.
        archive: qualsiasi oggetto con .iter_records().
        """
        matched = []
        total_scanned = 0
        for obs in archive.iter_records():
            total_scanned += 1
            if all(f(obs) for f in self._filters):
                matched.append(obs)
        # Ordinamento
        if self._order_by_field:
            matched.sort(
                key=lambda o: _field_value(o, self._order_by_field) or 0,
                reverse=self._descending,
            )
        # Paginazione
        paginated = matched[self._offset_n:]
        if self._limit_n is not None:
            paginated = paginated[:self._limit_n]
        return QueryResult(
            observations=paginated,
            total_scanned=total_scanned,
            total_matched=len(matched),
            query_repr=repr(self),
        )

    def execute_on(self, observations: Iterable) -> QueryResult:
        """Esegue la query su una sequenza di Observation già in memoria."""
        matched = [obs for obs in observations if all(f(obs) for f in self._filters)]
        total_scanned = len(matched)
        if self._order_by_field:
            matched.sort(
                key=lambda o: _field_value(o, self._order_by_field) or 0,
                reverse=self._descending,
            )
        paginated = matched[self._offset_n:]
        if self._limit_n is not None:
            paginated = paginated[:self._limit_n]
        return QueryResult(
            observations=paginated,
            total_scanned=total_scanned,
            total_matched=len(matched),
            query_repr=repr(self),
        )

    def __repr__(self) -> str:
        parts = [f"Query(filters={len(self._filters)}"]
        if self._order_by_field:
            parts.append(f"order_by={self._order_by_field!r}")
        if self._limit_n:
            parts.append(f"limit={self._limit_n}")
        return ", ".join(parts) + ")"
