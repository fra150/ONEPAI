"""Decoratori utili per il framework ONEPAI."""

from __future__ import annotations

import functools
import time
from typing import Any, Callable, Optional, TypeVar

import numpy as np

__all__ = ["timer", "cached_property_np", "validate_ndarray",
           "retry", "requires_numpy"]

F = TypeVar("F", bound=Callable[..., Any])


def timer(label: Optional[str] = None) -> Callable[[F], F]:
    """
    Decoratore che stampa il tempo di esecuzione della funzione.

    @timer("Shadow mapping")
    def compute_shadows(): ...
    """
    def decorator(fn: F) -> F:
        name = label or fn.__qualname__

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            t0 = time.perf_counter()
            result = fn(*args, **kwargs)
            elapsed = time.perf_counter() - t0
            print(f"[timer] {name}: {elapsed*1000:.2f} ms")
            return result

        return wrapper  # type: ignore
    return decorator


def cached_property_np(fn: F) -> property:
    """
    Come @property ma memorizza il risultato numpy la prima volta.
    Invalida la cache se l'oggetto ha un attributo _cache_version che cambia.
    """
    attr = "_cached_" + fn.__name__

    @property  # type: ignore
    @functools.wraps(fn)
    def wrapper(self):
        val = getattr(self, attr, None)
        if val is None:
            val = fn(self)
            setattr(self, attr, val)
        return val

    return wrapper


def validate_ndarray(
    arg_name: str = "x",
    min_dim: int = 1,
    max_dim: int = 4,
    dtype=None,
) -> Callable[[F], F]:
    """
    Valida che un argomento posizionale sia un numpy array con le dimensioni corrette.

    @validate_ndarray("weights", min_dim=2, max_dim=2)
    def analyze(self, weights): ...
    """
    def decorator(fn: F) -> F:
        params = list(fn.__code__.co_varnames[:fn.__code__.co_argcount])
        try:
            idx = params.index(arg_name)
        except ValueError:
            return fn

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            if arg_name in kwargs:
                val = kwargs[arg_name]
            elif idx < len(args):
                val = args[idx]
            else:
                return fn(*args, **kwargs)
            if not isinstance(val, np.ndarray):
                try:
                    val = np.asarray(val)
                except Exception as e:
                    raise TypeError(f"{arg_name!r} deve essere convertibile a ndarray: {e}") from e
            if not (min_dim <= val.ndim <= max_dim):
                raise ValueError(
                    f"{arg_name!r} deve avere {min_dim}-{max_dim} dimensioni, "
                    f"ricevuto {val.ndim}D"
                )
            if dtype is not None:
                val = val.astype(dtype)
            if arg_name in kwargs:
                kwargs[arg_name] = val
            else:
                args = args[:idx] + (val,) + args[idx + 1:]
            return fn(*args, **kwargs)

        return wrapper  # type: ignore
    return decorator


def retry(max_attempts: int = 3, exceptions: tuple = (Exception,),
          delay: float = 0.0) -> Callable[[F], F]:
    """
    Riprova la funzione fino a `max_attempts` volte in caso di eccezione.
    """
    def decorator(fn: F) -> F:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(max_attempts):
                try:
                    return fn(*args, **kwargs)
                except exceptions as e:
                    last_exc = e
                    if delay > 0 and attempt < max_attempts - 1:
                        time.sleep(delay)
            raise last_exc  # type: ignore
        return wrapper  # type: ignore
    return decorator


def requires_numpy(fn: F) -> F:
    """Lancia ImportError descrittivo se numpy non è disponibile."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            import numpy  # noqa: F401
        except ImportError:
            raise ImportError(
                f"{fn.__qualname__} richiede numpy. "
                "Installalo con: pip install numpy"
            )
        return fn(*args, **kwargs)
    return wrapper  # type: ignore
