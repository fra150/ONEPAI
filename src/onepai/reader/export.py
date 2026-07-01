"""
Esportazione di Observation e risultati di analisi in vari formati.

Formati supportati:
    JSON      — lista di oggetti JSON
    CSV       — tabella con header
    Markdown  — tabella Markdown (per README, report)
    HTML      — frammento tabella HTML (embedding in report)
    JSONL     — JSON Lines (streaming, un record per riga)
"""

from __future__ import annotations

import csv
import dataclasses
import io
import json
from pathlib import Path
from typing import Iterable, List, Optional, Union

__all__ = ["export_json", "export_csv", "export_jsonl",
           "export_markdown", "export_html_table",
           "Exporter"]


def _obs_to_dict(obs) -> dict:
    """Converte un'Observation in dict, tollerante a slots=True."""
    if dataclasses.is_dataclass(obs):
        return dataclasses.asdict(obs)
    if hasattr(obs, "__dict__"):
        return obs.__dict__.copy()
    return {"repr": repr(obs)}


def export_json(
    observations: Iterable,
    path: Optional[Union[str, Path]] = None,
    indent: int = 2,
) -> str:
    """Esporta come JSON array. Se path è dato, scrive su file."""
    data = [_obs_to_dict(o) for o in observations]
    text = json.dumps(data, indent=indent, default=str)
    if path:
        Path(path).write_text(text, encoding="utf-8")
    return text


def export_jsonl(
    observations: Iterable,
    path: Optional[Union[str, Path]] = None,
) -> str:
    """Esporta come JSONL (un record JSON per riga)."""
    lines = [json.dumps(_obs_to_dict(o), default=str) for o in observations]
    text = "\n".join(lines)
    if path:
        Path(path).write_text(text, encoding="utf-8")
    return text


def export_csv(
    observations: Iterable,
    path: Optional[Union[str, Path]] = None,
    fields: Optional[List[str]] = None,
) -> str:
    """Esporta come CSV. fields limita le colonne da includere."""
    rows = [_obs_to_dict(o) for o in observations]
    if not rows:
        return ""
    if fields is None:
        # Unione di tutti i campi trovati
        fields = list(dict.fromkeys(k for row in rows for k in row))
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fields, extrasaction="ignore",
                             lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    text = buf.getvalue()
    if path:
        Path(path).write_text(text, encoding="utf-8")
    return text


def export_markdown(
    observations: Iterable,
    fields: Optional[List[str]] = None,
    title: Optional[str] = None,
) -> str:
    """Esporta come tabella Markdown."""
    rows = [_obs_to_dict(o) for o in observations]
    if not rows:
        return "_Nessun dato_\n"
    if fields is None:
        fields = list(dict.fromkeys(k for row in rows for k in row))
    lines = []
    if title:
        lines.append(f"## {title}\n")
    # Header
    header = "| " + " | ".join(fields) + " |"
    sep = "| " + " | ".join("---" for _ in fields) + " |"
    lines.append(header)
    lines.append(sep)
    for row in rows:
        cells = [str(row.get(f, ""))[:50] for f in fields]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def export_html_table(
    observations: Iterable,
    fields: Optional[List[str]] = None,
    table_class: str = "onepai-table",
) -> str:
    """Esporta come frammento <table> HTML (senza <html>/<body>)."""
    rows = [_obs_to_dict(o) for o in observations]
    if not rows:
        return '<p class="onepai-empty">Nessun dato</p>'
    if fields is None:
        fields = list(dict.fromkeys(k for row in rows for k in row))
    th = "".join(f"<th>{f}</th>" for f in fields)
    tbody = "".join(
        "<tr>" + "".join(f"<td>{str(row.get(f, ''))[:80]}</td>" for f in fields) + "</tr>"
        for row in rows
    )
    return f'<table class="{table_class}"><thead><tr>{th}</tr></thead><tbody>{tbody}</tbody></table>'


class Exporter:
    """
    Interfaccia unificata per esportare risultati in multipli formati.

    Uso::

        exp = Exporter(registry.get("mlp.0"))
        exp.to_json("output/mlp0.json")
        exp.to_csv("output/mlp0.csv")
        md = exp.to_markdown()
    """

    def __init__(self, observations: Iterable):
        self._obs = list(observations)

    def to_json(self, path: Optional[Union[str, Path]] = None, **kwargs) -> str:
        return export_json(self._obs, path=path, **kwargs)

    def to_jsonl(self, path: Optional[Union[str, Path]] = None) -> str:
        return export_jsonl(self._obs, path=path)

    def to_csv(self, path: Optional[Union[str, Path]] = None,
                fields: Optional[List[str]] = None) -> str:
        return export_csv(self._obs, path=path, fields=fields)

    def to_markdown(self, fields: Optional[List[str]] = None,
                     title: Optional[str] = None) -> str:
        return export_markdown(self._obs, fields=fields, title=title)

    def to_html_table(self, fields: Optional[List[str]] = None) -> str:
        return export_html_table(self._obs, fields=fields)

    def count(self) -> int:
        return len(self._obs)

    def __repr__(self) -> str:
        return f"Exporter({len(self._obs)} observations)"
