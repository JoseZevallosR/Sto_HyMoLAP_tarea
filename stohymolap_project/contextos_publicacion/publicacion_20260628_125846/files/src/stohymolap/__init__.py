"""StoHyMoLAP: framework experimental RAMIS / lluvia-escorrentía.

El paquete expone módulos de datos, física hidrológica, calibración,
experimentos, métricas y validaciones anti-fuga. Mantener este archivo permite
que imports como ``import stohymolap`` funcionen tanto en instalación editable
como al ejecutar scripts desde el árbol del repositorio.
"""
from __future__ import annotations

try:  # pragma: no cover - depende de si el paquete está instalado
    from importlib.metadata import version

    __version__ = version("stohymolap")
except Exception:  # noqa: BLE001
    __version__ = "0.1.0"

__all__ = ["__version__"]

