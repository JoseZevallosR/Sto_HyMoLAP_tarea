"""Configuración centralizada de logging para toda la plataforma.

Se prefiere ``logging`` sobre ``print`` para que cada experimento pueda
escribir simultáneamente a consola y a un archivo dentro de su carpeta de
salida, manteniendo trazabilidad y reproducibilidad.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

_DEFAULT_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATEFMT = "%H:%M:%S"


def get_logger(name: str = "stohymolap") -> logging.Logger:
    """Devuelve un logger con nombre jerárquico bajo ``stohymolap``."""
    if not name.startswith("stohymolap"):
        name = f"stohymolap.{name}"
    return logging.getLogger(name)


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[Path] = None,
    name: str = "stohymolap",
) -> logging.Logger:
    """Configura el logger raíz del proyecto.

    Parameters
    ----------
    level:
        Nivel de logging (p.ej. ``logging.INFO``).
    log_file:
        Si se entrega, se añade un ``FileHandler`` a esa ruta.
    name:
        Nombre del logger raíz del proyecto.

    Returns
    -------
    logging.Logger
        Logger configurado y listo para usar.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(_DEFAULT_FORMAT, datefmt=_DATEFMT)

    console = logging.StreamHandler(stream=sys.stdout)
    console.setLevel(level)
    console.setFormatter(formatter)
    logger.addHandler(console)

    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
