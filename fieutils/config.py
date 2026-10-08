"""Configuracion comun a los cinco articulos de la linea B.

Centraliza semilla, dispositivo de computo y registro de ejecucion. Todos los
scripts `03_experiment.py` deben llamar a `fijar_semilla()` como primera accion:
el protocolo de la guia exige semilla 42 declarada en el manuscrito.
"""

from __future__ import annotations

import logging
import os
import random
from pathlib import Path

import numpy as np

SEMILLA = 42

# Raiz del arbol de trabajo (C:\lineaB). Los proyectos cuelgan de aqui.
RAIZ = Path(__file__).resolve().parent.parent

PROYECTOS = {
    "P7": RAIZ / "01-ingenius-P7",
    "P4": RAIZ / "02-serbian-jee-P4",
    "P1": RAIZ / "03-tehnicki-glasnik-P1",
    "P5": RAIZ / "04-sakarya-jcis-P5",
    "P3": RAIZ / "05-mev-P3",
}


def fijar_semilla(semilla: int = SEMILLA, determinista: bool = True) -> None:
    """Fija la semilla de Python, NumPy y PyTorch.

    Con `determinista=True` se desactiva el autotuner de cuDNN. Cuesta algo de
    velocidad pero garantiza que dos corridas den el mismo numero, que es lo que
    un revisor puede pedir. En los barridos largos (P1, P5) conviene dejarlo en
    False y declarar la variabilidad mediante las semillas multiples.
    """
    random.seed(semilla)
    np.random.seed(semilla)
    os.environ["PYTHONHASHSEED"] = str(semilla)

    try:
        import torch
    except ImportError:
        return

    torch.manual_seed(semilla)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(semilla)
    torch.backends.cudnn.deterministic = determinista
    torch.backends.cudnn.benchmark = not determinista


def dispositivo(verboso: bool = True):
    """Devuelve la GPU si esta disponible, si no la CPU.

    Aborta con un mensaje claro si PyTorch se instalo sin CUDA: es el fallo que
    mas tiempo hace perder, porque el entrenamiento arranca igual y solo se nota
    horas despues.
    """
    import torch

    if not torch.cuda.is_available():
        if verboso:
            print(
                "AVISO: no hay CUDA. torch =", torch.__version__,
                "\nSi la version termina en '+cpu', reinstalar con el indice cu126.",
            )
        return torch.device("cpu")

    dev = torch.device("cuda")
    if verboso:
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {props.name} | VRAM: {props.total_memory / 1024**3:.1f} GB")
    return dev


def vram_libre_gb() -> float:
    """VRAM libre en GB. Util para elegir el tamano de lote sin provocar OOM."""
    import torch

    if not torch.cuda.is_available():
        return 0.0
    libre, _total = torch.cuda.mem_get_info()
    return libre / 1024**3


def configurar_log(nombre: str, proyecto: str | Path) -> logging.Logger:
    """Registro que escribe a consola y a `results/logs/<nombre>.log`.

    El log es la bitacora de computo que pide la planificacion: sirve para
    rellenar el campo "tiempo de computo consumido" del README diario.
    """
    ruta = Path(proyecto) if isinstance(proyecto, Path) else PROYECTOS[proyecto]
    destino = ruta / "results" / "logs"
    destino.mkdir(parents=True, exist_ok=True)

    log = logging.getLogger(nombre)
    log.setLevel(logging.INFO)
    log.handlers.clear()

    formato = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S")

    archivo = logging.FileHandler(destino / f"{nombre}.log", encoding="utf-8")
    archivo.setFormatter(formato)
    log.addHandler(archivo)

    consola = logging.StreamHandler()
    consola.setFormatter(formato)
    log.addHandler(consola)

    return log
