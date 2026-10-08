"""Entrenamiento, evaluacion y determinismo para la extension AICE de P1.

Por que no se usa `fieutils.vision.entrenar` / `matriz_transferencia`
---------------------------------------------------------------------
`fieutils` lo comparte P5 y no se modifica (decision del 02/10, README). De
`fieutils.vision` solo se reutiliza lo que no cambia: `crear_modelo`,
`transformaciones` y las estadisticas de ImageNet. Lo que aqui es distinto:

- **Determinismo real.** `fieutils.config.fijar_semilla(determinista=True)`
  solo toca `cudnn.deterministic` y `cudnn.benchmark`. Aqui ademas se activa
  `torch.use_deterministic_algorithms(True)` y se exige
  `CUBLAS_WORKSPACE_CONFIG` (que el script principal fija ANTES de importar
  torch), y el orden de los lotes sale de un `torch.Generator` sembrado.
- **Lote fijo.** `fieutils.vision.lote_para_vram` calcula el lote con la VRAM
  libre en el momento, que varia entre ejecuciones; en la corrida original las
  cuatro arquitecturas llegaron al tope de 64. Aqui el lote es 64 siempre, para
  que dos ejecuciones identicas no difieran por la memoria libre.
- **Evaluacion detallada.** `evaluar_detallado` devuelve la probabilidad y la
  prediccion de cada imagen, con las que se calculan TP, TN, FP, FN, precision,
  recall y F1 (EXP2). El F1 sale de los conteos, con la misma definicion que
  `sklearn.metrics.f1_score(pos_label=1, zero_division=0)` de la version
  anterior.
"""

from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

LOTE = 64
CLASE_GRIETA = 1


# ---------------------------------------------------------------- semillas

def fijar_semilla_determinista(semilla: int) -> None:
    """Semilla de Python, NumPy y PyTorch, con algoritmos deterministas.

    Lanza un error si el entorno no tiene `CUBLAS_WORKSPACE_CONFIG`: sin esa
    variable cuBLAS no es determinista y PyTorch lo rechaza al primer `matmul`,
    pero mejor saberlo antes de cargar datos.
    """
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in (":4096:8", ":16:8"):
        raise RuntimeError("Fijar CUBLAS_WORKSPACE_CONFIG=:4096:8 antes de importar torch.")
    random.seed(semilla)
    np.random.seed(semilla)
    torch.manual_seed(semilla)
    torch.cuda.manual_seed_all(semilla)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def generador(semilla: int) -> torch.Generator:
    """Generador para el orden de lotes del DataLoader de entrenamiento."""
    g = torch.Generator()
    g.manual_seed(semilla)
    return g


def dispositivo() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch sin CUDA: el experimento no se corre en CPU.")
    return torch.device("cuda")


# ----------------------------------------------------------- entrenamiento

@dataclass
class Resultado:
    modelo: nn.Module
    mejor_epoca: int
    mejor_f1_val: float
    epocas_entrenadas: int
    historial: list[dict] = field(default_factory=list)
    segundos: float = 0.0
    mejores_pesos: dict | None = None


def entrenar(
    modelo: nn.Module,
    cargador_train: DataLoader,
    cargador_val: DataLoader,
    epocas: int = 12,
    lr: float = 3e-4,
    peso_decaimiento: float = 1e-4,
    paciencia: int = 3,
    precision_mixta: bool = True,
    log=None,
) -> Resultado:
    """Ajuste fino con parada temprana sobre el F1 de validacion de la clase grieta.

    Mismos hiperparametros que `fieutils.vision.entrenar` en la version
    anterior (AdamW, coseno con T_max = epocas, paciencia 3, precision mixta).
    Devuelve el modelo con los pesos de la mejor epoca.
    """
    dev = dispositivo()
    modelo = modelo.to(dev)
    criterio = nn.CrossEntropyLoss()
    optimizador = torch.optim.AdamW(modelo.parameters(), lr=lr, weight_decay=peso_decaimiento)
    planificador = torch.optim.lr_scheduler.CosineAnnealingLR(optimizador, T_max=epocas)
    escalador = torch.amp.GradScaler("cuda", enabled=precision_mixta)

    mejor_f1, mejor_epoca, sin_mejora = -1.0, -1, 0
    mejores_pesos, historial = None, []
    inicio = time.time()

    for epoca in range(epocas):
        modelo.train()
        perdida_total, n = 0.0, 0
        for x, y in cargador_train:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            optimizador.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=precision_mixta):
                perdida = criterio(modelo(x), y)
            escalador.scale(perdida).backward()
            escalador.step(optimizador)
            escalador.update()
            perdida_total += perdida.item() * x.size(0)
            n += x.size(0)
        planificador.step()

        val = metricas(*evaluar_detallado(modelo, cargador_val)[:2])
        registro = {"epoca": epoca, "perdida_train": perdida_total / max(n, 1),
                    "f1_val": val["f1"], "exactitud_val": val["exactitud"]}
        historial.append(registro)
        if log:
            log.info(f"epoca {epoca:>2} | perdida {registro['perdida_train']:.4f} | "
                     f"f1 val {val['f1']:.4f} | exactitud {val['exactitud']:.4f}")

        if val["f1"] > mejor_f1:
            mejor_f1, mejor_epoca, sin_mejora = val["f1"], epoca, 0
            mejores_pesos = {k: v.detach().cpu().clone() for k, v in modelo.state_dict().items()}
        else:
            sin_mejora += 1
            if sin_mejora >= paciencia:
                if log:
                    log.info(f"parada temprana en la epoca {epoca} (paciencia {paciencia})")
                break

    if mejores_pesos is not None:
        modelo.load_state_dict(mejores_pesos)

    return Resultado(modelo=modelo, mejor_epoca=mejor_epoca, mejor_f1_val=mejor_f1,
                     epocas_entrenadas=len(historial), historial=historial,
                     segundos=time.time() - inicio, mejores_pesos=mejores_pesos)


# --------------------------------------------------------------- evaluacion

@torch.no_grad()
def evaluar_detallado(modelo: nn.Module, cargador: DataLoader, precision_mixta: bool = True):
    """Etiqueta real, prediccion y probabilidad de grieta de cada imagen, en orden."""
    dev = dispositivo()
    modelo = modelo.to(dev).eval()
    reales, predichas, probs = [], [], []
    for x, y in cargador:
        x = x.to(dev, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=precision_mixta):
            salida = modelo(x)
        p = torch.softmax(salida.float(), dim=1)[:, CLASE_GRIETA]
        predichas.append(salida.argmax(1).cpu().numpy())
        probs.append(p.cpu().numpy())
        reales.append(y.numpy())
    return np.concatenate(reales), np.concatenate(predichas), np.concatenate(probs)


def metricas(y_real: np.ndarray, y_pred: np.ndarray) -> dict:
    """Conteos de la matriz de confusion y metricas de la clase grieta.

    Precision y recall con `zero_division=0` (si el modelo no predice ninguna
    grieta, precision = 0), igual que sklearn en la version anterior. El caso
    se puede reconocer porque TP + FP = 0.
    """
    tp = int(np.sum((y_pred == 1) & (y_real == 1)))
    tn = int(np.sum((y_pred == 0) & (y_real == 0)))
    fp = int(np.sum((y_pred == 1) & (y_real == 0)))
    fn = int(np.sum((y_pred == 0) & (y_real == 1)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn, "precision": precision,
            "recall": recall, "f1": f1, "exactitud": (tp + tn) / max(len(y_real), 1)}
