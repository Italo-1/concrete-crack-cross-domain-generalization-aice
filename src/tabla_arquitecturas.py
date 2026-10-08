"""Tabla de arquitecturas (M2): familia, parametros con cabeza de 2 clases,
dimension de la caracteristica de la sonda lineal, pesos de origen y lote.

Se calcula en CPU instanciando cada modelo de `timm` con `pretrained=False`
(los parametros no dependen de los pesos). La etiqueta de los pesos es la que
`timm` resuelve por defecto para cada nombre, la misma que esta en la cache
local (README, E0).

    python src/tabla_arquitecturas.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import timm
import torch

from fieutils import tablas

RAIZ = Path(__file__).resolve().parent.parent
TABLAS = RAIZ / "results" / "tables"

ARQUITECTURAS = {
    "resnet18": ("ResNet-18", "CNN, residual", "He et al. (2016)"),
    "efficientnet_b0": ("EfficientNet-B0", "CNN, compound-scaled", "Tan and Le (2019)"),
    "mobilenetv3_small_100": ("MobileNetV3-Small", "CNN, mobile search", "Howard et al. (2019)"),
    "vit_tiny_patch16_224": ("ViT-Tiny/16", "Vision transformer", "Dosovitskiy et al. (2021); Steiner et al. (2021)"),
}


def main() -> int:
    filas = []
    for nombre, (legible, familia, ref) in ARQUITECTURAS.items():
        m2 = timm.create_model(nombre, pretrained=False, num_classes=2)
        m0 = timm.create_model(nombre, pretrained=False, num_classes=0)
        etiqueta = m2.pretrained_cfg.get("tag") or ""
        filas.append({
            "Architecture": legible, "Family": familia,
            "Parameters (M)": round(sum(p.numel() for p in m2.parameters()) / 1e6, 2),
            # Dimension real de la salida con num_classes=0 (la que usa la sonda):
            # en MobileNetV3 no coincide con num_features, por la cabeza conv.
            "Probe feature dim.": int(m0.eval()(torch.zeros(1, 3, 224, 224)).shape[1]),
            "timm weights": f"{nombre}.{etiqueta}" if etiqueta else nombre,
            "Input": "224 x 224", "Batch": 64, "Reference": ref,
        })
        print(filas[-1])
    df = pd.DataFrame(filas).set_index("Architecture")
    tablas.exportar(df, TABLAS, "tabla_arquitecturas",
                    caption=("Architectures compared. Parameters counted with the 2-class head. All four "
                             "are initialised from ImageNet weights served by timm and trained with the same "
                             "optimiser, schedule and batch size."),
                    etiqueta="tab:arquitecturas", decimales=2, indice=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
