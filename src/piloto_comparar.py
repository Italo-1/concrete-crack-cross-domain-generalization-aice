"""Compara dos ejecuciones identicas del piloto (E0) para comprobar el determinismo.

Lee `results/piloto/a` y `results/piloto/b` (mismos argumentos, distinta
`--salida`) y compara, celda a celda, los conteos y el F1, y, imagen a imagen,
la prediccion y la probabilidad de grieta. Escribe el informe en
`results/piloto/comparacion.json`.

    python src/piloto_comparar.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
PILOTO = RAIZ / "results" / "piloto"
CLAVE = ["arquitectura", "semilla", "condicion", "particion", "dominio_train", "dominio_test"]


def main() -> int:
    a = pd.read_csv(PILOTO / "a" / "tables" / "resultados_matriz.csv")
    b = pd.read_csv(PILOTO / "b" / "tables" / "resultados_matriz.csv")
    m = a.merge(b, on=CLAVE, suffixes=("_a", "_b"))
    informe = {"celdas": len(m)}
    for col in ("tp", "tn", "fp", "fn", "f1", "mejor_epoca", "epocas_entrenadas"):
        informe[f"celdas_distintas_{col}"] = int((m[f"{col}_a"] != m[f"{col}_b"]).sum())
    informe["max_dif_abs_f1"] = float((m["f1_a"] - m["f1_b"]).abs().max())

    preds_a = sorted((PILOTO / "a" / "predicciones").rglob("*.csv.gz"))
    n_img, n_pred_dist, max_dif_prob = 0, 0, 0.0
    for pa in preds_a:
        pb = PILOTO / "b" / pa.relative_to(PILOTO / "a")
        da, db = pd.read_csv(pa), pd.read_csv(pb)
        assert (da[["dominio_test", "indice_en_dominio"]].values == db[["dominio_test", "indice_en_dominio"]].values).all()
        n_img += len(da)
        n_pred_dist += int((da["pred"] != db["pred"]).sum())
        max_dif_prob = max(max_dif_prob, float(np.abs(da["prob_grieta"] - db["prob_grieta"]).max()))
    informe.update({"imagenes": n_img, "predicciones_distintas": n_pred_dist,
                    "max_dif_abs_prob_grieta": max_dif_prob})

    ta = pd.read_csv(PILOTO / "a" / "tables" / "historial_entrenamiento.csv")
    tb = pd.read_csv(PILOTO / "b" / "tables" / "historial_entrenamiento.csv")
    informe["max_dif_abs_perdida_train"] = float((ta["perdida_train"] - tb["perdida_train"]).abs().max())
    informe["identico"] = (informe["predicciones_distintas"] == 0 and informe["max_dif_abs_prob_grieta"] == 0
                           and informe["max_dif_abs_perdida_train"] == 0)

    (PILOTO / "comparacion.json").write_text(json.dumps(informe, indent=2), encoding="utf-8")
    print(json.dumps(informe, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
