"""Analisis pedidos por la revision adversarial, ronda 2 (03/10/2026).

Todo sale de resultados ya guardados (matriz por celda y probabilidades por
imagen de la condicion base); no se reentrena nada. Se anaden despues de ver
los resultados y se declaran asi en el manuscrito.

1. Caida relativa anclada en la diagonal del DESTINO (misma arquitectura y
   semilla, modelo entrenado en el destino), junto a la anclada en el origen
   (ecuacion 1 del manuscrito), y perdida absoluta frente a la diagonal del
   destino.
2. Exactitud y coeficiente de correlacion de Matthews (MCC) por particion. Con
   tests de 300 + 300, la exactitud es igual a la exactitud balanceada; el
   clasificador que responde "grieta" a todo tiene exactitud 0.5 y MCC 0.
3. Umbral recalibrado con k = 50 etiquetas del destino (25 por clase, 200
   sorteos, semilla 42) por dos criterios: maximo F1 (el de 08_umbral_auroc.py)
   y maximo indice de Youden (TPR - FPR). Para cada uno: F1, exactitud y MCC en
   las 550 imagenes restantes.
4. Precision a la prevalencia real del destino (fraccion de parches con grieta
   de la coleccion completa tras quitar duplicados, seccion 3.1 del
   manuscrito), con la TPR y la FPR medidas en el test balanceado, y la F1 del
   clasificador trivial a esa prevalencia, 2*pi / (1 + pi).

Salidas: results/tables/ronda2.json y ronda2_particiones.csv.

    python src/09_ronda2.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
TABLAS = RAIZ / "results" / "tables"
PRED = RAIZ / "results" / "predicciones" / "baseline_original"
SEMILLA = 42
K = 50
SORTEOS = 200
CORRIDA = ["arquitectura", "semilla"]
# Fraccion con grieta de cada coleccion completa (manuscrito, seccion 3.1).
PREVALENCIA = {"sdnet_D": 0.149, "sdnet_P": 0.107, "sdnet_W": 0.212, "metu": 0.5}
ORDEN = ["dentro", "entre_superficie", "sdnet_a_metu", "metu_a_sdnet"]


def direccion(origen: str, destino: str) -> str:
    if origen == destino:
        return "dentro"
    if origen == "metu":
        return "metu_a_sdnet"
    if destino == "metu":
        return "sdnet_a_metu"
    return "entre_superficie"


def mcc(tp, tn, fp, fn) -> float:
    den = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    return float((tp * tn - fp * fn) / den) if den > 0 else 0.0


def metricas(y: np.ndarray, pred: np.ndarray) -> tuple[float, float, float]:
    tp = int(np.sum(pred & (y == 1)))
    fp = int(np.sum(pred & (y == 0)))
    fn = int(np.sum(~pred & (y == 1)))
    tn = int(np.sum(~pred & (y == 0)))
    f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
    acc = 0.5 * (tp / (tp + fn) + tn / (tn + fp))   # balanceada
    return f1, acc, mcc(tp, tn, fp, fn)


def umbral(y: np.ndarray, p: np.ndarray, criterio: str) -> float:
    """Umbral sobre "p >= t" que maximiza el criterio; candidatos: valores
    unicos de p. Empates: el umbral mas cercano a 0.5."""
    t = np.unique(p)
    pos, neg = int((y == 1).sum()), int((y == 0).sum())
    orden = np.argsort(p, kind="stable")
    ys = (y[orden] == 1).astype(int)
    sufijo = np.concatenate([np.cumsum(ys[::-1])[::-1], [0]])
    i = np.searchsorted(p[orden], t, side="left")
    tp = sufijo[i]
    fp = (len(p) - i) - tp
    if criterio == "f1":
        valor = np.where(tp > 0, 2 * tp / np.maximum(tp + fp + pos, 1), 0.0)
    else:   # youden
        valor = tp / pos - fp / neg
    mejor = np.flatnonzero(np.isclose(valor, valor.max()))
    return float(t[mejor[np.argmin(np.abs(t[mejor] - 0.5))]])


def main() -> int:
    m = pd.read_csv(TABLAS / "resultados_matriz.csv")
    b = m[(m.particion == "original") & (m.condicion == "baseline")].copy()
    assert len(b) == 192
    b["direccion"] = [direccion(s, t) for s, t in zip(b.dominio_train, b.dominio_test)]
    diag = b[b.direccion == "dentro"].set_index(CORRIDA + ["dominio_train"]).f1

    # 1-2. anclajes, exactitud y MCC por celda
    b["f1_diag_origen"] = [diag[(a, s, o)] for a, s, o in zip(b.arquitectura, b.semilla, b.dominio_train)]
    b["f1_diag_destino"] = [diag[(a, s, t)] for a, s, t in zip(b.arquitectura, b.semilla, b.dominio_test)]
    b["caida_origen"] = (b.f1_diag_origen - b.f1) / b.f1_diag_origen
    b["caida_destino"] = (b.f1_diag_destino - b.f1) / b.f1_diag_destino
    b["perdida_abs_destino"] = b.f1_diag_destino - b.f1
    b["exactitud"] = (b.tp + b.tn) / (b.tp + b.tn + b.fp + b.fn)
    b["mcc"] = [mcc(*r) for r in zip(b.tp, b.tn, b.fp, b.fn)]
    tpr, fpr = b.tp / (b.tp + b.fn), b.fp / (b.fp + b.tn)
    pi = b.dominio_test.map(PREVALENCIA)
    b["precision_prevalencia"] = np.where(tpr > 0, tpr * pi / (tpr * pi + fpr * (1 - pi)), np.nan)

    # 3. umbral recalibrado por F1 y por Youden
    rng = np.random.default_rng(SEMILLA)
    filas = []
    for f in sorted(PRED.glob("*.csv.gz")):
        arq, semilla, origen = re.fullmatch(r"(.+)_s(\d+)_(.+)", f.name.replace(".csv.gz", "")).groups()
        d = pd.read_csv(f)
        for destino, g in d.groupby("dominio_test"):
            y = g["etiqueta"].to_numpy()
            p = np.nan_to_num(g["prob_grieta"].to_numpy(), nan=0.0)   # como en 08_umbral_auroc.py
            pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
            acum = {c: [] for c in ("f1", "youden")}
            for _ in range(SORTEOS):
                el = np.concatenate([rng.choice(pos, K // 2, replace=False), rng.choice(neg, K // 2, replace=False)])
                resto = np.setdiff1d(np.arange(len(y)), el)
                for c in acum:
                    t = umbral(y[el], p[el], c)
                    acum[c].append(metricas(y[resto], p[resto] >= t))
            fila = {"arquitectura": arq, "semilla": int(semilla), "dominio_train": origen, "dominio_test": destino}
            for c, v in acum.items():
                v = np.array(v)
                fila[f"k50_{c}_f1"], fila[f"k50_{c}_exactitud"], fila[f"k50_{c}_mcc"] = v.mean(axis=0)
            filas.append(fila)
    rec = pd.DataFrame(filas)
    b = b.merge(rec, on=["arquitectura", "semilla", "dominio_train", "dominio_test"], validate="one_to_one")
    assert len(b) == 192

    columnas = ["f1", "caida_origen", "caida_destino", "perdida_abs_destino", "exactitud", "mcc",
                "precision", "recall", "precision_prevalencia",
                "k50_f1_f1", "k50_f1_exactitud", "k50_f1_mcc", "k50_youden_f1", "k50_youden_exactitud",
                "k50_youden_mcc"]
    part = b.groupby("direccion")[columnas].mean().loc[ORDEN]
    part.round(4).to_csv(TABLAS / "ronda2_particiones.csv")
    print(part.round(3).T.to_string())

    fuera = b[b.direccion != "dentro"]
    bajo = fuera[fuera.f1 < 2 / 3 - 1e-12]
    ms = b[b.direccion == "metu_a_sdnet"]
    salida = {
        "particiones": part.round(4).to_dict(orient="index"),
        "celdas_bajo_2_3": {
            "n": int(len(bajo)), "exactitud_min_max": [float(bajo.exactitud.min()), float(bajo.exactitud.max())],
            "mcc_min_max": [float(bajo.mcc.min()), float(bajo.mcc.max())],
            "con_exactitud_<=_0.5": int((bajo.exactitud <= 0.5).sum())},
        "fuera_exactitud_>_0.5": int((fuera.exactitud > 0.5).sum()),
        "fuera_n": int(len(fuera)),
        "metu_a_sdnet_por_destino": ms.groupby("dominio_test")[
            ["f1", "precision", "recall", "precision_prevalencia", "exactitud", "mcc"]].mean().round(4)
            .to_dict(orient="index"),
        "referencia_trivial_a_prevalencia": {d: round(2 * p / (1 + p), 4) for d, p in PREVALENCIA.items()},
        "prevalencias": PREVALENCIA, "k": K, "sorteos": SORTEOS, "semilla": SEMILLA,
    }
    (TABLAS / "ronda2.json").write_text(json.dumps(salida, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in salida.items() if k != "particiones"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
