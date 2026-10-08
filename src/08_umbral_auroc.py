"""Ordenacion frente a umbral: AUROC, AP y recalibracion del umbral por celda.

Control pedido por la revision adversarial (F5, ronda 1): con un umbral fijo
de 0.5, una caida de recall puede venir de que el modelo no separa las clases
del destino (AUROC bajo) o de que las ordena bien pero el umbral aprendido en
el origen no sirve (AUROC alto). Se calcula desde las probabilidades por imagen
ya guardadas; no se reentrena nada.

Por celda (arquitectura, semilla, origen, destino) y regimen (ajuste fino base
y sonda lineal):
  - AUROC y precision media (AP) de la clase grieta;
  - F1 con el umbral 0.5 (debe coincidir con resultados_matriz.csv);
  - F1 con umbral recalibrado con k imagenes etiquetadas del destino
    (k = 20, 50, 100; mitad de cada clase), elegido para maximizar F1 en esas
    k y evaluado en las 600 - k restantes; 200 sorteos con semilla 42;
  - F1 con el mejor umbral sobre todo el test (techo, no alcanzable).

Salidas: results/tables/umbral_auroc_celdas.csv, umbral_auroc_resumen.csv,
umbral_auroc.json.

    python src/08_umbral_auroc.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

RAIZ = Path(__file__).resolve().parent.parent
PRED = RAIZ / "results" / "predicciones"
TABLAS = RAIZ / "results" / "tables"
SEMILLA = 42
KS = [20, 50, 100]
SORTEOS = 200
# EXP-escala (03/10) va al final: el generador se consume en este orden, asi que
# los sorteos de los dos primeros regimenes no cambian al anadirlo.
REGIMENES = {"baseline_original": "fine-tuning", "sonda_lineal_original": "linear probe",
             "escala_original": "scale augmentation"}


def direccion(origen: str, destino: str) -> str:
    if origen == destino:
        return "dentro"
    if origen == "metu":
        return "metu_a_sdnet"
    if destino == "metu":
        return "sdnet_a_metu"
    return "entre_superficie"


def f1_umbral(y: np.ndarray, p: np.ndarray, t: float) -> float:
    pred = p >= t
    tp = int(np.sum(pred & (y == 1)))
    fp = int(np.sum(pred & (y == 0)))
    fn = int(np.sum(~pred & (y == 1)))
    return 2 * tp / (2 * tp + fp + fn) if tp else 0.0


def mejor_umbral(y: np.ndarray, p: np.ndarray) -> float:
    """Umbral que maximiza F1 en (y, p); candidatos: valores unicos de p.

    Vectorizado: con los valores unicos en orden descendente, TP y FP para
    "p >= t" son sumas acumuladas. Empates en F1: el umbral mas cercano a 0.5.
    """
    t = np.unique(p)
    orden = np.argsort(p, kind="stable")
    ys = (y[orden] == 1).astype(int)
    sufijo = np.concatenate([np.cumsum(ys[::-1])[::-1], [0]])   # positivos en ys[i:]
    i = np.searchsorted(p[orden], t, side="left")                # primer p >= t
    tp = sufijo[i]
    fp = (len(p) - i) - tp
    fn = int((y == 1).sum()) - tp
    f1 = np.where(tp > 0, 2 * tp / np.maximum(2 * tp + fp + fn, 1), 0.0)
    mejor = np.flatnonzero(f1 == f1.max())
    return float(t[mejor[np.argmin(np.abs(t[mejor] - 0.5))]])


def recalibrado(y: np.ndarray, p: np.ndarray, k: int, rng: np.random.Generator) -> float:
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    f1s = []
    for _ in range(SORTEOS):
        elegidos = np.concatenate([rng.choice(pos, k // 2, replace=False),
                                   rng.choice(neg, k // 2, replace=False)])
        resto = np.setdiff1d(np.arange(len(y)), elegidos)
        t = mejor_umbral(y[elegidos], p[elegidos])
        f1s.append(f1_umbral(y[resto], p[resto], t))
    return float(np.mean(f1s))


def main() -> int:
    rng = np.random.default_rng(SEMILLA)
    filas = []
    for carpeta, regimen in REGIMENES.items():
        for f in sorted((PRED / carpeta).glob("*.csv.gz")):
            # "mobilenetv3_small_100_s42_metu": el "_s" de "small" no es la semilla.
            arq, semilla, origen = re.fullmatch(r"(.+)_s(\d+)_(.+)",
                                                f.name.replace(".csv.gz", "")).groups()
            d = pd.read_csv(f)
            for destino, g in d.groupby("dominio_test"):
                y, p = g["etiqueta"].to_numpy(), g["prob_grieta"].to_numpy()
                # 34 de 691 200 predicciones tienen probabilidad NaN (desborde
                # de la inferencia en media precision; ver README 03/10). En la
                # matriz cuentan como "sin grieta" (pred = 0): aqui, prob = 0.
                p = np.nan_to_num(p, nan=0.0)
                fila = {"regimen": regimen, "arquitectura": arq, "semilla": int(semilla),
                        "dominio_train": origen, "dominio_test": destino,
                        "direccion": direccion(origen, destino),
                        "n_valores_unicos": int(len(np.unique(p))),
                        "auroc": roc_auc_score(y, p), "ap": average_precision_score(y, p),
                        # Prediccion guardada (argmax de los logits), que es la
                        # de la matriz; la probabilidad guardada esta redondeada
                        # en media precision y "p >= 0.5" difiere en pocos casos.
                        "f1_05": f1_umbral(y, g["pred"].to_numpy().astype(float), 0.5),
                        "discrepancias_05": int(((p >= 0.5).astype(int) != g["pred"]).sum()),
                        "f1_techo": f1_umbral(y, p, mejor_umbral(y, p))}
                for k in KS:
                    fila[f"f1_k{k}"] = recalibrado(y, p, k, rng)
                filas.append(fila)
    celdas = pd.DataFrame(filas)
    celdas.to_csv(TABLAS / "umbral_auroc_celdas.csv", index=False)

    # Control: el F1 a 0.5 desde las predicciones coincide con la matriz.
    m = pd.read_csv(TABLAS / "resultados_matriz.csv")
    m = m[(m.condicion == "baseline") & (m.particion == "original")]
    chk = celdas[celdas.regimen == "fine-tuning"].merge(
        m, on=["arquitectura", "semilla", "dominio_train", "dominio_test"])
    dif_max = float((chk.f1_05 - chk.f1).abs().max())
    assert len(chk) == 192 and dif_max < 1e-9, (len(chk), dif_max)

    columnas = ["auroc", "ap", "f1_05"] + [f"f1_k{k}" for k in KS] + ["f1_techo"]
    resumen = celdas.groupby(["regimen", "direccion"])[columnas].mean().round(3)
    resumen["celdas_auroc_bajo_0.7"] = celdas.groupby(["regimen", "direccion"]).auroc.apply(
        lambda s: int((s < 0.7).sum()))
    resumen.to_csv(TABLAS / "umbral_auroc_resumen.csv")
    print(resumen.to_string())

    # Por par origen-destino en METU -> SDNET (fine-tuning), para el texto.
    ms = celdas[(celdas.regimen == "fine-tuning") & (celdas.direccion == "metu_a_sdnet")]
    por_par = ms.groupby("dominio_test")[columnas].mean().round(3)
    print(por_par.to_string())
    info = {"control_f1_05_vs_matriz_dif_max": dif_max, "sorteos": SORTEOS, "ks": KS,
            "semilla": SEMILLA, "metu_a_sdnet_por_destino": por_par.to_dict(orient="index"),
            "valores_unicos_min": int(celdas.n_valores_unicos.min())}
    (TABLAS / "umbral_auroc.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
