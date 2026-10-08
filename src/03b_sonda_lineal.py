"""P1 (AICE) - EXP3: sonda lineal sobre caracteristicas ImageNet congeladas.

Separa lo que el ajuste fino aprende del dominio de origen de lo que la
representacion preentrenada ya trae: si la sonda cae menos que el ajuste fino
al cambiar de dominio, la perdida fuera de dominio se debe sobre todo al ajuste
fino (hipotesis H-EXP3 del README).

Diseno (fijado en el README antes de correr)
--------------------------------------------
1. Por (arquitectura, dominio) se extraen UNA vez las caracteristicas de train,
   val y test con el backbone `timm` preentrenado, `num_classes=0` (salida
   tras el pooling global, o el token de clase en ViT), en modo evaluacion y
   sin aumento de datos (transformacion de evaluacion de la condicion base).
   Se guardan en `data/processed/caracteristicas/`.
2. Por (arquitectura, semilla, origen) se entrena una cabeza `nn.Linear(d, 2)`
   sobre las caracteristicas de train del origen: AdamW (lr 1e-3, wd 1e-4),
   lote 256, hasta 100 epocas, parada temprana con paciencia 10 sobre el F1 de
   validacion del origen. La semilla fija la inicializacion de la cabeza y el
   orden de los lotes, asi que las tres semillas dan tres cabezas distintas.
3. Cada cabeza se evalua en el test de los cuatro dominios: misma matriz 4x4 y
   mismas columnas que `resultados_matriz.csv`.

La cabeza se entrena en CPU y en float32: es una capa lineal sobre <= 2800
vectores y asi no depende de cuDNN. Las caracteristicas se extraen en GPU con
precision mixta, como la evaluacion del ajuste fino.

    python src/03b_sonda_lineal.py --smoke     # 1 arquitectura, 1 semilla, salida aparte
    python src/03b_sonda_lineal.py             # 4 arq. x 3 semillas x 4 origenes
"""

from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import importlib
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
import entrenamiento as ent  # noqa: E402

exp = importlib.import_module("03_experiment")

RAIZ = exp.RAIZ
CARACTERISTICAS = RAIZ / "data" / "processed" / "caracteristicas"
LR, PESO_DECAIMIENTO, LOTE_SONDA, EPOCAS_SONDA, PACIENCIA_SONDA = 1e-3, 1e-4, 256, 100, 10


# ------------------------------------------------------ caracteristicas

@torch.no_grad()
def extraer(arquitectura: str, conjuntos: dict, log) -> dict[str, dict[str, np.ndarray]]:
    """Caracteristicas congeladas de train/val/test de los cuatro dominios (con cache)."""
    ruta = CARACTERISTICAS / f"{arquitectura}.npz"
    if ruta.exists():
        datos = np.load(ruta)
        log.info(f"[{arquitectura}] caracteristicas en cache: {ruta.name}")
        return {d: {p: datos[f"{d}__{p}"] for p in ("train", "val", "test", "y_train", "y_val", "y_test")}
                for d in exp.DOMINIOS}

    ent.fijar_semilla_determinista(42)
    import timm
    backbone = timm.create_model(arquitectura, pretrained=True, num_classes=0).to(ent.dispositivo()).eval()
    salida, plano = {}, {}
    for d in exp.DOMINIOS:
        salida[d] = {}
        for p in ("train", "val", "test"):
            # Sin aumento: transformacion de evaluacion tambien para train.
            ds = exp.DominioDataset(conjuntos[d][p].X, conjuntos[d][p].y, conjuntos[d][p].indices,
                                    exp.transform_evaluacion("baseline"))
            feats, ys = [], []
            for x, y in exp.cargador(ds):
                with torch.amp.autocast("cuda"):
                    f = backbone(x.to(ent.dispositivo()))
                feats.append(f.float().cpu().numpy())
                ys.append(y.numpy())
            salida[d][p], salida[d][f"y_{p}"] = np.concatenate(feats), np.concatenate(ys)
            plano[f"{d}__{p}"], plano[f"{d}__y_{p}"] = salida[d][p], salida[d][f"y_{p}"]
        log.info(f"[{arquitectura}] '{d}': caracteristicas {salida[d]['train'].shape[1]}-d extraidas")
    CARACTERISTICAS.mkdir(parents=True, exist_ok=True)
    np.savez(ruta, **plano)
    del backbone
    torch.cuda.empty_cache()
    return salida


# ---------------------------------------------------------------- sonda

def predecir(cabeza: nn.Module, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    with torch.no_grad():
        logits = cabeza(torch.from_numpy(X))
    return logits.argmax(1).numpy(), torch.softmax(logits, 1)[:, ent.CLASE_GRIETA].numpy()


def entrenar_sonda(Xtr, ytr, Xval, yval, semilla: int):
    ent.fijar_semilla_determinista(semilla)
    cabeza = nn.Linear(Xtr.shape[1], 2)
    opt = torch.optim.AdamW(cabeza.parameters(), lr=LR, weight_decay=PESO_DECAIMIENTO)
    criterio = nn.CrossEntropyLoss()
    cargador = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr).long()),
                          batch_size=LOTE_SONDA, shuffle=True, generator=ent.generador(semilla))
    mejor_f1, mejor_epoca, sin_mejora, mejores, historial = -1.0, -1, 0, None, []
    for epoca in range(EPOCAS_SONDA):
        cabeza.train()
        perdida_total = 0.0
        for x, y in cargador:
            opt.zero_grad(set_to_none=True)
            perdida = criterio(cabeza(x), y)
            perdida.backward()
            opt.step()
            perdida_total += perdida.item() * x.size(0)
        cabeza.eval()
        f1_val = ent.metricas(yval, predecir(cabeza, Xval)[0])["f1"]
        historial.append({"epoca": epoca, "perdida_train": perdida_total / len(ytr), "f1_val": f1_val})
        if f1_val > mejor_f1:
            mejor_f1, mejor_epoca, sin_mejora = f1_val, epoca, 0
            mejores = {k: v.clone() for k, v in cabeza.state_dict().items()}
        else:
            sin_mejora += 1
            if sin_mejora >= PACIENCIA_SONDA:
                break
    cabeza.load_state_dict(mejores)
    return cabeza.eval(), mejor_epoca, mejor_f1, historial


# ----------------------------------------------------------------- main

def main() -> int:
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("--arquitecturas", default=",".join(exp.ARQUITECTURAS))
    a.add_argument("--semillas", default=",".join(map(str, exp.SEMILLAS)))
    a.add_argument("--salida", default=str(RAIZ / "results"))
    a.add_argument("--reanudar", action="store_true", help="Salta las cabezas ya completas.")
    a.add_argument("--smoke", action="store_true", help="resnet18, semilla 42, salida results/smoke_sonda")
    args = a.parse_args()
    if args.smoke:
        args.arquitecturas, args.semillas, args.salida = "resnet18", "42", str(RAIZ / "results" / "smoke_sonda")

    salida = Path(args.salida)
    (salida / "tables").mkdir(parents=True, exist_ok=True)
    log = exp.configurar_log(salida, "03b_sonda_lineal")
    torch.set_num_threads(4)

    manifiesto = pd.read_csv(exp.MANIFIESTOS["original"])
    conjuntos = exp.construir_conjuntos(exp.cargar_arrays(), manifiesto, "baseline")
    arranque = time.perf_counter()

    for arquitectura in [x.strip() for x in args.arquitecturas.split(",") if x.strip()]:
        F = extraer(arquitectura, conjuntos, log)
        for semilla in [int(x) for x in args.semillas.split(",") if x.strip()]:
            for origen in exp.DOMINIOS:
                if args.reanudar:
                    filtro = {"arquitectura": arquitectura, "semilla": semilla, "dominio_train": origen}
                    pred = salida / "predicciones" / "sonda_lineal_original" / f"{arquitectura}_s{semilla}_{origen}.csv.gz"
                    if (exp.hecho(salida / "tables" / "resultados_sonda.csv", filtro, 4, [pred])
                            and exp.hecho(salida / "tables" / "historial_sonda.csv", filtro, 1, [])):
                        log.info(f"[{arquitectura}/sonda/s{semilla}] '{origen}' ya hecho, se salta")
                        continue
                inicio = time.perf_counter()
                cabeza, mejor_epoca, mejor_f1, historial = entrenar_sonda(
                    F[origen]["train"], F[origen]["y_train"], F[origen]["val"], F[origen]["y_val"], semilla)
                filas, predicciones = [], []
                for destino in exp.DOMINIOS:
                    pred, prob = predecir(cabeza, F[destino]["test"])
                    y = F[destino]["y_test"]
                    m = ent.metricas(y, pred)
                    filas.append({
                        "arquitectura": arquitectura, "semilla": semilla, "condicion": "sonda_lineal",
                        "particion": "original", "dominio_train": origen, "dominio_test": destino,
                        "diagonal": origen == destino,
                        **{k: m[k] for k in ("tp", "tn", "fp", "fn", "precision", "recall", "f1")},
                        "n_test": int(len(y)), "mejor_epoca": mejor_epoca,
                        "epocas_entrenadas": len(historial), "epocas_max": EPOCAS_SONDA,
                        "segundos_entrenamiento": round(time.perf_counter() - inicio, 1),
                    })
                    predicciones.append(pd.DataFrame({
                        "dominio_test": destino, "indice_en_dominio": conjuntos[destino]["idx_test"],
                        "etiqueta": y, "pred": pred, "prob_grieta": np.round(prob, 6)}))
                log.info(f"[{arquitectura}/sonda/s{semilla}] '{origen}': F1 val {mejor_f1:.4f} "
                         f"(epoca {mejor_epoca}/{len(historial)}) | "
                         + ", ".join(f"{f['dominio_test']} {f['f1']:.3f}" for f in filas))
                dir_pred = salida / "predicciones" / "sonda_lineal_original"
                dir_pred.mkdir(parents=True, exist_ok=True)
                pd.concat(predicciones).to_csv(dir_pred / f"{arquitectura}_s{semilla}_{origen}.csv.gz", index=False)
                exp.anexar(filas, salida / "tables" / "resultados_sonda.csv", exp.CLAVE)
                exp.anexar([{"arquitectura": arquitectura, "semilla": semilla, "condicion": "sonda_lineal",
                             "particion": "original", "dominio_train": origen, **h} for h in historial],
                           salida / "tables" / "historial_sonda.csv", exp.CLAVE[:5] + ["epoca"])

    log.info(f"03b_sonda_lineal.py terminado en {(time.perf_counter() - arranque) / 60:.1f} min.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
