"""P1 (AICE) - EXP4: entrenamiento con tres dominios, evaluacion en el cuarto.

Responde a la pregunta practica de un operador con imagenes de varios sitios:
entrenar con tres fuentes, ¿transfiere mejor a la cuarta que entrenar con una?
(hipotesis H-EXP4 del README).

Diseno (fijado en el README antes de correr)
--------------------------------------------
- Para cada dominio excluido, el train combinado tiene **2800 imagenes**, el
  mismo tamano que un entrenamiento de una sola fuente, para que el efecto de la
  diversidad no se confunda con el de la cantidad: 934/933/933 por fuente
  (en el orden de `DOMINIOS`), con 1400 grieta + 1400 no grieta en total y
  cada fuente equilibrada a +-1 imagen. Val combinado: 600 (100 + 100 por
  fuente).
- Las imagenes se sortean UNA vez con la semilla 42 dentro de las particiones
  train/val ya definidas en `manifiesto.csv` (particion original); las tres
  semillas de entrenamiento comparten ese sorteo, igual que en E1 comparten la
  particion.
- Condicion base, mismos hiperparametros, lote y determinismo que
  `03_experiment.py`. Se evalua en el test de los cuatro dominios; la celda
  principal es la del dominio excluido (`excluido = True`).

    python src/03c_multifuente.py --smoke     # resnet18, 1 semilla, 1 epoca, salida aparte
    python src/03c_multifuente.py             # 4 excluidos x 4 arq. x 3 semillas = 48
"""

from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import importlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import ConcatDataset

from fieutils import vision

sys.path.insert(0, str(Path(__file__).resolve().parent))
import entrenamiento as ent  # noqa: E402

exp = importlib.import_module("03_experiment")

RAIZ = exp.RAIZ
SEMILLA_SORTEO = 42
N_TRAIN, N_VAL = 2800, 600


def reparto(n_total: int, n_fuentes: int = 3) -> list[tuple[int, int]]:
    """(no grieta, grieta) por fuente, sumando n_total/2 por clase.

    Para 2800 (1400 por clase = 3 x 466 + 2): la no grieta lleva el +1 en las
    fuentes 0 y 1, la grieta en las fuentes 0 y 2 -> (467, 467), (467, 466),
    (466, 467), es decir 934/933/933 imagenes. Para 600: (100, 100) x 3.
    """
    por_clase = n_total // 2
    base, resto = divmod(por_clase, n_fuentes)
    extra_no_grieta = set(range(resto))
    extra_grieta = {0} | set(range(n_fuentes - resto + 1, n_fuentes)) if resto else set()
    pares = [(base + (i in extra_no_grieta), base + (i in extra_grieta)) for i in range(n_fuentes)]
    assert sum(p[0] for p in pares) == sum(p[1] for p in pares) == por_clase
    return pares


def sortear(manifiesto: pd.DataFrame, fuentes: list[str], particion: str, n_total: int,
            rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Indices (dentro del array de cada fuente) del subconjunto combinado."""
    salida = {}
    for fuente, (n0, n1) in zip(fuentes, reparto(n_total, len(fuentes))):
        sub = manifiesto[(manifiesto["dominio"] == fuente) & (manifiesto["particion"] == particion)]
        elegidos = [rng.choice(sub.loc[sub["etiqueta"] == c, "indice_en_dominio"].to_numpy(), size=n, replace=False)
                    for c, n in ((0, n0), (1, n1))]
        salida[fuente] = np.sort(np.concatenate(elegidos))
    return salida


def main() -> int:
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("--arquitecturas", default=",".join(exp.ARQUITECTURAS))
    a.add_argument("--semillas", default=",".join(map(str, exp.SEMILLAS)))
    a.add_argument("--excluidos", default=",".join(exp.DOMINIOS))
    a.add_argument("--epocas", type=int, default=exp.EPOCAS)
    a.add_argument("--salida", default=str(RAIZ / "results"))
    a.add_argument("--reanudar", action="store_true", help="Salta los modelos ya completos.")
    a.add_argument("--smoke", action="store_true")
    args = a.parse_args()
    if args.smoke:
        args.arquitecturas, args.semillas, args.excluidos, args.epocas = "resnet18", "42", "metu", 1
        args.salida = str(RAIZ / "results" / "smoke_multifuente")

    salida = Path(args.salida)
    (salida / "tables").mkdir(parents=True, exist_ok=True)
    log = exp.configurar_log(salida, "03c_multifuente")

    manifiesto = pd.read_csv(exp.MANIFIESTOS["original"])
    arrays = exp.cargar_arrays()
    conjuntos = exp.construir_conjuntos(arrays, manifiesto, "baseline")
    t_train, t_eval = exp.transform_entrenamiento("baseline"), exp.transform_evaluacion("baseline")

    # Sorteo unico, independiente de la arquitectura y la semilla de entrenamiento.
    rng = np.random.default_rng(SEMILLA_SORTEO)
    sorteos = {}
    for excluido in exp.DOMINIOS:
        fuentes = [d for d in exp.DOMINIOS if d != excluido]
        sorteos[excluido] = {"fuentes": fuentes,
                             "train": sortear(manifiesto, fuentes, "train", N_TRAIN, rng),
                             "val": sortear(manifiesto, fuentes, "val", N_VAL, rng)}
    (salida / "tables" / "multifuente_sorteo.json").write_text(json.dumps(
        {e: {p: {f: len(v) for f, v in s[p].items()} for p in ("train", "val")} for e, s in sorteos.items()},
        indent=2), encoding="utf-8")

    arranque = time.perf_counter()
    for arquitectura in [x.strip() for x in args.arquitecturas.split(",") if x.strip()]:
        for semilla in [int(x) for x in args.semillas.split(",") if x.strip()]:
            for excluido in [x.strip() for x in args.excluidos.split(",") if x.strip()]:
                if args.reanudar:
                    filtro = {"arquitectura": arquitectura, "semilla": semilla, "dominio_excluido": excluido}
                    pred = salida / "predicciones" / "multifuente_original" / f"{arquitectura}_s{semilla}_sin_{excluido}.csv.gz"
                    if (exp.hecho(salida / "tables" / "resultados_multifuente.csv", filtro, 4, [pred])
                            and exp.hecho(salida / "tables" / "historial_multifuente.csv", filtro, 1, [])):
                        log.info(f"[{arquitectura}/multifuente/s{semilla}] sin '{excluido}' ya hecho, se salta")
                        continue
                s = sorteos[excluido]
                train = ConcatDataset([exp.DominioDataset(*arrays[f], s["train"][f], t_train) for f in s["fuentes"]])
                val = ConcatDataset([exp.DominioDataset(*arrays[f], s["val"][f], t_eval) for f in s["fuentes"]])
                ent.fijar_semilla_determinista(semilla)
                modelo = vision.crear_modelo(arquitectura, n_clases=2)
                log.info(f"[{arquitectura}/multifuente/s{semilla}] excluido '{excluido}', "
                         f"fuentes {s['fuentes']}, train {len(train)}, val {len(val)}")
                res = ent.entrenar(modelo, exp.cargador(train, semilla), exp.cargador(val),
                                   epocas=args.epocas, log=log)

                filas, predicciones = [], []
                for destino in exp.DOMINIOS:
                    y, pred, prob = ent.evaluar_detallado(res.modelo, exp.cargador(conjuntos[destino]["test"]))
                    m = ent.metricas(y, pred)
                    filas.append({
                        "arquitectura": arquitectura, "semilla": semilla, "condicion": "multifuente",
                        "particion": "original", "dominio_excluido": excluido,
                        "dominio_train": "+".join(s["fuentes"]), "dominio_test": destino,
                        "excluido": destino == excluido,
                        **{k: m[k] for k in ("tp", "tn", "fp", "fn", "precision", "recall", "f1")},
                        "n_test": int(len(y)), "mejor_epoca": res.mejor_epoca,
                        "epocas_entrenadas": res.epocas_entrenadas, "epocas_max": args.epocas,
                        "segundos_entrenamiento": round(res.segundos, 1),
                    })
                    predicciones.append(pd.DataFrame({
                        "dominio_test": destino, "indice_en_dominio": conjuntos[destino]["idx_test"],
                        "etiqueta": y, "pred": pred, "prob_grieta": np.round(prob, 6)}))
                log.info(f"    F1: " + ", ".join(
                    f"{f['dominio_test']}{'*' if f['excluido'] else ''} {f['f1']:.3f}" for f in filas))

                dir_pred = salida / "predicciones" / "multifuente_original"
                dir_pred.mkdir(parents=True, exist_ok=True)
                pd.concat(predicciones).to_csv(dir_pred / f"{arquitectura}_s{semilla}_sin_{excluido}.csv.gz",
                                               index=False)
                clave = ["arquitectura", "semilla", "dominio_excluido", "dominio_test"]
                exp.anexar(filas, salida / "tables" / "resultados_multifuente.csv", clave)
                exp.anexar([{"arquitectura": arquitectura, "semilla": semilla, "dominio_excluido": excluido, **h}
                            for h in res.historial],
                           salida / "tables" / "historial_multifuente.csv", clave[:3] + ["epoca"])
                del modelo, res
                torch.cuda.empty_cache()

    log.info(f"03c_multifuente.py terminado en {(time.perf_counter() - arranque) / 60:.1f} min.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
