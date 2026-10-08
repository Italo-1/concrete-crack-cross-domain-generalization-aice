"""Partición agrupada por foto de origen para los tres dominios SDNET2018.

Análisis de sensibilidad de la fuga entre parches de una misma foto (README,
hallazgo 1 de F0). No sustituye a la partición original: la escribe aparte, en
`data/processed/manifiesto_agrupado.csv`, para entrenar con ella solo la
condición base.

Qué hace
--------
Trabaja sobre las MISMAS 4000 imágenes por dominio que ya están cacheadas en
`03-tehnicki-glasnik-P1/data/processed/` (no se tocan ni se duplican). Solo
reasigna la columna `particion`:

1. La foto de origen sale del nombre de archivo de SDNET2018
   (`7001-115.jpg` -> foto 7001). Todas las imágenes de una foto van a la misma
   partición.
2. Se buscan, entre `N_PERMUTACIONES` órdenes aleatorios de las fotos, el
   reparto voraz test -> val -> train que descarta menos imágenes al recortar
   test y val a 300 grieta + 300 no grieta exactos (los mismos tamaños que la
   partición original).
3. Train se queda con el resto de las fotos, se equilibra por clase y se
   iguala entre los tres dominios al mínimo común, para que el tamaño de
   entrenamiento no varíe entre dominios. Lo que sobra se marca `descartada`.

METU no trae la foto de origen en el nombre de archivo y no se puede agrupar:
sus filas se copian tal cual de la partición original.

    python src/02b_particion_agrupada.py
"""

from __future__ import annotations

import json
import ntpath
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
ORIGEN = Path(r"C:\lineaB\03-tehnicki-glasnik-P1\data\processed")
DESTINO = RAIZ / "data" / "processed"

SEMILLA = 42
N_EVAL_POR_CLASE = 300          # test y val: 300 grieta + 300 no grieta
N_PERMUTACIONES = 20_000
DOMINIOS_SDNET = ["sdnet_D", "sdnet_P", "sdnet_W"]


def foto_de_origen(ruta: str) -> str:
    return ntpath.basename(ruta).split("-")[0]


def repartir(conteos: pd.DataFrame, rng: np.random.Generator) -> tuple[list, list, int]:
    """Mejor reparto voraz de fotos a test y val entre muchas permutaciones.

    `conteos` tiene una fila por foto y columnas 0 y 1 (imágenes sin y con
    grieta). Devuelve las fotos de test, las de val y el número de imágenes que
    se descartan al recortar test y val a 300/300.
    """
    fotos = conteos.index.to_numpy()
    n0, n1 = conteos[0].to_numpy(), conteos[1].to_numpy()
    mejor = (None, None, np.inf)

    for _ in range(N_PERMUTACIONES):
        orden = rng.permutation(len(fotos))
        bloques, actual, c0, c1 = [], [], 0, 0
        for i in orden:
            if len(bloques) == 2:
                break
            actual.append(i)
            c0, c1 = c0 + n0[i], c1 + n1[i]
            if c0 >= N_EVAL_POR_CLASE and c1 >= N_EVAL_POR_CLASE:
                bloques.append((actual, c0 + c1 - 2 * N_EVAL_POR_CLASE))
                actual, c0, c1 = [], 0, 0
        if len(bloques) < 2:
            continue
        descarte = bloques[0][1] + bloques[1][1]
        if descarte < mejor[2]:
            mejor = ([fotos[i] for i in bloques[0][0]], [fotos[i] for i in bloques[1][0]], descarte)

    if mejor[0] is None:
        raise RuntimeError("Ninguna permutación reúne 300/300 en test y val.")
    return mejor


def main() -> int:
    DESTINO.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEMILLA)

    manifiesto = pd.read_csv(ORIGEN / "manifiesto.csv")
    manifiesto["particion_original"] = manifiesto["particion"]
    manifiesto["foto"] = ""
    resumen = {}

    for dominio in DOMINIOS_SDNET:
        sub = manifiesto[manifiesto["dominio"] == dominio].copy()
        sub["foto"] = sub["ruta"].map(foto_de_origen)
        conteos = sub.groupby(["foto", "etiqueta"]).size().unstack(fill_value=0)

        fotos_test, fotos_val, descarte = repartir(conteos, rng)
        sub["particion"] = "train"
        sub.loc[sub["foto"].isin(fotos_test), "particion"] = "test"
        sub.loc[sub["foto"].isin(fotos_val), "particion"] = "val"

        # Recorte exacto de test y val a 300/300; lo sobrante se descarta.
        for part in ("test", "val"):
            for clase in (0, 1):
                idx = sub.index[(sub["particion"] == part) & (sub["etiqueta"] == clase)]
                sobran = rng.choice(idx, size=len(idx) - N_EVAL_POR_CLASE, replace=False)
                sub.loc[sobran, "particion"] = "descartada"

        manifiesto.loc[sub.index, ["particion", "foto"]] = sub[["particion", "foto"]]
        resumen[dominio] = {
            "fotos": int(conteos.shape[0]),
            "fotos_test": len(fotos_test),
            "fotos_val": len(fotos_val),
            "descartadas_al_recortar_test_val": int(descarte),
        }

    # Train: equilibrio por clase e igualdad de tamaño entre los tres dominios.
    disponibles = {
        d: manifiesto[(manifiesto["dominio"] == d) & (manifiesto["particion"] == "train")]
        .groupby("etiqueta").size().min()
        for d in DOMINIOS_SDNET
    }
    n_train_por_clase = int(min(disponibles.values()))
    for dominio in DOMINIOS_SDNET:
        for clase in (0, 1):
            idx = manifiesto.index[(manifiesto["dominio"] == dominio)
                                   & (manifiesto["particion"] == "train")
                                   & (manifiesto["etiqueta"] == clase)]
            sobran = rng.choice(idx, size=len(idx) - n_train_por_clase, replace=False)
            manifiesto.loc[sobran, "particion"] = "descartada"
        resumen[dominio]["train_por_clase_disponible"] = int(disponibles[dominio])

    # Comprobación: ninguna foto aparece en dos particiones usadas.
    usadas = manifiesto[manifiesto["dominio"].isin(DOMINIOS_SDNET)
                        & (manifiesto["particion"] != "descartada")]
    cruce = usadas.groupby(["dominio", "foto"])["particion"].nunique()
    assert (cruce == 1).all(), "Una foto quedó repartida entre particiones."

    tabla = pd.crosstab([manifiesto["dominio"], manifiesto["etiqueta"]], manifiesto["particion"])
    print(tabla.to_string())

    manifiesto.to_csv(DESTINO / "manifiesto_agrupado.csv", index=False)
    resumen["n_train_por_clase_sdnet"] = n_train_por_clase
    resumen["metu"] = "sin cambios: no hay foto de origen en el nombre de archivo"
    resumen["semilla"] = SEMILLA
    resumen["n_permutaciones"] = N_PERMUTACIONES
    (DESTINO / "manifiesto_agrupado.json").write_text(
        json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(resumen, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
