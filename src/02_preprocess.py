"""P1 - preprocesado: cuatro dominios equiparados para la matriz de transferencia.

Los cuatro dominios son los tres subconjuntos de SDNET2018 -tableros de puente,
pavimentos y muros, cada uno con sus propias condiciones de captura- mas METU.
Eso separa dos preguntas que la ficha original mezclaba: transferencia entre
SUPERFICIES dentro de la misma campana, y transferencia entre CAMPANAS distintas.

Este script deja los cuatro dominios en igualdad de condiciones. Sin eso, la
matriz no medira solo cambio de dominio:

  1. **Tamano.** SDNET-D aporta 13 620 imagenes y METU 40 000. Entrenar con
     conjuntos de tamanos distintos contamina la comparacion con el efecto del
     tamano de muestra.
  2. **Prior de clase.** SDNET va del 10.7 % al 21.2 % de imagenes con fisura;
     METU esta exactamente al 50 %. Un modelo entrenado en METU y evaluado en
     SDNET sufre un cambio de prior que NO es cambio de dominio, y al reves
     igual. Es el confuso mas serio de este articulo y se elimina aqui.
  3. **Solape.** Si la misma imagen aparece en dos dominios, esa celda mide
     memorizacion en lugar de generalizacion. Se comprueba por hash.

    python src/02_preprocess.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from fieutils.config import SEMILLA, configurar_log, fijar_semilla
from fieutils.vision import cachear_imagenes, hash_archivos

RAIZ = Path(__file__).resolve().parent.parent
CRUDO = RAIZ / "data" / "raw"
PROCESADO = RAIZ / "data" / "processed"

TAMANO = 224
# Proporciones de la division dentro de cada dominio. El test de cada dominio es
# lo que se usa para evaluar los modelos entrenados en los otros tres.
PROPORCIONES = {"train": 0.70, "val": 0.15, "test": 0.15}

DOMINIOS = {
    "sdnet_D": {
        "raiz": CRUDO / "sdnet2018" / "D",
        "fisura": "CD", "sana": "UD",
        "descripcion": "SDNET2018, tableros de puente",
    },
    "sdnet_P": {
        "raiz": CRUDO / "sdnet2018" / "P",
        "fisura": "CP", "sana": "UP",
        "descripcion": "SDNET2018, pavimentos",
    },
    "sdnet_W": {
        "raiz": CRUDO / "sdnet2018" / "W",
        "fisura": "CW", "sana": "UW",
        "descripcion": "SDNET2018, muros",
    },
    "metu": {
        "raiz": CRUDO / "metu",
        "fisura": "Positive", "sana": "Negative",
        "descripcion": "METU, edificios del campus",
    },
}


def indexar(log) -> pd.DataFrame:
    """Lista todas las imagenes con su dominio y su clase."""
    filas = []
    for nombre, cfg in DOMINIOS.items():
        for etiqueta, carpeta in ((1, cfg["fisura"]), (0, cfg["sana"])):
            ruta = cfg["raiz"] / carpeta
            archivos = sorted(p for p in ruta.iterdir() if p.suffix.lower() == ".jpg")
            filas += [
                {"dominio": nombre, "etiqueta": etiqueta, "ruta": str(p)}
                for p in archivos
            ]
        del etiqueta

    df = pd.DataFrame(filas)
    resumen = df.pivot_table(index="dominio", columns="etiqueta",
                             values="ruta", aggfunc="count")
    resumen.columns = ["sana", "fisura"]
    resumen["total"] = resumen.sum(axis=1)
    resumen["% fisura"] = (100 * resumen["fisura"] / resumen["total"]).round(1)

    log.info("Inventario crudo:\n" + resumen.to_string())
    return df


def submuestrear(df: pd.DataFrame, n_por_clase: int, log) -> pd.DataFrame:
    """Toma `n_por_clase` imagenes de cada clase en cada dominio.

    El resultado tiene el mismo tamano y el mismo balance de clases en los
    cuatro dominios, que es la condicion para que las 16 celdas de la matriz
    sean comparables entre si.
    """
    rng = np.random.default_rng(SEMILLA)
    trozos = []

    for (dominio, etiqueta), grupo in df.groupby(["dominio", "etiqueta"]):
        if len(grupo) < n_por_clase:
            raise ValueError(
                f"{dominio} clase {etiqueta} solo tiene {len(grupo)} imagenes, "
                f"se piden {n_por_clase}"
            )
        elegidas = rng.choice(grupo.index.values, size=n_por_clase, replace=False)
        trozos.append(df.loc[elegidas])

    salida = pd.concat(trozos).reset_index(drop=True)
    log.info(f"Submuestreo: {n_por_clase} por clase y dominio "
             f"-> {len(salida)} imagenes ({n_por_clase * 2} por dominio)")
    return salida


def dividir(df: pd.DataFrame, log) -> pd.DataFrame:
    """Division estratificada train/val/test dentro de cada dominio y clase."""
    rng = np.random.default_rng(SEMILLA)
    df = df.copy()
    df["particion"] = ""

    for (_dominio, _etiqueta), grupo in df.groupby(["dominio", "etiqueta"]):
        indices = grupo.index.values.copy()
        rng.shuffle(indices)
        n = len(indices)
        n_train = int(round(n * PROPORCIONES["train"]))
        n_val = int(round(n * PROPORCIONES["val"]))

        df.loc[indices[:n_train], "particion"] = "train"
        df.loc[indices[n_train:n_train + n_val], "particion"] = "val"
        df.loc[indices[n_train + n_val:], "particion"] = "test"

    log.info("Particiones:\n" + pd.crosstab(df["dominio"], df["particion"]).to_string())
    return df


def deduplicar(df: pd.DataFrame, log) -> tuple[pd.DataFrame, dict]:
    """Elimina imagenes repetidas y comprueba el solape entre dominios.

    Se hace ANTES de submuestrear y dividir. Un duplicado exacto que caiga uno
    en entrenamiento y otro en prueba es fuga: el modelo ve en la evaluacion una
    imagen que ya memorizo. Aqui es poco frecuente, pero un articulo que critica
    la evaluacion laxa del campo no puede permitirse dejarla.

    Optimizacion: dos archivos identicos tienen forzosamente el mismo tamano, de
    modo que solo hace falta calcular el hash de aquellos cuyo tamano se repite.
    Eso reduce el trabajo de casi cien mil archivos a unos pocos miles.
    """
    df = df.copy()
    df["bytes"] = [Path(r).stat().st_size for r in df["ruta"]]

    tamanos_repetidos = df["bytes"].duplicated(keep=False)
    candidatos = df[tamanos_repetidos]
    log.info(f"Deduplicacion: {len(candidatos)} de {len(df)} archivos comparten "
             f"tamano con otro; solo esos necesitan hash")

    df["hash"] = None
    if not candidatos.empty:
        df.loc[candidatos.index, "hash"] = hash_archivos(
            candidatos["ruta"].tolist(), log=log
        )

    con_hash = df[df["hash"].notna()]
    repetidos = con_hash[con_hash.duplicated("hash", keep=False)]

    entre_dominios, detalle = 0, []
    for h, grupo in repetidos.groupby("hash"):
        dominios = sorted(set(grupo["dominio"]))
        if len(dominios) > 1:
            entre_dominios += len(grupo)
            detalle.append({"dominios": dominios, "n": int(len(grupo))})

    # Se conserva una copia de cada imagen dentro de cada dominio.
    antes = len(df)
    df = df[~df.duplicated(subset=["dominio", "hash"], keep="first")
            | df["hash"].isna()].reset_index(drop=True)
    eliminadas = antes - len(df)

    log.info(f"Duplicados intra-dominio eliminados: {eliminadas}")
    log.info(f"Duplicados ENTRE dominios: {entre_dominios}")
    if entre_dominios:
        log.info("ATENCION: hay imagenes compartidas entre dominios. Esas celdas "
                 "medirian memorizacion, no generalizacion.")
        for d in detalle[:10]:
            log.info(f"  {d['dominios']} x{d['n']}")

    resumen = {
        "duplicados_intra_dominio_eliminados": int(eliminadas),
        "duplicados_entre_dominios": int(entre_dominios),
        "detalle_entre_dominios": detalle[:50],
    }
    return df.drop(columns=["bytes", "hash"]), resumen


def main() -> int:
    PROCESADO.mkdir(parents=True, exist_ok=True)
    log = configurar_log("02_preprocess", RAIZ)
    fijar_semilla(SEMILLA)

    df = indexar(log)
    df, solape = deduplicar(df, log)

    # El dominio mas pobre fija el tamano: SDNET-D tiene 2025 imagenes con
    # fisura, asi que ese es el techo si se quiere igualdad entre los cuatro.
    minimo = int(df.groupby(["dominio", "etiqueta"]).size().min())
    n_por_clase = min(2000, minimo)
    log.info(f"Clase mas pequena tras deduplicar: {minimo} imagenes "
             f"-> se toman {n_por_clase} por clase y dominio")

    seleccion = submuestrear(df, n_por_clase, log)
    seleccion = dividir(seleccion, log)

    # Cache por dominio, en el orden del manifiesto para que las filas casen.
    seleccion = seleccion.sort_values(["dominio", "particion"]).reset_index(drop=True)
    seleccion["indice_en_dominio"] = seleccion.groupby("dominio").cumcount()

    for dominio in DOMINIOS:
        sub = seleccion[seleccion["dominio"] == dominio]
        log.info(f"[{dominio}] cacheando {len(sub)} imagenes a {TAMANO}x{TAMANO}")
        cachear_imagenes(sub["ruta"].tolist(), PROCESADO / f"{dominio}_X.npy",
                         tamano=TAMANO, log=log)
        np.save(PROCESADO / f"{dominio}_y.npy", sub["etiqueta"].values.astype(np.int64))

    seleccion.to_csv(PROCESADO / "manifiesto.csv", index=False)

    manifiesto = {
        "semilla": SEMILLA,
        "tamano_imagen": TAMANO,
        "n_por_clase_por_dominio": n_por_clase,
        "n_por_dominio": n_por_clase * 2,
        "proporciones": PROPORCIONES,
        "dominios": {k: v["descripcion"] for k, v in DOMINIOS.items()},
        "balance_de_clases": "50 % en los cuatro dominios, por construccion",
        "solape": solape,
        "nota_diseno": (
            "Tamano y prior de clase igualados entre dominios para que la "
            "matriz mida cambio de dominio y no tamano de muestra ni cambio "
            "de prior."
        ),
    }
    (PROCESADO / "manifiesto.json").write_text(
        json.dumps(manifiesto, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    log.info(f"Listo. {len(seleccion)} imagenes en 4 dominios de "
             f"{n_por_clase * 2} cada uno, balanceados al 50 %.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
