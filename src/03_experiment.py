"""P1 (AICE) - matriz de transferencia 4x4 con conteos, predicciones y checkpoints.

Entrena y evalua. No resume, no interpreta y no dibuja: escribe los resultados
crudos y ahi termina. El analisis vive en `04_stats.py` y las figuras en
`05_figures.py`.

Diseno (igual que la version de Tehnicki glasnik)
--------------------------------------------------
4 arquitecturas x 3 semillas x 4 dominios de origen = 48 entrenamientos por
condicion; cada modelo se evalua en el test de los 4 dominios (16 celdas por
arquitectura y semilla). Tres condiciones: `baseline`, `agresivo` (aumento de
datos solo en train) y `ecualizado` (ecualizacion de histograma en train, val
y test). Ver README de la version anterior, bitacora 25/08, para el defecto que
tuvo la primera version de `ecualizado` y su correccion.

Que cambia respecto a la version anterior (README, E0)
------------------------------------------------------
- Por celda se guardan TP, TN, FP, FN, precision, recall y F1, y por imagen la
  probabilidad y la prediccion (EXP2).
- Entrenamiento determinista (`entrenamiento.py`) y lote fijo de 64.
- Los pesos de la mejor epoca de la condicion base se guardan en
  `checkpoints/` (fuera de git).
- `--particion agrupada` usa `data/processed/manifiesto_agrupado.csv`
  (`02b_particion_agrupada.py`): misma imagenes, partido por foto de origen en
  SDNET2018. Es el analisis de sensibilidad de la fuga entre parches.
- Las filas se anexan al terminar CADA modelo (no cada matriz), asi que un
  corte a mitad de una matriz solo pierde el modelo en curso.

Datos: se leen por ruta de `03-tehnicki-glasnik-P1/data/processed/`, que no se
modifica.

    python src/03_experiment.py --smoke                       # 1 epoca, 4 arq., SDNET-D, salida aparte
    python src/03_experiment.py --condicion baseline          # matriz base
    python src/03_experiment.py --condicion todo              # las tres condiciones
    python src/03_experiment.py --condicion baseline --particion agrupada
"""

from __future__ import annotations

import os

# Antes de importar torch: sin esto cuBLAS no es determinista.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import json
import logging
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from fieutils import vision

sys.path.insert(0, str(Path(__file__).resolve().parent))
import entrenamiento as ent  # noqa: E402

RAIZ = Path(__file__).resolve().parent.parent
DATOS = Path(r"C:\lineaB\03-tehnicki-glasnik-P1\data\processed")
MANIFIESTOS = {
    "original": DATOS / "manifiesto.csv",
    "agrupada": RAIZ / "data" / "processed" / "manifiesto_agrupado.csv",
}
CHECKPOINTS = RAIZ / "checkpoints"

DOMINIOS = ["sdnet_D", "sdnet_P", "sdnet_W", "metu"]
ARQUITECTURAS = ["resnet18", "efficientnet_b0", "mobilenetv3_small_100", "vit_tiny_patch16_224"]
SEMILLAS = [42, 43, 44]
CONDICIONES = ["baseline", "agresivo", "ecualizado"]
# `escala` (EXP-escala, F5, protocolo en el README del 03/10) no entra en
# `--condicion todo`, que sigue siendo las tres condiciones originales.
CONDICIONES_EXTRA = ["escala"]
ESCALA_MIN, ESCALA_MAX = 0.5, 1.5
EPOCAS = 12
TAMANO = 224
CLAVE = ["arquitectura", "semilla", "condicion", "particion", "dominio_train", "dominio_test"]


# ------------------------------------------------------------------- dataset

class DominioDataset(Dataset):
    """Vista sobre el array cacheado (N, 224, 224, 3) uint8 de un dominio.

    Guarda los indices en vez de copiar `X[indices]`: los cuatro arrays ya
    ocupan 2,4 GB y se comparten entre arquitecturas y condiciones.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, indices: np.ndarray, transform):
        self.X, self.y, self.indices, self.transform = X, y, indices, transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i):
        j = self.indices[i]
        return self.transform(Image.fromarray(self.X[j])), int(self.y[j])


def _transform_ecualizado(entrenamiento: bool, tamano: int = TAMANO):
    """Ecualizacion de histograma en TODAS las particiones (version corregida
    del 25/08 en la version anterior; ver su README)."""
    pasos = [T.Resize((tamano, tamano)), T.Lambda(ImageOps.equalize)]
    if entrenamiento:
        pasos.append(T.RandomHorizontalFlip())
    pasos += [T.ToTensor(), T.Normalize(vision.MEDIA_IMAGENET, vision.DESV_IMAGENET)]
    return T.Compose(pasos)


class EscalaAleatoria:
    """Escala aleatoria s ~ U(ESCALA_MIN, ESCALA_MAX) sobre una imagen PIL cuadrada.

    s < 1: reduce y rellena por reflejo hasta el tamano original, con el
    contenido en posicion aleatoria (sin bordes negros: serian un atajo).
    s > 1: amplia y recorta en posicion aleatoria. Usa el generador global de
    torch, que `entrenamiento.fijar_semilla_determinista` fija, para no romper
    el determinismo.
    """

    def __init__(self, minimo: float = ESCALA_MIN, maximo: float = ESCALA_MAX):
        self.minimo, self.maximo = minimo, maximo

    def __call__(self, img: Image.Image) -> Image.Image:
        lado = img.size[0]
        s = self.minimo + (self.maximo - self.minimo) * torch.rand(1).item()
        nuevo = max(1, round(lado * s))
        img = img.resize((nuevo, nuevo), Image.BILINEAR)
        if nuevo < lado:
            hueco = lado - nuevo
            izq = int(torch.randint(0, hueco + 1, (1,)).item())
            arr = int(torch.randint(0, hueco + 1, (1,)).item())
            a = np.pad(np.asarray(img), ((arr, hueco - arr), (izq, hueco - izq), (0, 0)), mode="reflect")
            return Image.fromarray(a)
        if nuevo > lado:
            x = int(torch.randint(0, nuevo - lado + 1, (1,)).item())
            y = int(torch.randint(0, nuevo - lado + 1, (1,)).item())
            return img.crop((x, y, x + lado, y + lado))
        return img


def _transform_escala(tamano: int = TAMANO):
    return T.Compose([T.Resize((tamano, tamano)), EscalaAleatoria(), T.RandomHorizontalFlip(),
                      T.ToTensor(), T.Normalize(vision.MEDIA_IMAGENET, vision.DESV_IMAGENET)])


def transform_entrenamiento(condicion: str):
    if condicion == "ecualizado":
        return _transform_ecualizado(entrenamiento=True)
    if condicion == "escala":
        return _transform_escala()
    return vision.transformaciones(entrenamiento=True, tamano=TAMANO, agresivo=condicion == "agresivo")


def transform_evaluacion(condicion: str):
    if condicion == "ecualizado":
        return _transform_ecualizado(entrenamiento=False)
    return vision.transformaciones(entrenamiento=False, tamano=TAMANO)


def indices_particion(manifiesto: pd.DataFrame, dominio: str, particion: str) -> np.ndarray:
    sub = manifiesto[(manifiesto["dominio"] == dominio) & (manifiesto["particion"] == particion)]
    if sub.empty:
        raise ValueError(f"Sin filas de particion '{particion}' para el dominio '{dominio}'.")
    return np.sort(sub["indice_en_dominio"].to_numpy())


def cargar_arrays() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    return {d: (np.load(DATOS / f"{d}_X.npy"), np.load(DATOS / f"{d}_y.npy")) for d in DOMINIOS}


def construir_conjuntos(arrays, manifiesto: pd.DataFrame, condicion: str) -> dict[str, dict]:
    t_train, t_eval = transform_entrenamiento(condicion), transform_evaluacion(condicion)
    conjuntos = {}
    for d, (X, y) in arrays.items():
        idx = {p: indices_particion(manifiesto, d, p) for p in ("train", "val", "test")}
        conjuntos[d] = {
            "train": DominioDataset(X, y, idx["train"], t_train),
            "val": DominioDataset(X, y, idx["val"], t_eval),
            "test": DominioDataset(X, y, idx["test"], t_eval),
            "idx_test": idx["test"],
        }
    return conjuntos


def cargador(ds: Dataset, semilla: int | None = None) -> DataLoader:
    """`num_workers=0`: las transformaciones con `T.Lambda` no se pueden
    serializar bajo el `spawn` de Windows, y con las imagenes ya en memoria el
    cuello de botella es la GPU. Con semilla -> barajado reproducible."""
    if semilla is None:
        return DataLoader(ds, batch_size=ent.LOTE, shuffle=False, num_workers=0, pin_memory=True)
    return DataLoader(ds, batch_size=ent.LOTE, shuffle=True, num_workers=0, pin_memory=True,
                      generator=ent.generador(semilla))


# --------------------------------------------------------------------- salida

def anexar(filas: list[dict], ruta: Path, clave: list[str], intentos: int = 8) -> int:
    """Escribe o amplia un CSV; las filas con la misma clave se reemplazan.

    Escritura atomica (archivo temporal + `os.replace`) con reintentos. El
    02/10, E1 y E2 murieron con `OSError: [Errno 22] Invalid argument` al
    reabrir un CSV de historial para escribirlo: en Windows es un bloqueo
    transitorio del archivo (antivirus o indexador). Un reintento con espera
    lo resuelve; el temporal evita dejar un CSV a medio escribir.
    """
    nuevo = pd.DataFrame(filas)
    for intento in range(intentos):
        try:
            if ruta.exists():
                combinado = pd.concat([pd.read_csv(ruta), nuevo], ignore_index=True)
                combinado = combinado.drop_duplicates(subset=clave, keep="last")
            else:
                combinado = nuevo
            temporal = ruta.with_name(f".{ruta.name}.{os.getpid()}.tmp")
            combinado.to_csv(temporal, index=False)
            os.replace(temporal, ruta)
            return len(combinado)
        except OSError:
            if intento == intentos - 1:
                raise
            time.sleep(2 * (intento + 1))
    return 0


def hecho(ruta_csv: Path, filtro: dict, n_filas: int, extras: list[Path]) -> bool:
    """True si el CSV ya tiene `n_filas` con esa clave y existen los archivos
    extra (predicciones, checkpoint). Para `--reanudar`."""
    if not ruta_csv.exists() or not all(p.exists() for p in extras):
        return False
    df = pd.read_csv(ruta_csv)
    for k, v in filtro.items():
        df = df[df[k] == v]
    return len(df) >= n_filas


def configurar_log(salida: Path, nombre: str) -> logging.Logger:
    (salida / "logs").mkdir(parents=True, exist_ok=True)
    log = logging.getLogger(nombre)
    log.setLevel(logging.INFO)
    log.handlers.clear()
    formato = logging.Formatter("%(asctime)s | %(message)s", "%H:%M:%S")
    for h in (logging.FileHandler(salida / "logs" / f"{nombre}.log", encoding="utf-8"),
              logging.StreamHandler()):
        h.setFormatter(formato)
        log.addHandler(h)
    return log


def registrar_entorno(salida: Path, args) -> None:
    import timm
    import torchvision
    entorno = {
        "fecha": time.strftime("%Y-%m-%d %H:%M:%S"),
        "argumentos": vars(args),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "timm": timm.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(0),
        "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "lote": ent.LOTE,
    }
    (salida / "logs" / f"entorno_03_experiment_{time.strftime('%Y%m%d_%H%M%S')}.json").write_text(
        json.dumps(entorno, indent=2, ensure_ascii=False), encoding="utf-8")


# ----------------------------------------------------------------------- main

def entrenar_origen(conjuntos, origen, arquitectura, semilla, condicion, particion,
                    epocas, salida, guardar_pesos, log) -> None:
    ent.fijar_semilla_determinista(semilla)
    modelo = vision.crear_modelo(arquitectura, n_clases=2)
    res = ent.entrenar(modelo, cargador(conjuntos[origen]["train"], semilla),
                       cargador(conjuntos[origen]["val"]), epocas=epocas, log=log)
    log.info(f"[{arquitectura}/{condicion}/{particion}/s{semilla}] '{origen}': mejor F1 val "
             f"{res.mejor_f1_val:.4f} en la epoca {res.mejor_epoca}, "
             f"{res.epocas_entrenadas} epocas, {res.segundos / 60:.1f} min")

    filas, predicciones = [], []
    for destino in DOMINIOS:
        y, pred, prob = ent.evaluar_detallado(res.modelo, cargador(conjuntos[destino]["test"]))
        m = ent.metricas(y, pred)
        filas.append({
            "arquitectura": arquitectura, "semilla": semilla, "condicion": condicion,
            "particion": particion, "dominio_train": origen, "dominio_test": destino,
            "diagonal": origen == destino,
            **{k: m[k] for k in ("tp", "tn", "fp", "fn", "precision", "recall", "f1")},
            "n_test": int(len(y)), "mejor_epoca": res.mejor_epoca,
            "epocas_entrenadas": res.epocas_entrenadas, "epocas_max": epocas,
            "segundos_entrenamiento": round(res.segundos, 1),
        })
        predicciones.append(pd.DataFrame({
            "dominio_test": destino, "indice_en_dominio": conjuntos[destino]["idx_test"],
            "etiqueta": y, "pred": pred, "prob_grieta": np.round(prob, 6),
        }))
    log.info("    F1 por destino: " + ", ".join(f"{f['dominio_test']} {f['f1']:.3f}" for f in filas))

    dir_pred = salida / "predicciones" / f"{condicion}_{particion}"
    dir_pred.mkdir(parents=True, exist_ok=True)
    pd.concat(predicciones).to_csv(dir_pred / f"{arquitectura}_s{semilla}_{origen}.csv.gz", index=False)

    anexar(filas, salida / "tables" / "resultados_matriz.csv", CLAVE)
    hist = [{"arquitectura": arquitectura, "semilla": semilla, "condicion": condicion,
             "particion": particion, "dominio_train": origen, **h} for h in res.historial]
    anexar(hist, salida / "tables" / "historial_entrenamiento.csv", CLAVE[:5] + ["epoca"])

    if guardar_pesos:
        dir_ck = CHECKPOINTS / f"{condicion}_{particion}"
        dir_ck.mkdir(parents=True, exist_ok=True)
        torch.save(res.mejores_pesos, dir_ck / f"{arquitectura}_s{semilla}_{origen}.pt")

    del modelo, res
    torch.cuda.empty_cache()


def main() -> int:
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("--condicion", choices=CONDICIONES + CONDICIONES_EXTRA + ["todo"], default="baseline")
    a.add_argument("--particion", choices=list(MANIFIESTOS), default="original")
    a.add_argument("--arquitecturas", default=",".join(ARQUITECTURAS))
    a.add_argument("--semillas", default=",".join(map(str, SEMILLAS)))
    a.add_argument("--origenes", default=",".join(DOMINIOS),
                   help="Dominios de origen a entrenar (por defecto los cuatro).")
    a.add_argument("--epocas", type=int, default=EPOCAS)
    a.add_argument("--salida", default=str(RAIZ / "results"),
                   help="Carpeta de resultados (tables/, predicciones/, logs/).")
    a.add_argument("--checkpoints", choices=["auto", "si", "no"], default="auto",
                   help="auto = solo condicion baseline con particion original.")
    a.add_argument("--reanudar", action="store_true",
                   help="Salta los modelos que ya tienen sus 4 filas de resultados, su historial, "
                        "sus predicciones y, si corresponde, su checkpoint.")
    a.add_argument("--smoke", action="store_true",
                   help="1 epoca, 4 arquitecturas, semilla 42, origen SDNET-D, "
                        "salida en results/smoke, sin checkpoints.")
    args = a.parse_args()

    if args.smoke:
        args.condicion, args.semillas, args.origenes, args.epocas = "baseline", "42", "sdnet_D", 1
        args.salida, args.checkpoints = str(RAIZ / "results" / "smoke"), "no"

    salida = Path(args.salida)
    (salida / "tables").mkdir(parents=True, exist_ok=True)
    log = configurar_log(salida, "03_experiment")
    registrar_entorno(salida, args)

    arquitecturas = [x.strip() for x in args.arquitecturas.split(",") if x.strip()]
    semillas = [int(x) for x in args.semillas.split(",") if x.strip()]
    origenes = [x.strip() for x in args.origenes.split(",") if x.strip()]
    condiciones = CONDICIONES if args.condicion == "todo" else [args.condicion]

    manifiesto = pd.read_csv(MANIFIESTOS[args.particion])
    arrays = cargar_arrays()
    arranque = time.perf_counter()
    log.info(f"Particion '{args.particion}', condiciones {condiciones}, {len(arquitecturas)} arq. x "
             f"{len(semillas)} semillas x {len(origenes)} origenes, {args.epocas} epocas max., lote {ent.LOTE}")

    for condicion in condiciones:
        conjuntos = construir_conjuntos(arrays, manifiesto, condicion)
        guardar = (args.checkpoints == "si") or (
            args.checkpoints == "auto" and condicion == "baseline" and args.particion == "original")
        for arquitectura in arquitecturas:
            for semilla in semillas:
                inicio = time.perf_counter()
                for origen in origenes:
                    if args.reanudar:
                        filtro = {"arquitectura": arquitectura, "semilla": semilla, "condicion": condicion,
                                  "particion": args.particion, "dominio_train": origen}
                        extras = [salida / "predicciones" / f"{condicion}_{args.particion}"
                                  / f"{arquitectura}_s{semilla}_{origen}.csv.gz"]
                        if guardar:
                            extras.append(CHECKPOINTS / f"{condicion}_{args.particion}"
                                          / f"{arquitectura}_s{semilla}_{origen}.pt")
                        if (hecho(salida / "tables" / "resultados_matriz.csv", filtro, 4, extras)
                                and hecho(salida / "tables" / "historial_entrenamiento.csv", filtro, 1, [])):
                            log.info(f"[{arquitectura}/{condicion}/{args.particion}/s{semilla}] '{origen}' ya hecho, se salta")
                            continue
                    log.info(f"[{arquitectura}/{condicion}/{args.particion}/s{semilla}] entrenando en '{origen}'")
                    entrenar_origen(conjuntos, origen, arquitectura, semilla, condicion, args.particion,
                                    args.epocas, salida, guardar, log)
                log.info(f"[{arquitectura}/{condicion}/s{semilla}] {len(origenes)} origenes en "
                         f"{(time.perf_counter() - inicio) / 60:.1f} min")

    log.info(f"03_experiment.py terminado en {(time.perf_counter() - arranque) / 60:.1f} min.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
