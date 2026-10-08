"""Entrenamiento y transferencia de modelos de vision. Compartido por P1 y P5.

P1 (fisuras) construye una matriz 4x4: entrena en cada dominio y evalua en los
cuatro. P5 (plantas) entrena en laboratorio y evalua en campo, con fine-tuning
progresivo sobre k imagenes por clase.

Los dos hacen lo mismo en el fondo -entrenar en un dominio, medir en otro- asi
que el bucle de entrenamiento, la evaluacion y la construccion de la matriz
viven aqui una sola vez.

Dimensionado para 6 GB de VRAM (RTX 4050 Laptop): precision mixta activada por
defecto y `lote_para_vram()` para elegir el tamano de lote sin provocar OOM.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset

from fieutils.config import dispositivo, fijar_semilla

# Estadisticas de ImageNet: los modelos preentrenados las esperan.
MEDIA_IMAGENET = (0.485, 0.456, 0.406)
DESV_IMAGENET = (0.229, 0.224, 0.225)

# Consumo aproximado de VRAM por imagen a 224x224 con precision mixta, en MB.
# Medido de forma conservadora; sirve para elegir lote, no para predecir exacto.
_COSTE_VRAM_MB = {
    "resnet18": 11,
    "resnet50": 34,
    "efficientnet_b0": 19,
    "mobilenetv3_small_100": 6,
    "vit_tiny_patch16_224": 13,
}


# ------------------------------------------------------------------- modelos

def crear_modelo(nombre: str, n_clases: int = 2, preentrenado: bool = True) -> nn.Module:
    """Crea un modelo de timm con la cabeza ajustada a `n_clases`.

    Nombres usados en la linea B:
        P1  resnet18, efficientnet_b0, mobilenetv3_small_100, vit_tiny_patch16_224
        P5  resnet50, efficientnet_b0
    """
    import timm

    return timm.create_model(nombre, pretrained=preentrenado, num_classes=n_clases)


def lote_para_vram(nombre: str, margen: float = 0.75, maximo: int = 64) -> int:
    """Tamano de lote que cabe en la VRAM libre, con margen de seguridad.

    Con 6 GB reales el margen importa: PyTorch reserva memoria por bloques y un
    lote calculado al limite provoca OOM a mitad de una epoca, tirando horas de
    entrenamiento nocturno.
    """
    if not torch.cuda.is_available():
        return 8

    libre_mb = torch.cuda.mem_get_info()[0] / 1024**2
    coste = _COSTE_VRAM_MB.get(nombre, 25)
    lote = int((libre_mb * margen) / coste)

    # Potencia de 2 mas cercana por debajo, acotada a [4, maximo].
    lote = min(max(lote, 4), maximo)
    return 2 ** int(np.floor(np.log2(lote)))


# ------------------------------------------------------------------ resultado

@dataclass
class ResultadoEntrenamiento:
    """Lo que devuelve `entrenar`, listo para volcar a `results/`."""

    modelo: nn.Module
    mejor_epoca: int
    mejor_metrica: float
    historial: list[dict] = field(default_factory=list)
    segundos: float = 0.0

    def resumen(self) -> str:
        return (
            f"mejor metrica val = {self.mejor_metrica:.4f} en epoca {self.mejor_epoca} "
            f"({self.segundos / 60:.1f} min, {len(self.historial)} epocas)"
        )


# -------------------------------------------------------------- entrenamiento

def entrenar(
    modelo: nn.Module,
    cargador_train: DataLoader,
    cargador_val: DataLoader,
    epocas: int = 12,
    lr: float = 3e-4,
    paciencia: int = 3,
    clase_positiva: int = 1,
    precision_mixta: bool = True,
    metrica_seleccion: str = "f1",
    log=None,
) -> ResultadoEntrenamiento:
    """Fine-tuning completo con early stopping sobre una metrica de validacion.

    Se vigila F1 y no exactitud porque las clases estan desbalanceadas en los
    dos articulos: en SDNET2018 las fisuras son minoria clara, y con exactitud
    el early stopping premiaria al modelo que predice siempre "sin fisura".

    `metrica_seleccion` elige cual de las claves que devuelve `evaluar()`
    gobierna el early stopping y la eleccion de "mejor epoca". Por defecto es
    'f1' (F1 de `clase_positiva`), correcto para el problema binario de P1.
    P5 es multiclase con 18 clases sin una "clase positiva" que tenga sentido,
    asi que usa 'f1_macro'; con 'f1' ahi el early stopping se guiaria por el
    desempeno de una sola clase elegida por indice, no por el global.

    Devuelve el modelo con los pesos de la mejor epoca, no los de la ultima.
    """
    dev = dispositivo(verboso=False)
    modelo = modelo.to(dev)

    criterio = nn.CrossEntropyLoss()
    optimizador = torch.optim.AdamW(modelo.parameters(), lr=lr, weight_decay=1e-4)
    planificador = torch.optim.lr_scheduler.CosineAnnealingLR(optimizador, T_max=epocas)

    usar_amp = precision_mixta and dev.type == "cuda"
    escalador = torch.amp.GradScaler("cuda", enabled=usar_amp)

    mejor_metrica, mejor_epoca, sin_mejora = -1.0, -1, 0
    mejores_pesos = None
    historial = []
    inicio = time.time()

    for epoca in range(epocas):
        modelo.train()
        perdida_total, n = 0.0, 0

        for x, y in cargador_train:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            optimizador.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=usar_amp):
                perdida = criterio(modelo(x), y)

            escalador.scale(perdida).backward()
            escalador.step(optimizador)
            escalador.update()

            perdida_total += perdida.item() * x.size(0)
            n += x.size(0)

        planificador.step()
        metricas = evaluar(modelo, cargador_val, clase_positiva=clase_positiva)
        registro = {
            "epoca": epoca,
            "perdida_train": perdida_total / max(n, 1),
            **metricas,
        }
        historial.append(registro)

        if log:
            log.info(
                f"epoca {epoca:>2} | perdida {registro['perdida_train']:.4f} | "
                f"{metrica_seleccion} val {metricas[metrica_seleccion]:.4f} | "
                f"exactitud {metricas['exactitud']:.4f}"
            )

        if metricas[metrica_seleccion] > mejor_metrica:
            mejor_metrica, mejor_epoca, sin_mejora = metricas[metrica_seleccion], epoca, 0
            mejores_pesos = {k: v.detach().cpu().clone() for k, v in modelo.state_dict().items()}
        else:
            sin_mejora += 1
            if sin_mejora >= paciencia:
                if log:
                    log.info(f"early stopping en epoca {epoca} (paciencia {paciencia})")
                break

    if mejores_pesos is not None:
        modelo.load_state_dict(mejores_pesos)

    return ResultadoEntrenamiento(
        modelo=modelo,
        mejor_epoca=mejor_epoca,
        mejor_metrica=mejor_metrica,
        historial=historial,
        segundos=time.time() - inicio,
    )


@torch.no_grad()
def evaluar(modelo: nn.Module, cargador: DataLoader, clase_positiva: int = 1) -> dict:
    """Exactitud, F1 de la clase positiva (solo si el problema es binario) y F1-macro.

    `f1_score(..., pos_label=...)` fuerza `average='binary'` dentro de scikit-learn,
    que lanza `ValueError` si el cargador tiene mas de dos clases presentes. P1
    (grieta/no grieta) nunca dispara esto; P5 (18 clases) lo hace siempre.
    Encontrado el 26/08 al ejecutar el smoke test de P5: la funcion no
    completaba ni una epoca en un problema multiclase. Con mas de dos clases
    'f1' se deja en NaN -no esta definida para ese caso- y el llamador debe
    usar 'f1_macro', que es lo que ya hacen P5 y cualquier articulo multiclase
    futuro via `metrica_seleccion`.
    """
    from sklearn.metrics import accuracy_score, f1_score

    dev = dispositivo(verboso=False)
    modelo = modelo.to(dev).eval()

    reales, predichas = [], []
    for x, y in cargador:
        x = x.to(dev, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=dev.type == "cuda"):
            salida = modelo(x)
        predichas.append(salida.argmax(1).cpu().numpy())
        reales.append(y.numpy())

    y_real = np.concatenate(reales)
    y_pred = np.concatenate(predichas)
    es_binario = np.union1d(y_real, y_pred).size <= 2

    f1_binario = (
        float(f1_score(y_real, y_pred, pos_label=clase_positiva, zero_division=0))
        if es_binario else float("nan")
    )

    return {
        "exactitud": float(accuracy_score(y_real, y_pred)),
        "f1": f1_binario,
        "f1_macro": float(f1_score(y_real, y_pred, average="macro", zero_division=0)),
        "n": int(y_real.size),
    }


# ------------------------------------------------- matriz de transferencia

def matriz_transferencia(
    dominios: dict[str, tuple[DataLoader, DataLoader, DataLoader]],
    arquitectura: str,
    n_clases: int = 2,
    epocas: int = 12,
    semilla: int = 42,
    log=None,
) -> np.ndarray:
    """Entrena en cada dominio y evalua en todos. Devuelve la matriz de F1.

    `dominios` mapea nombre -> (train, val, test). La celda [i, j] es el F1 del
    modelo entrenado en el dominio i, evaluado sobre el test del dominio j.

    La diagonal reproduce lo que reporta la literatura; el articulo vive de lo
    que pasa fuera de ella. Coste: un entrenamiento por fila, no por celda, ya
    que evaluar es barato comparado con entrenar.
    """
    nombres = list(dominios)
    matriz = np.full((len(nombres), len(nombres)), np.nan)

    for i, origen in enumerate(nombres):
        fijar_semilla(semilla, determinista=False)
        train, val, _test = dominios[origen]

        if log:
            log.info(f"[{arquitectura}] entrenando en '{origen}' ({i + 1}/{len(nombres)})")

        modelo = crear_modelo(arquitectura, n_clases=n_clases)
        resultado = entrenar(modelo, train, val, epocas=epocas, log=log)

        if log:
            log.info(f"[{arquitectura}] '{origen}': {resultado.resumen()}")

        for j, destino in enumerate(nombres):
            _t, _v, test_destino = dominios[destino]
            matriz[i, j] = evaluar(resultado.modelo, test_destino)["f1"]

        del modelo, resultado
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return matriz


# ---------------------------------------------------------------- submuestreo

def submuestrear_balanceado(
    conjunto: Dataset, etiquetas: np.ndarray, n_por_clase: int, semilla: int = 42
) -> Subset:
    """Submuestreo con el mismo numero de ejemplos por clase.

    Es el recorte aprobado para P1: con SDNET2018 aportando 56 000 imagenes y
    METU 40 000, dejar los tamanos dispares contamina la comparacion de
    transferencia con el tamano del conjunto de origen. Igualarlos elimina ese
    factor de confusion, ademas de recortar el coste de computo.
    """
    rng = np.random.default_rng(semilla)
    indices = []

    for clase in np.unique(etiquetas):
        candidatos = np.flatnonzero(etiquetas == clase)
        n = min(n_por_clase, candidatos.size)
        indices.append(rng.choice(candidatos, size=n, replace=False))

    seleccion = np.concatenate(indices)
    rng.shuffle(seleccion)
    return Subset(conjunto, seleccion.tolist())


def cachear_imagenes(
    rutas: list, destino: Path, tamano: int = 224, log=None, cada: int = 2000
) -> np.ndarray:
    """Decodifica y redimensiona una lista de imagenes a un unico array uint8.

    Devuelve (N, tamano, tamano, 3) y lo guarda en `destino` como .npy.

    Merece la pena porque los dos articulos de vision reentrenan muchas veces
    sobre el mismo conjunto: P1 hace 48 ajustes de 12 epocas, lo que supone del
    orden de dos millones de lecturas de imagen. Decodificar el JPEG una sola
    vez y dejarlo en un array convierte la carga de datos en un problema de
    memoria y no de CPU, que es donde estaria el cuello de botella con una GPU
    modesta.

    Se guarda en uint8, no en float: ocupa cuatro veces menos y la conversion a
    tensor normalizado se hace por lote en el momento de entrenar.
    """
    from PIL import Image

    destino = Path(destino)
    destino.parent.mkdir(parents=True, exist_ok=True)

    if destino.exists():
        cacheado = np.load(destino, mmap_mode="r")
        if cacheado.shape[0] == len(rutas):
            if log:
                log.info(f"cache ya presente: {destino.name} {cacheado.shape}")
            return cacheado
        if log:
            log.info(f"cache obsoleto ({cacheado.shape[0]} != {len(rutas)}), se regenera")

    salida = np.empty((len(rutas), tamano, tamano, 3), dtype=np.uint8)
    for i, ruta in enumerate(rutas):
        with Image.open(ruta) as img:
            salida[i] = np.asarray(
                img.convert("RGB").resize((tamano, tamano), Image.BILINEAR),
                dtype=np.uint8,
            )
        if log and cada and (i + 1) % cada == 0:
            log.info(f"  {i + 1}/{len(rutas)} imagenes")

    np.save(destino, salida)
    if log:
        log.info(f"cache escrito: {destino.name} {salida.shape} "
                 f"({salida.nbytes / 1024**3:.2f} GB)")
    return salida


def hash_archivos(rutas: list, log=None) -> list[str]:
    """MD5 del contenido de cada archivo, para detectar imagenes repetidas.

    La ficha de P1 marca como riesgo que los datasets se solapen. Si la misma
    imagen aparece en dos dominios, la celda correspondiente de la matriz de
    transferencia deja de medir generalizacion y mide memorizacion.
    """
    import hashlib

    hashes = []
    for i, ruta in enumerate(rutas):
        hashes.append(hashlib.md5(Path(ruta).read_bytes()).hexdigest())
        if log and (i + 1) % 5000 == 0:
            log.info(f"  hash {i + 1}/{len(rutas)}")
    return hashes


def transformaciones(entrenamiento: bool, tamano: int = 224, agresivo: bool = False):
    """Transformaciones estandar. `agresivo=True` es la mitigacion de P1.

    La condicion agresiva -rotacion, brillo, contraste, ruido- se aplica solo al
    entrenamiento y se mide si reduce la caida fuera de dominio.
    """
    from torchvision import transforms as T

    if not entrenamiento:
        return T.Compose([
            T.Resize((tamano, tamano)),
            T.ToTensor(),
            T.Normalize(MEDIA_IMAGENET, DESV_IMAGENET),
        ])

    pasos = [T.Resize((tamano, tamano)), T.RandomHorizontalFlip()]
    if agresivo:
        pasos += [
            T.RandomRotation(30),
            T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2),
            T.RandomAffine(0, translate=(0.1, 0.1)),
        ]
    pasos += [T.ToTensor(), T.Normalize(MEDIA_IMAGENET, DESV_IMAGENET)]

    if agresivo:
        # Ruido gaussiano leve, despues de normalizar.
        pasos.append(T.Lambda(lambda t: t + torch.randn_like(t) * 0.05))

    return T.Compose(pasos)
