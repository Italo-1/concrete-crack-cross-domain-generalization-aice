"""Representaciones de senal en una dimension. Compartido por P7 y P4.

P7 (rodamientos) compara cuatro representaciones: estadisticos temporales,
magnitud FFT, wavelet discreta db4 y espectrograma STFT.
P4 (calidad de energia) compara tres: wavelet discreta, S-transform y
espectrograma STFT.

El solape entre ambos articulos es casi total; solo la S-transform es exclusiva
de P4. Cualquier cambio aqui afecta a los dos manuscritos: verificarlo antes de
tocar nada despues de que P7 este cerrado.
"""

from __future__ import annotations

import numpy as np
import pywt
from scipy import signal as sp_signal
from scipy import stats as sp_stats

NOMBRES_ESTADISTICOS = [
    "mean", "rms", "std", "skewness", "kurtosis", "peak",
    "peak_to_peak", "crest_factor", "shape_factor", "impulse_factor",
    "clearance_factor", "mean_amplitude_root",
]


# ------------------------------------------------------------------ segmentado

def segmentar(x: np.ndarray, ventana: int, solape: float = 0.5) -> np.ndarray:
    """Divide una senal en ventanas solapadas. Devuelve (n_ventanas, ventana).

    P7 usa ventana = 2048 muestras con 50 % de solape.
    P4 usa ventanas de 10 ciclos, que a 256 muestras por ciclo son 2560.

    El solape multiplica el numero de muestras de entrenamiento, pero las
    ventanas contiguas no son independientes: la division train/test debe
    hacerse por registro de origen, nunca por ventana, o el resultado sale
    inflado. Es el error mas comun en los articulos de este tema.
    """
    if x.ndim != 1:
        raise ValueError("Se espera una senal unidimensional.")
    paso = int(ventana * (1.0 - solape))
    if paso < 1:
        raise ValueError("El solape deja un paso menor que una muestra.")

    n = 1 + (x.size - ventana) // paso if x.size >= ventana else 0
    if n <= 0:
        return np.empty((0, ventana))

    indices = np.arange(ventana)[None, :] + paso * np.arange(n)[:, None]
    return x[indices]


# -------------------------------------------------- (a) estadisticos temporales

def estadisticos_temporales(x: np.ndarray) -> np.ndarray:
    """Los 12 descriptores clasicos del diagnostico de vibraciones.

    Acepta una ventana (n,) o un lote (n_ventanas, n). Devuelve (n_ventanas, 12)
    en el orden de NOMBRES_ESTADISTICOS.

    Curtosis y factor de cresta son los sensibles a impactos repetitivos, que es
    justo la firma de una falla localizada en pista.
    """
    X = np.atleast_2d(np.asarray(x, dtype=float))

    media = X.mean(axis=1)
    rms = np.sqrt(np.mean(X**2, axis=1))
    desv = X.std(axis=1, ddof=1)
    asimetria = sp_stats.skew(X, axis=1)
    curtosis = sp_stats.kurtosis(X, axis=1, fisher=False)
    pico = np.max(np.abs(X), axis=1)
    pico_pico = X.max(axis=1) - X.min(axis=1)

    media_abs = np.mean(np.abs(X), axis=1)
    raiz_amplitud = np.mean(np.sqrt(np.abs(X)), axis=1) ** 2

    # Se protege la division: una ventana de silencio da rms y media_abs nulos.
    seguro = lambda num, den: np.divide(
        num, den, out=np.zeros_like(num, dtype=float), where=den > 1e-12
    )

    return np.column_stack([
        media, rms, desv, asimetria, curtosis, pico, pico_pico,
        seguro(pico, rms),          # factor de cresta
        seguro(rms, media_abs),     # factor de forma
        seguro(pico, media_abs),    # factor de impulso
        seguro(pico, raiz_amplitud),  # factor de holgura
        raiz_amplitud,
    ])


# ------------------------------------------------------------------- (b) FFT

def magnitud_fft(x: np.ndarray, n_bandas: int | None = None) -> np.ndarray:
    """Magnitud del espectro de una senal real, hasta Nyquist.

    Con `n_bandas` se promedia el espectro en bandas de igual ancho, que reduce
    la dimension sin perder la estructura gruesa. Sin ese argumento devuelve el
    espectro completo (ventana // 2 + 1 valores), que para ventana = 2048 son
    1025 caracteristicas: demasiadas para un SVM con pocas muestras.
    """
    X = np.atleast_2d(np.asarray(x, dtype=float))
    espectro = np.abs(np.fft.rfft(X, axis=1))

    if n_bandas is None:
        return espectro

    bordes = np.linspace(0, espectro.shape[1], n_bandas + 1, dtype=int)
    return np.column_stack([
        espectro[:, bordes[i]:bordes[i + 1]].mean(axis=1) for i in range(n_bandas)
    ])


# --------------------------------------------------------------- (c) wavelet

def wavelet_db4(
    x: np.ndarray,
    niveles: int = 5,
    ondicula: str = "db4",
    solo_energia: bool = True,
    estadisticos_por_nivel: bool = False,
) -> np.ndarray:
    """Descomposicion wavelet discreta multinivel.

    Tres modos, y la eleccion no es inocente:

        solo_energia=True            energia RELATIVA de cada banda
                                     (niveles + 1 valores). Invariante a la
                                     amplitud por construccion, dimension baja.
                                     Adecuado cuando la falla redistribuye
                                     energia entre bandas, como en rodamientos.

        estadisticos_por_nivel=True  energia relativa, desviacion tipica y
                                     entropia de Shannon por nivel
                                     (3 x (niveles + 1) valores). Es el conjunto
                                     que usa habitualmente la literatura de
                                     calidad de energia.

                                     **La desviacion tipica SI depende de la
                                     amplitud**, y eso es justo lo que hace
                                     detectables fenomenos que no mueven energia
                                     entre bandas, como el flicker: una
                                     modulacion lenta de amplitud deja las
                                     energias relativas practicamente
                                     intactas.

        solo_energia=False           todos los coeficientes concatenados, para
                                     alimentar una red en lugar de un
                                     clasificador clasico.
    """
    X = np.atleast_2d(np.asarray(x, dtype=float))
    salida = []

    for fila in X:
        coefs = pywt.wavedec(fila, ondicula, level=niveles)

        if estadisticos_por_nivel:
            energias = np.array([np.sum(c**2) for c in coefs])
            total = energias.sum()
            relativas = energias / total if total > 1e-12 else energias

            desviaciones = np.array([np.std(c) for c in coefs])

            entropias = []
            for c in coefs:
                p = c**2
                s = p.sum()
                if s <= 1e-12:
                    entropias.append(0.0)
                    continue
                p = p / s
                entropias.append(float(-np.sum(p * np.log(p + 1e-12))))

            salida.append(np.concatenate([relativas, desviaciones, entropias]))

        elif solo_energia:
            energias = np.array([np.sum(c**2) for c in coefs])
            total = energias.sum()
            salida.append(energias / total if total > 1e-12 else energias)
        else:
            salida.append(np.concatenate(coefs))

    if estadisticos_por_nivel or solo_energia:
        return np.array(salida)

    # Sin energia, las filas pueden diferir en longitud por el padding: se recorta.
    minimo = min(len(f) for f in salida)
    return np.array([f[:minimo] for f in salida])


# ------------------------------------------------------------ (d) espectrograma

def parametros_stft(
    n: int, fs: float, nperseg: int | None = None, resolucion_hz: float | None = None
) -> dict:
    """Elige `nperseg` y devuelve las caracteristicas reales del espectrograma.

    **La resolucion en frecuencia depende solo de `nperseg`**, nunca del tamano
    de la imagen de salida: delta_f = fs / nperseg. Son dos cosas independientes
    y confundirlas lleva a redimensionar una imagen sin ganar informacion.

    Hay un compromiso ineludible: con una ventana de `n` muestras, mas
    resolucion en frecuencia significa menos tramas de tiempo. Conviene elegir
    `nperseg` a partir de la frecuencia que se necesita distinguir y comprobar
    despues que quedan tramas suficientes.

        resolucion_hz   se elige la potencia de 2 que alcance esa resolucion o
                        mejor, acotada a la longitud de la senal
        nperseg         longitud exacta de la ventana, tiene prioridad
        ninguno         n // 8 redondeado a potencia de 2

    Devuelve un diccionario con `nperseg`, `noverlap`, `delta_f_hz`,
    `n_frecuencias` y `n_tramas`, pensado para volcarlo tal cual al manifiesto
    del articulo: son los numeros que el manuscrito tiene que declarar.
    """
    if nperseg is None:
        if resolucion_hz is not None:
            objetivo = fs / float(resolucion_hz)
            nperseg = int(2 ** np.ceil(np.log2(max(objetivo, 8))))
        else:
            nperseg = int(2 ** np.round(np.log2(max(n // 8, 8))))

    nperseg = int(min(max(nperseg, 8), n))
    noverlap = nperseg // 2
    paso = nperseg - noverlap

    return {
        "nperseg": nperseg,
        "noverlap": noverlap,
        "delta_f_hz": fs / nperseg,
        "n_frecuencias": nperseg // 2 + 1,
        "n_tramas": 1 + max(n - nperseg, 0) // paso,
    }


def espectrograma_stft(
    x: np.ndarray,
    fs: float = 12_000.0,
    tamano: int = 64,
    en_db: bool = True,
    nperseg: int | None = None,
    resolucion_hz: float | None = None,
    normalizar: str = "imagen",
    devolver_info: bool = False,
):
    """Espectrograma redimensionado a una imagen cuadrada de `tamano` x `tamano`.

    Es la entrada de la CNN 2D en P7 y en P4.

    `tamano` fija SOLO el tamano de la imagen de salida. La resolucion en
    frecuencia la fija `nperseg` (o `resolucion_hz`), y se consulta con
    `parametros_stft`. Redimensionar no crea informacion: si la resolucion de la
    STFT no separa dos frecuencias, ampliar la imagen tampoco lo hara.

    El modo de normalizacion **depende de si la amplitud es ruido o es senal**,
    y elegirlo mal borra el fenomeno que se quiere medir:

        "imagen"   cada imagen a [0, 1] por separado. Descarta la amplitud
                   absoluta y conserva solo la forma tiempo-frecuencia. Es lo
                   adecuado en P7, donde la amplitud de la vibracion escala con
                   la carga y esa dependencia contaminaria la comparacion entre
                   cargas.
        "global"   una sola escala para todo el lote. Conserva las diferencias
                   de amplitud ENTRE senales. Es lo adecuado en P4: sag, swell e
                   interrupcion se definen precisamente por su amplitud, y
                   normalizar cada imagen por separado borraria lo que las
                   distingue de una senal sana.
        "ninguna"  sin normalizar.

    Con `devolver_info=True` devuelve `(imagenes, info)`, donde `info` es lo que
    entrega `parametros_stft`.
    """
    if normalizar not in ("imagen", "global", "ninguna"):
        raise ValueError(f"modo de normalizacion desconocido: {normalizar!r}")

    X = np.atleast_2d(np.asarray(x, dtype=float))
    n = X.shape[1]

    info = parametros_stft(n, fs, nperseg=nperseg, resolucion_hz=resolucion_hz)
    imagenes = []

    for fila in X:
        _f, _t, Z = sp_signal.stft(
            fila, fs=fs, nperseg=info["nperseg"], noverlap=info["noverlap"]
        )
        magnitud = np.abs(Z)
        if en_db:
            magnitud = 20 * np.log10(magnitud + 1e-12)

        img = _redimensionar(magnitud, tamano, tamano)
        if normalizar == "imagen":
            rango = img.max() - img.min()
            img = (img - img.min()) / rango if rango > 1e-12 else np.zeros_like(img)
        imagenes.append(img)

    salida = np.array(imagenes)

    if normalizar == "global":
        minimo, maximo = salida.min(), salida.max()
        rango = maximo - minimo
        salida = (salida - minimo) / rango if rango > 1e-12 else np.zeros_like(salida)
        info = {**info, "escala_global": {"min": float(minimo), "max": float(maximo)}}

    return (salida, info) if devolver_info else salida


def _redimensionar(m: np.ndarray, alto: int, ancho: int) -> np.ndarray:
    """Remuestreo bilineal.

    Antes se usaba el vecino mas proximo, que al ampliar pocas tramas de tiempo
    a una imagen grande producia bandas macizas y descartaba muestras al
    reducir. El bilineal no inventa resolucion -eso lo fija `nperseg`- pero no
    tira informacion disponible.
    """
    from scipy import ndimage

    filas = np.linspace(0, m.shape[0] - 1, alto)
    columnas = np.linspace(0, m.shape[1] - 1, ancho)
    malla = np.meshgrid(filas, columnas, indexing="ij")
    return ndimage.map_coordinates(m, malla, order=1, mode="nearest")


# ----------------------------------------------------------- (e) S-transform
# Exclusiva de P4.

def s_transform(x: np.ndarray, fs: float = 3_200.0, max_frec: int | None = None) -> np.ndarray:
    """S-transform de Stockwell: devuelve la matriz |S| de (frecuencias, tiempo).

    Combina la resolucion en fase de la STFT con la ventana adaptativa de la
    wavelet, y en la literatura de calidad de energia es la representacion de
    referencia para sags y swells.

    Advertencia de coste: es O(N^2 log N). Para una ventana de 2560 muestras
    tarda del orden de un segundo, asi que sobre miles de senales conviene
    limitar `max_frec` a las bandas de interes (los primeros armonicos).
    """
    x = np.asarray(x, dtype=float).ravel()
    n = x.size

    espectro = np.fft.fft(x)
    espectro = np.concatenate([espectro, espectro])  # periodico, para el desplazamiento

    tope = max_frec if max_frec is not None else n // 2
    salida = np.zeros((tope, n), dtype=complex)
    salida[0, :] = x.mean()

    frecuencias = np.arange(n)
    for k in range(1, tope):
        # Gaussiana cuyo ancho escala con la frecuencia: sigma = 1 / k.
        gauss = np.exp(-2 * (np.pi**2) * (frecuencias**2) / (k**2))
        salida[k, :] = np.fft.ifft(espectro[k:k + n] * gauss)

    return np.abs(salida)


# ------------------------------------------------------------------- registro

REPRESENTACIONES_P7 = {
    "estadisticos": estadisticos_temporales,
    "fft": magnitud_fft,
    "wavelet": wavelet_db4,
    "espectrograma": espectrograma_stft,
}

REPRESENTACIONES_P4 = {
    "wavelet": wavelet_db4,
    "s_transform": s_transform,
    "espectrograma": espectrograma_stft,
}
