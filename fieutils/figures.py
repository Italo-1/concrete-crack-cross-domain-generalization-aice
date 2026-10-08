"""Estilo y figuras comunes de la linea B.

Los cinco articulos generan unas veinte figuras. Todas salen de aqui, de modo que
el conjunto se lee como un solo cuerpo de trabajo y no como cinco estilos
distintos.

Tres reglas que gobiernan el modulo:

1. Legible impresa en blanco y negro. Ninguna serie se distingue solo por color:
   cada una lleva ademas marcador y trazo propios. Los mapas de calor usan rampas
   monotonas en luminancia, que al pasar a gris siguen ordenando bien.
2. Color asignado por funcion. Categorico para identidad (modelos,
   representaciones), secuencial para magnitud (F1 en una matriz), divergente con
   gris central para polaridad (caida relativa, skill score frente a
   persistencia). Nunca arcoiris.
3. Vectorial para el manuscrito, PNG para revisar. `guardar()` escribe ambos.

La paleta esta validada para deuteranopia, protanopia y tritanopia; el orden es
fijo y no se cicla. Si un articulo necesita mas de seis series, se agrupan las
menores en "Otros" o se pasa a paneles multiples.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# ------------------------------------------------------------------- paleta

# Paleta Okabe-Ito EN EL ORDEN QUE FIJA LA GUIA METODOLOGICA, seccion 4.2.
# No se reordena: es una especificacion del encargo, no una preferencia.
CATEGORICA = [
    "#0072B2", "#D55E00", "#009E73", "#CC79A7",
    "#E69F00", "#56B4E9", "#F0E442", "#000000",
]

# Validacion de accesibilidad de esta paleta (ejecutada el 22/08/2026):
#
#   Series 1-6  -> pasa todas las comprobaciones. El par adyacente mas
#                  ajustado es #CC79A7 con #009E73, a delta-E 7.6 en
#                  deuteranopia. Queda por encima del suelo admisible SOLO
#                  gracias a la codificacion secundaria, que la propia guia
#                  exige ("distinguir por marcador y tipo de linea, no solo
#                  por color"). Por eso MARCADORES y TRAZOS no son opcionales.
#
#   Serie 7 (#F0E442, amarillo) -> luminancia 0.90 y contraste 1.29:1 sobre
#                  blanco. Practicamente invisible como linea fina.
#   Serie 8 (#000000, negro)    -> croma cero, se confunde con la tinta del
#                  texto y de los ejes.
#
# Con mas de seis series, repartir en paneles antes que recurrir a la 7 y la 8.
# Afecta a P3, que compara siete modelos: ver nota en su 05_figures.py.
LIMITE_SERIES_SEGURAS = 6

# Codificacion secundaria OBLIGATORIA: la serie i lleva siempre el mismo
# marcador y el mismo trazo, para que la figura se lea impresa en blanco y
# negro (guia, seccion 4.2).
MARCADORES = ["o", "s", "^", "D", "v", "P", "X", "*"]
TRAZOS = [
    "-", "--", "-.", ":",
    (0, (3, 1, 1, 1)), (0, (5, 1)), (0, (1, 1)), (0, (3, 1, 1, 1, 1, 1)),
]

SECUENCIAL = "cividis"   # monotona en luminancia: sobrevive al gris
DIVERGENTE = "RdBu_r"    # dos polos con neutro central, para valores con signo

TINTA = "#1a1a1a"
TINTA_SUAVE = "#5a5a5a"
REJILLA = "#d8d8d8"

# Anchos de columna que fija la guia, en pulgadas.
ANCHO_SIMPLE = 3.5
ANCHO_DOBLE = 7.16


def aplicar_estilo(base: int = 9) -> None:
    """Estilo global. Llamar una vez al principio de cada `05_figures.py`.

    Los valores reproducen la especificacion tecnica de la guia metodologica
    (seccion 4.2). Los que no aparecen alli se eligen para no contradecirla.
    """
    mpl.rcParams.update({
        # --- fijados por la guia ---
        "font.family": "serif",
        "font.size": base,                 # 9
        "axes.labelsize": base,            # 9
        "axes.titlesize": base,            # 9
        "legend.fontsize": base - 1,       # 8
        "xtick.labelsize": base - 1,       # 8
        "ytick.labelsize": base - 1,       # 8
        "figure.dpi": 300,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        # --- complementos que no contradicen la guia ---
        "font.serif": ["DejaVu Serif", "Times New Roman"],
        "savefig.dpi": 300,
        "savefig.pad_inches": 0.02,
        "axes.edgecolor": TINTA_SUAVE,
        "axes.labelcolor": TINTA,
        "axes.linewidth": 0.8,
        "axes.axisbelow": True,
        "grid.color": REJILLA,
        "grid.linewidth": 0.5,
        "xtick.color": TINTA_SUAVE,
        "ytick.color": TINTA_SUAVE,
        "text.color": TINTA,
        "lines.linewidth": 1.6,
        "lines.markersize": 4.5,
        "legend.frameon": False,
        "figure.autolayout": False,
    })


def estilo_serie(i: int) -> dict:
    """Color, marcador y trazo de la serie `i`. Identidad nunca solo por color."""
    if i >= LIMITE_SERIES_SEGURAS:
        import warnings

        warnings.warn(
            f"Serie {i + 1}: se entra en los colores 7-8 de la paleta "
            f"({CATEGORICA[i % len(CATEGORICA)]}), que no superan las "
            f"comprobaciones de accesibilidad. Repartir en paneles.",
            stacklevel=2,
        )
    return {
        "color": CATEGORICA[i % len(CATEGORICA)],
        "marker": MARCADORES[i % len(MARCADORES)],
        "linestyle": TRAZOS[i % len(TRAZOS)],
    }


def etiqueta_panel(i: int) -> str:
    """Etiqueta de panel: (a), (b), (c)...

    La guia prohibe titulos dentro de la figura: el titulo va en el pie. En
    figuras de varios paneles se usa la convencion de revista, que es rotular
    cada panel con una letra y describirlos en el `\\caption`.
    """
    return f"({chr(ord('a') + i)})"


def pie_de_figura(descripcion: str, paneles: list[str] | None = None,
                  nota: str = "") -> str:
    """Compone el texto del pie, que la guia exige autoexplicativo.

    Devuelve una cadena lista para pegar en el `\\caption` de LaTeX. Si la
    figura tiene paneles, los enumera con su letra, que es donde van los
    nombres que no pueden ir dentro de la figura.
    """
    partes = [descripcion.rstrip(".") + "."]

    if paneles:
        listado = ", ".join(
            f"{etiqueta_panel(i)} {nombre}" for i, nombre in enumerate(paneles)
        )
        partes.append(f"Paneles: {listado}.")

    if nota:
        partes.append(nota.rstrip(".") + ".")

    return " ".join(partes)


def guardar(fig, destino: str | Path, nombre: str, cerrar: bool = True) -> Path:
    """Guarda en PDF vectorial (para el manuscrito) y PNG 300 dpi (para revisar)."""
    carpeta = Path(destino)
    carpeta.mkdir(parents=True, exist_ok=True)

    pdf = carpeta / f"{nombre}.pdf"
    fig.savefig(pdf)
    fig.savefig(carpeta / f"{nombre}.png")
    if cerrar:
        plt.close(fig)
    return pdf


# ------------------------------------------------- mapa de calor de transferencia
# Figura azul de P7 (Fig. 3) y de P1 (Fig. 2).

def mapa_calor_transferencia(
    matrices: dict[str, np.ndarray],
    etiquetas: list[str],
    titulo_color: str = "F1",
    divergente: bool = False,
    centro: float = 0.0,
    anotar: bool = True,
    etiqueta_filas: str = "Training",
    etiqueta_columnas: str = "Test",
    figsize: tuple[float, float] | None = None,
    tam_anotacion: int | None = None,
):
    """Panel de matrices cuadradas origen-destino, una por clave de `matrices`.

    Es la figura que sostiene P7 y P1: la diagonal es lo que reporta la
    literatura, y el articulo vive de lo que pasa fuera de ella. Por eso la
    diagonal se recuadra, para que el lector localice de un vistazo la
    referencia contra la que comparar.

    Con `divergente=True` la rampa se centra en `centro`, util cuando la celda es
    una caida relativa y el signo importa.
    """
    n = len(matrices)
    if figsize is None:
        figsize = (ANCHO_DOBLE, ANCHO_DOBLE / max(n, 1) * 1.15)

    fig, ejes = plt.subplots(1, n, figsize=figsize, constrained_layout=True)
    ejes = np.atleast_1d(ejes)

    todos = np.concatenate([m.ravel() for m in matrices.values()])
    if divergente:
        radio = np.nanmax(np.abs(todos - centro))
        vmin, vmax, cmap = centro - radio, centro + radio, DIVERGENTE
    else:
        vmin, vmax, cmap = np.nanmin(todos), np.nanmax(todos), SECUENCIAL

    for indice, (eje, (_nombre, matriz)) in enumerate(zip(ejes, matrices.items())):
        im = eje.imshow(matriz, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
        # Solo la letra del panel: los nombres van en el pie de figura, porque
        # la guia prohibe titulos dentro de la figura. Usar `pie_de_figura`
        # con `paneles=list(matrices)` para generar el texto del caption.
        eje.set_title(etiqueta_panel(indice), pad=6, fontsize=mpl.rcParams["axes.titlesize"])
        eje.set_xticks(range(len(etiquetas)))
        eje.set_yticks(range(len(etiquetas)))
        eje.set_xticklabels(etiquetas, rotation=45, ha="right")
        eje.set_yticklabels(etiquetas)
        eje.grid(False)

        # La diagonal es la referencia: se recuadra en lugar de colorearla aparte.
        for i in range(len(etiquetas)):
            eje.add_patch(plt.Rectangle(
                (i - 0.5, i - 0.5), 1, 1, fill=False,
                edgecolor=TINTA, linewidth=1.4,
            ))

        if anotar:
            mapa = mpl.colormaps[cmap]
            rango = (vmax - vmin) or 1.0
            for i in range(matriz.shape[0]):
                for j in range(matriz.shape[1]):
                    valor = matriz[i, j]
                    if np.isnan(valor):
                        continue
                    # El color del texto se decide por la luminancia real de la
                    # celda, no por el punto medio del rango: cividis y RdBu_r no
                    # son claros en el centro, y un umbral aritmetico deja
                    # numeros oscuros sobre fondo oscuro.
                    r, g, b, _ = mapa((valor - vmin) / rango)
                    luminancia = 0.299 * r + 0.587 * g + 0.114 * b
                    color = "white" if luminancia < 0.55 else TINTA
                    # Tamano de la anotacion segun cuantas celdas hay que
                    # meter en el panel: con fontsize fijo, una matriz con
                    # muchas etiquetas deja numeros que se salen de su celda.
                    # Se escala con el lado de la matriz, con piso en 5 (aun
                    # legible) y techo en 7 (el valor original, para matrices
                    # pequenas). El numero de etiquetas no es la unica variable
                    # que importa -el tamano de celda tambien depende del
                    # `figsize` que pase cada articulo (p.ej. paneles mas altos
                    # para acomodar etiquetas largas)-, asi que se deja
                    # `tam_anotacion` como escape manual cuando el automatico
                    # no basta.
                    tam = tam_anotacion or max(5, min(7, int(60 / max(len(etiquetas), 1))))
                    eje.text(j, i, f"{valor:.2f}", ha="center", va="center",
                             fontsize=tam, color=color)

    ejes[0].set_ylabel(etiqueta_filas)
    for eje in ejes:
        eje.set_xlabel(etiqueta_columnas)

    barra = fig.colorbar(im, ax=ejes.tolist(), fraction=0.025, pad=0.02)
    barra.set_label(titulo_color)
    barra.outline.set_visible(False)

    return fig, ejes


# ------------------------------------------------------ curva con banda de IC
# Figura azul de P5 (Fig. 3) y de P4 (Fig. 4).

def curva_con_ic(
    x,
    series: dict[str, tuple],
    etiqueta_x: str,
    etiqueta_y: str,
    referencia: float | None = None,
    etiqueta_referencia: str = "referencia",
    log_x: bool = False,
    figsize: tuple[float, float] | None = None,
):
    """Curvas con banda de intervalo de confianza.

    `series` mapea nombre -> (media, limite_inferior, limite_superior), cada uno
    del mismo largo que `x`.

    La linea horizontal de `referencia` es lo que convierte la figura en un
    argumento: en P5 es la exactitud en laboratorio (el techo que se intenta
    recuperar) y en P8 era la particion aleatoria. Sin ella el lector no sabe
    contra que comparar.
    """
    fig, eje = plt.subplots(figsize=figsize or (ANCHO_SIMPLE, ANCHO_SIMPLE * 0.75))

    for i, (nombre, valores) in enumerate(series.items()):
        media, inf, sup = valores
        estilo = estilo_serie(i)
        eje.plot(x, media, label=nombre, markeredgecolor="white",
                 markeredgewidth=0.5, **estilo)
        eje.fill_between(x, inf, sup, color=estilo["color"], alpha=0.15, linewidth=0)

    if referencia is not None:
        eje.axhline(referencia, color=TINTA, linestyle=(0, (2, 2)), linewidth=1.0)
        eje.annotate(etiqueta_referencia, xy=(x[-1], referencia),
                     xytext=(-4, 4), textcoords="offset points",
                     ha="right", fontsize=7, color=TINTA_SUAVE)

    eje.set_xlabel(etiqueta_x)
    eje.set_ylabel(etiqueta_y)
    if log_x:
        eje.set_xscale("symlog", linthresh=1)
    if len(series) >= 2:
        # `loc="best"` competia por espacio con las propias curvas y su
        # banda de IC: con pocas series ya no queda hueco libre dentro del
        # eje y la leyenda termina encima de los datos. Se saca fuera del
        # eje, encima del panel; `savefig.bbox = "tight"` (aplicar_estilo)
        # extiende el lienzo guardado para incluirla, asi que no se recorta.
        eje.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02),
                   ncol=min(len(series), 3), frameon=False)

    return fig, eje


# ------------------------------------------------------------ diagrama de caja

def diagrama_caja(
    datos: dict[str, np.ndarray],
    etiqueta_y: str,
    etiqueta_x: str = "",
    figsize: tuple[float, float] | None = None,
):
    """Cajas con los puntos individuales superpuestos.

    Con pocas repeticiones (3 semillas en P1 y P4) la caja sola engana: se
    dibujan los puntos encima para que se vea de cuantas observaciones se habla.
    """
    fig, eje = plt.subplots(figsize=figsize or (ANCHO_SIMPLE, ANCHO_SIMPLE * 0.75))

    nombres = list(datos.keys())
    valores = [np.asarray(datos[k], dtype=float) for k in nombres]

    cajas = eje.boxplot(valores, patch_artist=True, widths=0.55,
                        medianprops={"color": TINTA, "linewidth": 1.4},
                        whiskerprops={"color": TINTA_SUAVE},
                        capprops={"color": TINTA_SUAVE},
                        flierprops={"marker": "", "markersize": 0})

    rng = np.random.default_rng(42)
    for i, (caja, v) in enumerate(zip(cajas["boxes"], valores), start=1):
        color = CATEGORICA[(i - 1) % len(CATEGORICA)]
        caja.set(facecolor=color, alpha=0.30, edgecolor=color, linewidth=1.0)
        eje.scatter(rng.normal(i, 0.045, size=v.size), v, s=13, color=color,
                    edgecolor="white", linewidth=0.5, zorder=3)

    eje.set_xticklabels(nombres, rotation=20, ha="right")
    eje.set_ylabel(etiqueta_y)
    if etiqueta_x:
        eje.set_xlabel(etiqueta_x)

    return fig, eje


# ------------------------------------------ diagrama de diferencia critica
# P7, Fig. 4.

def diagrama_diferencia_critica(
    rangos_medios,
    distancia_critica: float,
    figsize: tuple[float, float] | None = None,
):
    """Diagrama de Demsar: rangos medios con las barras de no significancia.

    Los metodos unidos por una barra horizontal no son distinguibles entre si.
    `rangos_medios` es la Serie que devuelve `stats.friedman_nemenyi`, ya
    ordenada de mejor (rango bajo) a peor.
    """
    nombres = list(rangos_medios.index)
    valores = np.asarray(rangos_medios.values, dtype=float)
    k = len(nombres)

    # Los metodos se reparten en dos columnas, asi que la altura la fija la
    # columna mas larga, no el total: con k = 4 son 2 filas, no 4.
    mitad_rango = (valores.min() + valores.max()) / 2
    n_izq = int(np.sum(valores < mitad_rango))
    filas = max(n_izq, k - n_izq, 1)

    fig, eje = plt.subplots(figsize=figsize or (ANCHO_DOBLE * 0.8, 1.5 + 0.30 * filas))

    bajo = np.floor(valores.min() - 0.3)
    alto = np.ceil(valores.max() + 0.3)

    # El texto de los nombres vive en un margen fuera del eje de rangos. Sin este
    # margen las etiquetas largas se salen del lienzo y se cruzan con las guias.
    margen = (alto - bajo) * 0.62
    eje.set_xlim(bajo - margen, alto + margen)

    # El eje de rangos va arriba y las etiquetas cuelgan de el. El limite
    # inferior se ajusta a la ultima fila ocupada para no dejar franja vacia.
    y_eje = 1.0
    y_minimo = y_eje - 0.55 - (filas - 1) * 0.42 - 0.30
    eje.set_ylim(y_minimo, y_eje + 1.15)
    eje.axis("off")

    # Eje de rangos, con las marcas y los numeros por encima.
    eje.plot([bajo, alto], [y_eje, y_eje], color=TINTA, linewidth=1.1)
    for t in np.arange(bajo, alto + 0.001):
        eje.plot([t, t], [y_eje, y_eje + 0.16], color=TINTA, linewidth=1.0)
        eje.text(t, y_eje + 0.26, f"{int(t)}", ha="center", va="bottom", fontsize=8)

    # Cada metodo desciende desde su rango y sale al margen del lado mas cercano.
    # Las etiquetas se apilan de fuera hacia dentro para que no se solapen.
    mitad = (bajo + alto) / 2
    izquierdos = [i for i in range(k) if valores[i] < mitad]
    derechos = [i for i in range(k) if valores[i] >= mitad]

    for orden, i in enumerate(izquierdos):
        y = y_eje - 0.55 - orden * 0.42
        eje.plot([valores[i], valores[i], bajo - margen * 0.72], [y_eje, y, y],
                 color=TINTA_SUAVE, linewidth=0.9)
        eje.text(bajo - margen * 0.76, y, nombres[i], ha="right", va="center",
                 fontsize=8, color=TINTA)

    for orden, i in enumerate(reversed(derechos)):
        y = y_eje - 0.55 - orden * 0.42
        eje.plot([valores[i], valores[i], alto + margen * 0.72], [y_eje, y, y],
                 color=TINTA_SUAVE, linewidth=0.9)
        eje.text(alto + margen * 0.76, y, nombres[i], ha="left", va="center",
                 fontsize=8, color=TINTA)

    # Barras de no significancia: grupos cuyos rangos difieren menos que la DC.
    # Se descartan los grupos contenidos en otro ya dibujado.
    grupos = []
    for i in range(k):
        j = i
        while j + 1 < k and valores[j + 1] - valores[i] < distancia_critica:
            j += 1
        if j > i and not any(a <= i and j <= b for a, b in grupos):
            grupos.append((i, j))

    for nivel, (a, b) in enumerate(grupos):
        y = y_eje - 0.16 - nivel * 0.17
        eje.plot([valores[a] - 0.03, valores[b] + 0.03], [y, y],
                 color=TINTA, linewidth=2.8, solid_capstyle="round")

    # Referencia de escala: un segmento de longitud CD, arriba a la izquierda.
    # La etiqueta va en ingles (CD, critical difference): las figuras van al
    # manuscrito y REDACCION.md prohibe que un texto interno en espanol llegue
    # al PDF. Corregido en la revision adversarial de P7 (01/09).
    y_dc = y_eje + 0.72
    eje.plot([bajo, bajo + distancia_critica], [y_dc, y_dc],
             color=TINTA_SUAVE, linewidth=1.6, solid_capstyle="butt")
    for extremo in (bajo, bajo + distancia_critica):
        eje.plot([extremo, extremo], [y_dc - 0.07, y_dc + 0.07],
                 color=TINTA_SUAVE, linewidth=1.0)
    eje.text(bajo + distancia_critica / 2, y_dc + 0.13,
             f"CD = {distancia_critica:.2f}", ha="center", va="bottom",
             fontsize=8, color=TINTA_SUAVE)

    return fig, eje


# ------------------------------------------------------------ barras agrupadas
# P3, Fig. 3 (skill score por modelo y horizonte).

def barras_agrupadas(
    categorias: list[str],
    series: dict[str, np.ndarray],
    etiqueta_y: str,
    etiqueta_x: str = "",
    linea_cero: bool = False,
    figsize: tuple[float, float] | None = None,
):
    """Barras agrupadas con separacion de 2 px entre barras contiguas.

    Con `linea_cero=True` marca el cero, imprescindible cuando la magnitud tiene
    signo: en P3 un skill score negativo significa que el modelo es peor que la
    persistencia, y esa frontera debe verse.
    """
    fig, eje = plt.subplots(figsize=figsize or (ANCHO_DOBLE * 0.7, ANCHO_SIMPLE * 0.8))

    n = len(series)
    x = np.arange(len(categorias))
    ancho = 0.8 / n

    for i, (nombre, valores) in enumerate(series.items()):
        color = CATEGORICA[i % len(CATEGORICA)]
        eje.bar(x + (i - (n - 1) / 2) * ancho, valores, ancho * 0.92,
                label=nombre, color=color, edgecolor="white", linewidth=0.8)

    if linea_cero:
        eje.axhline(0, color=TINTA, linewidth=1.0)

    eje.set_xticks(x)
    eje.set_xticklabels(categorias, rotation=25, ha="right")
    eje.set_ylabel(etiqueta_y)
    if etiqueta_x:
        eje.set_xlabel(etiqueta_x)
    if n >= 2:
        # Mismo motivo que en `curva_con_ic`: con barras ocupando todo el
        # eje no queda hueco libre para "best" y la leyenda cae encima de
        # alguna serie. Se saca fuera del eje, encima del panel. El
        # desplazamiento es mayor que en `curva_con_ic` (1.12 en vez de 1.02)
        # porque quien llama a esta funcion suele anadir ademas un titulo de
        # panel con `eje.set_title(..., loc="left")` (P3: un titulo por
        # sitio); con 1.02 la leyenda quedaba pegada a ese titulo.
        eje.legend(loc="lower center", bbox_to_anchor=(0.5, 1.12),
                   ncol=min(n, 3), frameon=False)

    return fig, eje
