"""Exportacion de tablas en los dos formatos que exige la guia: .csv y .tex.

La guia (seccion 4.3) fija el formato de las tablas del manuscrito:

    - booktabs: \\toprule, \\midrule, \\bottomrule. Sin lineas verticales.
    - Mejor resultado por columna en negrita.
    - Media +/- desviacion estandar, no solo media.
    - Decimales consistentes en toda la tabla.
    - Unidades en el encabezado, no en cada celda.

Y la seccion 1.1 exige que `results/tables/` contenga tanto `.csv` como `.tex`.
El `.csv` es la fuente rastreable; el `.tex` se pega en el manuscrito.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _escapar(texto: str) -> str:
    """Escapa los caracteres que LaTeX interpreta."""
    reemplazos = {
        "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
        "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
        "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
    }
    for viejo, nuevo in reemplazos.items():
        texto = texto.replace(viejo, nuevo)
    return texto


def _texto_seguro(v) -> str:
    """`str(v)` a prueba de NaN.

    `serie.astype(str)` sobre una columna de tipo objeto con texto mezclado
    con NaN NO convierte el NaN a la cadena "nan": lo deja como `float`
    (comportamiento de pandas al mezclar `astype(str)` con nulos en columnas
    `object`). Una celda `float` se cuela hasta el `" & ".join(...)` final y
    truena con TypeError. Encontrado en P4 el 25/08 al exportar una tabla con
    una celda faltante (una combinacion representacion x SNR sin datos
    todavia). Afecta a los cinco articulos: cualquier tabla con una celda
    ausente en una columna no numerica.
    """
    return "--" if pd.isna(v) else str(v)


def media_desv(valores, decimales: int = 3) -> str:
    """Formatea 'media +/- desviacion' con decimales fijos, como pide la guia."""
    v = np.asarray(valores, dtype=float)
    v = v[~np.isnan(v)]
    if v.size == 0:
        return "--"
    if v.size == 1:
        return f"{v[0]:.{decimales}f}"
    return f"{v.mean():.{decimales}f} $\\pm$ {v.std(ddof=1):.{decimales}f}"


def media_desv_ic(valores, ic=None, decimales: int = 3) -> str:
    """Formatea 'media +/- desviacion [lo, hi]'.

    La guia separa dos reglas que es facil confundir, porque `REDACCION.md` las
    fusiono en una sola lista titulada "Figuras y tablas":

        - seccion 4.3, TABLAS:  "Media +/- desviacion estandar, no solo media."
        - seccion 4.2, FIGURAS: "Barras de error o intervalos de confianza
          siempre que se muestren promedios."

    La disyuntiva del intervalo de confianza pertenece a la regla de figuras, no
    a la de tablas. De ahi que una tabla que reporte solo `[95 % CI]` incumpla
    4.3 aunque el estimador sea mejor. Encontrado el 03/09 en la revision
    cruzada de P4 y P5, donde los dos revisores citaban bien la guia pero
    secciones distintas.

    La salida lleva las dos cosas: la desviacion estandar que pide 4.3 y el
    intervalo bootstrap ya calculado, que con tres semillas informa mas que una
    desviacion de n = 3. `ic` es la pareja (lo, hi); si se omite, esto degrada a
    `media_desv`.
    """
    v = np.asarray(valores, dtype=float)
    v = v[~np.isnan(v)]
    if v.size == 0:
        return "--"
    base = media_desv(v, decimales)
    if ic is None:
        return base
    lo, hi = ic
    if pd.isna(lo) or pd.isna(hi):
        return base
    return f"{base} [{float(lo):.{decimales}f}, {float(hi):.{decimales}f}]"


def exportar(
    df: pd.DataFrame,
    destino: str | Path,
    nombre: str,
    caption: str,
    etiqueta: str | None = None,
    decimales: int = 3,
    negrita_maximo: list[str] | None = None,
    negrita_minimo: list[str] | None = None,
    unidades: dict[str, str] | None = None,
    indice: bool = True,
    sin_escapar: list[str] | None = None,
) -> tuple[Path, Path]:
    """Escribe la tabla en `.csv` y en `.tex` con formato booktabs.

    `negrita_maximo` / `negrita_minimo` son las columnas cuyo mejor valor se
    resalta. Para F1 o exactitud interesa el maximo; para RMSE o error, el
    minimo. Si no se indica ninguna, no se resalta nada: es preferible eso a
    resaltar en la direccion equivocada.

    `unidades` mapea columna -> unidad, y se anade al encabezado entre
    corchetes, nunca a cada celda.

    `sin_escapar` lista las columnas de texto que ya traen LaTeX formado (por
    ejemplo "0.988 $\\pm$ 0.015"), para no escaparlas por segunda vez: sin esto,
    el `$` y la `\\` de una celda pre-formateada salen como texto literal en el
    PDF en vez de compilar como modo matematico.

    Soporta indice simple o MULTIINDICE. Con multiindice, cada nivel se vuelca
    en su propia columna de la izquierda con el nombre del nivel como
    encabezado; sin este tratamiento, `pandas` entrega la fila como una tupla de
    Python y el resultado literal `('Wavelet db4', 'MLP')` termina impreso en la
    tabla del manuscrito.

    Devuelve las rutas (csv, tex).
    """
    carpeta = Path(destino)
    carpeta.mkdir(parents=True, exist_ok=True)

    ruta_csv = carpeta / f"{nombre}.csv"
    df.to_csv(ruta_csv, index=indice)

    negrita_maximo = negrita_maximo or []
    negrita_minimo = negrita_minimo or []
    unidades = unidades or {}
    sin_escapar = sin_escapar or []

    # --- cuerpo con decimales consistentes y mejor valor en negrita ---------
    formateado = df.copy()
    for columna in formateado.columns:
        serie = formateado[columna]

        if not pd.api.types.is_numeric_dtype(serie):
            if columna in sin_escapar:
                formateado[columna] = serie.map(_texto_seguro)
            else:
                formateado[columna] = serie.map(_texto_seguro).map(_escapar)
            continue

        if columna in negrita_maximo:
            mejor = serie.max()
        elif columna in negrita_minimo:
            mejor = serie.min()
        else:
            mejor = None

        # Columnas de tipo entero (conteos: "n", "n_semillas"...) se formatean
        # sin decimales. Sin esto, `decimales` (pensado para metricas como F1
        # o exactitud) se aplica tambien a los conteos y una tabla con una
        # columna "n" imprime "30.000" en vez de "30" -encontrado el 27/08 en
        # P5, y ya presente sin detectar en la tabla 4 de P4 (Flicker F1, "n").
        es_entera = pd.api.types.is_integer_dtype(serie)

        def _celda(v, es_entera=es_entera, decimales=decimales, mejor=mejor):
            if pd.isna(v):
                return "--"
            texto = f"{int(v)}" if es_entera else f"{v:.{decimales}f}"
            if mejor is not None and np.isclose(v, mejor):
                return f"\\textbf{{{texto}}}"
            return texto

        formateado[columna] = [_celda(v) for v in serie]

    es_multiindice = indice and isinstance(df.index, pd.MultiIndex)
    n_niveles_indice = len(df.index.names) if es_multiindice else (1 if indice else 0)

    encabezados = []
    if es_multiindice:
        encabezados += [_escapar(str(n) or "") for n in df.index.names]
    elif indice:
        encabezados.append(_escapar(str(df.index.name or "")))
    for c in df.columns:
        texto = _escapar(str(c))
        if c in unidades:
            texto += f" [{_escapar(unidades[c])}]"
        encabezados.append(texto)

    n_columnas = len(df.columns) + n_niveles_indice
    # Columnas de etiquetas alineadas a la izquierda, valores a la derecha para
    # que los decimales queden en columna.
    alineacion = ("l" * n_niveles_indice + "r" * len(df.columns))

    lineas = [
        r"\begin{table}[htbp]",
        r"\centering",
        f"\\caption{{{caption}}}",
    ]
    if etiqueta:
        lineas.append(f"\\label{{{etiqueta}}}")
    lineas += [
        f"\\begin{{tabular}}{{{alineacion}}}",
        r"\toprule",
        " & ".join(encabezados) + r" \\",
        r"\midrule",
    ]

    fila_anterior = None
    for etiqueta_fila, fila in formateado.iterrows():
        if es_multiindice:
            # Un valor de nivel vacio cuando repite el de la fila anterior:
            # es la convencion habitual en tablas de revista para no repetir
            # la misma etiqueta de grupo en cada fila.
            niveles = []
            for i, valor in enumerate(etiqueta_fila):
                repite = fila_anterior is not None and fila_anterior[: i + 1] == etiqueta_fila[: i + 1]
                niveles.append("" if repite else _escapar(str(valor)))
            fila_anterior = etiqueta_fila
            etiquetas_fila = niveles
        elif indice:
            etiquetas_fila = [_escapar(str(etiqueta_fila))]
        else:
            etiquetas_fila = []

        celdas = etiquetas_fila + list(fila.astype(str))
        lineas.append(" & ".join(celdas) + r" \\")

    lineas += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]

    ruta_tex = carpeta / f"{nombre}.tex"
    ruta_tex.write_text("\n".join(lineas), encoding="utf-8")

    return ruta_csv, ruta_tex
