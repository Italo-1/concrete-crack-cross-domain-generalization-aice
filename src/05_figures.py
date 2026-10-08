"""P1 (AICE) - figuras del manuscrito, desde `results/` y los datos cacheados.

No entrena nada: la Fig. 4 sale de las predicciones por imagen que guarda
`03_experiment.py`, no de un modelo reentrenado aparte (en la version anterior
lo era, y su conjunto de errores cambiaba entre ejecuciones; README de
03-tehnicki-glasnik-P1, bitacora 02/09).

    Fig. 1  Una muestra con grieta y una sin grieta por dominio.
    Fig. 2  Matriz de transferencia: media sobre arquitecturas (a) y una por
            arquitectura (b-e), con la misma escala de color.
    Fig. 3  F1 por arquitectura y particion direccional, IC bootstrap 95 % y
            referencia del clasificador que marca todo como grieta.
    Fig. 4  Falsos positivos y negativos del par entre campanas con menor F1,
            de un modelo de la matriz.
    Fig. 5  Precision frente a recall de cada celda (EXP2), con curvas de F1
            constante y el punto del clasificador trivial.
    Fig. 6  Caida relativa del ajuste fino frente a la sonda lineal (EXP3).
    Fig. 7  Multi-fuente frente a una sola fuente, por dominio excluido (EXP4).

Estilo: la paleta, los anchos y las letras de panel salen de `fieutils.figures`;
la tipografia se cambia aqui a sans-serif (Arial/Helvetica, 8-9 pt) y el PNG a
600 dpi, porque lo piden las guias de AI in Civil Engineering (JOURNAL.md). No
se toca `fieutils`.

    python src/05_figures.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from fieutils import figures as ffig
from fieutils import stats as fstats
from fieutils.config import SEMILLA, configurar_log

RAIZ = Path(__file__).resolve().parent.parent
DATOS = Path(r"C:\lineaB\03-tehnicki-glasnik-P1\data\processed")
TABLAS = RAIZ / "results" / "tables"
PREDICCIONES = RAIZ / "results" / "predicciones"
DESTINO = RAIZ / "results" / "figures"

DOMINIOS_CORTOS = {"sdnet_D": "SDNET-D", "sdnet_P": "SDNET-P", "sdnet_W": "SDNET-W", "metu": "METU"}
TRAMAS = ["", "///", "...", "xxx"]


def _cargar_modulo(nombre_archivo: str, alias: str):
    ruta = Path(__file__).resolve().parent / nombre_archivo
    spec = importlib.util.spec_from_file_location(alias, ruta)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


ST = _cargar_modulo("04_stats.py", "stats_aice")
DOMINIOS = ST.DOMINIOS
ARQ = ST.ARQUITECTURAS_LEGIBLES
DIR = ST.DIRECCION_LEGIBLE
ORDEN_DIRECCION = ST.ORDEN_DIRECCION


# ---------------------------------------------------------------- estilo

# Ancho de caja de texto de sn-jnl (372 pt = 131 mm): las figuras anchas se
# generan a 129 mm, uno de los anchos de Springer, para que el PDF no las
# reduzca y el texto se quede en 8 pt (a 180 mm se reducian al 73 %).
ANCHO_CAJA = 129 / 25.4
ETIQUETAS_CORTAS = ["D", "P", "W", "M"]


def estilo() -> None:
    ffig.aplicar_estilo()
    ffig.mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 8,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "pdf.fonttype": 42, "ps.fonttype": 42,     # fuentes incrustadas como TrueType
        "savefig.dpi": 600,
    })


def guardar(fig, nombre: str) -> None:
    DESTINO.mkdir(parents=True, exist_ok=True)
    fig.savefig(DESTINO / f"{nombre}.pdf", bbox_inches="tight")
    fig.savefig(DESTINO / f"{nombre}.png", dpi=600, bbox_inches="tight")
    ffig.plt.close(fig)


def cargar() -> tuple[pd.DataFrame, pd.DataFrame]:
    manifiesto = pd.read_csv(DATOS / "manifiesto.csv")
    r = ST.anotar(pd.read_csv(TABLAS / "resultados_matriz.csv"))
    base = r[(r["particion"] == "original") & (r["condicion"] == "baseline")].copy()
    return manifiesto, base


# ------------------------------------------------------------------ Fig 1

def figura_1(manifiesto: pd.DataFrame, log) -> None:
    estilo()
    fig, ejes = ffig.plt.subplots(2, 4, figsize=(ANCHO_CAJA, ANCHO_CAJA * 0.56),
                                  constrained_layout=True)
    for col, d in enumerate(DOMINIOS):
        X = np.load(DATOS / f"{d}_X.npy", mmap_mode="r")
        for fila, etiqueta in enumerate([1, 0]):
            sub = manifiesto[(manifiesto["dominio"] == d) & (manifiesto["etiqueta"] == etiqueta)]
            img = X[int(sub.iloc[len(sub) // 2]["indice_en_dominio"])]
            eje = ejes[fila, col]
            eje.imshow(img)
            eje.set_xticks([]); eje.set_yticks([])
            eje.set_title(ffig.etiqueta_panel(fila * 4 + col), pad=3)
        ejes[1, col].set_xlabel(ST.NOMBRES_LEGIBLES[d])
    ejes[0, 0].set_ylabel("Crack")
    ejes[1, 0].set_ylabel("No crack")
    guardar(fig, "fig1_muestras_dominios")
    log.info("Fig 1: muestra central de cada dominio y clase, recorte cacheado de 224x224.")


# ------------------------------------------------------------------ Fig 2

def _matriz(df: pd.DataFrame) -> np.ndarray:
    return (df.groupby(["dominio_train", "dominio_test"])["f1"].mean()
            .unstack().reindex(index=DOMINIOS, columns=DOMINIOS).values)


def _panel_calor(eje, m, vmin, vmax, etiquetas, tam, ejes_x=True, ejes_y=True, numeros=True,
                 rotacion=45):
    mapa = ffig.mpl.colormaps[ffig.SECUENCIAL]
    im = eje.imshow(m, cmap=ffig.SECUENCIAL, vmin=vmin, vmax=vmax, aspect="equal")
    eje.set_xticks(range(4)); eje.set_yticks(range(4))
    eje.set_xticklabels(etiquetas if ejes_x else [], rotation=rotacion,
                        ha="right" if rotacion else "center")
    eje.set_yticklabels(etiquetas if ejes_y else [])
    eje.grid(False)
    for i in range(4):
        eje.add_patch(ffig.plt.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False,
                                         edgecolor=ffig.TINTA, linewidth=1.1))
        for j in range(4 if numeros else 0):
            r_, g_, b_, _ = mapa((m[i, j] - vmin) / ((vmax - vmin) or 1))
            claro = 0.299 * r_ + 0.587 * g_ + 0.114 * b_ >= 0.55
            # Tres decimales: con dos, SDNET-D -> METU (0.666) se leia como
            # 0.67, el F1 del clasificador trivial (revision F5, ronda 2).
            eje.text(j, i, f"{m[i, j]:.3f}", ha="center", va="center", fontsize=tam,
                     color=ffig.TINTA if claro else "white")
    return im


def figura_2(base: pd.DataFrame, log) -> None:
    estilo()
    matrices = {"Mean over architectures": _matriz(base)}
    matrices.update({ARQ[a]: _matriz(base[base["arquitectura"] == a]) for a in ARQ})
    todos = np.concatenate([m.ravel() for m in matrices.values()])
    vmin, vmax = float(np.nanmin(todos)), float(np.nanmax(todos))
    etiquetas = [DOMINIOS_CORTOS[d] for d in DOMINIOS]

    fig = ffig.plt.figure(figsize=(ANCHO_CAJA, ANCHO_CAJA * 0.62), constrained_layout=True)
    gs = fig.add_gridspec(2, 4)
    grande = fig.add_subplot(gs[:, :2])
    im = _panel_calor(grande, matrices["Mean over architectures"], vmin, vmax, etiquetas, 8)
    grande.set_title(ffig.etiqueta_panel(0), pad=3)
    grande.set_ylabel("Training domain"); grande.set_xlabel("Test domain")
    for k, (pos, nombre) in enumerate(zip([gs[0, 2], gs[0, 3], gs[1, 2], gs[1, 3]], list(matrices)[1:])):
        eje = fig.add_subplot(pos)
        # Sin numeros en los paneles pequenos: a este tamano saldrian por
        # debajo de los 8 pt que piden las guias de la revista. Los valores por
        # arquitectura estan en la tabla del apendice; aqui el panel muestra el
        # patron con la misma escala de color que (a).
        # Etiquetas cortas (D, P, W, M) en los paneles pequenos: a 129 mm las
        # largas se solapan; el pie de figura da la equivalencia.
        _panel_calor(eje, matrices[nombre], vmin, vmax, ETIQUETAS_CORTAS, 8,
                     ejes_x=k >= 2, ejes_y=k % 2 == 0, numeros=False, rotacion=0)
        eje.set_title(ffig.etiqueta_panel(k + 1), pad=2)
    barra = fig.colorbar(im, ax=fig.axes, fraction=0.025, pad=0.01)
    barra.set_label("F1 (crack class)")
    barra.outline.set_visible(False)
    guardar(fig, "fig2_matriz_transferencia")
    log.info(f"Fig 2: escala comun {vmin:.3f}-{vmax:.3f}; paneles {list(matrices)}.")


# ------------------------------------------------------------------ Fig 3

def figura_3(base: pd.DataFrame, log) -> None:
    estilo()
    arqs = [a for a in ARQ if a in set(base["arquitectura"])]
    fig, eje = ffig.plt.subplots(figsize=(ANCHO_CAJA, ffig.ANCHO_SIMPLE * 0.85))
    x, n = np.arange(len(arqs)), len(ORDEN_DIRECCION)
    ancho = 0.8 / n
    for i, niv in enumerate(ORDEN_DIRECCION):
        med, lo, hi = [], [], []
        for a in arqs:
            v, li, ls = fstats.ic_bootstrap(
                base.loc[(base["arquitectura"] == a) & (base["direccion"] == niv), "f1"].values, semilla=SEMILLA)
            med.append(v); lo.append(v - li); hi.append(ls - v)
        pos = x + (i - (n - 1) / 2) * ancho
        eje.bar(pos, med, ancho * 0.92, label=DIR[niv], color=ffig.CATEGORICA[i],
                edgecolor="white", linewidth=0.6, hatch=TRAMAS[i])
        eje.errorbar(pos, med, yerr=[lo, hi], fmt="none", ecolor=ffig.TINTA, elinewidth=0.8, capsize=2)
    eje.axhline(ST.F1_TRIVIAL, color=ffig.TINTA, linestyle="--", linewidth=0.8)
    eje.set_xticks(x); eje.set_xticklabels([ARQ[a] for a in arqs])
    eje.set_ylabel("F1 (crack class)"); eje.set_ylim(0, 1.05)
    eje.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4, frameon=False)
    guardar(fig, "fig3_f1_por_particion")
    log.info("Fig 3: media e IC bootstrap 95 % (10 000 remuestreos) por arquitectura y particion.")


# ------------------------------------------------------------------ Fig 4

def figura_4(base: pd.DataFrame, log) -> None:
    """Errores de un modelo de la matriz en el par entre campanas con menor F1.

    Par: el (arquitectura, origen, destino) entre campanas con menor F1 medio.
    Semilla: la de F1 mediano entre las tres, para no mostrar el peor caso.
    Ejemplos: los tres falsos positivos con mayor probabilidad de grieta y los
    tres falsos negativos con menor, es decir, los errores mas seguros del
    modelo; regla fija, no eleccion a mano.
    """
    fuera = base[base["categoria"] == "entre_campana"]
    medias = fuera.groupby(["arquitectura", "dominio_train", "dominio_test"])["f1"].mean()
    arq, origen, destino = medias.idxmin()
    filas = fuera[(fuera["arquitectura"] == arq) & (fuera["dominio_train"] == origen)
                  & (fuera["dominio_test"] == destino)].sort_values("f1")
    fila = filas.iloc[len(filas) // 2]
    semilla = int(fila["semilla"])

    pred = pd.read_csv(PREDICCIONES / "baseline_original" / f"{arq}_s{semilla}_{origen}.csv.gz")
    pred = pred[pred["dominio_test"] == destino]
    fp = pred[(pred["etiqueta"] == 0) & (pred["pred"] == 1)].sort_values("prob_grieta", ascending=False).head(3)
    fn = pred[(pred["etiqueta"] == 1) & (pred["pred"] == 0)].sort_values("prob_grieta").head(3)
    conteo = {k: int(fila[k]) for k in ("tp", "fp", "fn", "tn")}
    assert conteo["fp"] == int(((pred["etiqueta"] == 0) & (pred["pred"] == 1)).sum()), "conteo FP inconsistente"
    pd.DataFrame([{"architecture": arq, "source": origen, "target": destino, "seed": semilla,
                   **conteo, "precision": float(fila["precision"]), "recall": float(fila["recall"]),
                   "f1": float(fila["f1"]), "pair_mean_f1": float(medias.min()),
                   "fp_shown": fp["indice_en_dominio"].tolist(), "fn_shown": fn["indice_en_dominio"].tolist()}]
                 ).to_csv(TABLAS / "fig4_conteo_errores.csv", index=False)

    X = np.load(DATOS / f"{destino}_X.npy", mmap_mode="r")
    n_col = max(len(fp), len(fn), 1)
    estilo()
    fig, ejes = ffig.plt.subplots(2, n_col, figsize=(ffig.ANCHO_SIMPLE * n_col / 3 * 1.3, ffig.ANCHO_SIMPLE * 0.9),
                                  constrained_layout=True, squeeze=False)
    k = 0
    for f, sel in enumerate([fp, fn]):
        for c in range(n_col):
            eje = ejes[f, c]
            if c < len(sel):
                fila_img = sel.iloc[c]
                eje.imshow(X[int(fila_img["indice_en_dominio"])])
                eje.set_title(f"{ffig.etiqueta_panel(k)} p = {fila_img['prob_grieta']:.2f}", pad=2)
                eje.set_xticks([]); eje.set_yticks([])
                k += 1
            else:
                eje.set_axis_off()
    ejes[0, 0].set_ylabel("False positive")
    ejes[1, 0].set_ylabel("False negative")
    guardar(fig, "fig4_errores")
    log.info(f"Fig 4: {ARQ[arq]} {origen}->{destino}, semilla {semilla} (F1 {fila['f1']:.3f}, media del par "
             f"{medias.min():.3f}); conteos {conteo}.")


# ------------------------------------------------------------------ Fig 5

def figura_5(base: pd.DataFrame, log) -> None:
    estilo()
    fig, eje = ffig.plt.subplots(figsize=(ffig.ANCHO_SIMPLE, ffig.ANCHO_SIMPLE * 0.92))
    r_ = np.linspace(0.01, 1, 200)
    # Iso-F1, con la del clasificador trivial (2/3) discontinua: es la
    # referencia central del articulo (revision F5). Etiquetas a 8 pt, el
    # minimo de AICE (antes 6 pt).
    for f1, etiqueta, trazo in ((0.4, "0.4", "-"), (2 / 3, "0.667", "--"), (0.8, "0.8", "-")):
        p_ = f1 * r_ / (2 * r_ - f1)
        ok = (p_ > 0) & (p_ <= 1)
        eje.plot(r_[ok], p_[ok], color=ffig.TINTA_SUAVE if trazo == "--" else ffig.REJILLA,
                 linewidth=0.8, linestyle=trazo, zorder=0)
        # La de 0.667 termina en la estrella del trivial: etiqueta algo mas arriba.
        y_texto = p_[ok][-1] + (0.06 if trazo == "--" else 0)
        eje.text(r_[ok][-1], y_texto, f" F1={etiqueta}", fontsize=8, color=ffig.TINTA_SUAVE, va="center")
    for i, niv in enumerate(ORDEN_DIRECCION):
        g = base[base["direccion"] == niv]
        eje.scatter(g["recall"], g["precision"], s=10, marker=ffig.MARCADORES[i], color=ffig.CATEGORICA[i],
                    alpha=0.8, linewidths=0, label=DIR[niv])
    eje.scatter([1.0], [0.5], marker="*", s=60, color=ffig.TINTA, zorder=5, label="Always-crack classifier")
    eje.set_xlabel("Recall (crack class)"); eje.set_ylabel("Precision (crack class)")
    eje.set_xlim(0, 1.03); eje.set_ylim(0, 1.03)
    eje.legend(loc="lower left", frameon=False)
    guardar(fig, "fig5_precision_recall")
    log.info(f"Fig 5: {len(base)} celdas de la condicion base.")


# ------------------------------------------------------------------ Fig 6

def figura_6(base: pd.DataFrame, log) -> None:
    ruta = TABLAS / "resultados_sonda.csv"
    if not ruta.exists():
        log.info("Fig 6: sin resultados de la sonda lineal todavia; se omite.")
        return
    sonda = ST.anotar(pd.read_csv(ruta))
    estilo()
    fig, eje = ffig.plt.subplots(figsize=(ffig.ANCHO_SIMPLE, ffig.ANCHO_SIMPLE * 0.75))
    x = np.arange(len(ST.FUERA))
    for i, (nombre, df) in enumerate((("Fine-tuning", base), ("Linear probe", sonda))):
        caida = ST.caida_relativa(df).groupby(ST.CORRIDA + ["direccion"])["caida"].mean().reset_index()
        med, lo, hi = [], [], []
        for niv in ST.FUERA:
            v, li, ls = fstats.ic_bootstrap(caida.loc[caida["direccion"] == niv, "caida"].values, semilla=SEMILLA)
            med.append(v); lo.append(v - li); hi.append(ls - v)
        pos = x + (i - 0.5) * 0.38
        eje.bar(pos, med, 0.36, color=ffig.CATEGORICA[i], hatch=TRAMAS[i], edgecolor="white",
                linewidth=0.6, label=nombre)
        eje.errorbar(pos, med, yerr=[lo, hi], fmt="none", ecolor=ffig.TINTA, elinewidth=0.8, capsize=2)
    eje.axhline(0, color=ffig.TINTA, linewidth=0.6)
    eje.set_xticks(x); eje.set_xticklabels([DIR[n] for n in ST.FUERA])
    eje.set_ylabel("Relative F1 drop")
    eje.legend(frameon=False, loc="upper left")
    guardar(fig, "fig6_sonda_frente_ajuste")
    log.info("Fig 6: caida relativa media por corrida, IC bootstrap 95 %.")


# ------------------------------------------------------------------ Fig 7

def figura_7(base: pd.DataFrame, log) -> None:
    ruta = TABLAS / "resultados_multifuente.csv"
    if not ruta.exists():
        log.info("Fig 7: sin resultados multi-fuente todavia; se omite.")
        return
    mf = pd.read_csv(ruta)
    loo = mf[mf["excluido"]]
    una = base[base["dominio_train"] != base["dominio_test"]]
    agg = (una.groupby(["arquitectura", "semilla", "dominio_test"])["f1"].agg(["mean", "max"]).reset_index())
    series = {
        "Single source (mean of three)": lambda d: agg.loc[agg["dominio_test"] == d, "mean"].values,
        "Single source (best of three)": lambda d: agg.loc[agg["dominio_test"] == d, "max"].values,
        "Three sources, same size": lambda d: loo.loc[loo["dominio_excluido"] == d, "f1"].values,
    }
    estilo()
    fig, eje = ffig.plt.subplots(figsize=(ffig.ANCHO_SIMPLE, ffig.ANCHO_SIMPLE * 0.75))
    x = np.arange(len(DOMINIOS))
    for i, (nombre, f) in enumerate(series.items()):
        med, lo, hi = [], [], []
        for d in DOMINIOS:
            v, li, ls = fstats.ic_bootstrap(f(d), semilla=SEMILLA)
            med.append(v); lo.append(v - li); hi.append(ls - v)
        pos = x + (i - 1) * 0.26
        eje.bar(pos, med, 0.24, color=ffig.CATEGORICA[i], hatch=TRAMAS[i], edgecolor="white",
                linewidth=0.6, label=nombre)
        eje.errorbar(pos, med, yerr=[lo, hi], fmt="none", ecolor=ffig.TINTA, elinewidth=0.8, capsize=2)
    eje.axhline(ST.F1_TRIVIAL, color=ffig.TINTA, linestyle="--", linewidth=0.8)
    eje.set_xticks(x); eje.set_xticklabels([DOMINIOS_CORTOS[d] for d in DOMINIOS])
    eje.set_xlabel("Held-out test domain"); eje.set_ylabel("F1 (crack class)"); eje.set_ylim(0, 1.05)
    eje.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=1)
    guardar(fig, "fig7_multifuente")
    log.info("Fig 7: por dominio excluido, media e IC bootstrap 95 % sobre arquitecturas y semillas.")


# ------------------------------------------------------------------ main

def main() -> int:
    log = configurar_log("05_figures", RAIZ)
    manifiesto, base = cargar()
    if base.empty:
        log.info("Sin filas de la condicion base con particion original.")
        return 1
    log.info(f"{len(base)} celdas de la condicion base (se esperan 192).")
    figura_1(manifiesto, log)
    figura_2(base, log)
    figura_3(base, log)
    figura_4(base, log)
    figura_5(base, log)
    figura_6(base, log)
    figura_7(base, log)
    log.info(f"Figuras en {DESTINO} (PDF vectorial con fuentes incrustadas y PNG a 600 dpi).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
