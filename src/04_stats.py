"""P1 (AICE) - estadistica de E1-E3 sobre los resultados nuevos.

Consume lo que escriben `03_experiment.py`, `03b_sonda_lineal.py` y
`03c_multifuente.py` en `results/tables/`. No entrena nada; solo agrega y
prueba. Todo numero que vaya al manuscrito sale de una tabla de aqui o de
`results/tables/resumen_estadistico.json`.

Pruebas fijadas en el README (protocolo, "Prueba estadistica") antes de ver los
resultados nuevos
------------------------------------------------------------------------------
1. Replica del analisis de la version anterior sobre la condicion base con la
   particion original (192 celdas): ANOVA de dos vias a priori (arquitectura x
   par, con interaccion) con Levene y Shapiro-Wilk; Tukey HSD sobre las tres
   categorias a priori; Welch + Holm entre las cuatro particiones
   direccionales con delta de Cliff (celda como unidad) y repetido con la
   corrida (arquitectura x origen x semilla) como unidad; IC bootstrap;
   caida relativa anclada en la diagonal del propio origen y semilla;
   mitigaciones emparejadas por (arquitectura, par, semilla).
2. EXP2: precision y recall por particion; precision - recall emparejada por
   corrida, Wilcoxon de rangos con signo, Holm sobre las tres particiones fuera
   de diagonal, correlacion biserial de rangos.
3. EXP3: caida relativa del ajuste fino frente a la de la sonda lineal,
   emparejada por corrida y particion; Wilcoxon, Holm sobre tres particiones,
   biserial de rangos, IC bootstrap de la diferencia media.
4. EXP4: F1 del multi-fuente en el dominio excluido frente a la media y al
   maximo de los tres F1 de una sola fuente hacia ese dominio (mismas
   arquitectura y semilla); Wilcoxon sobre 48 parejas, Holm sobre dos
   comparaciones, biserial de rangos, IC bootstrap; desglose por dominio.
5. Sensibilidad a la fuga entre parches: condicion base con la particion
   agrupada por foto (SDNET2018) frente a la original.

Cada bloque se salta con un aviso si sus resultados aun no existen, para poder
correr el script con resultados parciales.

    python src/04_stats.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sstats

from fieutils import stats as fstats
from fieutils import tablas
from fieutils.config import SEMILLA, configurar_log

RAIZ = Path(__file__).resolve().parent.parent
TABLAS = RAIZ / "results" / "tables"

DOMINIOS = ["sdnet_D", "sdnet_P", "sdnet_W", "metu"]
NOMBRES_LEGIBLES = {"sdnet_D": "SDNET-Deck", "sdnet_P": "SDNET-Pavement",
                    "sdnet_W": "SDNET-Wall", "metu": "METU"}
ARQUITECTURAS_LEGIBLES = {
    "resnet18": "ResNet-18", "efficientnet_b0": "EfficientNet-B0",
    "mobilenetv3_small_100": "MobileNetV3-Small", "vit_tiny_patch16_224": "ViT-Tiny",
}
ORDEN_ARQ = list(ARQUITECTURAS_LEGIBLES)
CATEGORIA_LEGIBLE = {"dentro": "Within-domain", "entre_superficie": "Cross-surface",
                     "entre_campana": "Cross-campaign"}
ORDEN_CATEGORIA = ["dentro", "entre_superficie", "entre_campana"]
DIRECCION_LEGIBLE = {"dentro": "Within-domain", "entre_superficie": "Cross-surface",
                     "sdnet_a_metu": "SDNET to METU", "metu_a_sdnet": "METU to SDNET"}
ORDEN_DIRECCION = ["dentro", "entre_superficie", "sdnet_a_metu", "metu_a_sdnet"]
FUERA = ORDEN_DIRECCION[1:]
CONDICION_LEGIBLE = {"agresivo": "Aggressive augmentation", "ecualizado": "Histogram equalization"}

# F1 de la clase grieta de un clasificador que responde "grieta" a todo, sobre
# los tests de este diseno (300 grieta / 300 no grieta en cada dominio):
# precision 0.5, recall 1.0.
F1_TRIVIAL = 2 * 0.5 * 1.0 / (0.5 + 1.0)
CORRIDA = ["arquitectura", "semilla", "dominio_train"]


# ------------------------------------------------------------------ utilidades

def formatear_p(valor: float, decimales: int = 4) -> str:
    """'$<$0.0001' en vez de '0.0000', que en un manuscrito es enganoso."""
    umbral = 10 ** (-decimales)
    return f"$<${umbral:.{decimales}f}" if valor < umbral else f"{valor:.{decimales}f}"


def media_de(valores, negrita: bool = False, decimales: int = 3) -> str:
    v = np.asarray(valores, dtype=float)
    texto = f"{v.mean():.{decimales}f} $\\pm$ {v.std(ddof=1):.{decimales}f}"
    return f"\\textbf{{{texto}}}" if negrita else texto


def holm(p: list[float]) -> list[float]:
    orden, m, corriente, salida = np.argsort(p), len(p), 0.0, [0.0] * len(p)
    for rango, i in enumerate(orden):
        corriente = max(corriente, min(1.0, p[i] * (m - rango)))
        salida[i] = corriente
    return salida


def wilcoxon_emparejado(dif: np.ndarray) -> dict:
    """Wilcoxon de rangos con signo y correlacion biserial de rangos emparejada.

    r = (suma de rangos positivos - suma de rangos negativos) / suma total,
    sobre las diferencias distintas de cero (Kerby, 2014). Va de -1 a 1.
    """
    dif = np.asarray(dif, dtype=float)
    nz = dif[dif != 0]
    if len(nz) == 0:
        return {"n": len(dif), "W": np.nan, "p": 1.0, "r_rb": 0.0}
    rangos = sstats.rankdata(np.abs(nz))
    r_rb = (rangos[nz > 0].sum() - rangos[nz < 0].sum()) / rangos.sum()
    w = sstats.wilcoxon(dif, zero_method="wilcox", alternative="two-sided")
    return {"n": len(dif), "W": float(w.statistic), "p": float(w.pvalue), "r_rb": float(r_rb)}


def categoria_par(train: str, test: str) -> str:
    if train == test:
        return "dentro"
    if train.startswith("sdnet_") and test.startswith("sdnet_"):
        return "entre_superficie"
    return "entre_campana"


def direccion_par(train: str, test: str) -> str:
    cat = categoria_par(train, test)
    if cat != "entre_campana":
        return cat
    return "metu_a_sdnet" if train == "metu" else "sdnet_a_metu"


def anotar(r: pd.DataFrame) -> pd.DataFrame:
    r = r.copy()
    r["categoria"] = [categoria_par(t, d) for t, d in zip(r["dominio_train"], r["dominio_test"])]
    r["direccion"] = [direccion_par(t, d) for t, d in zip(r["dominio_train"], r["dominio_test"])]
    r["par"] = r["dominio_train"] + "->" + r["dominio_test"]
    return r


def leer(nombre: str, log) -> pd.DataFrame | None:
    ruta = TABLAS / nombre
    if not ruta.exists():
        log.info(f"AVISO: {nombre} no existe todavia; se salta lo que depende de el.")
        return None
    return pd.read_csv(ruta)


def completa(sub: pd.DataFrame, nombre: str, esperadas: int, log) -> bool:
    ok = len(sub) == esperadas
    log.info(f"{nombre}: {len(sub)} filas ({'completa' if ok else f'INCOMPLETA, se esperaban {esperadas}'}).")
    return ok


def caida_relativa(df: pd.DataFrame) -> pd.DataFrame:
    """Celdas fuera de diagonal con su caida relativa anclada en la diagonal del
    propio origen, misma arquitectura y semilla."""
    diag = df[df["dominio_train"] == df["dominio_test"]].set_index(CORRIDA)["f1"]
    fuera = df[df["dominio_train"] != df["dominio_test"]].copy()
    fuera["f1_origen"] = [diag.loc[k] for k in zip(*(fuera[c] for c in CORRIDA))]
    fuera["caida"] = [fstats.caida_relativa(d, f) for d, f in zip(fuera["f1_origen"], fuera["f1"])]
    return fuera


# ================================================== 1. replica (condicion base)

def tabla_f1_por_particion(base: pd.DataFrame) -> dict:
    """Tabla 1: F1 medio +- DE por arquitectura y particion direccional."""
    filas, medias = [], {}
    for niv in ORDEN_DIRECCION:
        medias[niv] = {a: base.loc[(base["arquitectura"] == a) & (base["direccion"] == niv), "f1"].mean()
                       for a in ORDEN_ARQ}
    for a in ORDEN_ARQ:
        fila = {"Architecture": ARQUITECTURAS_LEGIBLES[a]}
        for niv in ORDEN_DIRECCION:
            mejor = np.isclose(medias[niv][a], max(medias[niv].values()))
            fila[DIRECCION_LEGIBLE[niv]] = media_de(
                base.loc[(base["arquitectura"] == a) & (base["direccion"] == niv), "f1"], negrita=mejor)
        filas.append(fila)
    tabla = pd.DataFrame(filas).set_index("Architecture")
    tablas.exportar(
        tabla, TABLAS, "tabla_f1_por_particion",
        caption=("F1-score of the crack class (mean $\\pm$ SD over cells and 3 seeds) by architecture "
                 "and directional transfer partition, baseline condition. Best value per column in bold."),
        etiqueta="tab:f1-particion", indice=True, sin_escapar=list(tabla.columns))
    return {niv: {a: float(v) for a, v in d.items()} for niv, d in medias.items()}


def anova_y_supuestos(base: pd.DataFrame, log) -> dict:
    log.info("=== ANOVA a priori: arquitectura x par, con interaccion ===")
    datos = base[["arquitectura", "par", "semilla", "f1"]].copy()
    res = fstats.anova_factorial(datos, dependiente="f1", factores=["arquitectura", "par"], interaccion=True)
    log.info(f"\n{res.round(4).to_string()}")
    res.round(6).to_csv(TABLAS / "anova_dos_factores.csv")

    factores = {"C(arquitectura)": "Architecture", "C(par)": "Train-test pair",
                "C(arquitectura):C(par)": "Architecture x Pair", "Residual": "Residual"}
    t = res[["df", "F", "PR(>F)", "eta_sq_parcial"]].rename(index=factores).copy()
    t["df"] = t["df"].map(lambda v: "--" if pd.isna(v) else str(int(v)))
    t["PR(>F)"] = t["PR(>F)"].map(lambda v: "--" if pd.isna(v) else formatear_p(v))
    t = t.rename(columns={"PR(>F)": "p", "eta_sq_parcial": "Partial eta sq."})
    tablas.exportar(t, TABLAS, "anova_dos_factores_tex",
                    caption=("Two-way ANOVA fixed a priori on F1-score: architecture (4 levels) x train-test "
                             "pair (16 levels), with interaction; seed is the replicate (192 observations). "
                             "Assumption tests fail (see text); conclusions rest on the directional contrasts."),
                    etiqueta="tab:anova", decimales=2, indice=True, sin_escapar=["df", "p"])

    grupos = [base.loc[base["categoria"] == c, "f1"].values for c in ORDEN_CATEGORIA]
    lev = sstats.levene(*grupos)
    residuos = base["f1"] - base.groupby(["arquitectura", "par"])["f1"].transform("mean")
    sw = sstats.shapiro(residuos.values)
    var = {c: float(g.var(ddof=1)) for c, g in zip(ORDEN_CATEGORIA, grupos)}
    log.info(f"Levene W={lev.statistic:.3f} p={lev.pvalue:.2e}; Shapiro W={sw.statistic:.3f} p={sw.pvalue:.2e}; var {var}")
    return {
        "anova": {k: {"df": float(res.loc[k, "df"]), "F": float(res.loc[k, "F"]) if pd.notna(res.loc[k, "F"]) else None,
                      "p": float(res.loc[k, "PR(>F)"]) if pd.notna(res.loc[k, "PR(>F)"]) else None,
                      "eta2p": float(res.loc[k, "eta_sq_parcial"]) if pd.notna(res.loc[k, "eta_sq_parcial"]) else None}
                  for k in res.index},
        "levene_W": float(lev.statistic), "levene_p": float(lev.pvalue),
        "shapiro_W": float(sw.statistic), "shapiro_p": float(sw.pvalue), "varianza_por_categoria": var,
    }


def tukey(base: pd.DataFrame, log) -> dict:
    res = fstats.tukey_hsd(base, dependiente="f1", grupo="categoria")
    t = pd.DataFrame(res._results_table.data[1:], columns=res._results_table.data[0])
    log.info(f"Tukey HSD (a priori, 3 categorias):\n{t.to_string()}")
    t.to_csv(TABLAS / "tukey_categoria.csv", index=False)
    return {f"{a} vs {b}": {"diff": float(d), "p": float(p), "ci": [float(lo), float(hi)]}
            for a, b, d, p, lo, hi in zip(t["group1"], t["group2"], t["meandiff"], t["p-adj"], t["lower"], t["upper"])}


def direccional(base: pd.DataFrame, log) -> dict:
    """Seis contrastes Welch + Holm entre particiones: celda y corrida como unidad."""
    salida = {}
    for unidad in ("celda", "corrida"):
        datos = base if unidad == "celda" else (
            base.groupby(CORRIDA + ["direccion"])["f1"].mean().reset_index())
        filas, p = [], []
        for i, a in enumerate(ORDEN_DIRECCION):
            for b in ORDEN_DIRECCION[i + 1:]:
                xa = datos.loc[datos["direccion"] == a, "f1"].values
                xb = datos.loc[datos["direccion"] == b, "f1"].values
                t = sstats.ttest_ind(xa, xb, equal_var=False)
                delta, magnitud = fstats.cliffs_delta(xa, xb)
                filas.append({"Group 1": DIRECCION_LEGIBLE[a], "Group 2": DIRECCION_LEGIBLE[b],
                              "n 1": len(xa), "n 2": len(xb), "Mean diff.": float(xa.mean() - xb.mean()),
                              "t (Welch)": float(t.statistic), "Cliff's delta": float(delta),
                              "Magnitude": magnitud})
                p.append(float(t.pvalue))
        for f, q in zip(filas, holm(p)):
            f["p_holm"] = q
        salida[unidad] = filas
        tabla = pd.DataFrame(filas)
        tabla["p (Holm)"] = tabla.pop("p_holm").map(formatear_p)
        if unidad == "corrida":
            tabla = tabla.drop(columns=["Cliff's delta", "Magnitude"])
        for f in filas:
            log.info(f"[{unidad}] {f['Group 1']} vs {f['Group 2']}: {f['Mean diff.']:+.4f}, "
                     f"p_Holm={f['p_holm']:.2e}")
        nombre = "direccional_welch" if unidad == "celda" else "direccional_welch_corrida"
        tablas.exportar(
            tabla.set_index(["Group 1", "Group 2"]), TABLAS, nombre,
            caption=("Pairwise comparison of the four directional partitions: Welch's t-test with Holm "
                     "correction" + (" and Cliff's delta, cell as unit." if unidad == "celda" else
                                     ", training run (architecture x source x seed) as unit, averaging "
                                     "the cells each run contributes to a partition.")),
            etiqueta=f"tab:{nombre.replace('_', '-')}", decimales=3, indice=True, sin_escapar=["p (Holm)"])
    return salida


def resumen_direccional(base: pd.DataFrame, log) -> dict:
    """F1, IC bootstrap, DE y porcentaje de celdas sobre el clasificador trivial."""
    filas, salida = [], {}
    for niv in ORDEN_DIRECCION:
        g = base.loc[base["direccion"] == niv, "f1"].values
        v, li, ls = fstats.ic_bootstrap(g, semilla=SEMILLA)
        sobre = float(100 * (g > F1_TRIVIAL).mean())
        salida[niv] = {"n": int(len(g)), "f1": float(v), "ci": [float(li), float(ls)],
                       "sd": float(g.std(ddof=1)), "pct_sobre_trivial": sobre}
        filas.append({"Partition": DIRECCION_LEGIBLE[niv], "n": int(len(g)),
                      "F1": media_de(g), "95% CI": f"{li:.3f}, {ls:.3f}",
                      "Cells > 0.667 (%)": round(sobre, 1)})
        log.info(f"[{niv}] F1 {v:.4f} [{li:.4f}, {ls:.4f}], {sobre:.1f}% > trivial")
    fuera = base[base["direccion"] != "dentro"]
    salida["fuera_pct_sobre_trivial"] = float(100 * (fuera["f1"] > F1_TRIVIAL).mean())
    salida["fuera_pct_igual_o_bajo_trivial"] = 100 - salida["fuera_pct_sobre_trivial"]
    salida["entre_campana_agrupado"] = float(base.loc[base["categoria"] == "entre_campana", "f1"].mean())
    tablas.exportar(
        pd.DataFrame(filas).set_index("Partition"), TABLAS, "resumen_direccional",
        caption=("F1-score of the crack class by directional transfer partition (mean $\\pm$ SD), with "
                 "percentile bootstrap 95\\% CI and the share of cells above F1 = 0.667, the score of a "
                 "classifier that labels every image as cracked on these balanced test sets."),
        etiqueta="tab:direccional-resumen", decimales=1, indice=True, sin_escapar=["F1"])
    return salida


def tabla_caida_relativa(base: pd.DataFrame, log) -> dict:
    fuera = caida_relativa(base)
    filas, salida = [], {}
    medias = {niv: {a: fuera.loc[(fuera["arquitectura"] == a) & (fuera["direccion"] == niv), "caida"].mean()
                    for a in ORDEN_ARQ} for niv in FUERA}
    for a in ORDEN_ARQ:
        fila = {"Architecture": ARQUITECTURAS_LEGIBLES[a]}
        for niv in FUERA:
            g = fuera.loc[(fuera["arquitectura"] == a) & (fuera["direccion"] == niv), "caida"]
            fila[DIRECCION_LEGIBLE[niv]] = media_de(g, negrita=np.isclose(medias[niv][a], min(medias[niv].values())))
        filas.append(fila)
    for niv in FUERA:
        vals = medias[niv]
        salida[niv] = {"por_arquitectura": {a: float(v) for a, v in vals.items()},
                       "rango_entre_arquitecturas": float(max(vals.values()) - min(vals.values()))}
    salida["rango_entre_particiones_por_arquitectura"] = {
        a: float(max(medias[n][a] for n in FUERA) - min(medias[n][a] for n in FUERA)) for a in ORDEN_ARQ}
    tabla = pd.DataFrame(filas).set_index("Architecture")
    tablas.exportar(
        tabla, TABLAS, "caida_relativa",
        caption=("Relative F1 drop per cell, (F1 on the source domain - F1 on the target) / F1 on the "
                 "source domain, anchored on the diagonal of the cell's own source domain and seed; mean "
                 "$\\pm$ SD by architecture and directional partition. Smallest drop per column in bold."),
        etiqueta="tab:caida-relativa", indice=True, sin_escapar=list(tabla.columns))
    log.info(f"caida relativa: {json.dumps(salida, indent=1)}")
    return salida


def mitigaciones(r: pd.DataFrame, log) -> dict:
    base_fuera = r[(r["condicion"] == "baseline") & (r["categoria"] != "dentro")]
    filas, salida = [], {}
    for cond in ("agresivo", "ecualizado"):
        mit = r[(r["condicion"] == cond) & (r["categoria"] != "dentro")]
        if mit.empty:
            log.info(f"Sin filas de '{cond}' todavia.")
            continue
        e = base_fuera.merge(mit, on=["arquitectura", "par", "semilla"], suffixes=("_b", "_m"))
        dif = e["f1_m"].values - e["f1_b"].values
        v, li, ls = fstats.ic_bootstrap(dif, semilla=SEMILLA)
        delta, mag = fstats.cliffs_delta(e["f1_m"].values, e["f1_b"].values)
        diag_b = r[(r["condicion"] == "baseline") & (r["categoria"] == "dentro")]["f1"].mean()
        diag_m = r[(r["condicion"] == cond) & (r["categoria"] == "dentro")]["f1"].mean()
        salida[cond] = {"n": int(len(e)), "dif_media": float(v), "ic95": [float(li), float(ls)],
                        "cliff": float(delta), "magnitud": mag, "diagonal_base": float(diag_b),
                        "diagonal_condicion": float(diag_m)}
        filas.append({"Condition": CONDICION_LEGIBLE[cond], "n pairs": int(len(e)),
                      "F1 baseline": float(e["f1_b"].mean()), "F1 intervention": float(e["f1_m"].mean()),
                      "Mean diff.": float(v), "95% CI": f"{li:+.3f}, {ls:+.3f}",
                      "Cliff's delta": float(delta)})
        log.info(f"{cond}: {v:+.4f} [{li:+.4f}, {ls:+.4f}], delta {delta:+.3f} ({mag})")
    if filas:
        tablas.exportar(
            pd.DataFrame(filas).set_index("Condition"), TABLAS, "mitigacion_resumen",
            caption=("Training-time interventions against the baseline on the 144 out-of-domain cells: paired "
                     "F1 difference (intervention minus baseline) matched by architecture, domain pair and seed, "
                     "with 95\\% bootstrap CI and Cliff's delta."),
            etiqueta="tab:intervenciones", decimales=3, indice=True)
    return salida


# =========================================================== 2. EXP2: P y R

def precision_recall(base: pd.DataFrame, log) -> dict:
    log.info("=== EXP2: precision y recall por particion ===")
    sin_pred = int(((base["tp"] + base["fp"]) == 0).sum())
    filas, salida = [], {"celdas_sin_predicciones_positivas": sin_pred}
    for niv in ORDEN_DIRECCION:
        g = base[base["direccion"] == niv]
        filas.append({"Partition": DIRECCION_LEGIBLE[niv], "n": int(len(g)),
                      "Precision": media_de(g["precision"]), "Recall": media_de(g["recall"]),
                      "F1": media_de(g["f1"]),
                      "FN share of errors (%)": round(float(100 * g["fn"].sum() / max((g["fn"] + g["fp"]).sum(), 1)), 1)})
        salida[niv] = {"precision": float(g["precision"].mean()), "recall": float(g["recall"].mean()),
                       "fn_sobre_errores_pct": filas[-1]["FN share of errors (%)"]}
    # Por arquitectura (apendice / texto)
    pr_arq = (base.groupby(["arquitectura", "direccion"])[["precision", "recall"]].mean()
              .reset_index())
    pr_arq.to_csv(TABLAS / "precision_recall_por_arquitectura.csv", index=False)

    # H-EXP2: precision - recall emparejada por corrida.
    corr = base.groupby(CORRIDA + ["direccion"])[["precision", "recall"]].mean().reset_index()
    pruebas, p = [], []
    for niv in FUERA:
        g = corr[corr["direccion"] == niv]
        dif = (g["precision"] - g["recall"]).values
        w = wilcoxon_emparejado(dif)
        v, li, ls = fstats.ic_bootstrap(dif, semilla=SEMILLA)
        w.update({"particion": niv, "dif_media": float(v), "ic95": [float(li), float(ls)]})
        pruebas.append(w)
        p.append(w["p"])
    for w, q in zip(pruebas, holm(p)):
        w["p_holm"] = q
        log.info(f"P-R [{w['particion']}]: n={w['n']} dif={w['dif_media']:+.3f} "
                 f"[{w['ic95'][0]:+.3f}, {w['ic95'][1]:+.3f}] p_Holm={q:.2e} r_rb={w['r_rb']:+.2f}")
    salida["wilcoxon_precision_menos_recall"] = pruebas
    for f in filas:
        clave = next(k for k, v in DIRECCION_LEGIBLE.items() if v == f["Partition"])
        prueba = next((w for w in pruebas if w["particion"] == clave), None)
        f["P - R (runs)"] = "--" if prueba is None else f"{prueba['dif_media']:+.3f}"
        f["p (Holm)"] = "--" if prueba is None else formatear_p(prueba["p_holm"])
    tablas.exportar(
        pd.DataFrame(filas).set_index("Partition"), TABLAS, "precision_recall_particion",
        caption=("Precision, recall and F1 of the crack class by directional partition (mean $\\pm$ SD over "
                 "cells), share of errors that are false negatives, and the paired precision - recall "
                 "difference per training run with Wilcoxon signed-rank test (Holm over the three "
                 "out-of-domain partitions)."),
        etiqueta="tab:precision-recall", decimales=3, indice=True,
        sin_escapar=["Precision", "Recall", "F1", "p (Holm)"])
    return salida


# ===================================================== 3. EXP3: sonda lineal

def sonda_frente_ajuste(base: pd.DataFrame, sonda: pd.DataFrame, log) -> dict:
    log.info("=== EXP3: sonda lineal frente a ajuste fino ===")
    cf = caida_relativa(base).groupby(CORRIDA + ["direccion"])[["caida", "f1"]].mean().reset_index()
    cs = caida_relativa(sonda).groupby(CORRIDA + ["direccion"])[["caida", "f1"]].mean().reset_index()
    e = cf.merge(cs, on=CORRIDA + ["direccion"], suffixes=("_ft", "_lp"))
    diag = {reg: df[df["direccion"] == "dentro"]["f1"].mean() for reg, df in (("ft", base), ("lp", sonda))}
    filas, pruebas, p = [], [], []
    for niv in FUERA:
        g = e[e["direccion"] == niv]
        dif = (g["caida_ft"] - g["caida_lp"]).values
        w = wilcoxon_emparejado(dif)
        v, li, ls = fstats.ic_bootstrap(dif, semilla=SEMILLA)
        df1 = (g["f1_ft"] - g["f1_lp"]).values
        w1 = wilcoxon_emparejado(df1)
        w.update({"particion": niv, "dif_media_caida": float(v), "ic95": [float(li), float(ls)],
                  "caida_ft": float(g["caida_ft"].mean()), "caida_lp": float(g["caida_lp"].mean()),
                  "f1_ft": float(g["f1_ft"].mean()), "f1_lp": float(g["f1_lp"].mean()),
                  "f1_dif_media": float(df1.mean()), "f1_wilcoxon_p": w1["p"], "f1_r_rb": w1["r_rb"]})
        pruebas.append(w)
        p.append(w["p"])
    for w, q in zip(pruebas, holm(p)):
        w["p_holm"] = q
        log.info(f"[{w['particion']}] caida FT {w['caida_ft']:.3f} vs LP {w['caida_lp']:.3f}: dif "
                 f"{w['dif_media_caida']:+.3f} [{w['ic95'][0]:+.3f}, {w['ic95'][1]:+.3f}] "
                 f"p_Holm={q:.2e} r_rb={w['r_rb']:+.2f}; F1 FT {w['f1_ft']:.3f} vs LP {w['f1_lp']:.3f}")
        filas.append({"Partition": DIRECCION_LEGIBLE[w["particion"]], "Runs": int(w["n"]),
                      "F1 FT": w["f1_ft"], "F1 LP": w["f1_lp"],
                      "Drop FT": w["caida_ft"], "Drop LP": w["caida_lp"],
                      "Drop FT - LP": w["dif_media_caida"], "95% CI": f"{w['ic95'][0]:+.3f}, {w['ic95'][1]:+.3f}",
                      "p (Holm)": formatear_p(q), "r_rb": w["r_rb"]})
    tablas.exportar(
        pd.DataFrame(filas).set_index("Partition"), TABLAS, "sonda_frente_ajuste",
        caption=(f"Linear probe on frozen ImageNet features (LP) against full fine-tuning (FT). Within-domain "
                 f"F1: FT {diag['ft']:.3f}, LP {diag['lp']:.3f}. Per training run and partition: mean "
                 "out-of-domain F1 and relative drop for each regime, paired difference of the drops with 95\\% "
                 "bootstrap CI, Wilcoxon signed-rank test (Holm over three partitions) and matched-pairs "
                 "rank-biserial correlation."),
        etiqueta="tab:sonda", decimales=3, indice=True, sin_escapar=["p (Holm)"])
    return {"diagonal_ft": float(diag["ft"]), "diagonal_lp": float(diag["lp"]), "pruebas": pruebas}


# ===================================================== 4. EXP4: multi-fuente

def multifuente(base: pd.DataFrame, mf: pd.DataFrame, log) -> dict:
    log.info("=== EXP4: tres fuentes frente a una ===")
    loo = mf[mf["excluido"]].rename(columns={"f1": "f1_loo"})
    una = base[base["dominio_train"] != base["dominio_test"]]
    agg = (una.groupby(["arquitectura", "semilla", "dominio_test"])["f1"]
           .agg(f1_media_una="mean", f1_max_una="max").reset_index()
           .rename(columns={"dominio_test": "dominio_excluido"}))
    e = loo.merge(agg, on=["arquitectura", "semilla", "dominio_excluido"])
    pruebas, p = [], []
    for ref in ("f1_media_una", "f1_max_una"):
        dif = (e["f1_loo"] - e[ref]).values
        w = wilcoxon_emparejado(dif)
        v, li, ls = fstats.ic_bootstrap(dif, semilla=SEMILLA)
        w.update({"referencia": ref, "dif_media": float(v), "ic95": [float(li), float(ls)]})
        pruebas.append(w)
        p.append(w["p"])
    for w, q in zip(pruebas, holm(p)):
        w["p_holm"] = q
        log.info(f"LOO - {w['referencia']}: n={w['n']} {w['dif_media']:+.3f} [{w['ic95'][0]:+.3f}, "
                 f"{w['ic95'][1]:+.3f}] p_Holm={q:.2e} r_rb={w['r_rb']:+.2f}")
    filas = []
    for d in DOMINIOS:
        g = e[e["dominio_excluido"] == d]
        filas.append({"Held-out domain": NOMBRES_LEGIBLES[d], "n": int(len(g)),
                      "F1 three sources": media_de(g["f1_loo"]),
                      "F1 single source (mean)": media_de(g["f1_media_una"]),
                      "F1 single source (best)": media_de(g["f1_max_una"]),
                      "Recall three sources": media_de(mf[mf["excluido"] & (mf["dominio_excluido"] == d)]["recall"])})
    tablas.exportar(
        pd.DataFrame(filas).set_index("Held-out domain"), TABLAS, "multifuente",
        caption=("Leave-one-domain-out training on the other three domains, subsampled to the size of a "
                 "single-source training set (2800 images), against the three single-source models "
                 "transferring to the same held-out domain (same architecture and seed). Mean $\\pm$ SD over "
                 "4 architectures x 3 seeds."),
        etiqueta="tab:multifuente", indice=True,
        sin_escapar=["F1 three sources", "F1 single source (mean)", "F1 single source (best)", "Recall three sources"])
    return {"pruebas": pruebas,
            "por_dominio": {d: {"f1_loo": float(e.loc[e["dominio_excluido"] == d, "f1_loo"].mean()),
                                "f1_media_una": float(e.loc[e["dominio_excluido"] == d, "f1_media_una"].mean()),
                                "f1_max_una": float(e.loc[e["dominio_excluido"] == d, "f1_max_una"].mean())}
                            for d in DOMINIOS}}


# ========================================= 5. sensibilidad: particion agrupada

def sensibilidad_particion(original: pd.DataFrame, agrupada: pd.DataFrame, log) -> dict:
    log.info("=== Sensibilidad: particion agrupada por foto (SDNET2018) ===")
    clave = ["arquitectura", "semilla", "dominio_train", "dominio_test"]
    e = original.merge(agrupada, on=clave, suffixes=("_orig", "_agr"))
    e = anotar(e.rename(columns={"dominio_train": "dominio_train", "dominio_test": "dominio_test"}))
    salida, filas = {}, []
    for d in DOMINIOS:
        g = e[(e["dominio_train"] == d) & (e["dominio_test"] == d)]
        dif = (g["f1_agr"] - g["f1_orig"]).values
        v, li, ls = fstats.ic_bootstrap(dif, semilla=SEMILLA)
        salida[d] = {"diag_orig": float(g["f1_orig"].mean()), "diag_agr": float(g["f1_agr"].mean()),
                     "dif": float(v), "ic95": [float(li), float(ls)]}
        filas.append({"Domain": NOMBRES_LEGIBLES[d], "Diagonal, patch split": float(g["f1_orig"].mean()),
                      "Diagonal, photo-grouped split": float(g["f1_agr"].mean()),
                      "Difference": float(v), "95% CI": f"{li:+.3f}, {ls:+.3f}"})
    agr = anotar(agrupada)
    salida["particiones_agrupada"] = {niv: float(agr.loc[agr["direccion"] == niv, "f1"].mean()) for niv in ORDEN_DIRECCION}
    salida["contraste_corrida_agrupada"] = direccional_sin_tablas(agr)
    caida_o = caida_relativa(anotar(original)).groupby("direccion")["caida"].mean()
    caida_a = caida_relativa(agr).groupby("direccion")["caida"].mean()
    salida["caida_relativa"] = {niv: {"orig": float(caida_o[niv]), "agr": float(caida_a[niv])} for niv in FUERA}
    tablas.exportar(
        pd.DataFrame(filas).set_index("Domain"), TABLAS, "sensibilidad_particion",
        caption=("Sensitivity of the within-domain F1 to the split: original patch-level split against a split "
                 "grouped by source photograph (SDNET2018 subsets; METU file names carry no photograph "
                 "identifier and keep the original split). Mean over 4 architectures x 3 seeds; paired "
                 "difference (grouped minus original) with 95\\% bootstrap CI."),
        etiqueta="tab:sensibilidad-particion", decimales=3, indice=True)
    log.info(json.dumps(salida, indent=1))
    return salida


def direccional_sin_tablas(r: pd.DataFrame) -> dict:
    datos = r.groupby(CORRIDA + ["direccion"])["f1"].mean().reset_index()
    xa = datos.loc[datos["direccion"] == "sdnet_a_metu", "f1"].values
    xb = datos.loc[datos["direccion"] == "metu_a_sdnet", "f1"].values
    t = sstats.ttest_ind(xa, xb, equal_var=False)
    return {"dif": float(xa.mean() - xb.mean()), "t": float(t.statistic), "p_sin_corregir": float(t.pvalue)}


# ======================================================== apendice por celda

def apendice(base: pd.DataFrame) -> None:
    """Tabla completa por celda y semilla (F1, P, R) de la condicion base."""
    t = base.copy()
    t["Architecture"] = t["arquitectura"].map(ARQUITECTURAS_LEGIBLES)
    t["Source"] = t["dominio_train"].map(NOMBRES_LEGIBLES)
    t["Target"] = t["dominio_test"].map(NOMBRES_LEGIBLES)
    t = t.sort_values(["arquitectura", "dominio_train", "dominio_test", "semilla"])
    t[["Architecture", "Source", "Target", "semilla", "tp", "fp", "fn", "tn", "precision", "recall", "f1"]] \
        .rename(columns={"semilla": "Seed", "tp": "TP", "fp": "FP", "fn": "FN", "tn": "TN",
                         "precision": "Precision", "recall": "Recall", "f1": "F1"}) \
        .to_csv(TABLAS / "apendice_celdas_base.csv", index=False)


# ===================================================================== main

# ================================================ 5. respuesta a la revision F5

def revision_f5(base: pd.DataFrame, orig: pd.DataFrame, log) -> dict:
    """Analisis anadidos tras la revision adversarial (ronda 1, 03/10/2026).

    1. IC de cada particion remuestreando corridas, no celdas: las celdas de
       una misma corrida no son independientes (bootstrap por conglomerados;
       como cada corrida aporta el mismo numero de celdas a una particion, la
       media de las medias por corrida es la media de las celdas).
    2. Contrastes direccionales con la corrida como unidad y el diseno
       correcto: pareados (Wilcoxon de rangos con signo) cuando las dos
       particiones salen de las mismas corridas; Welch cuando salen de
       corridas distintas (fuente SDNET2018 frente a fuente METU). Holm sobre
       los seis.
    3. Intervenciones frente a la base pareadas por corrida (media de sus tres
       celdas fuera de diagonal): Wilcoxon, biserial de rangos emparejada e IC
       bootstrap, en lugar de Cliff (que es para muestras independientes).
    4. Empates con el clasificador trivial (F1 = 2/3 exacto).
    """
    salida: dict = {}
    por_corrida = base.groupby(CORRIDA + ["direccion"])["f1"].mean().reset_index()

    ic = {}
    for niv in ORDEN_DIRECCION:
        g = por_corrida.loc[por_corrida["direccion"] == niv, "f1"].values
        v, li, ls = fstats.ic_bootstrap(g, semilla=SEMILLA)
        ic[niv] = {"corridas": int(len(g)), "f1": float(v), "ci_corridas": [float(li), float(ls)]}
    salida["ic_por_corridas"] = ic

    filas, p = [], []
    for i, a in enumerate(ORDEN_DIRECCION):
        for b in ORDEN_DIRECCION[i + 1:]:
            xa = por_corrida[por_corrida["direccion"] == a]
            xb = por_corrida[por_corrida["direccion"] == b]
            e = xa.merge(xb, on=CORRIDA, suffixes=("_a", "_b"))
            if len(e):    # mismas corridas en las dos particiones: pareado
                dif = e["f1_a"].values - e["f1_b"].values
                w = wilcoxon_emparejado(dif)
                v, li, ls = fstats.ic_bootstrap(dif, semilla=SEMILLA)
                fila = {"a": a, "b": b, "diseno": "paired", "n": int(len(e)), "dif": float(dif.mean()),
                        "ic": [float(li), float(ls)], "estadistico": w["W"], "efecto": w["r_rb"],
                        "medida_efecto": "rank-biserial"}
                p.append(w["p"])
            else:         # corridas distintas: Welch sobre medias por corrida
                t = sstats.ttest_ind(xa["f1"].values, xb["f1"].values, equal_var=False)
                delta, _ = fstats.cliffs_delta(xa["f1"].values, xb["f1"].values)
                # IC de la diferencia de medias remuestreando corridas dentro de
                # cada grupo por separado (ronda 3 de la revision, 08/10).
                rng = np.random.default_rng(SEMILLA)
                va, vb = xa["f1"].to_numpy(), xb["f1"].to_numpy()
                boot = [rng.choice(va, len(va)).mean() - rng.choice(vb, len(vb)).mean() for _ in range(10_000)]
                fila = {"a": a, "b": b, "diseno": "independent (Welch)", "n": f"{len(xa)}/{len(xb)}",
                        "dif": float(va.mean() - vb.mean()),
                        "ic": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                        "estadistico": float(t.statistic), "efecto": float(delta),
                        "medida_efecto": "Cliff's delta"}
                p.append(float(t.pvalue))
            filas.append(fila)
    for f, q, p0 in zip(filas, holm(p), p):
        f["p"], f["p_holm"] = p0, q
        log.info(f"[F5 corrida] {f['a']} vs {f['b']} ({f['diseno']}, n={f['n']}): {f['dif']:+.4f}, "
                 f"efecto {f['efecto']:+.3f}, p_Holm={q:.2e}")
    salida["contrastes_corrida"] = filas

    base_c = orig[(orig["condicion"] == "baseline") & (orig["categoria"] != "dentro")] \
        .groupby(CORRIDA)["f1"].mean()
    interv = {}
    for cond in ("agresivo", "ecualizado"):
        m = orig[(orig["condicion"] == cond) & (orig["categoria"] != "dentro")].groupby(CORRIDA)["f1"].mean()
        e = pd.concat([base_c.rename("b"), m.rename("m")], axis=1).dropna()
        dif = (e["m"] - e["b"]).values
        w = wilcoxon_emparejado(dif)
        v, li, ls = fstats.ic_bootstrap(dif, semilla=SEMILLA)
        interv[cond] = {"corridas": int(len(e)), "dif": float(v), "ic": [float(li), float(ls)],
                        "p": w["p"], "r_rb": w["r_rb"]}
        log.info(f"[F5 intervencion] {cond}: {v:+.4f} [{li:+.4f}, {ls:+.4f}], p={w['p']:.2e}, r={w['r_rb']:+.3f}")
    p_int = holm([interv[c]["p"] for c in interv])
    for c, q in zip(interv, p_int):
        interv[c]["p_holm"] = q
    salida["intervenciones_por_corrida"] = interv

    fuera = base[base["categoria"] != "dentro"]
    exactos = fuera[np.isclose(fuera["f1"], F1_TRIVIAL, atol=1e-12)]
    salida["empates_trivial"] = {
        "celdas_f1_igual_2_3": int(len(exactos)),
        "celdas_por_encima": int((fuera["f1"] > F1_TRIVIAL + 1e-12).sum()),
        "celdas_por_debajo": int((fuera["f1"] < F1_TRIVIAL - 1e-12).sum()),
        "detalle_empates": exactos[CORRIDA + ["dominio_test", "tp", "fp", "fn", "tn", "f1"]].to_dict(orient="records"),
    }
    log.info(f"[F5] empates con el trivial: {salida['empates_trivial']}")

    # El "control" entre superficies tambien cambia de sitio en los pares con
    # tableros (laboratorio SMASH frente a campus; articulo de datos de
    # SDNET2018). F1 medio de los pares que conservan el sitio (muros y
    # pavimentos) frente a los pares con tableros.
    es = base[base["direccion"] == "entre_superficie"]
    con_tablero = (es["dominio_train"] == "sdnet_D") | (es["dominio_test"] == "sdnet_D")
    salida["entre_superficie_sitio"] = {
        "sin_tablero_mismo_sitio": float(es.loc[~con_tablero, "f1"].mean()),
        "con_tablero": float(es.loc[con_tablero, "f1"].mean()),
        "celdas": [int((~con_tablero).sum()), int(con_tablero.sum())]}
    log.info(f"[F5] entre superficies por sitio: {salida['entre_superficie_sitio']}")
    (TABLAS / "revision_f5.json").write_text(json.dumps(salida, indent=2, default=float), encoding="utf-8")
    return salida


# ===================================================== 6. EXP-escala (F5)

def exp_escala(orig: pd.DataFrame, log) -> dict:
    """EXP-escala, con el protocolo fijado en el README (03/10, antes de entrenar).

    Principal (H-escala): F1 medio de las tres celdas METU->SDNET de cada una
    de las 12 corridas de origen METU, escala frente a base, pareado por
    (arquitectura, semilla); Wilcoxon bilateral, alfa 0.05, biserial de rangos
    e IC bootstrap de la diferencia media.
    Secundarias (Wilcoxon por corrida, Holm entre las cinco): recall y AUROC
    de METU->SDNET; F1 entre superficies y SDNET->METU; F1 dentro de dominio.
    """
    esc = orig[orig["condicion"] == "escala"]
    if not completa(esc, "escala/original", 192, log):
        return {}
    base = orig[orig["condicion"] == "baseline"]
    auroc = pd.read_csv(TABLAS / "umbral_auroc_celdas.csv")
    regimen = {"baseline": "fine-tuning", "escala": "scale augmentation"}

    def por_corrida(direccion: str, metrica: str) -> pd.DataFrame:
        partes = []
        for cond, df in (("baseline", base), ("escala", esc)):
            if metrica == "auroc":
                df = auroc[auroc["regimen"] == regimen[cond]]
            g = df[df["direccion"] == direccion].groupby(CORRIDA)[metrica].mean().rename(cond)
            partes.append(g)
        return pd.concat(partes, axis=1).dropna()

    def contraste(e: pd.DataFrame) -> dict:
        dif = (e["escala"] - e["baseline"]).values
        w = wilcoxon_emparejado(dif)
        v, li, ls = fstats.ic_bootstrap(dif, semilla=SEMILLA)
        return {"corridas": int(len(e)), "base": float(e["baseline"].mean()), "escala": float(e["escala"].mean()),
                "dif": float(v), "ic": [float(li), float(ls)], "W": w["W"], "p": w["p"], "r_rb": w["r_rb"],
                "corridas_mejoran": int((dif > 0).sum())}

    salida: dict = {}
    e = por_corrida("metu_a_sdnet", "f1")
    salida["principal"] = contraste(e)
    salida["principal"]["por_arquitectura"] = {
        a: float((g["escala"] - g["baseline"]).mean()) for a, g in e.groupby(level="arquitectura")}
    cel = lambda df: df[df["direccion"] == "metu_a_sdnet"].groupby("dominio_test")["f1"].mean()
    salida["principal"]["por_destino"] = {d: {"base": float(cel(base)[d]), "escala": float(cel(esc)[d])}
                                          for d in DOMINIOS[:3]}
    log.info(f"[escala] PRINCIPAL METU->SDNET F1: {json.dumps(salida['principal'], default=float)}")

    secundarias = {"recall_metu_a_sdnet": ("metu_a_sdnet", "recall"),
                   "auroc_metu_a_sdnet": ("metu_a_sdnet", "auroc"),
                   "f1_entre_superficie": ("entre_superficie", "f1"),
                   "f1_sdnet_a_metu": ("sdnet_a_metu", "f1"),
                   "f1_dentro": ("dentro", "f1")}
    sec = {k: contraste(por_corrida(*v)) for k, v in secundarias.items()}
    for k, q in zip(sec, holm([s["p"] for s in sec.values()])):
        sec[k]["p_holm"] = q
        log.info(f"[escala] {k}: {sec[k]['base']:.3f} -> {sec[k]['escala']:.3f}, dif {sec[k]['dif']:+.4f} "
                 f"[{sec[k]['ic'][0]:+.4f}, {sec[k]['ic'][1]:+.4f}], r={sec[k]['r_rb']:+.3f}, p_Holm={q:.3g}")
    salida["secundarias"] = sec

    # Descriptivo, fuera de las familias fijadas: el mismo resumen por corrida
    # que las otras dos intervenciones (media de las tres celdas fuera de
    # diagonal de cada corrida), para la tabla comparativa.
    fuera = lambda df: df[df["categoria"] != "dentro"].groupby(CORRIDA)["f1"].mean()
    salida["descriptivo_fuera_por_corrida"] = contraste(
        pd.concat([fuera(base).rename("baseline"), fuera(esc).rename("escala")], axis=1).dropna())
    log.info(f"[escala] descriptivo fuera de dominio: {salida['descriptivo_fuera_por_corrida']}")
    (TABLAS / "exp_escala.json").write_text(json.dumps(salida, indent=2, default=float), encoding="utf-8")
    return salida


def main() -> int:
    TABLAS.mkdir(parents=True, exist_ok=True)
    log = configurar_log("04_stats", RAIZ)
    resumen: dict = {"semilla": SEMILLA, "f1_clasificador_trivial": F1_TRIVIAL}

    r = leer("resultados_matriz.csv", log)
    if r is None:
        return 1
    r = anotar(r)
    orig = r[r["particion"] == "original"]
    base = orig[orig["condicion"] == "baseline"].copy()
    resumen["base_completa"] = completa(base, "baseline/original", 192, log)
    for cond in ("agresivo", "ecualizado"):
        completa(orig[orig["condicion"] == cond], f"{cond}/original", 192, log)

    if not base.empty:
        resumen["f1_por_particion_y_arquitectura"] = tabla_f1_por_particion(base)
        resumen["direccional"] = resumen_direccional(base, log)
        resumen["diagonal_por_dominio"] = {d: float(base[(base["dominio_train"] == d) & (base["dominio_test"] == d)]["f1"].mean())
                                           for d in DOMINIOS}
        resumen["diagonal_por_arquitectura_y_dominio"] = {
            f"{a}|{d}": float(v) for (a, d), v in
            base[base["categoria"] == "dentro"].groupby(["arquitectura", "dominio_train"])["f1"].mean().items()}
        resumen["contrastes"] = direccional(base, log)
        resumen["caida_relativa"] = tabla_caida_relativa(base, log)
        resumen["precision_recall"] = precision_recall(base, log)
        apendice(base)
        if resumen["base_completa"]:
            resumen.update(anova_y_supuestos(base, log))
            resumen["tukey"] = tukey(base, log)
    resumen["mitigaciones"] = mitigaciones(orig, log)
    if resumen.get("base_completa"):
        resumen["revision_f5"] = revision_f5(base, orig, log)
        if (TABLAS / "umbral_auroc_celdas.csv").exists():
            resumen["exp_escala"] = exp_escala(orig, log)

    sonda = leer("resultados_sonda.csv", log)
    if sonda is not None and not base.empty:
        resumen["sonda"] = sonda_frente_ajuste(base, anotar(sonda), log)

    mf = leer("resultados_multifuente.csv", log)
    if mf is not None and not base.empty:
        resumen["multifuente"] = multifuente(base, mf, log)

    agr = r[(r["particion"] == "agrupada") & (r["condicion"] == "baseline")]
    if not agr.empty and not base.empty:
        resumen["sensibilidad_particion"] = sensibilidad_particion(base, agr, log)

    (TABLAS / "resumen_estadistico.json").write_text(
        json.dumps(resumen, indent=2, ensure_ascii=True, default=float), encoding="utf-8")
    log.info("04_stats.py completado.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
