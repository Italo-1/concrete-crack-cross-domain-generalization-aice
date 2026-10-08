"""Genera paper/apendice.tex (tablas del apendice) desde results/tables/.

Las cifras del apendice no se copian a mano: salen de los mismos CSV que usa
04_stats.py. Ejecutar despues de 04_stats.py; main.tex lo incluye con \\input.

    python src/06_apendice.py

Tablas:
  A1-A2  condicion base, particion original: F1 por semilla, F1 medio,
         precision y recall medios, por arquitectura y par origen-destino
         (64 filas; los TP/FP/FN/TN por semilla estan en
         results/tables/apendice_celdas_base.csv).
  A3     matrices medias 4x4 de los demas regimenes: aumentacion agresiva,
         ecualizacion, aumentacion de escala (EXP-escala, 03/10), sonda
         lineal y particion agrupada por foto.
  A4     tres fuentes con un dominio excluido: F1 en el dominio excluido por
         arquitectura (media de 3 semillas).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
TABLAS = RAIZ / "results" / "tables"
SALIDA = RAIZ / "paper" / "apendice.tex"

DOMINIOS = ["sdnet_D", "sdnet_P", "sdnet_W", "metu"]
CORTO = {"sdnet_D": "SDNET-D", "sdnet_P": "SDNET-P", "sdnet_W": "SDNET-W", "metu": "METU"}
ARQ = {"resnet18": "ResNet-18", "efficientnet_b0": "EfficientNet-B0",
       "mobilenetv3_small_100": "MobileNetV3-Small", "vit_tiny_patch16_224": "ViT-Tiny"}
SEMILLAS = [42, 43, 44]


def f3(x: float) -> str:
    return f"{x:.3f}"


def tabla_celdas(base: pd.DataFrame, arqs: list[str], etiqueta: str, titulo: str) -> list[str]:
    filas = [r"\begin{table}[p]",
             rf"\caption{{{titulo}}}\label{{{etiqueta}}}",
             r"\begin{tabular*}{\textwidth}{@{\extracolsep\fill}llcccccc@{}}",
             r"\toprule",
             r"Source & Target & F1 s42 & F1 s43 & F1 s44 & F1 mean & Precision & Recall \\",
             r"\midrule"]
    for k, a in enumerate(arqs):
        if k:
            filas.append(r"\midrule")
        filas.append(rf"\multicolumn{{8}}{{@{{}}l}}{{\textit{{{ARQ[a]}}}}} \\")
        for s in DOMINIOS:
            for t in DOMINIOS:
                c = base[(base.arquitectura == a) & (base.dominio_train == s) & (base.dominio_test == t)]
                assert sorted(c.semilla) == SEMILLAS, (a, s, t)
                f1 = [f3(c[c.semilla == x].f1.iloc[0]) for x in SEMILLAS]
                media = f3(c.f1.mean())
                if s == t:   # diagonal en negrita para ubicarla de un vistazo
                    media = rf"\textbf{{{media}}}"
                filas.append(f"{CORTO[s]} & {CORTO[t]} & " + " & ".join(f1)
                             + f" & {media} & {f3(c.precision.mean())} & {f3(c.recall.mean())} \\\\")
    filas += [r"\botrule", r"\end{tabular*}", r"\end{table}", ""]
    return filas


def matriz(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby(["dominio_train", "dominio_test"]).f1.mean().unstack().loc[DOMINIOS, DOMINIOS]


def tabla_regimenes(regimenes: dict[str, pd.DataFrame]) -> list[str]:
    filas = [r"\begin{table}[t]",
             r"\caption{Mean crack-class F1 matrix (over 4 architectures $\times$ 3 seeds) of the other "
             r"training regimes. Rows: training domain; columns: test domain}\label{tab:A-regimes}",
             r"\begin{tabular*}{\textwidth}{@{\extracolsep\fill}llcccc@{}}",
             r"\toprule",
             r"Regime & Source & SDNET-D & SDNET-P & SDNET-W & METU \\",
             r"\midrule"]
    for k, (nombre, df) in enumerate(regimenes.items()):
        if k:
            filas.append(r"\midrule")
        m = matriz(df)
        assert len(df) == 192, (nombre, len(df))
        for i, s in enumerate(DOMINIOS):
            valores = []
            for t in DOMINIOS:
                v = f3(m.loc[s, t])
                valores.append(rf"\textbf{{{v}}}" if s == t else v)
            filas.append(f"{nombre if i == 0 else ''} & {CORTO[s]} & " + " & ".join(valores) + r" \\")
    filas += [r"\botrule", r"\end{tabular*}",
              r"\footnotetext{Diagonal in bold. The photo-grouped split changes the SDNET2018 "
              r"training and test sets only; METU models and test set are those of the patch split.}",
              r"\end{table}", ""]
    return filas


def tabla_multifuente(mf: pd.DataFrame) -> list[str]:
    x = mf[mf.excluido.astype(str) == "True"]
    assert len(x) == 48, len(x)
    m = x.groupby(["arquitectura", "dominio_excluido"]).f1.mean().unstack()
    filas = [r"\begin{table}[t]",
             r"\caption{Three-source training with one domain held out: crack-class F1 on the held-out "
             r"domain by architecture (mean over 3 seeds)}\label{tab:A-multi}",
             r"\begin{tabular*}{\textwidth}{@{\extracolsep\fill}lcccc@{}}",
             r"\toprule",
             r"Architecture & SDNET-D & SDNET-P & SDNET-W & METU \\",
             r"\midrule"]
    for a in ARQ:
        filas.append(f"{ARQ[a]} & " + " & ".join(f3(m.loc[a, d]) for d in DOMINIOS) + r" \\")
    filas += [r"\botrule", r"\end{tabular*}", r"\end{table}", ""]
    return filas


def main() -> int:
    r = pd.read_csv(TABLAS / "resultados_matriz.csv")
    orig = r[r.particion == "original"]
    base = orig[orig.condicion == "baseline"]
    assert len(base) == 192
    sonda = pd.read_csv(TABLAS / "resultados_sonda.csv")
    mf = pd.read_csv(TABLAS / "resultados_multifuente.csv")

    arqs = list(ARQ)
    nota = ("Crack-class F1 of every cell of the baseline condition by seed (s42, s43, s44: seeds 42, "
            "43 and 44), mean F1, and mean precision and recall over the three seeds; within-domain "
            "cells in bold. Confusion counts per seed are in the released results")
    lineas = [
        "% Generado por src/06_apendice.py desde results/tables/. No editar a mano.",
        "",
        *tabla_celdas(base, arqs[:2], "tab:A-cells1", f"{nota} ({ARQ[arqs[0]]}, {ARQ[arqs[1]]})"),
        *tabla_celdas(base, arqs[2:], "tab:A-cells2", f"{nota} ({ARQ[arqs[2]]}, {ARQ[arqs[3]]})"),
        *tabla_regimenes({
            "Aggressive augmentation": orig[orig.condicion == "agresivo"],
            "Histogram equalization": orig[orig.condicion == "ecualizado"],
            "Scale augmentation": orig[orig.condicion == "escala"],
            "Linear probe": sonda,
            "Photo-grouped split": r[(r.particion == "agrupada") & (r.condicion == "baseline")],
        }),
        *tabla_multifuente(mf),
    ]
    SALIDA.write_text("\n".join(lineas), encoding="utf-8")
    print(f"{SALIDA} escrito ({len(lineas)} lineas).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
