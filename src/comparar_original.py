"""E1: compara la matriz reentrenada con la de la version de Tehnicki glasnik.

Aplica los criterios de "cambio cualitativo" fijados en el README (protocolo,
"Prueba estadistica", punto 2) ANTES de ver los resultados nuevos:

  (a) la diferencia SDNET->METU menos METU->SDNET cambia de signo o deja de
      ser significativa tras Holm (seis contrastes de Welch entre las cuatro
      particiones direccionales; se comprueba con la celda y con la corrida
      como unidad);
  (b) cambia el orden de las cuatro particiones direccionales por F1 medio;
  (c) el IC 95 % bootstrap de "agresivo - base" deja de cruzar cero, o el de
      "ecualizado - base" pasa a cruzarlo o a ser positivo;
  (d) la fraccion de celdas fuera de diagonal con F1 > 0.667 sale de 35-65 %.

Tambien resume la diferencia celda a celda (media, media absoluta, maximo).
Solo usa la particion original de los resultados nuevos. Escribe
`results/tables/comparacion_original.json` y lo imprime.

    python src/comparar_original.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sstats

from fieutils import stats as fstats

RAIZ = Path(__file__).resolve().parent.parent
NUEVO = RAIZ / "results" / "tables" / "resultados_matriz.csv"
ORIGINAL = Path(r"C:\lineaB\03-tehnicki-glasnik-P1\results\tables\resultados_matriz.csv")
SEMILLA, F1_TRIVIAL = 42, 2 / 3
ORDEN = ["dentro", "entre_superficie", "sdnet_a_metu", "metu_a_sdnet"]
CLAVE = ["arquitectura", "semilla", "condicion", "dominio_train", "dominio_test"]


def direccion(train: str, test: str) -> str:
    if train == test:
        return "dentro"
    if train.startswith("sdnet_") and test.startswith("sdnet_"):
        return "entre_superficie"
    return "metu_a_sdnet" if train == "metu" else "sdnet_a_metu"


def holm(p: list[float]) -> list[float]:
    orden, m, corriente, salida = np.argsort(p), len(p), 0.0, [0.0] * len(p)
    for rango, i in enumerate(orden):
        corriente = max(corriente, min(1.0, p[i] * (m - rango)))
        salida[i] = corriente
    return salida


def contrastes(base: pd.DataFrame, unidad: str) -> dict:
    """Welch + Holm entre las cuatro particiones; devuelve el contraste clave."""
    if unidad == "corrida":
        base = base.groupby(["arquitectura", "semilla", "dominio_train", "direccion"])["f1"].mean().reset_index()
    pares, p = [], []
    for i, a in enumerate(ORDEN):
        for b in ORDEN[i + 1:]:
            xa, xb = base.loc[base["direccion"] == a, "f1"], base.loc[base["direccion"] == b, "f1"]
            pares.append((a, b, float(xa.mean() - xb.mean())))
            p.append(float(sstats.ttest_ind(xa, xb, equal_var=False).pvalue))
    ph = holm(p)
    todos = {f"{a} vs {b}": {"dif": d, "p_holm": q} for (a, b, d), q in zip(pares, ph)}
    return {"sdnet_a_metu vs metu_a_sdnet": todos["sdnet_a_metu vs metu_a_sdnet"], "seis": todos}


def resumen(r: pd.DataFrame) -> dict:
    base = r[r["condicion"] == "baseline"].copy()
    medias = base.groupby("direccion")["f1"].mean()
    fuera = base[base["direccion"] != "dentro"]
    mitig = {}
    for cond in ("agresivo", "ecualizado"):
        m = r[(r["condicion"] == cond) & (r["direccion"] != "dentro")]
        e = fuera.merge(m, on=["arquitectura", "semilla", "dominio_train", "dominio_test"], suffixes=("_b", "_m"))
        if len(e):
            v, li, ls = fstats.ic_bootstrap(e["f1_m"].values - e["f1_b"].values, semilla=SEMILLA)
            mitig[cond] = {"n": len(e), "dif_media": v, "ic95": [li, ls]}
    return {
        "filas_base": len(base),
        "f1_medio_por_particion": {k: float(medias[k]) for k in ORDEN},
        "orden_particiones": list(medias.sort_values(ascending=False).index),
        "contraste_celda": contrastes(base, "celda"),
        "contraste_corrida": contrastes(base, "corrida"),
        "mitigaciones": mitig,
        "fraccion_fuera_sobre_trivial": float((fuera["f1"] > F1_TRIVIAL).mean()),
    }


def main() -> int:
    nuevo = pd.read_csv(NUEVO)
    nuevo = nuevo[nuevo["particion"] == "original"].drop(columns="particion")
    original = pd.read_csv(ORIGINAL)
    for r in (nuevo, original):
        r["direccion"] = [direccion(t, d) for t, d in zip(r["dominio_train"], r["dominio_test"])]

    completo = {c: int((nuevo["condicion"] == c).sum()) for c in ("baseline", "agresivo", "ecualizado")}
    n, o = resumen(nuevo), resumen(original)

    celdas = original.merge(nuevo, on=CLAVE, suffixes=("_orig", "_nuevo"))
    dif = celdas["f1_nuevo"] - celdas["f1_orig"]

    ka_n, ka_o = n["contraste_celda"]["sdnet_a_metu vs metu_a_sdnet"], o["contraste_celda"]["sdnet_a_metu vs metu_a_sdnet"]
    kc_n = n["contraste_corrida"]["sdnet_a_metu vs metu_a_sdnet"]
    criterios = {
        "a_signo_cambia": bool(np.sign(ka_n["dif"]) != np.sign(ka_o["dif"])),
        "a_no_significativo_celda": bool(ka_n["p_holm"] >= 0.05),
        "a_no_significativo_corrida": bool(kc_n["p_holm"] >= 0.05),
        "b_orden_cambia": n["orden_particiones"] != o["orden_particiones"],
        "c_agresivo_deja_de_cruzar_cero": bool("agresivo" in n["mitigaciones"] and not (
            n["mitigaciones"]["agresivo"]["ic95"][0] <= 0 <= n["mitigaciones"]["agresivo"]["ic95"][1])),
        "c_ecualizado_cruza_cero_o_positivo": bool("ecualizado" in n["mitigaciones"]
                                                    and n["mitigaciones"]["ecualizado"]["ic95"][1] >= 0),
        "d_fraccion_fuera_de_35_65": not (0.35 <= n["fraccion_fuera_sobre_trivial"] <= 0.65),
    }
    informe = {
        "filas_nuevas_por_condicion (esperado 192)": completo,
        "celdas_emparejadas": len(celdas),
        "diferencia_por_celda": {"media": float(dif.mean()), "media_abs": float(dif.abs().mean()),
                                 "max_abs": float(dif.abs().max()),
                                 "por_condicion_media_abs": celdas.assign(d=dif.abs()).groupby("condicion")["d"].mean().to_dict()},
        "nuevo": n, "original": o,
        "criterios_cambio_cualitativo": criterios,
        "hay_cambio_cualitativo": any(criterios.values()),
    }
    salida = RAIZ / "results" / "tables" / "comparacion_original.json"
    salida.write_text(json.dumps(informe, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: informe[k] for k in ("filas_nuevas_por_condicion (esperado 192)", "diferencia_por_celda",
                                              "criterios_cambio_cualitativo", "hay_cambio_cualitativo")},
                     indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
