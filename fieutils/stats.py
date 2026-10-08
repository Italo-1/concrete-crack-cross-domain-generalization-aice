"""Pruebas estadisticas comunes a los cinco articulos.

Cada ficha declara su prueba ANTES de ver resultados; este modulo las implementa
una sola vez para que los cinco manuscritos las apliquen de forma identica.

Cobertura por articulo:
  P7  Friedman + Nemenyi entre representaciones, ANOVA de dos factores
  P1  ANOVA de dos factores, Tukey HSD, IC bootstrap sobre F1
  P5  IC bootstrap, ANOVA sobre estrategia de mitigacion
  P3  skill score frente a persistencia, Diebold-Mariano, Friedman + Nemenyi
  P4  ANOVA de dos factores (representacion x clasificador)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------- comparacion
# multiple de modelos sobre varios conjuntos (Demsar 2006)

@dataclass
class ResultadoFriedman:
    """Salida de `friedman_nemenyi`, lista para volcar a tabla del manuscrito."""

    estadistico: float
    p_valor: float
    rangos_medios: pd.Series
    nemenyi: pd.DataFrame
    distancia_critica: float
    n_conjuntos: int

    def hay_diferencia(self, alfa: float = 0.05) -> bool:
        return self.p_valor < alfa

    def resumen(self) -> str:
        return (
            f"Friedman: chi2 = {self.estadistico:.3f}, p = {self.p_valor:.4g}, "
            f"k = {len(self.rangos_medios)} metodos sobre N = {self.n_conjuntos} "
            f"conjuntos. Distancia critica (Nemenyi, alfa=0.05) = "
            f"{self.distancia_critica:.3f}."
        )


def friedman_nemenyi(tabla: pd.DataFrame, alfa: float = 0.05) -> ResultadoFriedman:
    """Friedman sobre rendimiento con post-hoc Nemenyi.

    `tabla` tiene una fila por conjunto de datos (o fold) y una columna por
    metodo comparado; las celdas son la metrica, donde mas alto es mejor.

    El valor que se dibuja en el diagrama de diferencia critica es la distancia
    critica: dos metodos cuyos rangos medios difieren menos que ella no son
    distinguibles con los datos disponibles.
    """
    import scikit_posthocs as sp

    datos = tabla.dropna()
    if datos.shape[0] < 2:
        raise ValueError("Se necesitan al menos 2 conjuntos para Friedman.")

    estadistico, p = stats.friedmanchisquare(*[datos[c].values for c in datos.columns])

    # Rango 1 = mejor. Se invierte el signo porque rankdata ordena ascendente.
    rangos = datos.apply(lambda fila: stats.rankdata(-fila), axis=1, result_type="expand")
    rangos.columns = datos.columns
    rangos_medios = rangos.mean().sort_values()

    nemenyi = sp.posthoc_nemenyi_friedman(datos.values)
    nemenyi.index = datos.columns
    nemenyi.columns = datos.columns

    k, n = datos.shape[1], datos.shape[0]
    cd = _distancia_critica(k, n, alfa)

    return ResultadoFriedman(float(estadistico), float(p), rangos_medios, nemenyi, cd, n)


def _distancia_critica(k: int, n: int, alfa: float = 0.05) -> float:
    """Distancia critica de Nemenyi: q_alfa * sqrt(k(k+1) / 6N)."""
    # Valores criticos del rango studentizado divididos por sqrt(2), alfa = 0.05.
    q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949,
           8: 3.031, 9: 3.102, 10: 3.164, 11: 3.219, 12: 3.268}
    q10 = {2: 1.645, 3: 2.052, 4: 2.291, 5: 2.459, 6: 2.589, 7: 2.693,
           8: 2.780, 9: 2.855, 10: 2.920, 11: 2.978, 12: 3.030}
    tabla = q05 if alfa <= 0.05 else q10
    if k not in tabla:
        raise ValueError(f"No hay valor critico tabulado para k = {k}.")
    return tabla[k] * np.sqrt(k * (k + 1) / (6.0 * n))


# --------------------------------------------------------------- intervalos de
# confianza por remuestreo

def ic_bootstrap(
    datos: np.ndarray,
    estadistico=np.mean,
    n_muestras: int = 10_000,
    alfa: float = 0.05,
    semilla: int = 42,
) -> tuple[float, float, float]:
    """IC percentil por bootstrap. Devuelve (valor, limite_inferior, limite_superior).

    Se usa en P1 y P5 sobre F1 y exactitud, donde el numero de repeticiones es
    pequeno (3 semillas) y asumir normalidad no esta justificado.
    """
    datos = np.asarray(datos, dtype=float)
    datos = datos[~np.isnan(datos)]
    if datos.size == 0:
        return np.nan, np.nan, np.nan

    rng = np.random.default_rng(semilla)
    remuestras = rng.choice(datos, size=(n_muestras, datos.size), replace=True)
    valores = np.apply_along_axis(estadistico, 1, remuestras)

    return (
        float(estadistico(datos)),
        float(np.percentile(valores, 100 * alfa / 2)),
        float(np.percentile(valores, 100 * (1 - alfa / 2))),
    )


# ------------------------------------------------------------------ factorial

def anova_factorial(
    df: pd.DataFrame, dependiente: str, factores: list[str], interaccion: bool = True
) -> pd.DataFrame:
    """ANOVA de tipo II sobre un diseno factorial.

    Con `interaccion=True` incluye todos los cruces. En P6 la interaccion era el
    hallazgo principal; en P1 y P4 tambien conviene mirarla antes de concluir que
    un factor manda sobre el otro.
    """
    import statsmodels.api as sm
    from statsmodels.formula.api import ols

    unir = " * " if interaccion else " + "
    formula = f"{dependiente} ~ " + unir.join(f"C({f})" for f in factores)
    modelo = ols(formula, data=df).fit()
    tabla = sm.stats.anova_lm(modelo, typ=2)

    # Eta cuadrado parcial: el tamano del efecto que se reporta junto al valor p.
    tabla["eta_sq_parcial"] = tabla["sum_sq"] / (tabla["sum_sq"] + tabla.loc["Residual", "sum_sq"])
    return tabla


def tukey_hsd(df: pd.DataFrame, dependiente: str, grupo: str, alfa: float = 0.05):
    """Tukey HSD para comparaciones por pares tras un ANOVA significativo (P1)."""
    from statsmodels.stats.multicomp import pairwise_tukeyhsd

    return pairwise_tukeyhsd(df[dependiente].values, df[grupo].values, alpha=alfa)


# ------------------------------------------------------------ series temporales

def skill_score(error_modelo: float, error_referencia: float) -> float:
    """Mejora relativa frente a una linea base. 0 = igual que la referencia.

    En P3 la referencia obligatoria es la persistencia. Es la metrica honesta del
    articulo: el RMSE absoluto solo no dice si el modelo aporta algo.
    """
    if error_referencia == 0:
        return np.nan
    return 1.0 - (error_modelo / error_referencia)


def diebold_mariano(
    errores_a: np.ndarray, errores_b: np.ndarray, horizonte: int = 1, potencia: int = 2
) -> tuple[float, float]:
    """Test de Diebold-Mariano con correccion de Harvey para muestras pequenas.

    Contrasta si dos series de pronostico difieren de forma significativa.
    Devuelve (estadistico, p_valor); p < 0.05 indica precision distinta.
    """
    e_a = np.asarray(errores_a, dtype=float)
    e_b = np.asarray(errores_b, dtype=float)
    if e_a.shape != e_b.shape:
        raise ValueError("Las dos series de errores deben tener igual longitud.")

    d = np.abs(e_a) ** potencia - np.abs(e_b) ** potencia
    n = d.size
    media = d.mean()

    # Varianza de largo plazo: autocovarianzas hasta horizonte - 1.
    gamma0 = np.sum((d - media) ** 2) / n
    acum = gamma0
    for k in range(1, horizonte):
        gamma_k = np.sum((d[k:] - media) * (d[:-k] - media)) / n
        acum += 2 * gamma_k

    if acum <= 0:
        return np.nan, np.nan

    dm = media / np.sqrt(acum / n)

    # Correccion de Harvey, Leybourne y Newbold (1997).
    factor = np.sqrt((n + 1 - 2 * horizonte + horizonte * (horizonte - 1) / n) / n)
    dm_corregido = dm * factor
    p = 2 * (1 - stats.t.cdf(np.abs(dm_corregido), df=n - 1))

    return float(dm_corregido), float(p)


# ------------------------------------------------------------ tamanos de efecto

def cliffs_delta(a: np.ndarray, b: np.ndarray) -> tuple[float, str]:
    """Cliff's delta con su interpretacion cualitativa.

    Tamano del efecto no parametrico, adecuado cuando las distribuciones son
    sesgadas y la media no representa al grupo.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mayores = np.sum(a[:, None] > b[None, :])
    menores = np.sum(a[:, None] < b[None, :])
    delta = (mayores - menores) / (a.size * b.size)

    # Etiquetas en ingles: la funcion alimenta tablas de manuscritos en ingles
    # en los cinco articulos, y ya se colo una vez sin traducir (P7, 24/08).
    magnitud = abs(delta)
    if magnitud < 0.147:
        etiqueta = "negligible"
    elif magnitud < 0.33:
        etiqueta = "small"
    elif magnitud < 0.474:
        etiqueta = "medium"
    else:
        etiqueta = "large"

    return float(delta), etiqueta


def mcnemar(y_real: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> tuple[float, float]:
    """Test de McNemar: dos clasificadores sobre LAS MISMAS muestras.

    Es la prueba correcta cuando se comparan dos modelos evaluados sobre el
    mismo conjunto, porque los aciertos estan pareados y una prueba para
    muestras independientes sobreestima la significancia.

    Devuelve (estadistico, p_valor). Usa la correccion de continuidad de Edwards
    cuando hay pocos discordantes.
    """
    y_real = np.asarray(y_real)
    a_acierta = np.asarray(pred_a) == y_real
    b_acierta = np.asarray(pred_b) == y_real

    solo_a = int(np.sum(a_acierta & ~b_acierta))
    solo_b = int(np.sum(~a_acierta & b_acierta))
    n = solo_a + solo_b

    if n == 0:
        return 0.0, 1.0

    # Con pocos discordantes la aproximacion chi-cuadrado falla: prueba binomial.
    if n < 25:
        p = 2 * min(stats.binom.cdf(min(solo_a, solo_b), n, 0.5), 0.5)
        return float(min(solo_a, solo_b)), float(min(p, 1.0))

    estadistico = (abs(solo_a - solo_b) - 1) ** 2 / n
    return float(estadistico), float(1 - stats.chi2.cdf(estadistico, df=1))


def kendall_tau_ic(
    ranking_a: np.ndarray,
    ranking_b: np.ndarray,
    n_muestras: int = 10_000,
    alfa: float = 0.05,
    semilla: int = 42,
) -> tuple[float, float, float, float]:
    """Tau de Kendall entre dos rankings, con IC bootstrap.

    Devuelve (tau, limite_inferior, limite_superior, p_valor).

    Mide concordancia entre ordenaciones. El IC importa porque con listas
    cortas -diez variables, siete modelos- tau tiene mucha varianza y un valor
    puntual no dice nada por si solo.
    """
    a = np.asarray(ranking_a, dtype=float)
    b = np.asarray(ranking_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("Los dos rankings deben tener igual longitud.")

    tau, p = stats.kendalltau(a, b)

    rng = np.random.default_rng(semilla)
    n = a.size
    valores = []
    for _ in range(n_muestras):
        idx = rng.integers(0, n, n)
        if np.unique(a[idx]).size < 2 or np.unique(b[idx]).size < 2:
            continue
        t, _ = stats.kendalltau(a[idx], b[idx])
        if not np.isnan(t):
            valores.append(t)

    if not valores:
        return float(tau), np.nan, np.nan, float(p)

    return (
        float(tau),
        float(np.percentile(valores, 100 * alfa / 2)),
        float(np.percentile(valores, 100 * (1 - alfa / 2))),
        float(p),
    )


def test_page(matriz: np.ndarray) -> tuple[float, float]:
    """Test de Page para tendencia monotona a lo largo de una secuencia.

    `matriz` tiene una fila por sujeto (replica) y una columna por condicion,
    con las condiciones EN EL ORDEN en que se espera la tendencia.

    A diferencia de Friedman, que solo detecta que las condiciones difieren,
    Page contrasta la hipotesis ordenada: que el rendimiento crece (o decrece)
    de forma monotona conforme avanza la secuencia.

    Devuelve (L, p_valor) con aproximacion normal.
    """
    datos = np.asarray(matriz, dtype=float)
    n, k = datos.shape
    if k < 3:
        raise ValueError("El test de Page necesita al menos 3 condiciones.")

    rangos = np.apply_along_axis(stats.rankdata, 1, datos)
    suma_rangos = rangos.sum(axis=0)
    pesos = np.arange(1, k + 1)

    L = float(np.sum(pesos * suma_rangos))

    media = n * k * (k + 1) ** 2 / 4
    varianza = n * (k**3 - k) ** 2 / (144 * (k - 1))
    if varianza <= 0:
        return L, np.nan

    z = (L - media) / np.sqrt(varianza)
    return L, float(1 - stats.norm.cdf(z))


def caida_relativa(dentro: float, fuera: float) -> float:
    """(F1 en dominio propio - F1 fuera de dominio) / F1 en dominio propio.

    Es la metrica central de P1 y aparece tambien en P7 y P5. Positiva significa
    perdida al cambiar de dominio.
    """
    if dentro == 0:
        return np.nan
    return (dentro - fuera) / dentro
