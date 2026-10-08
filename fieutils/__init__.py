"""fieutils - infraestructura comun de la linea B.

Cinco articulos (P7, P4, P1, P5, P3) comparten estos modulos. La regla es que
nada especifico de un articulo vive aqui: si solo lo usa uno, va en su `src/`.

    config       semilla, dispositivo, rutas, registro
    stats        Friedman-Nemenyi, bootstrap, ANOVA, Diebold-Mariano
    signal_repr  representaciones 1D    (P7, P4)
    figures      estilo y figuras       (los cinco)
    tablas       exportacion .csv + .tex (los cinco)
    vision       entrenamiento y transferencia CNN  (P1, P5)
"""

from fieutils import config, figures, signal_repr, stats, tablas, vision

__all__ = ["config", "figures", "signal_repr", "stats", "tablas", "vision"]
__version__ = "0.1.0"
