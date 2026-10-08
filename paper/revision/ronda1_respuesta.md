# Revisión adversarial, ronda 1 (03/10/2026): objeciones y qué se hizo

Revisor: agente en contexto limpio, sin acceso más que a `paper/main.pdf`, con
el prompt principal de la guía metodológica §5 y rol de revisor de *AI in
Civil Engineering*. Recomendación: **revisión mayor**. Cada afirmación de
hecho se verificó contra la fuente antes de aceptarla (detalle en el README,
bitácora del 03/10).

Estados: **Aplicado** (corregido en el manuscrito), **Limitación** (declarado
en la sección 6), **Experimento** (EXP-escala, en curso), **No aplicado**
(con motivo).

## Objeciones principales

| # | Objeción | Verificación | Estado |
|---|---|---|---|
| O1a | Una sola campaña independiente; el p-valor mide ruido de semilla | Correcto | Aplicado (sin p en el resumen; "for this pair of collections") + Limitación 1 |
| O1b | SDNET2018 no es una sola campaña: tableros en laboratorio | **Confirmado** en el artículo de datos | Aplicado (Intro, Método 3.1, Discusión 5.2 con pares sin tableros 0.624 frente a 0.669) |
| O1c | Varianza del muestreo de datos ignorada | Correcto (0.055 al cambiar solo el test) | Limitación 3 (con la cifra) |
| O1d | "A priori" contradictorio; la partición por dirección se añadió después | Correcto | Aplicado (Método: réplica de un hallazgo exploratorio) |
| O2a | F1 con umbral fijo; falta AUROC/AP y recalibración | Correcto | Aplicado: análisis nuevo (`08_umbral_auroc.py`), subsección 4.5 y tabla |
| O2b | Recuento de 78 frágil por empates | Una celda con F1 = 2/3 exacto | Aplicado (77 por debajo, 1 empate) |
| O3a | Encuadre sin probar | Correcto; Santos et al. (2022) lo describe | Aplicado (cita) + **Experimento** EXP-escala: no se cumple la prueba principal (+0.055, p = 0.052; AUROC sin cambio), la explicación se debilita (Discusión 5.1 y 5.3, Limitación 2) |
| O3b | METU: 458 frente a 500 fotos; Zhang et al. (2016) | **Confirmado** (ficha Mendeley frente a ISARC) | Aplicado (ambas cifras; método de recorte vía Santos et al.) |
| O3c | Aumentación sin escala | Correcto | **Experimento** EXP-escala, hecho (Método 3.5, Resultados 4.6, Tabla `tab:scale`) |
| O3d | Ruido de etiqueta de SDNET2018 | No auditado | Limitación 2 y nota en la Fig. 6 |

## Metodología y estadística

| Objeción | Estado |
|---|---|
| "Fell in every partition" falso por celda | Aplicado ("on average"; SDNET-P→METU supera su diagonal en las 4 arquitecturas) |
| Ecualización "did not respond" con IC que excluye 0 | Aplicado (por corrida: −0.027, IC [−0.051, −0.004], p_Holm = 0.095) |
| "Matched the best single source" sin equivalencia | Aplicado ("did not differ"; composición de fuentes) |
| "Not created by fine-tuning" no se extiende a modelos fundacionales | Aplicado ("for these ImageNet-pretrained models") + Limitación 4 |
| ±0.06 supone independencia | Aplicado (muestreo repartido entre fotos) |
| Welch por corrida con grupos no independientes | Aplicado: contrastes pareados (Wilcoxon) donde comparten corridas. **Cambio:** dentro frente a SDNET→METU pasa a p_Holm = 0.12 |
| ANOVA con pseudorreplicación | Aplicado (se reporta como a priori y no se basan conclusiones en él) |
| IC bootstrap sobre celdas | Aplicado (Tabla 3 por corridas; Fig. 4 declara que es por celdas) |
| Cliff's δ para comparaciones pareadas | Aplicado (intervenciones con Wilcoxon y r por corrida) |
| Notación W ambigua | Aplicado |
| Modelo mixto | No aplicado: los contrastes por corrida con diseño pareado responden a la misma objeción con menos supuestos |
| Tercera campaña, todos los datos, modelos fundacionales | No aplicado (decisión del usuario 03/10): Limitaciones 1, 3 y 4 |

## Redacción, figuras, tablas y referencias

| Objeción | Estado |
|---|---|
| Frases con rasgos de IA (p. 2, 3, 10, 19-22) | Aplicado (reescritas) |
| Mezcla de ortografía británica y estadounidense | Aplicado (estadounidense) |
| Notación de particiones | Aplicado ("SDNET to METU" en tablas, con nota) |
| Script con nombre en español en el apéndice | Aplicado |
| Abreviaturas sin definir | Aplicado (F1, TP/FP/FN/TN, CNN, ViT, timm, ANOVA, HSD, CI, SD, AUROC, AP, LP/FT, η²p, GPU; ROC en el resumen) |
| Fig. 5 sin iso-F1 de 0.667 | Aplicado (discontinua; etiquetas a 8 pt, estaban a 6) |
| Fig. 4: unidad del bootstrap | Aplicado (pie) |
| Fig. 8: "best of three" es un oráculo | Aplicado (pie) |
| Tabla 2: recuento de celdas | Aplicado |
| Tabla 9: "split unchanged" | Aplicado ("n/a" y nota) |
| Santos et al. (2022, AICE) omitido | **Confirmado** y aplicado (Intro, 2.2, 3.1, 5.1, 5.2, 5.3) |
| Geirhos: posición, no tamaño | **Confirmado** y aplicado |
| Özgenel: efecto de arquitectura fuerte | **Confirmado** y aplicado (5.2) |
| CODEBRIM y SCD sin definir | Aplicado (descritos sin sigla) |
| Versiones publicadas de Gulrajani, Tan & Le, Steiner | Pendiente: requiere abrir cada versión antes de cambiar la cita |
| Otras referencias propuestas (WILDS, Miller et al., Guo et al., CrackSeg9k...) | No aplicado: no se abrieron; no se cita sin leer |
| Marcadores de autoría y repositorio | Pendiente de decisión del usuario |
