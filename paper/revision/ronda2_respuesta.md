# Revisión adversarial, ronda 2 (03/10/2026): objeciones y qué se hizo

Tres informes en contexto limpio (`ronda2_revisor.md`, `ronda2_hostil.md`,
`ronda2_antiIA.md`); verificación en `ronda2_verificacion.md`. El usuario
aprobó aplicar las correcciones y mantener el título. Análisis nuevos en
`src/09_ronda2.py` (sin GPU; sorteos propios, para no alterar los de
`08_umbral_auroc.py`).

Estados: **Aplicado**, **Limitación**, **No aplicado** (con motivo),
**Refutado**.

## Cambios de conclusión

| Objeción | Verificación | Estado |
|---|---|---|
| Caída según el anclaje | Confirmada: anclada en el destino, SDNET→METU 0.217 frente a 0.204 entre superficies; METU→SDNET la mayor con cualquier anclaje (0.562/0.474) | Aplicado: Tabla 5 nueva (`tab:anchor`), 4.3, 5.2, resumen, contribución 1, Conclusión, carta |
| "No mejor que el trivial" solo en F1 | Confirmada: las 144 celdas superan la exactitud balanceada de 0.5; las 77 bajo 2/3 tienen BA 0.557-0.735 y MCC 0.17-0.52 | Aplicado: "por debajo del F1 del trivial" + BA y MCC (4.2, Tabla 5, resumen, contribución 2, 5.4, Conclusión, carta) |
| Prevalencia real | Confirmada: precisión METU→SDNET 0.538 (0.364-0.773), entre superficies 0.454 | Aplicado: 4.4, 5.4 ("about half would be false alarms"), Limitación 3 |
| Umbral que maximiza F1 sesgado hacia "todo grieta" | Confirmada: el F1-umbral baja el MCC de METU→SDNET de 0.332 a 0.205; con Youden, BA 0.629 → 0.656 | Aplicado: 4.5 reescrita, pie de la Tabla 8, Youden en la Tabla 5; la conclusión "recalibrar no basta" se mantiene con métricas que no favorecen al trivial |

## Metodología y encuadre

| Objeción | Estado |
|---|---|
| Escala: IC excluye 0 con p = 0.052 | Aplicado: "inconcluso, no refutado"; se mantiene la lectura fijada (debilitada); se quita "raises recall" |
| Sonda: "ya presente en las características" | Aplicado: las sondas entrenadas en SDNET separan SDNET (0.781-0.830); falla la frontera aprendida en METU |
| Tamaño confundido con preentrenamiento (ViT en ImageNet-21k, recetas) | Aplicado: Limitación 4; orden casi inverso al tamaño en METU→SDNET (4.3) |
| "Replication" | Aplicado: "confirmatory retraining… not an independent replication" |
| Campaña = METU | Aplicado: Limitación 1 |
| "Change of site did not make… worse" sin prueba | Aplicado: "not tested" |
| Contradicción sobre elegir la mejor fuente | Aplicado: 5.3 y paso 3 de 5.4 |
| Rashid et al. ya lo hicieron (hostil) | **Refutado** con sus tablas y usado: 2.1 remite a 5.2, donde se explica por qué su diseño no mostraba la asimetría |
| IC sin ajustar frente a p_Holm (Tabla 4) | Aplicado: nota en la tabla |
| Unidad de inferencia, modelo mixto, bootstrap sobre fotos | No aplicado: Limitación 3 ya declara que los IC cubren variación entre corridas, no entre sorteos de datos |
| Prerregistro verificable | No aplicado: el texto dice "fixed before", no "preregistered"; el protocolo consta en la bitácora fechada |
| ANOVA/Tukey | No aplicado: ya se declara que no se basan conclusiones en ellos |
| Experimentos (tercera colección, ajuste fino con k etiquetas, todos los datos, FP32, auditoría de etiquetas, color, escala física) | No aplicado: requieren GPU o datos nuevos; Limitaciones 1-3 y 6 |

## Redacción, figuras, abreviaturas

| Objeción | Estado |
|---|---|
| Títulos de 5.1 y 5.2, "not one quantity/number", "pooled figure hides", "does not survive" | Aplicado |
| 2.4 anáfora y repetición | Aplicado (condensada; reconoce las respuestas generales de Kumar, Gulrajani y Kapoor) |
| Spencer "turn images into information…" en 5.4 | Aplicado (eliminado de 5.4) |
| "Based on these findings", "prevents the deployment" | Aplicado |
| "lost recall, not precision" | Aplicado ("much more recall than precision") |
| "has little reason to learn" | Aplicado ("may not learn") |
| "report this second figure much less often" sin respaldo | Aplicado (eliminado) |
| "These studies leave several questions…" | Aplicado (eliminada) |
| "And…" al inicio de frase; "Three results bear on both" | Aplicado |
| "scores an F1 score" | Aplicado |
| Fig. 1 sin reescalado; abreviaturas en la figura | Aplicado (caja y pie) |
| Fig. 3: 0.666 se leía 0.67 | Aplicado (tres decimales en el panel a) |
| Tabla 5 (caída): negrita sobre valor negativo | Aplicado (nota) |
| W sobrecargada | Aplicado (SR en la Tabla 4) |
| P − R, FN | Aplicado (pie de la Tabla 7) |
| METU y F1 sin definir en el cuerpo; MD5, cuBLAS, cuDNN, AdamW | Aplicado (AdamW definido, sin cita: no se abrió su fuente) |
| Fig. 2 sin escala, Fig. 5 superposición, Fig. 6 "p" | No aplicado (menores; sin datos de escala física) |
| Referencias adicionales | **Aplicado en parte (03/10, pedido del usuario):** Lipton et al. 2014 (umbral para F1; 3.8, 4.5, 5.4), Chicco & Jurman 2020 (MCC; 3.8), Miller et al. 2021 (correlación dentro/fuera entre modelos, citado como contraste; 5.2), Loshchilov & Hutter (AdamW; 3.3), todas leídas. Youden 1950 no se cita: Wiley con verificación anti-bots, no se leyó. Recht, WILDS, Zech, etc.: no abiertas, no se citan |

## Refutadas

- Santos et al. (2022): "de 54 % a 84 %" es correcto; su "54 %" es mejora relativa.
- SDNET-Deck: 54 fotos es correcto (52 con 252 parches, una con 256, otra con 260).
- Hasan & Lu (2025): su sección 3.2 pide un test representativo e independiente, y su 3.3 un mecanismo de comprobación tras el despliegue; la cita es correcta.
- 54.9 % repetido: coincidencia real (79/144 en la sonda y en la partición agrupada).
