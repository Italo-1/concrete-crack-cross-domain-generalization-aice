# Ronda 3: revisor principal de AICE (08/10/2026) y verificación

Agente en contexto limpio, solo con `paper/main.pdf` (38 páginas) y la web.
Recomendación: revisión mayor, al límite del rechazo. Recalculó las cifras a
partir de las Tablas A1-A3: casi todo cuadra.

## Objeciones principales (resumen)

1. Una sola pareja de campañas: título y contribución 1 prometen más; la
   asimetría es esperable; novedad limitada frente a Rashid; modelos pequeños
   de ImageNet sin modelos de fundación; la contribución 4 es práctica
   estándar.
2. Incertidumbre sin la variabilidad del test (0.055 al cambiar el test, igual
   que el efecto de escala); fuga confundida con el cambio de test;
   pseudorreplicación (arquitectura × semilla); Fig. 4 por celdas;
   "confirmatory retraining" frente a entrenamiento determinista; "fixed
   before" sin registro; ausencias de diferencia presentadas como equivalencia.
3. Diagonales SDNET bajas (infraentrenamiento); etiquetas sin auditar;
   "78 de 144" es artefacto de la métrica; "realistic prevalence" sin
   respaldo; NaN en media precisión sin corregir.

## Verificación de las afirmaciones nuevas

| Afirmación | Verificación | Estado |
|---|---|---|
| 5.2 cita la Tabla 5 con pérdidas absolutas (0.216/0.169/0.387) que la tabla no contiene; en absoluto SDNET→METU pierde un 28 % más | Cierto | Aplicado: 5.2 usa Δt de la Tabla 5 (0.217/0.204/0.474) |
| 4.2 atribuye a la Tabla 5 rangos por celda que no contiene | Cierto | Aplicado |
| 0.055/0.054, 0.165/0.166, 0.023/0.024, −0.034/−0.035 | Diferencias calculadas sin redondear | Aplicado: nota en 3.8 |
| Ecualización: "mostly on SDNET-Deck and -Wall" | Cierto: Pavement 0.855 → 0.806 con ecualización | Aplicado |
| "Both figures are bounded below by 0.667" | Cierto: solo "Best" lo está (k = 20: 0.651) | Aplicado |
| "9 of the 12 runs improving, by +0.054 to +0.102…" ambiguo | Cierto | Aplicado |
| Tabla 2: "mostly within one SD" | Engañoso (η²p = 0.38) | Aplicado (eliminado) |
| "Confirmatory retraining" con entrenamiento determinista | El original NO era determinista (`determinista=False`, README) y no reproduce bit a bit | Aplicado: se aclara en el texto |
| Dorafshan 2018: 99 % sin el desbalance | Cierto: 319 de 3420 con grieta (nota de verificación) | Aplicado |
| "Realistic prevalence" | Prevalencia de la colección, no de campo | Aplicado |
| "Two have not been measured apart", "absence of a size effect" | Afirmaciones fuertes | Aplicado (atenuadas) |
| Welch sin IC (Tabla 4) | Cierto | Aplicado: IC bootstrap remuestreando corridas dentro de cada grupo |
| SDNET corto definido solo para tablas; s42-s44; p_Holm | Cierto | Aplicado |
| Nota de la Tabla 5 frente a 3.9 | Referencias correctas (análisis en 3.8, revisión en 3.9) | Aplicado: redacción aclarada |
| "One source, best" sin rotular como oráculo | Cierto en la Tabla 11 | Aplicado |
| Santos +54 puntos; 54 fotos × 252; 54.9 % repetido | **Refutadas en la ronda 2** (texto completo, datos crudos, recuento) | No aplicado |
| Zhu 2026: año del volumen 2027 | Crossref: en línea 09/07/2026, impreso 2027; se cita el año en línea | No aplicado |
| Gulrajani, Steiner, Loshchilov publicados en congreso/revista | Versiones publicadas no abiertas (anti-bots) | No aplicado |

## No aplicado (experimentos, decisión del usuario)

Tercera campaña, modelos de fundación, bootstrap jerárquico o modelo mixto,
reevaluación en FP32, curva de aprendizaje, auditoría de etiquetas, control de
color, "Name That Dataset", barrido de escala en test con los modelos METU
guardados (barato: candidato para el usuario). Citas de métodos (Wilcoxon,
Levene, PyTorch, timm…) y trabajos de grietas sugeridos: no abiertos, no se
citan.
