# Ronda 2: revisor principal de AICE (03/10/2026, sin aplicar todavía)

Agente en contexto limpio, solo con `paper/main.pdf` (34 páginas) y la web,
prompt 1 de `ronda2_prompts.md`. Transcribió las Tablas A1-A2 y recalculó las
cifras. Resumen del informe; la verificación de cada afirmación de hecho está
en `ronda2_verificacion.md`.

**Recomendación:** revisión mayor, en el límite del rechazo (incluye los
marcadores pendientes de autoría y repositorio como motivo de devolución).

## Objeciones principales

1. **La asimetría depende del anclaje y de que METU es fácil.** Caída anclada
   en el destino (recalculada por el revisor): entre superficies 0.204,
   SDNET→METU 0.217, METU→SDNET 0.474. Con ese anclaje, SDNET→METU pierde lo
   mismo que entre superficies; contradice la Conclusión, la Tabla 4
   (−0.128) y la p. 22. "Property of the ordered pair" no se distingue de
   "METU es un destino fácil y un mal origen". Pide los dos anclajes,
   reformular y una tercera campaña.
2. **La referencia "always-crack" y el F1 hacen parecer triviales a modelos
   informativos.** Las 24 celdas medias por debajo de 0.667 tienen exactitud
   0.57-0.72 y MCC 0.22-0.48. El F1 en el mejor umbral es ≥ 0.667 por
   construcción, y elegir el umbral maximizando F1 tiende a "todo grieta"
   (Lipton et al. 2014). Con prevalencia del 10 %, la precisión METU→SDNET
   bajaría a unos 0.40 y la referencia 2π/(1+π) a unos 0.18. Pide MCC o
   exactitud balanceada, umbral de Youden o con costes, y sensibilidad a la
   prevalencia.
3. **Unidad de inferencia e incertidumbre.** El muestreo del test mueve
   METU→SDNET 0.055, igual que el efecto de escala; las corridas comparten
   arquitectura y semilla; ausencia de significación leída como ausencia de
   efecto (escala: IC excluye 0 con p = 0.052, debería ser "no concluyente";
   el recall con escala p_Holm = 0.21 pero se afirma "raises recall"; LP = FT
   con p = 0.15 sin equivalencia).

## Metodología (selección)

- "Loss already present in the pretrained features": las sondas entrenadas en
  SDNET separan SDNET (0.781-0.830); lo que no transfiere es la frontera
  aprendida en METU. La sonda ordena mejor (AUROC 0.735).
- Tamaño confundido con familia, con ImageNet-21k (ViT) y con la receta de
  preentrenamiento de cada peso. En METU→SDNET el orden de la caída es casi el
  inverso del tamaño.
- Diagonales de SDNET bajas; falta control con todos los datos o más épocas.
- Encuadre: la hipótesis sigue como "a likely contributor" tras fallar su
  prueba; la escala física podría estimarse (SDNET ~60 mm por parche; METU a
  ~1 m, 4032×3024, parches de 227 px).
- "Replication" mal usado (mismo código y datos); lenguaje a priori sin
  registro verificable.
- ANOVA con par de 16 niveles y Tukey sobre un factor de 3 categorías que no
  está en el modelo.
- Familias de Holm ad hoc; Fig. 4 con bootstrap sobre celdas.
- NaN en precisión mixta (hasta 0.025 por celda): debería evaluarse en FP32.
- Contradicción: "la mejor fuente solo se identifica probando en el destino",
  pero el protocolo propone tener una muestra etiquetada del destino.
- "Needs training data from the site, not a new threshold" sin experimento;
  "the change of site did not make cross-surface transfer worse" sin prueba.
- Controles que faltan: ajuste fino con k etiquetas del destino, todos los
  datos, prevalencia realista, control de color, auditoría de etiquetas,
  tercera colección, bootstrap sobre fotos.
- Split agrupado: el test de SDNET-Deck sale de 8 fotos.

## Alcance

Encaja temáticamente; sin aportación metodológica de IA; la aportación civil
es limitada (clasificación de parches); el valor está en el rigor, que debe
ser impecable.

## Redacción

Frases aforísticas ("not one number", "not one quantity", títulos de 5.1 y
5.2, "The pooled figure hides…", "Within-collection figures are high.");
construcción "X, not Y" repetida; "lost recall, not precision" es falso entre
superficies (precisión 0.886 → 0.807); contribución 3 de unas 100 palabras;
resumen con unas 15 cifras; "those five things"; metanarrativa de "fixed
before" repetida unas diez veces; declaración de IA a precisar. Juicios sin
referencia: "ranked poorly" (AUROC 0.70), "lost little", "almost all".
Afirmaciones sin respaldo: "report this second figure much less often";
Hasan & Lu citado para representatividad del test; "checking the patches it
flags will not reveal"; "has little reason to learn"; "prevents the
deployment"; "Capture Campaigns" en plural; "persisted" con el split agrupado
(0.345 → 0.265); "scores an F1 score".

## Figuras y tablas

- Fig. 1: falta el reescalado y el split agrupado; pie "three regimes";
  tipografía pequeña.
- Fig. 2: sin escala física; diferencias de color sin comentar; (a) ambigua.
- Fig. 3: SDNET-D→METU muestra 0.67 y vale 0.666; barra de color 0.4-0.9 sin
  cubrir 0.33-1.00; (b)-(e) sin valores.
- Fig. 4: bootstrap sobre celdas; línea 0.667 sin rótulo.
- Fig. 5: superposición; arquitectura no distinguible.
- Fig. 6: "p" para la probabilidad (choca con el valor p).
- Fig. 8: "best of three" debería distinguirse visualmente.
- Tabla 2: DE sobre destinos de dificultad distinta. Tabla 3: mezcla DE de
  celdas con IC de corridas. Tabla 4: sin IC en filas de Welch; IC sin
  ajustar; "W" sobrecargado (Wilcoxon, Shapiro-Wilk, SDNET-Wall). Tabla 5:
  negrita sobre un valor negativo. Tabla 6: "P − R" sin definir. Tabla 7:
  sin dispersión; "Best" ≥ 0.667 por construcción. Tabla 8: IC excluye 0 con
  p = 0.052 sin comentario. Tabla A3: sin DE.

## Referencias

- Seis citas de AICE; Hasan & Lu, Luleci & Catbas y Fountoukidou "traídas con
  calzador".
- **Santos et al. (2022):** el resumen publicado dice que DA-Crack mejora la
  exactitud en 54 puntos; el manuscrito dice de 54 % a 84 %. Verificar.
- Faltan: Recht 2019, Miller 2021 ("Accuracy on the line"), Taori 2020, Koh
  2021 (WILDS), Zech 2018 (origen del ejemplo de neumonía), Lipton 2014, Flach
  & Kull 2015, Chicco & Jurman 2020, Saito & Rehmsmeier 2015, Demšar 2006,
  Benavoli 2017, Ganin 2016, Ben-David 2010, Koch 2015, CODEBRIM, dacl10k,
  Crack500, DeepCrack, Zhang et al. 2025 (arXiv:2508.10256), AdamW
  (Loshchilov & Hutter), Wightman 2021 ("ResNet strikes back").
- Formato: Gulrajani y Steiner como arXiv; DOI con y sin hipervínculo; "&"
  inconsistente; corte de Dorafshan 2018b; verificar Zhu 2026.

## Coherencia de cifras

Coinciden las principales (lista en el informe). Problemas señalados:
1. SDNET-Deck: 54 fotos × 252 = 13 608 < 13 620 parches.
2. Resumen: 0.867 → 0.654 compara con una diagonal que incluye METU; con la
   diagonal de SDNET (0.824) la caída es 0.169 (también Tukey 0.213).
3. "At most 0.055 (SDNET-Deck, 0.754 to 0.808)" da 0.054.
4. "At most 0.165" da 0.166 con la Tabla 5.
5. Tabla 9: −0.034 frente a 0.562 − 0.597 = −0.035.
6. "0.258" frente a 0.867 − 0.610 = 0.257.
7. "54.9 %" aparece dos veces con significados distintos.
8. "Raises recall" (contribución 3, Conclusión) frente a p_Holm = 0.21.
9. "Holds"/"persisted" frente a 0.345 → 0.265.
10. "Lost recall, not precision" frente a la Tabla 6.
11. Fig. 1 sin reescalado.
12. Santos et al.: cifras citadas frente al resumen publicado.

## Abreviaturas

F1 (definida en p. 10), AUROC (usada antes de definirse en el cuerpo), METU
(no redefinida en el cuerpo), TP/FP/FN/TN y CI en la Fig. 1, MD5, CUDA,
cuBLAS, cuDNN, AdamW (sin cita), P − R, W sobrecargada, SDNET-D/-Deck sin
unificar, CRediT, half-precision.

## Cambios mínimos que pide

Autoría y repositorio; los dos anclajes; MCC o exactitud balanceada; umbral
sin sesgo; bootstrap jerárquico; modelo mixto; tercera colección; ajuste fino
con pocas etiquetas del destino; prevalencia realista; medir escala física y
anchura; quitar "replicación" y "a priori" sin registro; corregir
inconsistencias, abreviaturas y frases aforísticas.
