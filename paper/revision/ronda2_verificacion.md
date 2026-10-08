# Ronda 2: verificación de las afirmaciones de hecho (03/10/2026)

Cada afirmación de los tres informes (`ronda2_revisor.md`, `ronda2_hostil.md`,
`ronda2_antiIA.md`) que se puede contrastar con datos o fuentes, contrastada
antes de aceptarla. Cálculos sobre `results/tables/resultados_matriz.csv`
(condición base, partición original).

## Confirmadas, cambian conclusiones cualitativas (se avisa al usuario)

1. **La caída depende del anclaje.**

   | Partición | F1 | Caída anclada en el origen | Caída anclada en el destino | Pérdida absoluta frente a la diagonal del destino |
   |---|---|---|---|---|
   | Entre superficies | 0.654 | 0.205 | 0.204 | 0.169 |
   | SDNET→METU | 0.782 | 0.055 | **0.217** | 0.216 |
   | METU→SDNET | 0.437 | 0.562 | 0.474 | 0.387 |

   METU→SDNET es la peor con cualquier anclaje. Pero "SDNET→METU pierde menos
   que entre superficies" solo vale con el anclaje en el origen: anclada en el
   destino, SDNET→METU pierde lo mismo o algo más. Parte de los 0.345 de F1
   absoluto refleja que METU es un destino fácil.
2. **"No mejor que el clasificador trivial" es cierto solo en F1.** Las 77
   celdas por debajo de 2/3 tienen exactitud 0.557-0.735 y MCC 0.17-0.52;
   ninguna tiene exactitud ≤ 0.5 ni MCC ≤ 0.05. Con test balanceado, la
   exactitud es igual a la exactitud balanceada, y la del clasificador trivial
   es 0.5. Medias por partición: exactitud 0.709 / 0.812 / 0.629; MCC 0.448 /
   0.638 / 0.332.
3. **Prevalencia.** Con el punto de operación medio METU→SDNET (recall 0.320,
   tasa de falsos positivos 0.053), la precisión sería 0.38, 0.48 y 0.58 con
   las prevalencias de SDNET-P, -D y -W (10.7, 14.9 y 21.2 %). "Usually right
   when it does" (5.4) no se sostiene a la prevalencia real. La referencia
   2π/(1+π) baja a 0.19, 0.26 y 0.35.
4. **El "mejor umbral" da F1 ≥ 2/3 por construcción** (el umbral mínimo lo
   marca todo como grieta), y maximizar F1 con k etiquetas empuja hacia "todo
   grieta". "La recalibración solo llega al nivel trivial" es en parte un
   artefacto del criterio. El AUROC (0.699) no depende de esto.

## Confirmadas, de redacción o presentación

- Escala: el IC excluye 0 con p = 0.052; "no se cumple" sigue (lectura fijada),
  pero "did not support" debe ser "no concluyente"; "raises recall" en la
  contribución 3 y la Conclusión contradice p_Holm = 0.21.
- Sonda: las sondas entrenadas en SDNET separan SDNET (diagonales 0.781-0.830,
  Tabla A3); lo que no transfiere es la frontera aprendida en METU, no "las
  características". "Already present in the pretrained features" exagera.
- ViT-Tiny usa pesos `augreg_in21k_ft_in1k` (ImageNet-21k); tamaño confundido
  también con preentrenamiento. Añadir a la Limitación 4.
- "Replication" para un reentrenamiento del mismo código y datos: el término
  correcto es reanálisis o reentrenamiento confirmatorio.
- Fig. 1 no incluye el reescalado (verificar `fig_pipeline.tex`).
- Contribución 3 y Conclusión: frases demasiado largas; títulos de 5.1/5.2 y
  "not one quantity" (coinciden revisor principal y pasada anti-IA).
- Abreviaturas: MD5, CUDA, cuBLAS, cuDNN, AdamW (sin cita), "P − R", W
  sobrecargada, METU no redefinida en el cuerpo.

## Refutadas

- **Santos et al. (2022), "+54 puntos":** el texto completo dice "improves the
  mean accuracy … from 54% to 84%. This corresponds to a relevant improvement
  of 54%", es decir, relativa (30/54). La cita del manuscrito es correcta.
- **SDNET-Deck, 54 fotos × 252 ≠ 13 620:** contado sobre los datos crudos
  (solo lectura en `03-tehnicki-glasnik-P1/data/raw`): 54 fotos; 52 con 252
  parches, una con 256 y otra con 260 (números de parche repetidos entre
  carpetas). El recuento es correcto.
- **Rashid et al. (2025) ya muestra la asimetría (revisor hostil):** en
  exactitud, SCD (METU)→SDNET2018 iguala o supera a la dirección contraria en
  los 4 modelos (76/84/81/82 frente a 56/77/50/82); en F1 la dirección del
  manuscrito sale en 2 de 4 (VGG16 82 frente a 61; ResNet50 76 frente a 64).
  Su test de SDNET2018 conserva el desbalance (estratificado, §4.4; "Imbalanced",
  Tabla 3), así que la exactitud premia predecir "sin grieta". Sirve para el
  encuadre.

## Confirmada parcialmente

- **SDNET-P→METU (0.941) supera la diagonal de SDNET-P (0.855):** cierto; el
  manuscrito ya lo dice en 5.1.

## No verificables sin experimentos (no se aplican salvo decisión del usuario)

Tercera colección; evaluación en FP32 (solo se guardaron los pesos de la
base); modelo mixto; entrenamiento con todos los datos; ajuste fino con k
etiquetas del destino; auditoría de etiquetas; control de color; escala física
medida.
