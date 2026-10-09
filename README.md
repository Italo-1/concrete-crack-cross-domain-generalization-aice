# Protocolo - P1

**Título:** Cross-Domain Generalization of Concrete Crack Classifiers: A
Transfer Matrix Across Surfaces and Capture Campaigns
**Revista destino:** *AI in Civil Engineering* (Springer Nature / Tongji
University), Original Article. Ficha de la revista en [`JOURNAL.md`](JOURNAL.md).
**Fecha de inicio:** 2 de octubre de 2026
**Responsable:** practicante línea B
**Estado:** manuscrito completo tras tres rondas de revisión adversarial
([`paper/main.pdf`](paper/main.pdf)). Pendientes antes del envío: lista de
autores y DOI de Zenodo para código, resultados y pesos
([`ENVIO.md`](ENVIO.md)).

> **Antecedente.** La matriz de la condición base se ejecutó primero en una
> versión anterior del estudio, no publicada y entrenada sin determinismo. Aquí
> todo se reentrenó desde cero, de forma determinista, con el diseño y las
> pruebas fijados antes de lanzar. Ninguna cifra procede de la versión
> anterior. El registro completo, con fechas, está en
> [`BITACORA.md`](BITACORA.md).

## Pregunta de investigación

Un clasificador de grietas en hormigón que supera el 95 % dentro de su propia
colección, ¿cuánto conserva al cambiar de superficie y al cambiar de campaña de
captura (cámara, distancia, sitio, país)? ¿Depende la pérdida de la dirección
del cambio? ¿Cómo falla el modelo fuera de su dominio, y qué parte de la
pérdida se debe al ajuste fino frente a lo que ya trae la representación
preentrenada?

## Hipótesis

Fijadas antes de entrenar. Junto a cada una, lo que se encontró.

| Hipótesis | Resultado |
|---|---|
| F1 dentro de dominio ≥ 0.97 en los cuatro dominios | Solo en METU (0.998); SDNET2018: 0.791-0.855 |
| Caída moderada entre superficies, mayor entre campañas | No: depende de la dirección (SDNET→METU 0.782, METU→SDNET 0.437) |
| Mayor caída cuanto más grande la arquitectura | No detectada (4 modelos, 1.5-11.2 M parámetros) |
| **H-EXP2.** De METU a SDNET2018 el fallo es pérdida de recall con precisión alta | Sí: recall 0.320, precisión 0.857 (test balanceado) |
| **H-EXP3.** La sonda lineal cae menos que el ajuste fino respecto a su diagonal | En las particiones con origen SDNET2018 sí; de METU a SDNET2018, no |
| **H-EXP4.** Tres fuentes a igual presupuesto ≥ media de una sola fuente | Sí (+0.134); no difiere de la mejor fuente (−0.004) |
| **H-escala.** Aumentación de escala sube el F1 de METU a SDNET2018 (añadida tras la ronda 1; protocolo en `BITACORA.md`, 03/10) | No se cumple: +0.055, p = 0.052; AUROC sin cambio. No concluyente |

## Variables

- **Independientes:** arquitectura (ResNet-18, EfficientNet-B0,
  MobileNetV3-Small, ViT-Tiny), dominio de origen, dominio de destino,
  condición de entrenamiento (base, aumentación agresiva, ecualización de
  histograma, aumentación de escala), régimen (ajuste fino o sonda lineal),
  número de fuentes (1 o 3), partición (por parche o agrupada por foto).
- **Dependientes:** F1 de la clase grieta (principal); precisión, recall,
  conteos de la matriz de confusión, exactitud balanceada, MCC, AUROC y
  precisión media; caída relativa anclada en la diagonal del origen y del
  destino.
- **Controladas:** semillas 42, 43 y 44; entrada 224×224; pesos ImageNet de
  `timm`; 2000 + 2000 parches por dominio (2800/600/600, test con 300 + 300);
  mismos hiperparámetros para todas las arquitecturas (AdamW, lr 3e-4,
  wd 1e-4, coseno, hasta 12 épocas, paciencia 3 sobre F1 de validación);
  entrenamiento determinista.

## Diseño

Cuatro dominios: los tres subconjuntos de SDNET2018 (tableros de puente,
pavimentos y muros, misma cámara y equipo) y la colección METU (otro país,
cámara y equipo). Cada modelo se entrena en un dominio y se prueba en los
cuatro: matriz de transferencia 4×4.

| Bloque | Condiciones | Entrenamientos | Celdas |
|---|---|---|---|
| Matriz base | 4 arquitecturas × 3 semillas × 4 orígenes | 48 | 192 |
| Intervenciones | aumentación agresiva, ecualización | 96 | 384 |
| Aumentación de escala | factor U(0.5, 1.5), recorte o reflejo | 48 | 192 |
| Sonda lineal | cabeza lineal sobre rasgos congelados | 48 cabezas | 192 |
| Tres fuentes | un dominio excluido, 2800 imágenes en total | 48 | 48 en el dominio excluido |
| Partición agrupada por foto | condición base, SDNET2018 agrupado | 48 | 192 |

Las celdas fuera de diagonal se agrupan en tres particiones direccionales:
entre superficies (dentro de SDNET2018, el control), SDNET2018→METU y
METU→SDNET2018. Con test balanceado, un clasificador que responde "grieta" a
todo obtiene F1 = 2/3, que se usa como referencia junto con la exactitud
balanceada (0.5 para ese clasificador).

## Prueba estadística

*Decidida antes de ver resultados*, salvo donde se indica.

- **Unidad de análisis:** la corrida (arquitectura × origen × semilla),
  promediando las celdas que aporta a cada partición.
- **Particiones direccionales:** Wilcoxon de rangos con signo cuando salen de
  las mismas corridas y Welch cuando no; Holm sobre las seis comparaciones;
  biserial de rangos o δ de Cliff; IC bootstrap remuestreando corridas
  (10 000 remuestreos, semilla 42).
- **Precisión frente a recall, sonda frente a ajuste fino, tres fuentes frente
  a una:** diferencias emparejadas por corrida, Wilcoxon, Holm dentro de cada
  familia, biserial de rangos e IC bootstrap.
- **ANOVA de dos vías** (arquitectura × par) con Levene, Shapiro-Wilk y Tukey:
  se reporta, pero no se basa ninguna conclusión en él (celdas no
  independientes; supuestos rechazados).
- **Añadidos con los resultados ya conocidos**, y declarados así en el
  artículo: AUROC, AP y umbral recalibrado con k etiquetas del destino
  (ronda 1); caída anclada en el destino, exactitud balanceada, MCC, umbral de
  Youden y precisión a la prevalencia de la colección (ronda 2).

## Criterio de interés

- **Dirección:** si la transferencia entre campañas depende de la dirección,
  una sola cifra por par de conjuntos de datos describe mal el problema, y una
  evaluación cruzada debe dar las dos direcciones.
- **Modo de fallo:** si fuera de dominio se pierde recall con precisión alta,
  revisar los parches marcados no revela el problema; hacen falta imágenes
  etiquetadas del sitio que incluyan grietas.
- **Origen de la pérdida:** si la sonda lineal falla igual, la pérdida no la
  crea el ajuste fino; si tres fuentes ayudan a igual presupuesto, un operador
  con varios sitios puede aprovecharlo.

## Datasets

| Nombre | Fuente | Licencia | Uso |
|---|---|---|---|
| SDNET2018 | Utah State University, DOI 10.15142/T3TD19; artículo de datos Dorafshan et al. (2018), *Data in Brief* 21 | Uso académico, citar | Tres dominios: tableros (13 620 parches, 54 fotos), pavimentos (24 334, 104), muros (18 138, 72) |
| METU (Concrete Crack Images for Classification) | Mendeley Data, DOI 10.17632/5y9wdsg2zt.2 | CC BY 4.0 | Cuarto dominio (40 000 parches, 50 % con grieta) |

- Duplicados exactos eliminados por MD5 antes de muestrear (1611, ninguno
  entre dominios).
- Submuestra balanceada de 2000 + 2000 parches por dominio, partida 70/15/15
  por clase. La lista de parches de cada partición (por parche y agrupada por
  foto) está en `data/processed/manifiesto_agrupado.csv`.
- Las imágenes no se incluyen en el repositorio: se descargan de sus fuentes
  con `src/01_download.py`.

## Citas obligatorias

Seis trabajos de *AI in Civil Engineering*, leídos completos antes de
citarlos: Abeysuriya et al. (2026), revisión sistemática que pide evaluación
entre conjuntos de datos; Oliveira Santos et al. (2022), adaptación de dominio
de un clasificador entrenado en METU (54 % → 84 % en una presa); Lateef & Yu
(2026), defectos de puentes, pide evaluación entre puentes y sitios;
Fountoukidou et al. (2024), separación por puente y distancia de captura;
Luleci & Necati Catbas (2023), transferencia entre puentes en monitorización
estructural; Hasan & Lu (2025), requisitos del conjunto de prueba y
comprobación tras el despliegue. Cada referencia de `paper/refs.bib` lleva una
nota con cómo se verificó y qué se le atribuye.

## Resultados principales

| Partición | F1 | IC 95 % (corridas) | Celdas > 0.667 | Exactitud balanceada |
|---|---|---|---|---|
| Dentro de dominio | 0.867 | 0.845-0.891 | 100 % | 0.871 |
| Entre superficies | 0.654 | 0.631-0.678 | 48.6 % | 0.709 |
| SDNET2018 → METU | 0.782 | 0.733-0.830 | 69.4 % | 0.812 |
| METU → SDNET2018 | 0.437 | 0.407-0.464 | 16.7 % | 0.629 |

- Las dos direcciones entre colecciones difieren en 0.345 de F1 (0.265 con la
  partición agrupada por foto). Medida desde la diagonal del destino,
  SDNET→METU pierde tanto como entre superficies: parte de la asimetría se
  debe a que METU es un destino fácil. METU→SDNET pierde más con cualquier
  anclaje.
- 78 de las 144 celdas fuera de diagonal quedan en o bajo el F1 del
  clasificador trivial, 77 con precisión mayor que recall; todas superan el
  azar en exactitud balanceada.
- De METU a SDNET2018 el AUROC es 0.699; un umbral de Youden elegido con 50
  parches etiquetados del destino solo sube la exactitud balanceada de 0.629 a
  0.656.
- Ninguna intervención reduce la pérdida de forma significativa; la sonda
  lineal falla igual que el ajuste fino de METU a SDNET2018.

## Figuras

| Figura | Contenido | Script |
|---|---|---|
| Fig. 1 | Pipeline experimental | `paper/fig_pipeline.tex` |
| Fig. 2 | Un parche con y sin grieta por dominio | `src/05_figures.py` |
| Fig. 3 | Matriz de transferencia, media y por arquitectura | `src/05_figures.py` |
| Fig. 4 | F1 por arquitectura y partición | `src/05_figures.py` |
| Fig. 5 | Precisión frente a recall de las 192 celdas | `src/05_figures.py` |
| Fig. 6 | Errores extremos de un modelo METU→SDNET-Wall | `src/05_figures.py` |
| Fig. 7 | Sonda lineal frente a ajuste fino | `src/05_figures.py` |
| Fig. 8 | Tres fuentes frente a una | `src/05_figures.py` |

Versiones para el envío (EPS y TIFF a 600 dpi) en `paper/envio/`, generadas
con `src/07_figuras_envio.py`. Ninguna figura procede de un modelo de
generación de imágenes.

## Riesgos y limitaciones

- **Un solo par de campañas independientes:** la asimetría descansa en
  SDNET2018 frente a METU y no se puede separar de las propiedades de METU
  (encuadre centrado, criterio de etiquetado, dificultad). Hace falta una
  tercera colección de hormigón (no Crack500, que es asfalto).
- **Etiquetas sin auditar** y explicación por encuadre sin medir (anchura y
  posición de la grieta).
- **Presupuesto de datos fijo** (2800 imágenes de entrenamiento, 12 épocas):
  las diagonales de SDNET2018 quedan por debajo de lo publicado con todos los
  datos.
- **Un solo sorteo de datos:** los IC cubren la variación entre corridas, no
  entre muestras; cambiar solo el test movió METU→SDNET 0.055.
- **METU no se puede agrupar por foto:** su diagonal puede incluir fuga entre
  parches.
- **Modelos pequeños preentrenados en ImageNet;** sin modelos de fundación.

## Pipeline

| Script | Qué hace |
|---|---|
| `src/01_download.py` | Descarga SDNET2018 y METU |
| `src/02_preprocess.py` | Deduplicación, submuestra balanceada y partición por parche |
| `src/02b_particion_agrupada.py` | Partición agrupada por foto de SDNET2018 |
| `src/entrenamiento.py` | Entrenamiento determinista y evaluación (módulo común) |
| `src/03_experiment.py` | Matriz 4×4 por condición (base, agresiva, ecualización, escala) |
| `src/03b_sonda_lineal.py` | Sonda lineal sobre rasgos congelados |
| `src/03c_multifuente.py` | Tres fuentes con un dominio excluido |
| `src/lanzar_f1.py` | Lanzador sin consola con reintentos (`--reanudar`) |
| `src/piloto_comparar.py` | Comprueba el determinismo (dos corridas idénticas) |
| `src/comparar_original.py` | Compara con la versión anterior según criterios fijados |
| `src/04_stats.py` | Estadística y tablas |
| `src/05_figures.py` | Figuras 2-8 |
| `src/06_apendice.py` | Tablas del apéndice (`paper/apendice.tex`) |
| `src/07_figuras_envio.py` | Figuras en EPS y TIFF para el envío |
| `src/08_umbral_auroc.py` | AUROC, AP y recalibración del umbral |
| `src/09_ronda2.py` | Anclaje en el destino, exactitud balanceada, MCC, Youden, prevalencia |
| `src/tabla_arquitecturas.py` | Tabla 1 (parámetros y dimensión de rasgos) |

**Reproducir.** Python 3.11 con `requirements.txt` (PyTorch con CUDA 12.6) y
`pip install -e .` para la librería común `fieutils/`. Los scripts leen las
imágenes de una ruta local de datos que hay que ajustar al descargar las
colecciones. Entrenamiento con `CUBLAS_WORKSPACE_CONFIG=:4096:8`. Todas las
cifras del artículo salen de `results/tables/` (estadística en
`resumen_estadistico.json`). Los pesos de los 48 modelos base (1 GB) se
publicarán en Zenodo.

## Revisión y uso de herramientas de IA

El manuscrito pasó tres rondas de revisión adversarial en contexto limpio
(revisor de la revista, revisor hostil al encuadre y pasada anti-IA). Los
informes, la verificación de cada afirmación contra los datos y las fuentes, y
lo que se aplicó o no con su motivo están en `paper/revision/`. El código, la
redacción y las revisiones se hicieron con un asistente de modelo de lenguaje
(Claude, Anthropic), como declara el artículo en su sección 3.9; todas las
cifras salen de ejecutar el código sobre las colecciones citadas.
