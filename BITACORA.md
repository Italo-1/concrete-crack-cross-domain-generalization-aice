# Bitácora - P1 · versión para *AI in Civil Engineering*

> Registro cronológico del proyecto, conservado tal como se escribió: cada
> decisión, verificación de referencias, ejecución y ronda de revisión, con su
> fecha. El protocolo vigente y el resumen del estudio están en
> [`README.md`](README.md). Algunos archivos que se mencionan aquí
> (`PLAN_AICE.md`, `PROMPT_INICIO.md`, `referencias/`, `checkpoints/`) no están
> en el repositorio público: contienen nombres de autores aún no decididos,
> textos con derechos de autor o pesos de modelos (1 GB, previstos para
> Zenodo).

---
# Protocolo - P1 · extensión para *AI in Civil Engineering*

> **Sobre este repositorio.** Versión del artículo P1 preparada para *AI in
> Civil Engineering* (borrador previo al envío; la autoría y el DOI de Zenodo
> están pendientes). Contiene el código (`src/`), la librería común
> `fieutils/`, los resultados por celda, las predicciones por imagen y las
> figuras (`results/`), el manuscrito LaTeX y su PDF (`paper/main.pdf`), la
> lista de parches y particiones (`data/processed/manifiesto_agrupado.*`) y
> las revisiones adversariales con su verificación (`paper/revision/`). No
> contiene las colecciones de imágenes (públicas: SDNET2018, DOI
> 10.15142/T3TD19; METU, DOI 10.17632/5y9wdsg2zt.2), ni los textos de los
> trabajos citados (derechos de autor), ni los pesos de los modelos (1 GB,
> previstos para Zenodo). Entorno: `requirements.txt` y `pip install -e .`
> para `fieutils`. Los scripts leen las imágenes desde la carpeta de datos de
> la versión anterior (`03-tehnicki-glasnik-P1/data/raw`); hay que ajustar esa
> ruta al reproducir. Esta bitácora está en español y registra cada decisión
> con su fecha.

**Revista destino:** *AI in Civil Engineering* (Springer Nature / Tongji
University), Original Article. Ficha en [`JOURNAL.md`](JOURNAL.md).
**Fecha de inicio:** 02/10/2026
**Responsable:** `[Author list pending]`
**Orden de trabajo:** [`PLAN_AICE.md`](PLAN_AICE.md). Donde la guía del
encargado (`referencias/guia_expansion_p1.pdf`) y el plan difieren, manda el
plan.
**Versión anterior:** `C:\lineaB\03-tehnicki-glasnik-P1\` (nunca enviada). Es
de solo lectura: los datos se leen por ruta desde ahí y no se modifica nada.

## Qué cambia respecto a la versión de *Tehnički glasnik*

El diseño base (4 dominios × 4 arquitecturas × 3 semillas, matriz 4×4, tres
condiciones de entrenamiento) se mantiene y se **vuelve a ejecutar entero**,
con tres añadidos:

1. **EXP2 · Precisión y recall por celda.** El CSV original solo guarda F1.
   Cada celda guarda ahora TP, TN, FP, FN, precisión, recall y F1, y las
   predicciones por imagen del test.
2. **EXP3 · Sonda lineal.** Backbone ImageNet congelado, solo se entrena la
   cabeza lineal. Misma matriz 4×4.
3. **EXP4 · Multi-fuente *leave-one-out*.** Se entrena con tres dominios y se
   evalúa en el cuarto, con el conjunto combinado reducido al tamaño de un
   entrenamiento de una sola fuente.

EXP1 (tercer conjunto de datos) queda como trabajo futuro declarado.

### Por qué se reentrena todo (no solo lo nuevo)

- No hay checkpoints ni predicciones de la corrida original: P/R exige
  reentrenar (`PLAN_AICE.md` §3.1).
- El original se entrenó con `determinista=False`. Una base nueva no reproduce
  la vieja, y el análisis de mitigaciones empareja base y mitigación por
  (arquitectura, semilla, celda): hay que reentrenar las tres condiciones
  juntas (§3.2). Todas las cifras del artículo salen de los resultados nuevos.

### Decisiones tomadas el 02/10 con el usuario

- **`fieutils` no se toca** (lo comparte P5). La instrumentación (conteos,
  predicciones, checkpoints, determinismo) vive en código propio de
  `06-aice-P1/src/`.
- **Sonda lineal con varianza entre semillas**: características extraídas una
  vez por (arquitectura, dominio, partición); la cabeza lineal se entrena con
  optimizador y parada temprana, variando la semilla (inicialización y orden de
  lotes). Una regresión logística con solver determinista daría el mismo
  número con las tres semillas.

## Pregunta de investigación

¿Cuánto F1 conserva un clasificador de grietas al cambiar de superficie y al
cambiar de campaña de captura, depende de la dirección del cambio, y qué parte
de la pérdida se debe al ajuste fino sobre el dominio de origen frente a lo que
ya trae la representación preentrenada?

## Hipótesis

Las de la versión anterior se conservan como estaban (diagonal ≥0.97, caída
moderada entre superficies, mayor entre campañas, mayor para arquitecturas más
grandes) y se reportan como hipótesis a priori. Para lo nuevo:

- **H-EXP2.** En la dirección METU→SDNET2018, el fallo dominante es pérdida de
  recall con precisión alta, no sobredetección (la única celda medida en la
  versión anterior, Fig. 4, dio precisión 0.97 y recall 0.11).
- **H-EXP3.** La sonda lineal tiene F1 dentro de dominio menor que el ajuste
  fino y **caída relativa fuera de dominio menor**, porque no puede
  sobreajustarse a rasgos propios del origen.
- **H-EXP4.** Entrenar con tres dominios a igual tamaño de muestra da en el
  dominio excluido un F1 al menos igual a la media de las tres transferencias
  de una sola fuente hacia ese dominio.

## Variables

- **Independientes:** arquitectura (ResNet-18, EfficientNet-B0,
  MobileNetV3-Small, ViT-Tiny), dominio de origen, dominio de destino,
  condición de entrenamiento (base, aumento agresivo, ecualización), régimen
  (ajuste fino / sonda lineal), número de fuentes (1 / 3).
- **Dependientes:** F1 de la clase grieta (principal), precisión, recall,
  conteos de la matriz de confusión, caída relativa.
- **Controladas:** semillas 42, 43, 44; recorte 224×224; pesos ImageNet de
  `timm`; 2800/600/600 imágenes por dominio, balanceadas; mismos
  hiperparámetros para todas las arquitecturas (AdamW, lr 3e-4, wd 1e-4,
  coseno, ≤12 épocas, paciencia 3 sobre F1 de validación, precisión mixta).

## Diseño

| Bloque | Condiciones | Entrenamientos | Celdas |
|---|---|---|---|
| E1 · EXP2 (reentrenamiento con P/R) | base, agresivo, ecualizado × 4 arq. × 3 semillas × 4 orígenes | 144 | 576 |
| E2 · EXP3 (sonda lineal) | 4 arq. × 3 semillas × 4 orígenes | 48 cabezas lineales | 192 |
| E3 · EXP4 (*leave-one-out*) | 4 dominios excluidos × 4 arq. × 3 semillas | 48 | 48 |

- **Determinismo:** `torch.use_deterministic_algorithms(True)`,
  `CUBLAS_WORKSPACE_CONFIG=:4096:8`, cuDNN determinista, sin autotuner,
  generador del `DataLoader` sembrado. El piloto (E0) corre **dos veces** la
  misma configuración y compara: si no coinciden bit a bit, se declara "casi
  determinista" y se mide la discrepancia.
- **Checkpoints:** mejor época de la condición base (48 modelos) en
  `checkpoints/` (fuera de git). Sirven para EXP1 futuro y para que la Fig. 4
  salga de un modelo de la matriz, no de uno reentrenado aparte.
- **Sonda lineal (EXP3), fijado ahora:** backbone `timm` con `num_classes=0`
  en modo evaluación, sin aumento; características de train/val/test extraídas
  una vez por (arquitectura, dominio). Cabeza `nn.Linear(d, 2)`, AdamW
  (lr 1e-3, wd 1e-4), lote 256, hasta 100 épocas, paciencia 10 sobre F1 de
  validación del dominio de origen. La semilla fija inicialización y orden de
  lotes.
- **Multi-fuente (EXP4), fijado ahora:** train combinado de 2800 imágenes
  (934/933/933 por fuente, mitad grieta y mitad no grieta en cada fuente) y val
  combinado de 600 (200 por fuente), muestreados con la semilla 42 de los
  conjuntos train/val ya definidos. Test: las 600 imágenes del dominio
  excluido. Condición base, mismos hiperparámetros que E1.

## Prueba estadística (fijada antes de ver resultados)

**Unidad de análisis.** Cada modelo aporta varias celdas, así que los
contrastes principales usan la **corrida** (arquitectura × origen × semilla)
como unidad, promediando las celdas que aporta a cada partición. Las pruebas a
nivel de celda se reportan como secundarias.

1. **Réplica de la versión anterior (E1).** Mismo análisis: ANOVA de dos vías
   a priori (con Levene y Shapiro-Wilk), Welch con Holm entre las cuatro
   particiones direccionales con δ de Cliff, IC bootstrap (10 000 remuestreos),
   caída relativa anclada en la diagonal del propio origen y semilla, y
   mitigaciones emparejadas por (arquitectura, par, semilla).
2. **Comparación con el original (E1).** Diferencia por celda entre F1 nuevo y
   original (media, media absoluta, máximo). **Se considera cambio cualitativo,
   y se para a avisar**, si ocurre cualquiera de estos:
   (a) la diferencia SDNET→METU menos METU→SDNET cambia de signo o deja de ser
   significativa tras Holm;
   (b) cambia el orden de las cuatro particiones direccionales por F1 medio;
   (c) el IC 95 % de "agresivo − base" deja de cruzar cero, o el de
   "ecualizado − base" pasa a cruzarlo o a ser positivo;
   (d) la fracción de celdas fuera de diagonal por encima de 0.667 sale del
   intervalo 35-65 %.
3. **EXP2.** Precisión y recall medios (± DE, IC bootstrap) por partición
   direccional y arquitectura. Para H-EXP2: diferencia emparejada
   precisión − recall por corrida en cada partición, Wilcoxon de rangos con
   signo, Holm sobre las tres particiones fuera de diagonal, correlación
   biserial de rangos como tamaño del efecto.
4. **EXP3.** Para cada corrida, diferencia emparejada entre la caída relativa
   del ajuste fino (condición base de E1) y la de la sonda lineal, por
   partición. Wilcoxon de rangos con signo, Holm sobre las tres particiones,
   correlación biserial de rangos e IC bootstrap de la diferencia media. Se
   reporta también la comparación del F1 absoluto fuera de dominio.
5. **EXP4.** Para cada (arquitectura, semilla, dominio excluido), F1 del
   modelo multi-fuente frente a (i) la media y (ii) el máximo de los tres F1 de
   una sola fuente hacia ese dominio (condición base de E1). Wilcoxon de rangos
   con signo sobre las 48 parejas, Holm sobre las dos comparaciones,
   correlación biserial de rangos e IC bootstrap. Se desglosa por dominio
   excluido.

- **Tamaño del efecto:** δ de Cliff (no emparejado), correlación biserial de
  rangos (emparejado), η² parcial (ANOVA).
- **Software:** scipy, statsmodels; semilla 42 en todos los remuestreos.

## Criterio de interés

- **EXP3, si la sonda cae menos:** el ajuste fino aprende rasgos propios del
  origen que no transfieren; la recomendación práctica cambia (congelar más
  capas). **Si cae igual o más:** la pérdida está ya en la representación
  preentrenada y el ajuste fino no es el culpable.
- **EXP4, si el multi-fuente supera a la media:** la diversidad de fuentes
  ayuda a igual cantidad de datos, lo que un operador con varios sitios puede
  aprovechar. **Si no:** juntar campañas no sustituye a validar en el sitio.
- **EXP2:** separar sobredetección de pérdida de recall cambia el riesgo
  operativo (falsas alarmas frente a grietas no vistas) en cualquier caso.

## Datasets

| Nombre | Fuente | Licencia | Verificado |
|---|---|---|---|
| SDNET2018 | digitalcommons.usu.edu; artículo de datos Dorafshan et al. (2018), *Data in Brief* 21, doi:10.1016/j.dib.2018.11.015 (verificado 01/09 en la versión anterior) | Uso académico, citar | 20/08 (versión anterior); se leen de `03-…/data/` |
| METU / Concrete Crack Images for Classification | Mendeley Data, DOI 10.17632/5y9wdsg2zt.2 | CC BY 4.0 | 21/08 (versión anterior) |

Preprocesados (sin duplicar): `C:\lineaB\03-tehnicki-glasnik-P1\data\processed\`
(`*_X.npy`, `*_y.npy`, `manifiesto.csv`, `manifiesto.json`).

## Citas obligatorias de la revista destino

Cinco, leídas completas el 02/10: Abeysuriya et al. 2026 (C1), Luleci &
Necati Catbas 2023 (C2), Lateef & Yu 2026 (C4), Fountoukidou et al. 2024 (C7),
Hasan & Lu 2025 (C8). Detalle y descartes en
[`referencias/aice/LECTURAS_AICE.md`](referencias/aice/LECTURAS_AICE.md).

## Pipeline

| Script | Estado |
|---|---|
| `src/01_download.py`, `src/02_preprocess.py` | Copias de la versión anterior; **no se ejecutan** (los datos ya existen en `03-…/data/`) |
| `src/03_experiment.py` | Copia; se adapta en E0 (rutas, conteos, predicciones, checkpoints, determinismo) |
| `src/04_stats.py`, `src/05_figures.py` | Copias; se adaptan en E4 |

---

# Hallazgos

## 02/10 · F0

1. **La partición original es por parche, no por foto de origen.**
   Los nombres de archivo de SDNET2018 codifican la foto de la que sale cada
   parche (`7001-115.jpg` → foto 7001): SDNET-Deck tiene 54 fotos de origen
   (crack y no crack), SDNET-Pavement 104, SDNET-Wall 72 (contado sobre
   `data/raw/` el 02/10). METU no trae ese
   dato, pero según Özgenel y Gönenç Sorguç sus 40 000 parches salen de 500
   fotografías. `02_preprocess.py` sortea train/val/test **por parche**
   (`rng.shuffle` de índices dentro de dominio y clase), así que parches vecinos
   de una misma foto caen en train y en test. **Medido sobre `manifiesto.csv`:
   en SDNET-Deck, -Pavement y -Wall, las 600 imágenes de test (100 %) salen de
   una foto que también aporta imágenes al entrenamiento**; todas las fotos con
   imágenes en test tienen también imágenes en train. En METU no se puede
   medir.
   - **Viabilidad de agrupar por foto (SDNET):** con las mismas 4000 imágenes
     cacheadas por dominio hay 54/104/72 fotos, con una mediana de 34/18/20
     parches con grieta por foto (máximo 83/51/87). Una partición 70/15/15 por
     foto es posible; el test exacto 300/300 obliga a descartar algunos parches
     de las fotos de test, y el train queda algo por debajo de 2800.
   - **A qué afecta:** solo a las **diagonales** (dentro de dominio). Las
     celdas fuera de diagonal comparan dominios distintos y no comparten fotos.
     Como la caída relativa se ancla en la diagonal, una diagonal inflada
     infla también la caída.
   - Abeysuriya et al. (2026, §4.1.3) señalan esta misma fuga como problema
     de la literatura.
   - **Pendiente de decisión del usuario** antes de E1 (ver bitácora).
2. **Tiempos de cómputo: el plan los sobrestima.** Según
   `segundos_matriz` del CSV original: base 1.98 h, agresivo 3.89 h,
   ecualizado 2.24 h → **~8.1 h las tres**, no ~24 h. El determinismo añadirá
   algo; se mide en el piloto.
3. **`fijar_semilla(determinista=True)` no garantiza reproducibilidad.** Solo
   ajusta `cudnn.deterministic` y `cudnn.benchmark`; no llama a
   `torch.use_deterministic_algorithms` ni fija `CUBLAS_WORKSPACE_CONFIG`. El
   código de E0 lo hace por su cuenta.
4. **Datos de AICE que corrigen la guía del encargado:** abstract de
   **150-250** palabras (no 250-300); citas **autor-año APA**, lista
   alfabética (no numeradas); **biografías de autores obligatorias**; uso de
   LLM **documentado en Métodos**; revisión **single-blind**; no cobra APC
   (lo cubre Tongji University). Detalle en `JOURNAL.md`.
5. **Citas AICE de la guía, leídas:** C1 son 46 estudios, no 47, y su ámbito
   es post-sismo; **C2 no contiene la asimetría** ni la idea "complejo→simple
   conserva más" que la guía propone atribuirle; C5 no trata lo que la guía
   dice y se descarta; la frase que la guía pone en boca de C6 está en el Aims &
   Scope de la revista, no en el editorial. Detalle en `LECTURAS_AICE.md`.

---

# Bitácora

## 02/10 - `[Author list pending]`
- Hecho: lectura completa de plan, guía del encargado, guía metodológica,
  manuscrito de *Tehnički glasnik*, plantilla Springer y código original.
  Estructura de `06-aice-P1/` creada y `src/` copiado. Web de AICE revisada y
  guardada en `paper/evidencia/` (guías de envío, tarifas, Aims & Scope, ética,
  checklist, política de IA de Springer). Las 8 citas AICE de la guía leídas
  completas; `JOURNAL.md` completado. Protocolo de la extensión, con las
  pruebas de EXP2-EXP4 fijadas antes de entrenar.
- Decisión del usuario sobre el hallazgo 1: **opción A**. La matriz principal
  (E1-E3) usa la partición original; la condición base se entrena además con
  una partición agrupada por foto en SDNET2018 como análisis de sensibilidad
  (48 entrenamientos más).
- **E0 hecho hasta el piloto:**
  - `src/02b_particion_agrupada.py` → `data/processed/manifiesto_agrupado.csv`.
    Mismas imágenes cacheadas; fotos de test/val/train disjuntas (comprobado
    con `assert`). Test y val 300/300 exactos; train **1390 por clase** en los
    tres SDNET (2780, frente a 2800 en METU, que no cambia); 10/4/14 imágenes
    descartadas al recortar test y val en D/P/W. Fotos en test: 8/16/11.
  - `src/entrenamiento.py` (propio, no toca `fieutils`): determinismo completo,
    lote fijo 64, conteos por celda y probabilidad por imagen.
  - `src/03_experiment.py` reescrito: `--particion`, conteos TP/TN/FP/FN, P, R
    y F1 por celda, predicciones por imagen en `results/predicciones/`,
    historial por época, checkpoints de la base con partición original,
    registro del entorno en `results/logs/entorno_*.json`. Las filas se anexan
    al terminar cada modelo.
  - **Lote fijo de 64:** `fieutils.vision.lote_para_vram` lo calculaba con la
    VRAM libre del momento, que cambia entre ejecuciones. En la corrida
    original las cuatro arquitecturas llegaron al tope de 64, así que el valor
    no cambia; solo deja de depender de la memoria libre.
  - **Prueba de humo** (1 época, 4 arquitecturas, SDNET-D): ninguna
    arquitectura usa operaciones sin versión determinista; no hay errores.
  - **Coste del determinismo: despreciable.** Medido por lote (64 imágenes,
    AMP): EfficientNet-B0 176 ms sin determinismo frente a 196 ms con él;
    MobileNetV3 86 frente a 110; ResNet-18 90 frente a 94. Lo que sí cuesta es
    un arranque de ~30 s por cada forma de lote nueva en EfficientNet-B0 (cuDNN
    eligiendo algoritmo), una vez por proceso; por eso la prueba de humo de una
    sola época parecía 6 veces más lenta. `channels_last` es más lento en esta
    GPU y no se usa.
  - Pesos preentrenados de `timm` en caché local: `resnet18.a1_in1k`,
    `efficientnet_b0.ra_in1k`, `mobilenetv3_small_100.lamb_in1k`,
    `vit_tiny_patch16_224.augreg_in21k_ft_in1k` (van a la tabla de
    arquitecturas, M2).
- **Piloto** (ResNet-18, semilla 42, base, partición original, 4 orígenes),
  ejecutado dos veces con salidas separadas (`results/piloto/a` y `/b`;
  comparación en `results/piloto/comparacion.json`): **idéntico bit a bit**.
  16 celdas con los mismos conteos, 9600 predicciones iguales, diferencia
  máxima de probabilidad 0 y pérdidas por época iguales. Tiempo: 10.4 min por
  matriz, igual que en la corrida original (10.4 min). El determinismo no
  cuesta tiempo apreciable.
  - Frente al F1 original de esa misma configuración: diferencia absoluta
    media 0.022 por celda (máximo 0.074, METU→SDNET-W). Es la variación que
    introducía la falta de determinismo de la corrida original; confirma que
    hay que reentrenar las tres condiciones juntas.
  - Las salidas se escriben bien: conteos, P/R/F1, predicciones por imagen
    (68 KB por modelo), historial por época.
- **Pruebas de humo de EXP3 y EXP4** (`03b_sonda_lineal.py --smoke`,
  `03c_multifuente.py --smoke`): funcionan. Sonda: 1.4 min por arquitectura con
  la extracción de características. Multi-fuente: reparto 934/933/933 de train
  y 200/200/200 de val por fuente, comprobado en
  `results/smoke_multifuente/tables/multifuente_sorteo.json`.
- **F1 lanzado a las 17:26** como proceso independiente
  (`src/lanzar_f1.bat`), en serie: E1 (tres condiciones, partición original,
  144 entrenamientos) → E1b (base con partición agrupada, 48) → E2 (sonda) →
  E3 (multi-fuente, 48). Estimado: ~8.5 h + ~2 h + ~0.1 h + ~2 h. Progreso en
  `results/logs/lanzar_f1.log`. EXP3 y EXP4 corren en la misma cadena para no
  dejar la GPU parada; la comparación de E1 con el original (criterios a-d del
  protocolo) se hace igual antes de usar nada de lo demás.
- `src/comparar_original.py`: aplica los criterios (a)-(d) a E1 y escribe
  `results/tables/comparacion_original.json`. Validado aplicándolo al CSV
  original: reproduce las cifras publicadas (0.866 / 0.652 / 0.759 / 0.472;
  agresivo −0.014 [−0.033, +0.006]; ecualizado −0.030 [−0.054, −0.006];
  49.3 % de celdas fuera de diagonal por encima de 0.667).
- Tiempo de cómputo consumido: ~35 min GPU (humo, mediciones y piloto).

## 02/10 (tarde) - F1 relanzado, F2 hecho, 04/05 adaptados - `[Author list pending]`

- **Incidente con el lanzamiento.** El primer `lanzar_f1.bat` (17:26) murió en
  la época 1 del primer modelo: `e1_stdout.txt` termina en `^C`. El `cmd`
  lanzado con `Start-Process` recibió la interrupción de consola al terminar la
  llamada que lo creó. No llegó a escribir ninguna fila. **Relanzado a las
  18:00** con `Win32_Process.Create` (WMI), que crea el proceso fuera del árbol
  y de la consola de la sesión. Comprobado: sigue vivo después de cerrar la
  llamada, y la época 0 y la 1 dan exactamente los números del piloto (F1 val
  0.7174 y 0.6977), así que el determinismo se mantiene en la ejecución larga.
  Pérdida: ~35 min de GPU.
- **F2 hecho: `paper/refs.bib` con 36 entradas**, cada una con su nota de
  verificación (qué se leyó, texto completo o resumen, y qué se le atribuye).
  Metadatos crudos en `referencias/nuevas/metadatos/`, textos en
  `referencias/nuevas/`.
  - Se conservan 16 de las 18 de la versión anterior (2 de ellas, Kos y
    Drapaluk, de *Tehnički glasnik*, como opcionales); se retiran Kolar,
    Svalina y Munđar (solo cumplían el mínimo de citas de aquella revista).
  - 5 de AICE (C1, C2, C4, C7, C8).
  - 15 nuevas: Rashid et al. 2025, ConCrackNet 2026, Chun & Kikuta 2024,
    Yang et al. 2020 (Crack500), Spencer et al. 2019, Zhou et al. 2023 (DG),
    Gulrajani & Lopez-Paz, Geirhos et al. 2020, Kumar et al. 2022, Kornblith et
    al. 2019, Kapoor & Narayanan 2023, Bouthillier et al. 2021, Welch 1947,
    Holm 1979, Cliff 1993; más el DOI del conjunto SDNET2018
    (10.15142/T3TD19), que pide la política de datos de AICE.
  - **No citadas por no poder leerlas:** Weng et al. 2023 (ScienceDirect pide
    CAPTCHA) y Phares et al. 2004 (ascelibrary bloquea). Sin resumen en ninguna
    fuente abierta. Se pueden añadir si alguien las abre a mano.
- **Hallazgos de F2 que cambian el texto:**
  1. **Ya existe una matriz de evaluación cruzada para clasificación de
     grietas**: Rashid, Mokji & Rasheed (2025, *J. Mech. Behav. Mater.*),
     6x6 con SDNET2018 y SCD (= METU). El hueco que proponía la guía ("nadie ha
     construido una matriz de transferencia completa") **ya no se puede
     escribir**. Lo que sigue siendo propio: separar superficie de campaña
     (ellos tratan SDNET2018 como un bloque), varias semillas con pruebas
     estadísticas (ellos: una ejecución), igualar prior y tamaño, el
     clasificador trivial como referencia, P/R como modo de fallo, sonda frente
     a ajuste fino y multi-fuente a igual tamaño. Sus números van en la misma
     dirección que nuestra asimetría (VGG16: F1 82 SDNET->SCD frente a 61
     SCD->SDNET).
  2. **Dorafshan et al. (2018, *Constr. Build. Mater.*) no usa SDNET2018**: 19
     imágenes de alta definición. La versión anterior le atribuía cifras dentro
     de SDNET2018; esas están en el artículo de datos (AlexNet, 87.5-95.5 %).
  3. **Cha et al. (2017) sí probó en otra estructura** (55 imágenes de una
     estructura no usada en entrenamiento). La versión anterior lo contaba
     entre los que solo evalúan dentro de su colección.
  4. SDNET2018 son parches de 256x256 (~60x60 mm) cortados de fotos de
     4068x3456 tomadas a 500 mm (artículo de datos): confirma por escrito la
     estructura padre-parche del hallazgo 1 de F0.
  5. Para la discusión: Geirhos et al. (2020) dan el marco de "atajo" para la
     lectura del recorte de METU (su ejemplo de juguete invalida una regla de
     tamaño de objeto con un test de tamaño aleatorizado); Kapoor & Narayanan
     (2023, [L3.2]) llaman fuga a la no independencia train/test, que es el
     caso de los parches de una misma foto; Kumar et al. (2022) es la hipótesis
     H-EXP3 publicada.
- **E4 (código): `04_stats.py` y `05_figures.py` reescritos** para el formato
  nuevo y probados con las salidas del piloto y de las pruebas de humo (prueba
  de código en el scratchpad; ningún número de esas pruebas se usa).
  - `04_stats.py`: réplica completa del análisis anterior + EXP2 (P/R por
    partición, Wilcoxon P-R por corrida) + EXP3 (caída FT frente a sonda) +
    EXP4 (tres fuentes frente a media y máximo de una) + sensibilidad a la
    partición agrupada + tabla de apéndice por celda y semilla + todos los
    números en `resumen_estadistico.json`. Corre con resultados parciales.
  - `05_figures.py`: 7 figuras, rotulado Arial 8 pt, PDF con fuentes
    incrustadas y PNG a 600 dpi (guías de AICE). La Fig. 2 es el mapa unificado
    (media sobre arquitecturas con números + las cuatro arquitecturas con la
    misma escala, sin números porque a ese tamaño quedarían por debajo de
    8 pt). La **Fig. 4 sale de las predicciones guardadas de un modelo de la
    matriz**, sin reentrenar: par entre campañas con menor F1 medio, semilla de
    F1 mediano, los tres FP más seguros y los tres FN más seguros (regla fija).
    Nuevas: Fig. 5 (precisión frente a recall por celda), Fig. 6 (sonda frente a
    ajuste fino), Fig. 7 (multi-fuente).
  - `src/tabla_arquitecturas.py` → `results/tables/tabla_arquitecturas`
    (M2): parámetros con cabeza de 2 clases (11.18 / 4.01 / 1.52 / 5.52 M),
    dimensión real de la característica de la sonda (512 / 1280 / 1024 / 192;
    en MobileNetV3 no es `num_features`=576, comprobado con la salida real),
    etiqueta de pesos `timm`.
- Siguiente: esperar E1 → `comparar_original.py` (criterios a-d) → si no hay
  cambio cualitativo, `04_stats.py` y `05_figures.py` con todo; F3.

## 02/10 (noche) - la cadena de F1 volvió a caer; nuevo lanzador; F3 empezado - `[Author list pending]`

- **Qué pasó con la cadena de las 18:00** (`results/logs/lanzar_f1.log` y
  `e*_stdout.txt`):
  - **E1 cayó a las 18:34** con `OSError [Errno 22] Invalid argument` al
    reescribir `historial_entrenamiento.csv`, tras 23 modelos de la base
    (ResNet-18 entero y EfficientNet-B0 hasta s44/SDNET-W). El último dejó sus
    filas de resultados pero no su historial ni su checkpoint.
  - La cadena siguió: **E1b terminó bien** (192 filas, 55 min). **E2 cayó**
    con el mismo error en `historial_sonda.csv`. **E3 murió con `^C`** hacia
    las 19:54, con 13 de 48 modelos.
  - Causa probable del Errno 22: bloqueo transitorio del archivo en Windows
    (antivirus o indexador) al reescribir un CSV pocos segundos después.
    Causa del `^C`: no identificada; el `cmd` del .bat tenía consola y la
    recibió (segunda vez, tras la de las 17:26).
- **Correcciones:**
  1. `anexar()` escribe a un temporal y lo sustituye con `os.replace`, con
     8 reintentos con espera creciente. La usan los tres scripts.
  2. `--reanudar` en `03_experiment.py`, `03b_sonda_lineal.py` y
     `03c_multifuente.py`: salta un modelo solo si tiene sus 4 filas de
     resultados, su historial, sus predicciones y (en la base original) su
     checkpoint.
  3. `src/lanzar_f1.py` sustituye al .bat (guardado como
     `results/logs/lanzar_f1_bat_retirado.txt`): arranca con `pythonw` vía
     WMI, sin consola; cada paso es un hijo con `DETACHED_PROCESS` y su propio
     grupo de procesos, así que no puede recibir un Ctrl+C; reintenta cada paso
     hasta 3 veces con `--reanudar`. Orden: E2 → E3 → E1.
- **Relanzado a las 19:58.** E2 terminó a las 20:02 (192/192). La sonda de
  EfficientNet s42/SDNET-W, que se repitió porque le faltaba el historial,
  dio exactamente los mismos números que antes del fallo: determinismo entre
  procesos. E3 se reanudó saltando los 13 modelos completos. El modelo de E1
  que se repetirá (EfficientNet s44/SDNET-W) se puede comparar con
  `results/logs/resultados_matriz_antes_reanudar.csv`, copia del CSV previa
  al relanzamiento.
- **F3 empezado: `paper/main.tex`** con `sn-jnl` y `sn-apa`; compila sin
  errores ni avisos (12 páginas con marcadores).
  - **Secc. 2 Related work** (~1100 palabras): tres bloques (exactitud alta
    dentro de una colección; sesgo de conjunto y cambio de distribución;
    transferencia entre estructuras civiles) y cierre con el hueco, ya
    reformulado tras Rashid et al. (2025): no se reclama "la primera matriz",
    sino cinco cosas que esa literatura no ha medido.
  - **Secc. 3 Method** (~2100 palabras, 3.1-3.9): datos, preprocesado y
    particiones (con la fuga entre parches y la partición agrupada como
    sensibilidad), arquitecturas y entrenamiento (tabla de arquitecturas,
    determinismo), matriz y particiones direccionales, intervenciones, sonda
    lineal, multi-fuente, análisis estadístico y uso de IA (AICE lo exige en
    Métodos).
  - **Fig. 1 del manuscrito: `paper/fig_pipeline.tex`** (TikZ propio,
    standalone, cajas grises, sans-serif) → `fig_pipeline.pdf`.
  - Dos cifras del Método anterior estaban mal y se corrigen contra
    `02_preprocess.log`: la clase más pequeña tras deduplicar son **2022**
    imágenes (no 2025, que es antes de deduplicar), y las 13 620 imágenes de
    SDNET-Deck son el total **antes** de deduplicar. Tamaño real de los
    parches descargados medido: METU 227x227, SDNET2018 256x256.
  - Cada afirmación sobre una referencia se repasó contra su nota del `.bib`;
    se rebajaron cinco que decían más de lo leído (Cha: "qualitatively";
    ConCrackNet: "six cross-surface pairs"; tope de la diagonal por el
    submuestreo, no medido; la prueba de determinismo era de una sola
    configuración; Bouthillier y Luleci, ajustados a su texto).
  - Comprobado: ninguna palabra de la lista prohibida, cero guiones largos.
  - Formato de la bibliografía: `sn-apacite.bst` solo imprime el DOI en
    `@article`; en el resto se añadió `url = https://doi.org/...` (sin tocar el
    .bst) y se protegieron las mayúsculas de `booktitle`. Pendiente de pulir:
    "Retrieved from" (APA 6) y la primera cita con todos los autores.

## 03/10 - F1 terminado; comparación con el original sin cambio cualitativo

- Cadena de `src/lanzar_f1.py` completa sin más fallos: E3 terminó el 02/10 a
  las 21:39 (192 filas), E1 el 03/10 a las 06:38 (538 min, código 0).
  `resultados_matriz.csv`: 192 filas por condición (baseline, agresivo,
  ecualizado) con partición original y 192 de baseline con partición agrupada.
- Las 284 filas previas al fallo de las 18:34 se rehicieron con `--reanudar` y
  salen idénticas (TP, TN, FP, FN coinciden en las 284).
- `src/comparar_original.py`: **ningún criterio (a)-(d) se activa**. Nuevo frente
  a original, F1 medio por partición: dentro 0.867/0.866, entre superficies
  0.654/0.652, SDNET→METU 0.782/0.759, METU→SDNET 0.437/0.472; mismo orden.
  Agresivo −0.005 [−0.024, +0.014] (antes −0.014 [−0.033, +0.006]); ecualizado
  −0.027 [−0.047, −0.006] (antes −0.030 [−0.054, −0.006]); fuera de dominio
  sobre el trivial 45.8 % (antes 49.3 %). Diferencia absoluta media por celda
  0.048 (máx. 0.54): mismo diseño, distinta ejecución (determinista ahora).
  Los números nuevos sustituyen a los publicados en todo el manuscrito; no se
  mezclan.
- `04_stats.py` y `05_figures.py` corridos sobre los resultados completos
  (tablas en `results/tables/`, figuras 1-7 en `results/figures/`).

## 03/10 (mañana) - F3: Results, Discussion y Limitations

- `paper/main.tex`: secc. 4 Results (4.1-4.8: diagonal, dirección, caída
  relativa, precisión/recall, intervenciones, sonda lineal, multi-fuente,
  sensibilidad a la partición), secc. 5 Discussion (5.1-5.4, con 5.4
  *Implications for automated infrastructure inspection*) y secc. 6
  Limitations (cinco, reordenadas). Toda cifra sale de `results/tables/` del
  03/10; ninguna de la versión de Tehnički glasnik.
- Frase rota de la Discusión anterior reescrita: la dirección de transferencia
  es propiedad del par ordenado de dominios, no de una distancia simétrica.
- No se usa la atribución de la guía §6.6 a Luleci & Catbas (asimetría
  complejo→simple): no está en el artículo (nota en `refs.bib`).
- Cifras nuevas calculadas para el texto (de `resultados_matriz.csv`): 78 de
  144 celdas fuera de diagonal en o bajo 0.667, y 77 de esas 78 con precisión
  mayor que recall (P media 0.838, R 0.384); METU→SDNET-D/W: recall máx.
  0.317 en las 24 celdas; mínimo de predicciones positivas por celda 40/600.
- Figuras: `05_figures.py` genera Fig. 1-3 a 129 mm (`ANCHO_CAJA`), porque a
  180 mm el PDF las reducía al 73 % y el texto quedaba por debajo de 8 pt;
  paneles pequeños de Fig. 2 con etiquetas D/P/W/M (explicadas en el pie).
  Copias en `paper/figures/`.
- Compila sin errores ni cajas desbordadas (24 páginas con tablas y figuras).
  Sin palabras prohibidas, cero guiones largos. Extensión aproximada sin
  Introducción/Conclusión/Abstract: ~5400 palabras.

## 03/10 (mediodía) - F3: Introduction y Conclusion

- Introducción (6 párrafos + 4 contribuciones numeradas): abre con el caso de
  la agencia (guía 6.2), cifras dentro de colección (Cha, SDNET2018, Özgenel),
  fuga y representatividad del test (Kapoor, Abeysuriya, Hasan & Lu), sesgo de
  conjunto (Torralba; revisión AICE de 46 estudios, no 47; Lateef & Yu), hueco
  y diseño. Contribuciones reescritas sobre los resultados nuevos (asimetría
  0.345; 77 de 78 celdas bajo el trivial fallan por recall; origen de la
  pérdida; protocolo), no las de la guía (0.287, artefacto de ecualización).
- Conclusión: hallazgos, protocolo de validación local (con la referencia
  trivial general 2π/(1+π) para muestras no balanceadas) y trabajo futuro. La
  frase de la guía sobre la sonda lineal como trabajo futuro se elimina porque
  ya está hecha; la de "costs a fraction of one manual inspection cycle" no se
  usa (sin dato que la respalde). Chun & Kikuta: adaptación NO supervisada,
  imágenes sin etiquetar (corregido en la redacción).
- kos2020 y drapaluk2019: no se citan (decisión F3, anotada en refs.bib).
- Compila sin errores ni cajas desbordadas, 26 páginas. Quedan con \pend:
  Abstract, keywords, Acknowledgements y Declarations.

## 03/10 (tarde) - F3: Abstract, keywords y apéndice

- Abstract de 241 palabras (límite AICE 150-250), sin subtítulos ni
  referencias; METU definido; cifras de results/tables del 03/10.
- Keywords (6): Concrete crack classification, Domain shift, Cross-dataset
  evaluation, Transfer learning, Infrastructure inspection, SDNET2018.
- Apéndice A generado por `src/06_apendice.py` → `paper/apendice.tex`
  (\input en main.tex): A1-A2 las 192 celdas base (F1 por semilla, F1 medio,
  P y R medios); A3 matrices medias de aumentación, ecualización, sonda lineal
  y partición agrupada; A4 multi-fuente por arquitectura. El script comprueba
  que cada celda tenga las tres semillas y que cada régimen tenga 192 filas.
- Compila sin errores ni cajas desbordadas, 30 páginas. Quedan con \pend solo
  Acknowledgements y Declarations (F4).

## 03/10 (tarde) - Bibliografía en APA 7 y declaraciones

- `apacite` (cargado por sn-jnl con `sn-apa`) sigue APA 6. Ajustes:
  `\renewcommand{\BRetrievedFrom}{}` (quita "Retrieved from", 14 casos);
  `\shortcites{...}` con las 20 entradas de tres o más autores ("et al." desde
  la primera cita: antes salía "Cha, Choi, and Büyüköztürk (2017)"); en la
  copia `paper/sn-apacite.bst` el truncado de la lista pasa de 7 a 20 autores
  (APA 7), comentado en el propio .bst. La copia de `paper/template/` no se
  toca. Comprobado en el PDF: 0 "Retrieved", listas completas (Howard 12,
  Bouthillier 17 autores), DOI de ConCrackNet con guion bajo correcto.
- "Statements and Declarations" (nombre de AICE): ética y consentimiento "not
  applicable" (imágenes públicas de superficies, sin personas); datos con los
  DOI de SDNET2018 y METU; uso de IA remitido a la secc. 3.9. Con \pend:
  financiación, confirmación de "no competing interests", URL/DOI del
  repositorio, CRediT, agradecimientos y biografías (obligatorias en AICE).
- Pesos de los modelos base: `checkpoints/` ocupa 1.1 GB; para publicarlos,
  Zenodo mejor que GitHub (decisión pendiente del usuario).
- Compila sin errores, 30 páginas.

## 03/10 (tarde) - F4: trámite

- `ENVIO.md` (formato del de P1-Tehnički): revista, Editorial Manager, tipo,
  APC, archivos fuente que AICE exige, formulario y lo que bloquea.
- `paper/declaraciones.md`: texto y origen de cada bloque de "Statements and
  Declarations". Agradecimientos, financiación y conflicto de intereses
  tomados de `C:\lineaB\TRAMITE.md` (común a los cinco artículos), sin "line
  B" ni fechas.
- `paper/tramite/carta_presentacion.md`: borrador de la guía §10 con las
  cifras nuevas, sin nombres, sin atribuir a Luleci & Catbas una asimetría
  que no miden; añade sonda lineal, multi-fuente y partición agrupada.
- `src/07_figuras_envio.py` → `paper/envio/Fig1..Fig8` (EPS vectorial; TIFF
  600 dpi para las dos figuras con fotografías), orden leído de los
  \includegraphics de main.tex.
- `fig_pipeline.tex`: de 147 a 125 mm y helvet sin escalar, para que el
  texto quede en 8 pt sin reducción (antes ~6.6 pt); sin guionado en cajas.
- Quitada del PDF la ruta `results/tables/tabla_arquitecturas.csv` (pie de la
  Tabla 1), pasada a comentario.
- `C:\lineaB\verificar_envio.py`: nueva entrada "P1-AICE" (main.tex,
  apendice.tex, main.pdf, carta) y patrones de restos de la plantilla sn-jnl.
  Resultado: 13 bloqueantes, todos de autoría, repositorio o la definición de
  \pend; ningún aviso de contenido.

## 03/10 (mediodía) - F5, ronda 1 de revisión adversarial: verificación y análisis nuevos

Revisión en contexto limpio (agente sin acceso más que a `paper/main.pdf`, rol
revisor de AICE, prompt de la guía metodológica §5): **revisión mayor**.
Antes de aplicar nada se verificó cada afirmación de hecho:

- **Confirmado.** SDNET2018 no es una sola campaña en sentido estricto:
  tableros fotografiados en el laboratorio SMASH (secciones de tablero
  almacenadas), muros y pavimentos en el campus de USU; misma cámara y
  distancia (`referencias/nuevas/sdnet_dib.txt`). El texto decía "share
  camera, protocol and campaign": hay que matizarlo.
- **Confirmado.** METU: la ficha de Mendeley V2 dice 458 fotos de 4032×3024,
  parches con el método de Zhang et al. (2016); el artículo de congreso dice
  500 fotos. Santos et al. (2022, AICE 1:8, verificado en Crossref y texto
  completo, `referencias/aice/s43503-022-00008-6.*`) describe los positivos
  de METU como "mainly close ups" con "Cracks ... centered in the patch", y un
  modelo entrenado en METU baja de >99 % a 54 % de exactitud en una presa
  (Itaipú). Es el antecedente de AICE que faltaba y respalda con una fuente
  la hipótesis del encuadre.
- **Confirmado.** Geirhos et al.: en el ejemplo de juguete la red aprende la
  POSICIÓN; el tamaño (contar píxeles blancos) es una regla hipotética. El
  texto lo describía mal.
- **No es un error.** Parches por foto de SDNET2018 (252 = 18×14; 234 en
  pavimentos), numerados 1..252 en los nombres: la partición agrupada es
  válida; la resolución "4068" del artículo de datos parece errata de 4608.
- **Hallazgo propio al calcular el AUROC:** 34 de 691 200 predicciones con
  probabilidad NaN (desborde de la inferencia en media precisión, casi todas
  EfficientNet-B0); en la matriz cuentan como "sin grieta". Cota: ninguna
  celda cambia más de 0.0064 de F1. Se declara en el Método.

Análisis nuevos (sin reentrenar):

- `src/08_umbral_auroc.py` → `umbral_auroc_*.csv/json`: AUROC/AP por celda y
  F1 con umbral recalibrado con k = 20/50/100 imágenes etiquetadas del
  destino. METU→SDNET: AUROC 0.699 (SDNET-D 0.679, SDNET-W 0.627); con k = 50
  el F1 pasa de 0.437 a 0.673, el nivel del trivial; techo 0.696. El fallo es
  de ordenación, no solo de umbral. Entre superficies 0.654 → 0.705 (AUROC
  0.782); SDNET→METU 0.782 → 0.799 (AUROC 0.847).
- `04_stats.py`, bloque `revision_f5` → `revision_f5.json`: contrastes por
  corrida con diseño correcto (pareados si comparten corridas). **Cambio:**
  dentro de dominio frente a SDNET→METU, pareado sobre las 36 corridas SDNET,
  +0.042, p_Holm = 0.115 (antes p = 0.0031 con Welch). Los otros cinco
  siguen significativos; la asimetría (+0.345) no cambia. Intervenciones por
  corrida: agresiva −0.005 (p = 0.76); ecualización −0.027 [−0.051, −0.004],
  p = 0.048, p_Holm = 0.095. Empates: una celda con F1 = 2/3 exacto, así que
  quedan 66 por encima, 1 igual y 77 por debajo.

## 03/10 (tarde) - F5: correcciones de la ronda 1 aplicadas (sin experimentos nuevos)

Decisión del usuario: aplicar solo las correcciones; ningún experimento nuevo.
Cambios en `paper/main.tex`:

- Resumen (240 palabras): sin p-valor; empate con el trivial; AUROC y umbral;
  "was not reduced by" en lugar de "did not respond to"; fuera "matched the
  best single source"; ROC definido.
- Introducción: sin las frases tipo aforismo; SDNET2018 descrito como mismo
  grupo, cámara y distancia, tableros en laboratorio y muros/pavimentos en
  campus; Santos et al. (2022); contribuciones reescritas.
- Trabajos relacionados: Özgenel con los tres conjuntos externos reales y el
  efecto de arquitectura (VGG16 0.96-0.98 frente a ResNet101/152 0.54-0.77);
  Geirhos corregido (posición, no tamaño); Santos et al. (2022); sin CODEBRIM
  ni SCD sin definir.
- Método: origen del diseño declarado (la partición por dirección se
  introdujo tras los primeros entrenamientos; esto es una réplica); SDNET y
  METU corregidos; desborde en media precisión (34 predicciones, ≤ 0.007);
  abreviaturas definidas; estadística reescrita (corrida como unidad, pareado
  cuando corresponde, IC por corridas, intervenciones por corrida, AUROC/AP y
  umbral recalibrado declarados como añadidos a posteriori).
- Resultados: Tabla 3 con IC por corridas; Tabla 4 nueva (pareada/Welch);
  subsección 4.5 nueva "Ranking and decision threshold" con su tabla;
  intervenciones por corrida; multi-fuente sin afirmar equivalencia y con la
  composición de fuentes; 0.055 por cambiar solo los parches de prueba.
- Discusión: encuadre apoyado en Santos et al. (centrado, ≥ 5 px del borde) y
  en el artículo de SDNET2018 (grietas de 0.06 mm); pares entre superficies
  sin tableros 0.624 frente a 0.669 con tableros (el cambio de sitio no
  empeora); AUROC en la lectura de la asimetría; arquitectura frente a
  Özgenel; protocolo con recalibración del umbral y con muestreo por fotos.
- Limitaciones: seis puntos (dos campañas; encuadre sin medir, sin
  aumentación de escala, sin auditoría de etiquetas; un solo sorteo de datos
  con el 0.055 como referencia; modelos pequeños ImageNet; partición agrupada
  solo base; sin datos del destino y solo clasificación de parches).
- Ortografía estadounidense unificada (labeled, normalized, generalize...).
- Figura de precisión-recall: iso-F1 de 0.667 discontinua y etiquetas a 8 pt
  (estaban a 6 pt). Tablas: recuentos por arquitectura en el pie de la
  Tabla 2, "n/a" y nota en la Tabla 9.
- `refs.bib`: santos2022 (verificado) y nota ampliada de ozgenel2018.
- Compila sin errores ni cajas desbordadas, 33 páginas.

## 03/10 (tarde) - EXP-escala: protocolo fijado ANTES de entrenar

Pedido por el usuario tras la ronda 1 de F5 (respuesta a la objeción O3 del
revisor: la explicación por encuadre no estaba puesta a prueba). Se fija aquí,
antes de lanzar ninguna corrida, para que conste que no se ajustó a los
resultados.

- **Condición `escala`** (`src/03_experiment.py`): sobre la imagen ya llevada a
  224×224, factor de escala s ~ U(0.5, 1.5) por imagen. s < 1: se reduce a
  round(224·s) y se rellena por reflejo hasta 224 con desplazamiento
  aleatorio (sin bordes negros, que serían un atajo nuevo). s > 1: se amplía y
  se recorta 224×224 en posición aleatoria. Después, volteo horizontal como en
  la base. Solo en entrenamiento; validación y prueba sin cambios. Todo lo
  demás igual que la base (arquitecturas, semillas 42-44, receta, partición
  original, parada temprana). Aleatoriedad con el generador de torch para
  conservar el determinismo.
- **Corridas:** 48 (4 arquitecturas × 3 semillas × 4 orígenes), prueba en los
  4 dominios: 192 celdas. Se guardan predicciones por imagen.
- **Hipótesis principal (H-escala):** entrenar en METU con escala aleatoria
  sube el F1 de METU→SDNET2018 frente a la base. Prueba: Wilcoxon de rangos
  con signo sobre las 12 corridas de origen METU (media de sus 3 celdas
  METU→SDNET), bilateral, α = 0.05, biserial de rangos e IC bootstrap de la
  diferencia media.
- **Secundarias** (descriptivas, Wilcoxon por corrida con Holm entre ellas):
  recall y AUROC de METU→SDNET; F1 de las otras dos particiones fuera de
  dominio; F1 dentro de dominio.
- **Lectura fijada de antemano:** si H-escala se cumple, la explicación por
  encuadre gana apoyo (no queda probada: la escala no es la única diferencia).
  Si no se cumple, se debilita y se dice así en la Discusión y en
  Limitaciones. En ningún caso se cambian el diseño ni el rango de escala
  después de ver resultados.
- **Comprobaciones antes de lanzar** (14:00-14:18): la transformación es
  determinista con el generador de torch (dos secuencias con la misma semilla,
  idénticas) y devuelve 224×224. Inspección visual de 6 muestras de METU: con
  s cercano a 0.5 el relleno por reflejo duplica la grieta en espejo. Se
  mantiene el diseño fijado: la etiqueta sigue siendo correcta y el artefacto
  no depende de la clase más allá de la presencia de grieta. Prueba corta
  (ResNet-18, semilla 42, METU, 1 época, en `results/smoke_escala/`): corre y
  guarda predicciones.
- **Lanzado** el 03/10 a las 14:19 con `lanzar_f1.py escala` (pythonw, WMI,
  `--reanudar`); salida en `results/logs/escala_stdout.txt`.
- **Ronda 2 preparada, sin lanzar:** `paper/revision/ronda2_prompts.md`
  (prompt principal, pasada anti-IA y revisor hostil opcional; condiciones
  para lanzarla). `paper/revision/ronda1_respuesta.md`: cada objeción de la
  ronda 1, su verificación y su estado (aplicado, limitación, experimento o no
  aplicado con motivo).
- **Referencias:** EfficientNet pasa a su versión publicada (ICML 2019, PMLR
  97:6105-6114, verificada en PMLR). Gulrajani (ICLR 2021) y Steiner (TMLR
  2022) siguen como arXiv: OpenReview y DBLP pidieron verificación anti-bots y
  no se abrió la versión publicada.

## 03/10 (17:27-17:50) - EXP-escala: resultado

- **Ejecución:** terminó a las 17:27 (187 min, un solo intento, código 0);
  192 filas `escala` en `resultados_matriz.csv` y predicciones en
  `results/predicciones/escala_original/`.
- **AUROC:** `08_umbral_auroc.py` incluye el régimen `escala_original` al
  final; las 384 filas previas (base y sonda) son idénticas a las de antes y el
  F1 a 0.5 de escala coincide con la matriz (diferencia máxima 0).
- **Análisis** (`04_stats.py`, función `exp_escala`, salida
  `results/tables/exp_escala.json`); el resto de `resumen_estadistico.json` no
  cambia.
- **H-escala (principal), METU→SDNET, 12 corridas:** F1 0.437 → 0.492,
  diferencia +0.055, IC bootstrap [+0.010, +0.098], W = 14, **p = 0.052**,
  r = +0.64; 9 de 12 corridas mejoran. Por arquitectura: ResNet-18 −0.001,
  EfficientNet-B0 +0.054, MobileNetV3 +0.102, ViT-Tiny +0.064. Por destino:
  +0.054 en los tres.
- **Secundarias (Holm sobre 5):** recall METU→SDNET 0.320 → 0.379 (+0.060,
  IC [+0.013, +0.106], p_Holm = 0.21); AUROC METU→SDNET 0.699 → 0.706 (+0.006,
  IC [−0.013, +0.026], p_Holm = 0.86); entre superficies +0.008 (0.86);
  SDNET→METU −0.022 (0.86); dentro +0.003 (0.25).
- **Lectura fijada de antemano:** p = 0.052 > 0.05, así que **H-escala no se
  cumple** y la explicación por encuadre se debilita. Matices que se reportan
  sin cambiar esa lectura: el efecto tiene la dirección esperada y un tamaño
  moderado, pero el AUROC no cambia. La escala mueve el punto de operación
  (más recall), no mejora la ordenación de las imágenes de SDNET2018, y cierra
  unos 0.055 de una brecha de unos 0.43 frente a la diagonal.
- **Pendiente:** se para y se avisa antes de incorporarlo al manuscrito
  (regla del plan: cambia el peso de la explicación por encuadre en la
  Discusión).

## 03/10 (18:00) - EXP-escala incorporado al manuscrito; ronda 2 lanzada

- El usuario aprueba incorporarlo con la lectura fijada (explicación por
  encuadre debilitada).
- **main.tex:** resumen ("nor significantly by random rescaling", 245
  palabras); Introducción (tres intervenciones, la tercera añadida para probar
  una explicación; contribución 3); Método 3.5 (descripción, artefacto de
  espejo, diseño fijado antes de entrenar) y 3.8 (prueba principal y
  secundarias); nota de NaN actualizada (67 de 806 400 predicciones; escala
  añade 33, todas de EfficientNet-B0 con origen SDNET-W; efecto máximo por
  celda 0.025; ninguna celda con origen METU); Resultados 4.6 con la tabla
  nueva `tab:scale`; Discusión 5.1 y 5.3 (no se cumple; el AUROC no cambia; con
  umbral recalibrado con k = 50, 0.675 frente a 0.673; cierra una sétima parte
  del hueco de 0.387 frente a la diagonal media de SDNET, 0.824); Limitación 2;
  Conclusión y trabajo futuro.
- **Descriptivos de escala:** el 50.0 % de las celdas fuera de diagonal supera
  0.667 (base: 45.8 %); 72 celdas en 0.667 o por debajo, 71 con P > R. Diagonal:
  0.870.
- **Apéndice:** la Tabla A3 incluye la matriz de escala (`06_apendice.py`).
- Figuras regeneradas (sin cambios de contenido), PDF de 34 páginas sin
  errores ni cajas desbordadas; `verificar_envio.py`: los mismos 13 bloqueantes
  de autoría, repositorio y `\pend` (más dos falsos positivos de DOI cortado
  por salto de línea).

## 03/10 (18:00-18:45) - Ronda 2 de revisión: informes, verificación y correcciones

- Tres agentes en contexto limpio (revisor de AICE, anti-IA, revisor hostil).
  Informes en `paper/revision/ronda2_*.md`; verificación de cada afirmación de
  hecho en `ronda2_verificacion.md`; estado de cada objeción en
  `ronda2_respuesta.md`.
- **Cambios cualitativos confirmados y avisados al usuario antes de aplicar**:
  la caída depende del anclaje (SDNET→METU anclada en el destino 0.217, igual
  que entre superficies); "no mejor que el trivial" solo vale en F1 (las 144
  celdas superan la exactitud balanceada de 0.5); precisión a la prevalencia
  real ~0.5; el umbral por máximo F1 está sesgado hacia "todo grieta" (Youden:
  BA 0.629 → 0.656). El usuario aprobó aplicar todo y **mantener el título**.
- **Refutadas:** Santos (+54 % es relativo), 54 fotos de SDNET-Deck (contado
  sobre `03-tehnicki-glasnik-P1/data/raw` en solo lectura), Hasan & Lu, el
  54.9 % repetido. Rashid et al.: su matriz no muestra la asimetría en
  exactitud; se usa en 5.2.
- **Nuevo:** `src/09_ronda2.py` → `results/tables/ronda2.json` y
  `ronda2_particiones.csv` (sorteos propios, semilla 42; `08` intacto).
  Tabla 5 nueva (`tab:anchor`). Fig. 1 con el reescalado; Fig. 3 con tres
  decimales. `paper/figures/` no se actualiza sola: se copiaron las PDF de
  `results/figures/`.
- Resumen 249 palabras; PDF de 36 páginas sin errores ni cajas desbordadas;
  carta de presentación actualizada; `verificar_envio.py`: los mismos 13
  bloqueantes de autoría y repositorio.

## 03/10 (19:00) - Referencias de métricas pedidas por el revisor

- Pedido del usuario. Cada una abierta y leída antes de citarla; metadatos y
  textos en `referencias/nuevas/` (y `metadatos/`):
  - **Lipton, Elkan y Naryanaswamy (2014)**, LNCS, ECML PKDD, pp. 225-239
    (Crossref); texto completo de arXiv:1402.1892. Teorema 3 (clasificador no
    informativo → el umbral óptimo para F1 lo marca todo positivo) y sección 7
    (winner's curse con muestras pequeñas). Citado en 3.8, 4.5 y 5.4.
  - **Chicco y Jurman (2020)**, BMC Genomics 21:6 (Crossref); texto completo en
    Europe PMC (BMC devolvió HTML en lugar del PDF). MCC alto solo si las cuatro
    celdas son buenas; F1 ignora los verdaderos negativos. Citado en 3.8.
  - **Miller et al. (2021)**, ICML, PMLR 139:7721-7735 (página de PMLR);
    texto completo de arXiv:2107.04649. La correlación es entre modelos con un
    mismo cambio, no entre fuentes: se cita como contraste en 5.2.
  - **Loshchilov y Hutter**, AdamW, arXiv:1711.05101 (DataCite; texto
    completo, v3 aceptada en ICLR 2019). Citado como arXiv: OpenReview pidió
    verificación anti-bots. Citado en 3.3.
  - **Youden (1950): no se cita.** Crossref sí, pero Wiley dio 403 y luego
    verificación anti-bots en el navegador; no se saltó. El texto define el
    índice.
- PDF de 38 páginas sin errores, cajas desbordadas ni citas indefinidas.
  Con tres o más autores, `sn-apacite` omite el "&" en todas las referencias
  (comportamiento del estilo, no de las entradas nuevas).

## 08/10 - Ronda 3 de revisión y repositorio público

- El usuario pide subir el proyecto a un repo público nuevo (como el de P1)
  para que lo revise el ingeniero, con una ronda 3 antes.
- Ronda 3: revisor de AICE y pasada anti-IA en contexto limpio
  (`paper/revision/ronda3_revisor.md`, `ronda3_antiIA.md`, con la
  verificación de cada afirmación). Repite tres objeciones ya refutadas en la
  ronda 2 (Santos, 54 fotos, 54.9 %). Confirmadas y aplicadas: 5.2 citaba la
  Tabla 5 con pérdidas absolutas (ahora Δt y aparte la absoluta); rangos por
  celda atribuidos a la Tabla 5; redondeos (nota en 3.8); la ecualización baja
  también SDNET-Pavement (0.855 → 0.806); "bounded below" solo vale para el
  mejor umbral; frase de escala por arquitectura; "within one SD" eliminada;
  el reentrenamiento no es copia bit a bit porque el original no era
  determinista; Dorafshan 2018: 319 de 3420 con grieta (99 % frente a ~91 % de
  "todo sin grieta"); prevalencia "realista" → de la colección; atenuadas
  "have not been measured apart" y "absence of a size effect"; IC para las
  filas de Welch de la Tabla 4 (`04_stats.py`: remuestreo de corridas dentro de
  cada grupo; solo cambia `revision_f5`); oráculo rotulado en la Tabla 11;
  s42-s44 en el apéndice; $p_\mathrm{Holm}$ definido. Anti-IA: Introducción,
  2.2, 2.3, 2.4, 5.2, 5.4, contribución 4, Conclusión y resumen (250
  palabras).
- No aplicado (experimentos o fuentes no abiertas): tercera campaña, modelos
  de fundación, bootstrap jerárquico, FP32, curva de aprendizaje, auditoría de
  etiquetas, color, barrido de escala en test, citas de métodos.
- PDF de 38 páginas sin errores ni cajas desbordadas; verificador: 14
  bloqueantes, todos de autoría y repositorio (el `\pend` de las
  declaraciones ahora también se detecta en la página 30 del PDF).
- **Repo:** `.gitignore` nuevo: fuera `data/` (salvo el manifiesto de
  particiones), `referencias/` (derechos de autor), `checkpoints/`,
  `PLAN_AICE.md` (contiene los nombres de los autores), `PROMPT_INICIO.md`,
  la plantilla y las capturas de Springer y las pruebas de humo. Se copian
  `fieutils/` y `pyproject.toml` de `C:\lineaB` (copia; el original no se
  modifica), como en el repo de P1.
