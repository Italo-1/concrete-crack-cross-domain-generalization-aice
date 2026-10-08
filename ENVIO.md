# Instrucciones de envío — P1 a *AI in Civil Engineering*

> Archivo pedido por `propuesta_planificación_sugerida.pdf`, §6 («Paquete de
> entrega por artículo»), con el formato del `ENVIO.md` de P1 para Tehnički
> glasnik. Lo usa quien realiza el envío.
>
> Los **[corchetes en negrita]** son lo que falta antes de enviar. Todo lo
> demás está verificado en la web de la revista el 02/10/2026 (`JOURNAL.md`,
> evidencias en `paper/evidencia/`).

- **Revista:** *AI in Civil Engineering* (AICE), Springer Nature, revista de
  Tongji University. eISSN 2730-5392. https://link.springer.com/journal/43503
- **Sistema de envío:** Editorial Manager, https://www.editorialmanager.com/aice
- **Tipo de artículo:** Original Article.
- **¿Cobra APC?** No: la cubre Tongji University (cita literal y evidencia en
  `JOURNAL.md`). Licencia CC BY o CC BY-NC-ND. Reconfirmar al enviar.
- **Revisión:** simple ciego. El manuscrito puede llevar la URL del
  repositorio y la afiliación.

## Archivos a subir

AICE exige las **fuentes editables completas** en cada envío y revisión; si
faltan, el artículo no se envía a revisión.

1. **Manuscrito (fuente LaTeX):** `paper/main.tex`, con
   - `paper/apendice.tex` (lo genera `src/06_apendice.py`),
   - `paper/refs.bib` y `paper/main.bbl`,
   - `paper/sn-jnl.cls` y `paper/sn-apacite.bst` (este último con el ajuste
     APA 7 de 20 autores, comentado en el archivo),
   - `paper/fig_pipeline.pdf` y `paper/figures/*.pdf` (las que incluye
     `main.tex`).
   Compilar: `pdflatex main && bibtex main && pdflatex main && pdflatex main`.
2. **PDF compilado:** `paper/main.pdf`.
3. **Figuras sueltas:** `paper/envio/Fig1.eps` … `Fig8.eps` / `.tif`, en el
   orden del texto (las genera `src/07_figuras_envio.py`; correspondencia en
   `paper/envio/LEEME_figuras.txt`). EPS vectorial para los gráficos; TIFF RGB
   a 600 dpi para las dos con fotografías (Fig. 2 y Fig. 6). Todo el rotulado
   en Helvetica/Arial de 8 pt o más a tamaño de impresión: las figuras anchas
   se generan a 125-129 mm y se incluyen sin escalar.
4. **Carta de presentación:** texto de `paper/tramite/carta_presentacion.md`
   (sin el comentario inicial), en el campo de Editorial Manager.

## Datos del formulario

- **Título:** Cross-Domain Generalization of Concrete Crack Classifiers: A
  Transfer Matrix Across Surfaces and Capture Campaigns
- **Autores en orden, con afiliación, correo y ORCID (16 dígitos):**
  **[pendiente: orden de autores no decidido]**. La portada (`main.tex`) usa
  el marcador `[Author list pending]`.
- **Autor de correspondencia y correo institucional activo:** **[pendiente]**
- **Resumen:** copiar de `main.tex` (249 palabras tras la ronda 2; límite
  150-250: no añadir nada sin quitar otra cosa).
- **Palabras clave (6; límite 4-6):** Concrete crack classification; Domain
  shift; Cross-dataset evaluation; Transfer learning; Infrastructure
  inspection; SDNET2018.
- **Financiación:** ninguna (declararla también en el formulario).
- **Conflicto de intereses:** ninguno.
- **Revisores sugeridos:** opcionales en AICE. Si se sugieren: independientes
  del trabajo, de países e instituciones distintos, con correo institucional
  o enlace verificable. **[decidir si se sugieren]**

## Lo que bloquea el envío hoy

`python C:\lineaB\verificar_envio.py P1-AICE` (03/10/2026): todos los
bloqueantes son de autoría o del repositorio, ninguno de contenido.

1. **Autoría:** nombres, afiliaciones, correo, ORCID, CRediT y biografías
   (obligatorias en AICE, < 100 palabras cada una). Marcadores en la portada,
   en *Authors' contributions*, en *Author biographies* y en la firma de la
   carta.
2. **Repositorio:** URL y DOI de Zenodo en *Data availability*, *Code
   availability* y en la carta. Los pesos de los modelos base ocupan 1.1 GB
   (`checkpoints/`): Zenodo. Si no se publican, quitar la mención en *Code
   availability* y en el Método (3.9).
3. Cuando no quede ningún `\pend{...}` en el texto, **borrar la definición de
   `\pend`** en el preámbulo de `main.tex`: el verificador la
   detecta a propósito, para que no se envíe el macro.

## Notas

- *Statements and Declarations* va antes de las referencias, como piden las
  guías; texto de cada bloque y su origen en `paper/declaraciones.md`.
- Las guías de Word piden los agradecimientos en la portada; la plantilla
  LaTeX de Springer los pone en `\backmatter`, que es lo que se usa aquí.
- El uso del asistente de IA está declarado en el Método (3.9), como exige la
  revista, y repetido en las declaraciones.
- P1 nunca se envió a *Tehnički glasnik*: no hay versión publicada ni en
  revisión que declarar en la carta.
- Antes de enviar: recompilar desde cero, pasar `verificar_envio.py P1-AICE`
  sin bloqueantes y revisar el PDF entero en pantalla.
