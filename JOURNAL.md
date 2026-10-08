# Ficha de revista

Plantilla de la guía metodológica, §1.3, ampliada con lo que pide
`PLAN_AICE.md` §3.5. Todo verificado el **02/10/2026** en la web oficial de la
revista; las páginas están guardadas con fecha en `paper/evidencia/` (HTML
original y su texto extraído, `.txt`).

- **Nombre completo:** *AI in Civil Engineering* (AICE). Springer Nature,
  revista de Tongji University. eISSN 2730-5392, ISSN 2097-0943 (de los
  metadatos `citation_issn` de los artículos guardados). Abreviatura usada en
  sus propias citas: *AI Civ. Eng.*
- **URL oficial:** https://link.springer.com/journal/43503
- **Tipo de artículo:** Original Article. La revista acepta Original Article y
  Review (las guías de envío solo listan esos dos; el Aims & Scope menciona
  también *comments* y *perspectives*).
- **¿Cobra APC?** `[x] No` `[ ] Sí -> MONTO:`
  Frase literal: *"There is no fee to publish in AI in Civil Engineering because
  the article processing charge (APC) is covered by the Tongji University."*
  Licencia CC BY o CC BY-NC-ND.
- **Fuente de la respuesta:** `paper/evidencia/aice_how-to-publish-with-us_2026-10-02.html`
  (sección *Fees and funding*). Reconfirmar en el momento del envío.
- **¿Plantilla LaTeX oficial?** `[x] Sí -> URL:` las guías dicen *"Manuscripts
  with mathematical content can also be submitted in LaTeX. We recommend using
  Springer Nature's LaTeX template"*. Es la de `paper/template/sn-article-template/`
  (v3.1, dic. 2024). Hay que enviar las fuentes editables (.tex) en cada envío.
- **Límite de páginas / palabras:**
  - Artículo: **no hay límite publicado** (ni en guías de envío ni en el
    checklist). Referencia de los artículos originales recientes leídos: entre
    ~5 100 y ~7 800 palabras de cuerpo, 22-49 referencias, 7-14 figuras.
  - **Abstract: 150 a 250 palabras**, sin abreviaturas sin definir ni
    referencias. (La guía del encargado pedía 250-300: **fuera del límite**.)
  - **Palabras clave: 4 a 6.**
- **Estilo de citas:** **autor-año entre paréntesis** ("(Thompson, 1990)"),
  lista **alfabética** por primer autor, **APA 7**, revistas en cursiva, DOI
  como enlace completo `https://doi.org/...`. Opción de la plantilla:
  **`sn-apa`** (usa `sn-apacite.bst`, presente en `bst/`). La guía del
  encargado decía "APA numerado": no es numerado.
- **¿Requiere abstract estructurado?** No. Abstract sin subtítulos.
- **¿Requiere declaración de uso de IA?** Sí. Las guías: los LLM no pueden ser
  autores y su uso **"should be properly documented in the Methods section"**.
  La política de Springer (`paper/evidencia/springer_politica_IA_2026-10-02.html`)
  clasifica por riesgo: pulir el lenguaje es verde; redactar, sugerir enfoques
  metodológicos o comparar con la literatura es ámbar (permitido con
  supervisión humana, verificación y declaración); fabricar datos, citas o
  resultados es rojo. Nuestro uso (redacción, código, revisiones adversariales)
  es ámbar: **declarar en Métodos**, además del bloque de declaraciones.
- **¿Requiere declaración de disponibilidad de datos?** Sí, obligatoria en
  todo artículo de investigación. Citar los conjuntos públicos con
  identificador persistente (DOI) en la lista de referencias, formato DataCite.
- **Otras declaraciones:** sección **"Statements and Declarations"** antes de
  las referencias, con *Competing interests* (obligatoria; si falta, se
  devuelve el manuscrito), *Funding*, *Ethics approval*, *Consent*,
  *Data/Code availability*, *Authors' contributions* (CRediT recomendado).
  La financiación también se declara en el sistema de envío.
- **Biografías de autores:** **obligatorias** para artículos de investigación,
  preferiblemente <100 palabras cada una, al final del texto. *No estaba en el
  plan.* Queda con el marcador `[Author list pending]` hasta decidir autoría.
- **Revisión:** **single-blind** (los revisores ven a los autores). El
  manuscrito **puede** citar la URL de GitHub, no solo el DOI de Zenodo.
- **Formato de figuras:** vectorial preferido en EPS, semitonos en TIFF; arte
  combinado ≥600 dpi, semitonos ≥300 dpi; RGB 8 bits; rotulado en **Helvetica
  o Arial**, 8-12 pt (difiere de la guía metodológica §4.2, que pide serif:
  **manda la revista**); sin títulos dentro de la figura; pie que empieza por
  **Fig.** en negrita, sin punto final; patrones además de color; contraste del
  rotulado ≥4.5:1; archivos `Fig1.eps`, `Fig2.eps`... Figuras dentro del texto.
- **Tablas:** numeración arábiga, citadas en orden, notas con letras
  minúsculas en superíndice.
- **Encabezados:** sistema decimal, **máximo tres niveles**.
- **Sistema de envío:** Editorial Manager, https://www.editorialmanager.com/aice
- **Fecha de cierre del próximo número:** no aplica (publicación continua por
  artículo).
- **Otras notas:** la página de carta al editor debe declarar reutilización de
  material propio (el manuscrito de *Tehnički glasnik* nunca se envió, así que
  no hay texto publicado que declarar).

## 5 artículos recientes leídos (últimos dos años)

Leídos completos el 02/10/2026 sobre el HTML de Springer Nature Link
(`referencias/aice/`). Fichas detalladas, con lo que cada uno sostiene y lo
que no, en [`referencias/aice/LECTURAS_AICE.md`](referencias/aice/LECTURAS_AICE.md).

| # | Cita | Tipo | Palabras cuerpo* | Refs | Fig. | Tab. | Estructura |
|---|---|---|---|---|---|---|---|
| 1 | Abeysuriya, K. G., Ciupala, M. A., Ghorashi, S. A., & Ilki, A. (2026). Deep learning for image-based structural element damage assessments in post-earthquake buildings: a systematic review. *AI Civ. Eng.*, 5, 15. https://doi.org/10.1007/s43503-026-00099-5 | Review | ~12 900 | 99 | 17 | 8 | Intro · Method · Results · Discussions · Conclusions |
| 2 | Lateef, J., & Yu, X. (2026). Novel computer vision algorithm for accurate multi-label classification of concrete bridge defects. *AI Civ. Eng.*, 5, 10. https://doi.org/10.1007/s43503-026-00095-9 | Original | ~5 100 | 22 | 7 | 8 | Intro · Methodology · Results · Discussion · Conclusion |
| 3 | Bazrafshan, P., Melag, K., & Ebrahimkhanlou, A. (2025). Semantic and lexical analysis of pre-trained vision language artificial intelligence models for automated image descriptions in civil engineering. *AI Civ. Eng.*, 4, 17. https://doi.org/10.1007/s43503-025-00063-9 | Original | ~7 300 | 49 | 12 | 7 | Intro · Método · Dataset · Experiments · Results · Discussion · Conclusion |
| 4 | Hasan, M., & Lu, M. (2025). Bridging AI and explainability in civil engineering: the Yin-Yang of predictive power and interpretability. *AI Civ. Eng.*, 4, 21. https://doi.org/10.1007/s43503-025-00066-6 | Review | ~7 400 | 75 | 5 | 2 | Temática, 7 secciones |
| 5 | Fountoukidou, T., Tkachenko, I., Poli, B., & Miguet, S. (2024). Extensible portal frame bridge synthetic dataset for structural semantic segmentation. *AI Civ. Eng.*, 3, 23. https://doi.org/10.1007/s43503-024-00041-7 | Original | ~7 800 | 36 | 14 | 13 | Intro · Model · Dataset · Experiments · Results · Conclusions + Apéndice |

\* Recuento aproximado del cuerpo extraído, incluye pies de figura.

Leídos además, por ser candidatos de la guía aunque tienen más de dos años o
no son artículos de investigación: Luleci & Necati Catbas (2023, art. 7),
Luleci & Catbas (2023, art. 9, *Communication*) y Zhao (2022, editorial).

**Lo que se saca de los cinco para el manuscrito:**

- Estructura convencional numerada (Introduction, Method(ology), Results,
  Discussion, Conclusion). Las contribuciones van en lista numerada al final de
  la introducción (Fountoukidou et al.) o en prosa (Lateef & Yu); ambas formas
  se publican.
- Los originales tienen **una sección de Discusión propia** y la limitación
  suele ir dentro de Discusión o Conclusión, no como sección aparte. Una
  sección *Limitations* separada es aceptable pero no es la norma de la revista.
- Las declaraciones publicadas usan "Availability of data and materials" o
  "Data availability", y "Competing interests" con fórmula estándar.
- Los originales cuentan con apéndice cuando hay tablas por arquitectura
  (Fountoukidou et al.): encaja con el apéndice de tablas por celda y semilla.
- Ninguno de los cinco hace evaluación cruzada entre campañas de captura
  reales; Lateef & Yu la piden como trabajo futuro ("cross-bridge and
  cross-site generalization") y Abeysuriya et al. la señalan como la brecha
  principal de su campo.

## Citas obligatorias de la revista destino

Ver `referencias/aice/LECTURAS_AICE.md`. Decisión tras la lectura: **C1, C2,
C4, C7, C8** (cinco); C3 opcional; C5 y C6 descartadas.
