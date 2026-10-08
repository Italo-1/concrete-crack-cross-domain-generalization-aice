# Ronda 2: pasada anti-IA (03/10/2026, sin aplicar todavía)

Agente en contexto limpio, solo con `paper/main.pdf` (34 páginas), prompt 2 de
`ronda2_prompts.md`. Informe resumido: pasaje, página y reescritura propuesta.
Las propuestas se revisarán junto con las de los otros dos revisores antes de
aplicarlas.

Valoración del revisor: pocas marcas léxicas típicas; quedan marcas de estilo
(frases sentenciosas, títulos tipo eslogan, la muletilla "not one quantity",
una anáfora triple, frases que anuncian, dos afirmaciones más amplias que la
evidencia) y la misma lista de vacíos repetida cuatro veces (resumen,
Introducción, 2.4, Conclusión).

## Prioridad alta

1. **P. 5, 2.4:** anáfora "it has not… it has not… it has not… Nor has it";
   repite el cuarto párrafo de la Introducción. Propuesta: condensar en una
   frase con "confound… do not compare… do not report"; "We address these
   points on two public collections."
2. **P. 25, 5.4, y p. 5, 2.3:** "they turn images into information that
   inspection decisions can rely on, the goal Spencer et al. (2019) set…"
   atribuye demasiado a dos pasos y se repite. Propuesta: "The first two steps
   follow from the requirement… and respond to the call for checks… (Hasan &
   Lu, 2025)." En 2.3: "Spencer et al. (2019) review computer-vision-based
   inspection and monitoring of civil infrastructure."
3. **P. 26, 7:** "Based on these findings, we recommend…" y "it prevents the
   deployment…". Propuesta: "We recommend that a crack classifier deployed at a
   new site be validated locally before…"; "it identifies models that perform
   no better than that reference…, as was the case in more than half…".
4. **Títulos de 5.1 y 5.2, y "not one quantity" (p. 22 y 26):** propuesta
   "Within-domain performance by domain"; "Direction of transfer and
   architecture"; "because the two cross-campaign directions differed"; "but
   its size depended on the partition:".
5. **P. 24, 5.4:** "A model in that state reports few cracks and is usually
   right when it does, so checking… will not reveal…; only labeled images…
   will." Propuesta: "Such a model reports few cracks, most of which are real.
   Its errors are missed cracks, which can be detected only with labeled
   images that include cracks, not by reviewing the patches it flags."

## Prioridad media

6. **P. 2, Intro, párrafo 1:** "What the agency needs to know… is how…".
   Propuesta: "Before using it, the agency needs to know how the model performs
   on images of other structures acquired with different equipment and by
   different crews."
7. **P. 2, Intro, párrafo 4:** "These studies leave several questions an
   operator would ask unanswered." Propuesta: eliminarla.
8. **P. 26, 7:** "…got there by missing cracks". Propuesta: "In almost all
   out-of-domain cells at or below the always-crack classifier, precision
   exceeded recall, so the errors were mainly missed cracks".
9. **P. 4, 2.2:** "bear directly on crack images" y "applies that instrument".
   Propuesta: "Two of their examples are analogous to the crack setting:"; "We
   apply the same design to images of concrete surfaces."
10. **P. 3, contribución 3:** rótulo poco natural y cuatro cláusulas con punto
    y coma. Propuesta: "Tests of where the loss originates." y frases
    separadas.
11. **P. 23-24:** frases que empiezan con "And". Propuesta: "The linear probe
    trained on METU also fails…"; "Even if real, the gain would close…".

## Prioridad baja

12. **Resumen:** "which says little about a new site" → "a figure that does
    not indicate performance at a new site".
13. **P. 3, 2.1:** "headline figure" → "reported accuracy".
14. **P. 11, 4.1:** "The pooled figure hides a split among the four
    diagonals." → "The pooled value averages over diagonals that differ
    widely."
15. **P. 22, 5.2:** "which is why Torralba and Efros (2011) reported…" → "as
    in the matrix format used by Torralba and Efros (2011)".
16. **P. 23:** "Three results of this study bear on both." / "Two further
    results point the same way." → "Three results are relevant to both
    explanations." / "Two other results are consistent with this."

## Redundancia

La lista de vacíos y hallazgos aparece en el resumen, la Introducción (párrafo
4 y contribuciones), 2.4 y la Conclusión. Propuesta: condensar 2.4 a dos
frases o fundirla con la Introducción.
