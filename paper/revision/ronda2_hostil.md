# Ronda 2: revisor hostil al encuadre (03/10/2026, sin aplicar todavía)

Agente en contexto limpio, solo con `paper/main.pdf` (34 páginas), prompt 3 de
`ronda2_prompts.md`. Resumen del informe; sus afirmaciones de hecho están
pendientes de verificar contra los datos y las fuentes.

## Parte I. Argumento para rechazar por falta de novedad

1. **Diseño conocido:** es la matriz de Torralba y Efros; Rashid et al. (2025)
   ya hicieron una prueba cruzada con SDNET2018 y METU en las dos direcciones.
   Lo que se añade (semillas, SDNET partido en tres, balance, pruebas) es de
   procedimiento: una réplica más rigurosa. Santos et al. (2022) y Özgenel
   (2018) ya midieron el fallo de modelos entrenados en METU.
2. **La sección 2.4 es una lista de comprobaciones no hechas, no preguntas
   abiertas:** clasificador trivial (práctica básica), LP frente a FT
   (Kornblith; Kumar), agregación de fuentes (Gulrajani), fuga entre parches
   (Kapoor; Abeysuriya), réplicas (Bouthillier).
3. **Resultados mayoritariamente nulos o confirmatorios:** falla la hipótesis
   a priori; ninguna intervención funciona (escala, p = 0.052); LP = FT en la
   dirección crítica; tres fuentes = mejor fuente; modelos pequeños.
4. **La asimetría es n = 1 y admite una lectura trivial:** METU es fácil (LP
   0.997), así que entrenar en lo fácil y probar en lo difícil transfiere
   peor; efectos aditivos de dificultad de origen y destino bastarían. Afirma
   que SDNET-P→METU (0.94) supera la diagonal de SDNET-P. La caída relativa,
   anclada en una diagonal de METU de 0.998 que puede tener fuga, infla el
   efecto. La única prueba del mecanismo (escala) no lo apoyó.
5. **Cambio de definición de etiqueta:** METU exige grietas alejadas del
   borde; SDNET2018 incluye grietas de 0.06 mm en cuadrícula. Se mide una
   incompatibilidad de anotación, no un límite de generalización.
6. **El 0.667 es un artefacto del balance:** con las prevalencias reales de
   SDNET2018 (10.7-21.2 %), la referencia baja a 0.19-0.35 y la precisión
   cambia.
7. **La premisa no se cumple en sus datos:** diagonales de SDNET2018 de
   0.79-0.86, lejos del 95 % que motiva el artículo; el contraste solo vale
   para METU.
8. **El protocolo de validación local es práctica estándar.**

Veredicto: rechazo por contribución insuficiente, o nota de replicación.

## Parte II. Cambios de encuadre propuestos (sin experimentos)

1. Declararlo estudio de medición confirmatorio: separar en 2.4 y en las
   contribuciones lo que se confirma (Kumar, Gulrajani, Kapoor, Torralba) de lo
   nuevo; reducir a una afirmación principal y una implicación metodológica.
2. Párrafo o tabla "Rashid et al. frente a este estudio": si sus cifras ya
   muestran la asimetría, decir que aquí se replica y se prueba; si no, por qué
   su diseño la ocultaba.
3. Tesis metodológica central: el agregado engaña (0.610 fuera del IC de las
   dos direcciones); la historia del propio estudio lo muestra.
4. Nombrar la diferencia de definición de etiqueta como hallazgo ("los
   conjuntos públicos no son intercambiables"); la escala descarta solo una
   parte.
5. "La colección más fácil es la peor maestra": asumir la lectura de
   dificultad como hipótesis generada por un caso, comprobable con una tercera
   campaña.
6. Anticipar las objeciones: F1 absoluto y diferencia entre direcciones antes
   que la caída relativa; llevar 2π/(1+π) al planteamiento; sacar "78 de 144"
   del resumen como cifra de impacto.
7. Dar prioridad al fallo silencioso (P > R en 77 de 78) como aporte práctico:
   revisar los positivos no lo revela.
8. Título y resumen: el título promete "capture campaigns" en plural con una
   sola campaña externa; propuesta "Public crack datasets are not
   interchangeable: direction-dependent transfer between SDNET2018 and METU";
   abrir el resumen con la tesis; "entre colecciones" en lugar de "entre
   campañas" donde corresponda.
9. Declarar en la Introducción que el presupuesto igualado no reproduce el
   95 % en SDNET2018, y por qué era necesario.
10. Contribución 4 como "implicaciones para el despliegue", con cada paso
    ligado a un resultado.
