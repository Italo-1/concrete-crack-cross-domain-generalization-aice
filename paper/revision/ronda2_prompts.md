# Ronda 2 de revisión adversarial y pasada anti-IA: prompts preparados

Se lanzan **después** de incorporar EXP-escala al manuscrito y recompilar
`paper/main.pdf`, cada uno en un contexto limpio (agente nuevo sin acceso más
que al PDF). Prompts de la guía metodológica §5, con dos añadidos que ya se
usaron en la ronda 1: coherencia de cifras y abreviaturas sin definir. El
revisor de la ronda 2 no recibe el informe de la ronda 1 ni la respuesta: la
guía pide que no tenga contexto previo.

## 1. Ronda 2: prompt principal (revisor de AICE)

> Actúa como revisor par para la revista AI in Civil Engineering (Springer
> Nature, Tongji University). Te adjunto un manuscrito enviado a esa revista:
> es el PDF C:\lineaB\06-aice-P1\paper\main.pdf (léelo completo con la
> herramienta Read usando el parámetro pages por tramos y mira las figuras).
> Lee SOLO ese PDF: no abras ningún otro archivo del disco ni busques
> información sobre el proyecto, y no modifiques nada. Puedes consultar en la
> web los trabajos citados si necesitas comprobar algo concreto.
>
> Evalúalo con el rigor de un revisor exigente, no con amabilidad.
> Estructura tu revisión así:
> 1. Recomendación: aceptar / revisión menor / revisión mayor / rechazar
> 2. Tres objeciones principales, ordenadas por gravedad
> 3. Problemas metodológicos: ¿las conclusiones se sostienen con los datos
>    presentados? ¿falta alguna condición de control? ¿la prueba estadística es
>    la adecuada?
> 4. ¿Encaja realmente en el alcance de esta revista?
> 5. Problemas de redacción: pasajes que suenan generados por IA, adjetivos
>    innecesarios, afirmaciones sin respaldo
> 6. Problemas de figuras y tablas
> 7. Referencias: ¿faltan trabajos evidentes? ¿cita trabajos de esta misma
>    revista?
> Sé duro. Es preferible que las objeciones vengan de ti y no del editor.
>
> Además: comprueba la coherencia interna de las cifras (que un mismo número
> diga lo mismo en el resumen, la introducción, los resultados, las tablas,
> las figuras y la conclusión) y señala cualquier abreviatura usada sin definir
> en su primera aparición. Para cada problema indica la página y la frase o
> tabla concreta. Responde en español.

## 2. Pasada anti-IA: prompt complementario

> Lee el manuscrito C:\lineaB\06-aice-P1\paper\main.pdf (solo ese archivo, sin
> modificar nada) e identifica los pasajes que parecen escritos por un modelo
> de lenguaje: adjetivos valorativos, tríadas, frases de relleno, párrafos que
> anuncian lo que van a decir, vocabulario inflado. Para cada pasaje señalado,
> indica la página, cita la frase y propón una reescritura sobria en el mismo
> registro académico. No reescribas el manuscrito completo; solo los pasajes
> marcados. Responde en español; las reescrituras, en inglés.

## 3. Opcional: revisor hostil al encuadre

> Actúa como un revisor escéptico que sospecha que este artículo
> (C:\lineaB\06-aice-P1\paper\main.pdf; solo ese archivo, sin modificar nada)
> no aporta novedad suficiente. Construye el argumento más fuerte posible para
> rechazarlo por falta de contribución original. Después, indica qué tendría
> que cambiar el manuscrito, en encuadre y no en experimentos, para desarmar
> ese argumento. Responde en español.

## Antes de lanzar

1. EXP-escala terminado (192 filas `escala` en `resultados_matriz.csv`),
   análisis según el protocolo del README y resultado incorporado a Método,
   Resultados, Discusión y Limitaciones, sea cual sea.
2. `04_stats.py`, `05_figures.py`, `06_apendice.py`, `07_figuras_envio.py`
   corridos de nuevo; `main.pdf` compilado sin errores ni cajas desbordadas.
3. `verificar_envio.py P1-AICE` sin bloqueantes nuevos (solo autoría y
   repositorio).
