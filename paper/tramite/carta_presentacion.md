<!--
Carta de presentación para AI in Civil Engineering (Editorial Manager).
Parte del borrador de la guía del encargado, §10, con estos cambios:
- cifras de los resultados nuevos (03/10/2026), no las de la versión anterior
  (0.287 → 0.345; "half the cells" → 78 de 144, con el modo de fallo);
- la cita a Luleci & Catbas se limita a lo que su artículo dice (transferencia
  entre estructuras planteada como generalización de dominio), no a una
  asimetría que no mide;
- se añaden la sonda lineal, la condición multi-fuente y la partición agrupada;
- ronda 2 de revisión (03/10): anclaje en el destino, exactitud balanceada,
  umbral de Youden y reescalado no significativo;
- sin nombres de autores: firma con el marcador hasta decidir la autoría
  (verificar_envio.py lo bloquea a propósito);
- URL del repositorio: se completa cuando se decida entre GitHub y Zenodo.
Copiar el texto de abajo en el campo de carta de Editorial Manager.
-->

Dear Editor-in-Chief,

We submit for your consideration the manuscript entitled "Cross-Domain Generalization of Concrete Crack Classifiers: A Transfer Matrix Across Surfaces and Capture Campaigns" for publication as an Original Article in *AI in Civil Engineering*.

A recent systematic review in your journal (Abeysuriya et al., 2026) found weak evidence that deep-learning models for structural damage assessment generalize to unseen data, and asked for cross-dataset testing. Our manuscript measures that gap for concrete crack classification. We fine-tuned four architectures on each of four image domains, three surfaces of SDNET2018 and the independent METU collection, with three seeds each, and tested every model on all four domains. Domains were balanced so that a classifier labeling every image as cracked scores F1 = 0.667 on every test set.

The two directions of transfer between the image collections differ by 0.345 F1 (0.782 against 0.437), a difference that disappears when the directions are pooled, as a single cross-dataset figure implicitly does. Part of it reflects that METU is an easy target, but METU-trained models lose most whether the loss is measured from the source or the target. Of the 144 out-of-domain results, 78 score at or below the F1 of the always-crack classifier, although all remain better than chance in balanced accuracy, and 77 of those miss most cracks, so reviewing the images a model flags would not reveal the problem. From METU to SDNET2018 the models also rank target images weakly, and a decision threshold recalibrated on labeled images of the target changes balanced accuracy little. The loss does not follow architecture size, is not reduced by aggressive augmentation or histogram equalization, nor significantly by random rescaling, and a linear probe on frozen features fails in the same way. The asymmetry persists, smaller, when the test split is grouped by parent photograph to remove leakage between patches. The direction we observe agrees with the drop reported in your journal by Santos et al. (2022) for a classifier trained on the same METU collection and tested on a concrete dam.

We believe the manuscript fits the scope of the journal because it addresses the evaluation of AI tools for infrastructure inspection and ends with local validation steps, each tied to a measured result, that an agency can apply before using a classifier's output for maintenance decisions. It also responds to the call for cross-site evaluation in recent work on concrete bridge defects in your journal (Lateef & Yu, 2026).

The manuscript is original, has not been published, and is not under consideration elsewhere. Both image collections are public and cited with their DOIs. The code, the data splits, the per-cell results and the per-image predictions are available at [repository URL and Zenodo DOI pending]. The use of a large language model assistant is described in the Methods section, as the journal's guidelines require. The authors declare no competing interests.

Sincerely,

[Author list pending]
