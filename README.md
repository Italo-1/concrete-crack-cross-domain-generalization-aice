# Cross-domain generalization of concrete crack classifiers

Data and code for the article *Cross-Domain Generalization of Concrete Crack Classifiers: A Transfer Matrix Across Surfaces and Capture Campaigns*, prepared for *AI in Civil Engineering* (Springer Nature). It extends a first version of the study, whose code is in [concrete-crack-cross-domain-generalization](https://github.com/Italo-1/concrete-crack-cross-domain-generalization).

Escuela Superior Politécnica de Chimborazo (ESPOCH), Riobamba, Ecuador.

## What the study does

Four ImageNet-pretrained architectures (ResNet-18, EfficientNet-B0, MobileNetV3-Small, ViT-Tiny) are fine-tuned on each of four image domains, the bridge-deck, pavement and wall subsets of SDNET2018 and an independent collection from the Middle East Technical University (METU), with three seeds, and each model is tested on all four. Besides the baseline, the study runs aggressive augmentation, histogram equalization and random rescaling; a linear classifier on frozen features; multi-source training on three domains with the fourth held out; and a split grouped by source photo to check leakage between patches. Every training run is deterministic and stores per-image predictions, from which precision and recall, AUROC, threshold recalibration with a few labeled target images and Matthews correlation are computed.

## Repository layout

| Path | Content |
|---|---|
| `src/01_download.py` | Download SDNET2018 and METU |
| `src/02_preprocess.py` | Four domains matched in size, with fixed train, validation and test splits |
| `src/02b_particion_agrupada.py` | Alternative split of the SDNET2018 domains grouped by source photo |
| `src/03_experiment.py` | 4 × 4 transfer matrix with counts, per-image predictions and checkpoints (`--condicion`, `--particion`) |
| `src/03b_sonda_lineal.py` | Linear classifier on frozen ImageNet features |
| `src/03c_multifuente.py` | Training on three domains, testing on the held-out fourth |
| `src/entrenamiento.py` | Training, evaluation and determinism settings shared by the three experiment scripts |
| `src/tabla_arquitecturas.py` | Table of architectures (parameters, feature dimension, weights) |
| `src/08_umbral_auroc.py` | AUROC, average precision and threshold recalibration per cell |
| `src/09_ronda2.py` | Drop anchored on the target diagonal, accuracy and MCC by partition |
| `src/04_stats.py` | ANOVA, Tukey HSD, Welch and Wilcoxon tests with Holm correction, bootstrap intervals |
| `src/05_figures.py` | Figures 2-8 (Figure 1 is drawn in `paper/fig_pipeline.tex`) |
| `src/06_apendice.py` | Appendix tables (`paper/apendice.tex`) from `results/tables/` |
| `src/07_figuras_envio.py` | Figure files in EPS and TIFF for the journal's submission system |
| `src/comparar_original.py` | Comparison with the matrix of the first version (needs its results) |
| `src/piloto_comparar.py` | Check that two identical pilot runs give identical predictions |
| `src/run_all.py` | Runs the pipeline in order and logs package versions to `results/logs/` |
| `fieutils/` | Shared vision, statistics and plotting code |
| `data/processed/manifiesto_agrupado.csv` | The photo-grouped split |
| `results/tables/`, `results/figures/` | Every table and figure reported in the article |
| `results/predicciones/` | Per-image predictions of every run, by condition |
| `paper/main.tex`, `paper/main.pdf` | LaTeX source (journal template) and compiled manuscript |

## Reproducing the analysis

```bash
pip install -r requirements.txt   # PyTorch with CUDA 12.6: see the note at the top of the file
pip install -e .                  # installs the shared fieutils package
python src/run_all.py
```

`python src/run_all.py --list` prints the steps and `--from 08` resumes from a given step. The statistics work from the tables and predictions already in `results/`; the figures that show image patches also need the downloaded collections. Training needs a CUDA GPU and runs with `CUBLAS_WORKSPACE_CONFIG=:4096:8` for deterministic results (set by `03_experiment.py`). Seeds 42, 43 and 44. The trained weights are not included in the repository.

SDNET2018 (Utah State University) and the METU Concrete Crack Images collection (Mendeley Data, DOI [10.17632/5y9wdsg2zt.2](https://doi.org/10.17632/5y9wdsg2zt.2), CC BY 4.0) are downloaded from their original repositories by `01_download.py` and are not redistributed here.
