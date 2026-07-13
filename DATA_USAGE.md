# Data usage guide



This document describes the datasets, derived feature databases, model outputs, and recommended usage of this repository.



## 1. Overview



This repository provides data and reproducibility materials for a layer-aware reconstructable geometric representation framework for landslide profiles and landslide polygons. The repository contains two main experimental parts:



1\. Synthetic layered landslide-profile experiments based on MatDEM-generated deposit images.

2\. RER2023-derived landslide polygon experiments based on single-layer field representation and latent reconstruction.



The main purpose of the repository is to support reproducible geometric representation, latent feature extraction, inverse reconstruction, and downstream tabular analysis of landslide profile or polygon geometries.



## 2. Git LFS requirement



Many numerical feature databases, generated images, model checkpoints, and evaluation outputs are stored using Git LFS. Users should install Git LFS before cloning the repository:



```bash

git lfs install

git clone https://github.com/QiningDeng/Layer-aware-reconstructable-landslide-geometry.git

cd Layer-aware-reconstructable-landslide-geometry

git lfs pull

```



If Git LFS is not installed, only lightweight pointer files may be downloaded for large files.



## 3. Repository data structure



### 3.1 Synthetic layered-profile data



\* `data/synthetic\_profiles/raw\_images/`



&#x20; Raw synthetic MatDEM layered-profile images. These images are derived from MatDEM-based landslide deposition simulations originally developed for AI-enhanced landslide deposition prediction.



\* `data/synthetic\_profiles/background\_cleaned\_images/`



&#x20; Background-cleaned synthetic profile images after removal of irrelevant regions and non-study visual elements.



\* `data/synthetic\_profiles/layer\_polygon\_database/`



&#x20; Layer-wise polygon records extracted from the synthetic layered-profile images. This folder contains polygon JSON files and associated boundary-extraction outputs.



\* `data/synthetic\_stageA\_feature\_database/`



&#x20; Stage A SDF-corner-mask feature database constructed from the layer-wise polygon records. This database is used as the main training input for the synthetic latent representation models.



\* `data/synthetic\_no\_learning\_representation\_database/`



&#x20; No-learning representation and self-check reconstruction outputs based on the Stage A geometric database. These files are used to verify whether the field-based SDF-corner-mask representation can reconstruct polygonal layer geometry before latent representation learning.



\* `results/synthetic\_stageB\_latent\_dimension\_sweep/`



&#x20; Stage B latent representation learning results for VAE and joint implicit auto-decoder models under multiple latent dimensions. The tested latent dimensions include 1, 2, 4, 8, 16, 32, 64, 128, 256, and 512.



### 3.2 RER2023-derived polygon data



\* `data/rer2023/single\_layer\_field\_database/`



&#x20; A derivative single-layer SDF-corner-mask feature database generated from the RER2023 landslide polygon inventory. Each sample represents one landslide polygon using a single-layer field representation.



\* `results/rer2023\_single\_layer\_latent\_model/`



&#x20; Latent reconstruction model outputs for the RER2023-derived single-layer field database. This folder includes model checkpoints, training curves, evaluation results, and related training history.



\* `results/rer2023\_reconstruction\_evaluation\_visuals/`



&#x20; Full reconstruction evaluation outputs, outlier-filtered metric tables, selected visual cases, exported masks, and publication-oriented reconstruction figures for the RER2023 polygon experiment.



\* `data/rer2023/latent\_feature\_tables/`



&#x20; Exported latent feature tables and related tabular datasets used for downstream morphology-based analysis and classification experiments.



## 4. Important data-source notes



### 4.1 Synthetic MatDEM data



The raw synthetic MatDEM layered-profile images were derived from a previously developed landslide deposition simulation dataset. Users who use the raw synthetic images or derivative datasets generated from them should cite the original synthetic-data source:



Cui, Y., Gong, C., Zheng, J., Wang, K., Han, J., Liu, W., Zhou, Y., 2026. AI-enhanced landslide deposition prediction: a novel framework integrating discrete element method and generative adversarial networks. Engineering Geology, 108752. https://doi.org/10.1016/j.enggeo.2026.108752.



The present repository further processes these images into background-cleaned profiles, layer-wise polygon records, SDF-corner-mask feature databases, no-learning reconstruction outputs, latent representation learning results, and tabular reconstructable geometric descriptors.



### 4.2 RER2023 data



The original RER2023 landslide polygon shapefile is not redistributed in this repository because its redistribution terms should be checked from the original data provider. This repository provides derivative feature databases, reconstruction results, and reproducibility scripts for the analyses reported in the manuscript.



Users who wish to reproduce the raw shapefile preprocessing step should obtain the original RER2023 landslide inventory from its official source and cite it according to the corresponding data-use terms.



The RER2023 dataset is described in:



Berti, M., Pizziolo, M., Scaroni, M., Generali, M., Critelli, V., Mulas, M., Tondo, M., Lelli, F., Fabbiani, C., Ronchetti, F., Ciccarese, G., Dal Seno, N., Ioriatti, E., Rani, R., Zuccarini, A., Simonelli, T., Corsini, A., 2025. RER2023: the landslide inventory dataset of the May 2023 Emilia-Romagna meteorological event. Earth System Science Data, 17(3), 1055–1074. https://doi.org/10.5194/essd-17-1055-2025.



The official public release is available through Zenodo at: https://doi.org/10.5281/zenodo.13742643.



## 5. Script-to-data correspondence



The main scripts are stored in `scripts/`.



### 5.1 Synthetic layered-profile workflow



\* `scripts/stageA\_sdf\_feature\_database.py`



&#x20; Builds the Stage A SDF-corner-mask feature database from layer-wise polygon JSON files. The outputs include numerical feature arrays, metadata, reconstructed polygons, self-check visualizations, and evaluation reports.



\* `scripts/stageB\_joint\_implicit\_train\_eval\_predict\_timing\_resume\_realtime.py`



&#x20; Trains and evaluates the Stage B latent representation models, including the VAE-based model and the joint implicit auto-decoder.



\* `scripts/batch\_train\_stageB\_all\_dimensions.py`



&#x20; Runs batch training across multiple latent dimensions for the synthetic Stage B experiments.



\* `scripts/stageB\_section4\_4\_tabular\_reconstruction\_batch.py`



&#x20; Exports tabular latent features, performs reconstruction evaluation, and generates representative reconstruction overlays for selected latent dimensions.



### 5.2 RER2023-derived polygon workflow



\* `scripts/01\_build\_rer2023\_single\_layer\_field\_database.py`



&#x20; Builds a single-layer SDF-corner-mask database from landslide polygon shapefiles.



\* `scripts/02\_train\_rer2023\_single\_layer\_field\_model.py`



&#x20; Trains and evaluates single-layer latent reconstruction models on the RER2023-derived feature database.



\* `scripts/03\_evaluate\_rer2023\_reconstruction\_visuals.py`



&#x20; Performs full train/validation/test reconstruction evaluation, outlier filtering, summary-table generation, and visualization of representative cases.



\* `scripts/05\_export\_rer2023\_latent\_features.py`



&#x20; Exports latent feature tables and related tabular datasets for downstream analysis.



## 6. Suggested use cases



Users may use this repository for the following purposes:



1\. Reproducing the synthetic layered-profile representation workflow.

2\. Examining the SDF-corner-mask field representation before and after latent dimensionality reduction.

3\. Evaluating reconstructable latent descriptors under different latent dimensions.

4\. Using exported latent feature tables for downstream landslide polygon morphology analysis.

5\. Comparing VAE-based and auto-decoder-based reconstructable geometric descriptors.

6\. Extending the workflow to other polygonal geohazard inventories or geotechnical geometry datasets.



## 7. Recommended usage order



For users interested in the synthetic layered-profile experiment:



1\. Start from `data/synthetic\_profiles/raw\_images/` and `data/synthetic\_profiles/background\_cleaned\_images/`.

2\. Inspect `data/synthetic\_profiles/layer\_polygon\_database/` for extracted layer-wise polygon records.

3\. Use `data/synthetic\_stageA\_feature\_database/` as the field-based training database.

4\. Inspect `data/synthetic\_no\_learning\_representation\_database/` to verify field-level reconstruction without representation learning.

5\. Use `results/synthetic\_stageB\_latent\_dimension\_sweep/` to compare VAE and auto-decoder reconstruction performance under different latent dimensions.

6\. Use `scripts/stageB\_section4\_4\_tabular\_reconstruction\_batch.py` to reproduce tabular latent feature export and reconstruction visualizations.



For users interested in the RER2023-derived polygon experiment:



1\. Obtain the original RER2023 shapefile from its official data source if raw preprocessing is required.

2\. Use `scripts/01\_build\_rer2023\_single\_layer\_field\_database.py` to rebuild the single-layer field database from the raw shapefile.

3\. Use `data/rer2023/single\_layer\_field\_database/` directly if only the processed derivative database is needed.

4\. Use `scripts/02\_train\_rer2023\_single\_layer\_field\_model.py` to train or evaluate the latent reconstruction model.

5\. Use `scripts/03\_evaluate\_rer2023\_reconstruction\_visuals.py` to reproduce reconstruction evaluation tables and visual cases.

6\. Use `data/rer2023/latent\_feature\_tables/` for downstream tabular analysis.



## 8. Notes on derivative data



The processed databases in this repository are derivative research data. They are intended to support the manuscript’s computational workflow, reconstruction experiments, and downstream analyses. Users should carefully cite the original data sources when using the synthetic MatDEM-derived images or RER2023-derived feature databases.



