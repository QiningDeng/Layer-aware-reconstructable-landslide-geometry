# Layer-aware reconstructable landslide geometry

This repository provides the datasets, processing scripts, model outputs, and reproducibility materials associated with the study:

**From layered deposit boundaries to reconstructable low-dimensional features: a layer-aware implicit geometric framework for landslide profile representation**

The repository focuses on a geometry-centered workflow for converting landslide profile and polygon boundaries into compact, reconstructable, and tabular geometric descriptors. The workflow includes background-cleaned synthetic layered-profile images, layer-wise polygon extraction, SDF-corner-mask feature construction, no-learning self-check reconstruction, latent representation learning with VAE and joint implicit auto-decoder models, inverse reconstruction, and downstream landslide-polygon feature-table experiments.

## Repository contents

The repository contains two main experimental parts.

First, the synthetic layered-profile experiment provides 1,200 MatDEM-generated three-layer landslide deposit profiles and their derived geometric databases. These files include raw synthetic images, background-cleaned images, layer-wise polygon records, no-learning SDF-corner-mask representation databases, Stage A feature databases, and Stage B latent-dimension sweep results for VAE and joint implicit auto-decoder models.

Second, the RER2023 landslide-polygon experiment provides derivative single-layer field databases, latent model training outputs, reconstruction evaluation results, visualization outputs, and exported latent feature tables. These materials are intended to support the polygon-level reconstruction and downstream morphology-based analysis reported in the manuscript.

## Data organization

* `data/synthetic_profiles/raw_images/`: raw synthetic MatDEM layered-profile images.
* `data/synthetic_profiles/background_cleaned_images/`: profile images after removal of irrelevant background and non-study regions.
* `data/synthetic_profiles/layer_polygon_database/`: layer-wise polygon records extracted from the synthetic profiles.
* `data/synthetic_stageA_feature_database/`: Stage A SDF-corner-mask feature database for latent model training.
* `data/synthetic_no_learning_representation_database/`: no-learning representation and self-check reconstruction outputs based on the Stage A geometric database.
* `results/synthetic_stageB_latent_dimension_sweep/`: VAE and joint implicit auto-decoder results under latent dimensions of 1, 2, 4, 8, 16, 32, 64, 128, 256, and 512.
* `data/rer2023/single_layer_field_database/`: derivative single-layer SDF-corner-mask database generated from the RER2023 landslide polygon inventory.
* `results/rer2023_single_layer_latent_model/`: latent reconstruction model outputs for the RER2023 polygon experiment.
* `results/rer2023_reconstruction_evaluation_visuals/`: full reconstruction evaluation, outlier-filtered metric tables, and visualization outputs.
* `data/rer2023/latent_feature_tables/`: exported latent feature tables and related tabular data for downstream analysis.
* `scripts/`: Python scripts used for database construction, model training, reconstruction evaluation, and feature-table export.

## Reproducibility

The scripts are organized according to the main workflow of the manuscript. The synthetic part covers background cleaning, polygon-based geometric representation, Stage A feature database construction, Stage B latent representation learning, and tabular reconstruction evaluation. The RER2023 part covers single-layer field database construction from polygon inventories, latent reconstruction model training, full reconstruction evaluation, and latent feature-table export.

Most numerical feature databases and model-related binary files are tracked using Git LFS. Users should install Git LFS before cloning the repository if they need to download the complete feature databases and model outputs.

## Synthetic MatDEM data note

The synthetic layered-profile images used in this repository were derived from MatDEM-based landslide deposition simulations originally developed for AI-enhanced landslide deposition prediction. Users who use the raw synthetic MatDEM images or derivative datasets generated from them should cite the original synthetic-data source:

Cui, Y., Gong, C., Zheng, J., Wang, K., Han, J., Liu, W., Zhou, Y., 2026. AI-enhanced landslide deposition prediction: a novel framework integrating discrete element method and generative adversarial networks. Engineering Geology, 108752. https://doi.org/10.1016/j.enggeo.2026.108752.

The present repository further processes these synthetic MatDEM images into background-cleaned images, layer-wise polygon records, SDF-corner-mask feature databases, no-learning reconstruction outputs, latent representation learning results, and tabular reconstructable geometric descriptors.

## RER2023 data note

The RER2023 dataset is described in the following publication: Berti, M., Pizziolo, M., Scaroni, M., Generali, M., Critelli, V., Mulas, M., Tondo, M., Lelli, F., Fabbiani, C., Ronchetti, F., Ciccarese, G., Dal Seno, N., Ioriatti, E., Rani, R., Zuccarini, A., Simonelli, T., Corsini, A., 2025. RER2023: the landslide inventory dataset of the May 2023 Emilia-Romagna meteorological event. Earth System Science Data, 17(3), 1055–1074. https://doi.org/10.5194/essd-17-1055-2025.

The official public release of the dataset can be accessed via Zenodo at: https://doi.org/10.5281/zenodo.13742643. Users should consult the original publication and data repository for detailed metadata, licensing conditions, and proper citation requirements.

## Citation

If this repository is useful, please cite.

If you use the raw synthetic MatDEM layered-profile images or derivative datasets generated from them, please also cite:

Cui, Y., Gong, C., Zheng, J., Wang, K., Han, J., Liu, W., Zhou, Y., 2026. AI-enhanced landslide deposition prediction: a novel framework integrating discrete element method and generative adversarial networks. Engineering Geology, 108752. https://doi.org/10.1016/j.enggeo.2026.108752.

If you use the RER2023-derived data or reproduce the RER2023 preprocessing workflow, please obtain and cite the original RER2023 dataset according to its official publication and data repository.

