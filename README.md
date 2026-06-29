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

## RER2023 data note

The original RER2023 landslide polygon inventory is not redistributed in this repository unless its redistribution terms are explicitly confirmed. The repository provides derivative data and reproducibility scripts for the analyses reported in the manuscript. Users who wish to reproduce the raw shapefile preprocessing step should obtain the original RER2023 inventory from its official data source and cite it according to the corresponding data-use terms.

## Citation

If this repository is useful for your research, please cite the associated manuscript once it becomes available.
