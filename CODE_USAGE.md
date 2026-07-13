# Code usage and running examples



This document explains how to use the main Python scripts in this repository. The scripts are organized according to two experimental workflows:



1\. Synthetic layered-profile geometry representation and reconstruction.

2\. RER2023-derived landslide polygon representation and reconstruction.



The examples assume that the current working directory is the root of the repository:



```bash

cd Layer-aware-reconstructable-landslide-geometry

```



Large feature databases, model checkpoints, and generated figures are tracked using Git LFS. Before running the scripts, make sure the LFS files have been downloaded:



```bash

git lfs install

git lfs pull

```



All examples write newly generated outputs to the `outputs/` folder to avoid overwriting the precomputed manuscript results under `data/` and `results/`.



## 1. Recommended environment



The scripts require Python scientific-computing, deep-learning, image-processing, and geospatial packages.



A typical conda environment can be created as follows:



```bash

conda create -n landslide\_geometry python=3.10

conda activate landslide\_geometry



pip install numpy pandas matplotlib opencv-python pillow tqdm scikit-learn openpyxl

pip install torch torchvision torchaudio

pip install geopandas shapely pyproj fiona

```



For GPU acceleration, install the PyTorch version that matches the local CUDA version. CPU execution is possible for small tests, but full training and posterior latent fitting are computationally expensive.



## 2. Script overview



| Script                                                                                                                                             | Main purpose                                                          | Main input                                 | Main output                                                                      |
| -------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------- | ------------------------------------------ | -------------------------------------------------------------------------------- |
| [`stageA_sdf_feature_database.py`](scripts/stageA_sdf_feature_database.py)                                                                         | Builds the Stage A SDF-corner-mask database                           | Layered polygon JSON files                 | Feature files, metadata, self-check results, and evaluation reports              |
| [`stageB_joint_implicit_train_eval_predict_timing_resume_realtime.py`](scripts/stageB_joint_implicit_train_eval_predict_timing_resume_realtime.py) | Trains, evaluates, and predicts layered-profile reconstruction models | Stage A feature database                   | Checkpoints, logs, evaluation results, predicted fields, and reconstructed masks |
| [`batch_train_stageB_all_dimensions.py`](scripts/batch_train_stageB_all_dimensions.py)                                                             | Runs VAE and auto-decoder experiments across latent dimensions        | Stage A feature database                   | Latent-dimension sweep results                                                   |
| [`stageB_section4_4_tabular_reconstruction_batch.py`](scripts/stageB_section4_4_tabular_reconstruction_batch.py)                                   | Exports latent tables and reconstruction results for Section 4.4      | Stage B model folders and Stage A database | Latent tables, reconstruction summaries, and visual examples                     |
| [`01_build_rer2023_single_layer_field_database.py`](scripts/01_build_rer2023_single_layer_field_database.py)                                       | Builds the RER2023 single-layer feature database                      | RER2023 shapefile                          | RER2023 feature database                                                         |
| [`02_train_rer2023_single_layer_field_model.py`](scripts/02_train_rer2023_single_layer_field_model.py)                                             | Trains and evaluates RER2023 VAE or auto-decoder models               | RER2023 feature database                   | Checkpoints, training history, and evaluation results                            |
| [`03_evaluate_rer2023_reconstruction_visuals.py`](scripts/03_evaluate_rer2023_reconstruction_visuals.py)                                           | Evaluates reconstruction performance and generates visual cases       | RER2023 database and best checkpoint       | Metrics, summary tables, and visual cases                                        |
| [`05_export_rer2023_latent_features.py`](scripts/05_export_rer2023_latent_features.py)                                                             | Exports latent features using posterior fitting                       | RER2023 database and best checkpoint       | Latent feature tables and IoU-filtered tables                                    |




## 3. Synthetic layered-profile workflow



### 3.1 Build the Stage A SDF-corner-mask feature database



This script converts layer-wise polygon JSON files into a multi-channel field representation. Each sample is represented by SDF, corner, and mask channels.



Input folder:



```text

data/synthetic\_profiles/layer\_polygon\_database/

```



Example command:



```bash

python scripts/stageA\_sdf\_feature\_database.py \\

&#x20; --input\_dir data/synthetic\_profiles/layer\_polygon\_database \\

&#x20; --output\_dir outputs/rebuild\_synthetic\_stageA\_feature\_database \\

&#x20; --grid\_width 512 \\

&#x20; --grid\_height 512 \\

&#x20; --padding 16 \\

&#x20; --sdf\_clip 64 \\

&#x20; --corner\_sigma 2.5

```



Expected outputs include:



```text

outputs/rebuild\_synthetic\_stageA\_feature\_database/

├── features/

├── metadata/

├── selfcheck\_polygons/

├── selfcheck\_visuals/

├── feature\_visuals/

└── reports/

```



The precomputed version used in the manuscript is available at:



```text

data/synthetic\_stageA\_feature\_database/

```



### 3.2 Train a synthetic Stage B VAE model



This command trains a VAE with a joint implicit decoder.



```bash

python scripts/stageB\_joint\_implicit\_train\_eval\_predict\_timing\_resume\_realtime.py train \\

&#x20; --model\_type vae \\

&#x20; --feature\_root data/synthetic\_stageA\_feature\_database \\

&#x20; --output\_root outputs/synthetic\_stageB\_vae\_d16 \\

&#x20; --latent\_dim 16 \\

&#x20; --encoder\_input\_size 128 \\

&#x20; --eval\_grid\_size 256 \\

&#x20; --batch\_size 4 \\

&#x20; --epochs 160 \\

&#x20; --learning\_rate 5e-5 \\

&#x20; --points\_per\_sample 4096 \\

&#x20; --patience 25 \\

&#x20; --device cuda

```



For a short CPU test:



```bash

python scripts/stageB\_joint\_implicit\_train\_eval\_predict\_timing\_resume\_realtime.py train \\

&#x20; --model\_type vae \\

&#x20; --feature\_root data/synthetic\_stageA\_feature\_database \\

&#x20; --output\_root outputs/synthetic\_stageB\_vae\_d16\_cpu\_test \\

&#x20; --latent\_dim 16 \\

&#x20; --epochs 5 \\

&#x20; --batch\_size 2 \\

&#x20; --points\_per\_sample 1024 \\

&#x20; --device cpu

```



### 3.3 Train a synthetic Stage B joint implicit auto-decoder model



```bash

python scripts/stageB\_joint\_implicit\_train\_eval\_predict\_timing\_resume\_realtime.py train \\

&#x20; --model\_type autodecoder \\

&#x20; --feature\_root data/synthetic\_stageA\_feature\_database \\

&#x20; --output\_root outputs/synthetic\_stageB\_autodecoder\_d64 \\

&#x20; --latent\_dim 64 \\

&#x20; --encoder\_input\_size 128 \\

&#x20; --eval\_grid\_size 256 \\

&#x20; --batch\_size 4 \\

&#x20; --epochs 160 \\

&#x20; --learning\_rate 5e-5 \\

&#x20; --latent\_learning\_rate 5e-3 \\

&#x20; --points\_per\_sample 4096 \\

&#x20; --patience 25 \\

&#x20; --device cuda

```



For a short CPU test:



```bash

python scripts/stageB\_joint\_implicit\_train\_eval\_predict\_timing\_resume\_realtime.py train \\

&#x20; --model\_type autodecoder \\

&#x20; --feature\_root data/synthetic\_stageA\_feature\_database \\

&#x20; --output\_root outputs/synthetic\_stageB\_autodecoder\_d64\_cpu\_test \\

&#x20; --latent\_dim 64 \\

&#x20; --epochs 5 \\

&#x20; --batch\_size 2 \\

&#x20; --points\_per\_sample 1024 \\

&#x20; --device cpu

```



### 3.4 Resume a synthetic Stage B training run



When extending a capped or interrupted run, use `--resume\_checkpoint`. A typical checkpoint is `latest\_model.pt` or `best\_model.pt`, depending on the output folder structure.



```bash

python scripts/stageB\_joint\_implicit\_train\_eval\_predict\_timing\_resume\_realtime.py train \\

&#x20; --model\_type autodecoder \\

&#x20; --feature\_root data/synthetic\_stageA\_feature\_database \\

&#x20; --output\_root outputs/synthetic\_stageB\_autodecoder\_d64 \\

&#x20; --latent\_dim 64 \\

&#x20; --epochs 300 \\

&#x20; --resume\_checkpoint outputs/synthetic\_stageB\_autodecoder\_d64/checkpoints/latest\_model.pt \\

&#x20; --device cuda

```



If the optimizer state is not compatible or a fresh optimizer is preferred, add:



```bash

\--resume\_reset\_optimizer

```



### 3.5 Evaluate a synthetic Stage B checkpoint



Example for validation split:



```bash

python scripts/stageB\_joint\_implicit\_train\_eval\_predict\_timing\_resume\_realtime.py evaluate \\

&#x20; --checkpoint outputs/synthetic\_stageB\_autodecoder\_d64/checkpoints/best\_model.pt \\

&#x20; --feature\_root data/synthetic\_stageA\_feature\_database \\

&#x20; --output\_root outputs/synthetic\_stageB\_autodecoder\_d64 \\

&#x20; --split\_name val

```



Example for test split:



```bash

python scripts/stageB\_joint\_implicit\_train\_eval\_predict\_timing\_resume\_realtime.py evaluate \\

&#x20; --checkpoint outputs/synthetic\_stageB\_autodecoder\_d64/checkpoints/best\_model.pt \\

&#x20; --feature\_root data/synthetic\_stageA\_feature\_database \\

&#x20; --output\_root outputs/synthetic\_stageB\_autodecoder\_d64 \\

&#x20; --split\_name test

```



### 3.6 Export predictions from a synthetic Stage B checkpoint



```bash

python scripts/stageB\_joint\_implicit\_train\_eval\_predict\_timing\_resume\_realtime.py predict \\

&#x20; --checkpoint outputs/synthetic\_stageB\_autodecoder\_d64/checkpoints/best\_model.pt \\

&#x20; --feature\_root data/synthetic\_stageA\_feature\_database \\

&#x20; --output\_root outputs/synthetic\_stageB\_autodecoder\_d64 \\

&#x20; --split\_name test

```



### 3.7 Run the full latent-dimension sweep



This batch launcher trains VAE and auto-decoder models under multiple latent dimensions. By default, the intended latent dimensions are:



```text

1, 2, 4, 8, 16, 32, 64, 128, 256, 512

```



A dry run can be used first to inspect the training plan without launching training:



```bash

python scripts/batch\_train\_stageB\_all\_dimensions.py \\

&#x20; --trainer\_script scripts/stageB\_joint\_implicit\_train\_eval\_predict\_timing\_resume\_realtime.py \\

&#x20; --feature\_root data/synthetic\_stageA\_feature\_database \\

&#x20; --output\_root outputs/synthetic\_stageB\_latent\_dimension\_sweep \\

&#x20; --models vae autodecoder \\

&#x20; --latent\_dims 1 2 4 8 16 32 64 128 256 512 \\

&#x20; --epochs 5000 \\

&#x20; --patience 40 \\

&#x20; --dry\_run

```



To run the actual batch training:



```bash

python scripts/batch\_train\_stageB\_all\_dimensions.py \\

&#x20; --trainer\_script scripts/stageB\_joint\_implicit\_train\_eval\_predict\_timing\_resume\_realtime.py \\

&#x20; --feature\_root data/synthetic\_stageA\_feature\_database \\

&#x20; --output\_root outputs/synthetic\_stageB\_latent\_dimension\_sweep \\

&#x20; --models vae autodecoder \\

&#x20; --latent\_dims 1 2 4 8 16 32 64 128 256 512 \\

&#x20; --epochs 5000 \\

&#x20; --patience 40 \\

&#x20; --batch\_size 4 \\

&#x20; --points\_per\_sample 4096 \\

&#x20; --existing\_run\_policy archive \\

&#x20; --continue\_on\_error

```



The repository already contains precomputed sweep results under:



```text

results/synthetic\_stageB\_latent\_dimension\_sweep/

```



### 3.8 Export Section 4.4 latent tables and reconstruction visualizations



This script integrates latent feature table export, reconstruction evaluation, and representative visualization for selected VAE and auto-decoder dimensions.



Full example:



```bash

python scripts/stageB\_section4\_4\_tabular\_reconstruction\_batch.py \\

&#x20; --stageb\_root results/synthetic\_stageB\_latent\_dimension\_sweep \\

&#x20; --feature\_root data/synthetic\_stageA\_feature\_database \\

&#x20; --output\_root outputs/synthetic\_section4\_4\_tabular\_reconstruction \\

&#x20; --selected\_vae\_dim 16 \\

&#x20; --selected\_autodecoder\_dim 64 \\

&#x20; --splits train val test \\

&#x20; --run\_latent\_export \\

&#x20; --run\_evaluation \\

&#x20; --run\_visual\_examples \\

&#x20; --examples\_per\_split 2 \\

&#x20; --sample\_names 59 358 573 768 997 1173 \\

&#x20; --polygon\_json\_root data/synthetic\_profiles/layer\_polygon\_database \\

&#x20; --original\_image\_root data/synthetic\_profiles/raw\_images \\

&#x20; --device cuda

```



Quick selected-model test:



```bash

python scripts/stageB\_section4\_4\_tabular\_reconstruction\_batch.py \\

&#x20; --stageb\_root results/synthetic\_stageB\_latent\_dimension\_sweep \\

&#x20; --feature\_root data/synthetic\_stageA\_feature\_database \\

&#x20; --output\_root outputs/synthetic\_section4\_4\_quick\_test \\

&#x20; --selected\_vae\_dim 16 \\

&#x20; --selected\_autodecoder\_dim 64 \\

&#x20; --only\_selected\_models \\

&#x20; --splits val test \\

&#x20; --run\_evaluation \\

&#x20; --run\_visual\_examples \\

&#x20; --polygon\_json\_root data/synthetic\_profiles/layer\_polygon\_database \\

&#x20; --device cuda

```



## 4. RER2023-derived landslide polygon workflow



The original RER2023 shapefile is not redistributed in this repository. To rebuild the RER2023-derived database from raw polygons, obtain the original RER2023 inventory from its official source and cite it according to the original data-use terms.



The repository provides the processed derivative database at:



```text

data/rer2023/single\_layer\_field\_database/

```



### 4.1 Build a single-layer field database from an original shapefile



This script reads landslide polygons from a shapefile, extracts valid polygons, computes shape descriptors, assigns or reads classes, normalizes polygons to a fixed grid, and exports SDF, corner, and mask feature arrays.



Example using all valid polygons:



```bash

python scripts/01\_build\_rer2023\_single\_layer\_field\_database.py \\

&#x20; --shp /path/to/RER2023.shp \\

&#x20; --output outputs/rer2023\_single\_layer\_field\_database \\

&#x20; --total-samples 0 \\

&#x20; --resolution 256 \\

&#x20; --padding 12 \\

&#x20; --sdf-clip 32 \\

&#x20; --corner-sigma 2.0 \\

&#x20; --corner-angle-threshold 35 \\

&#x20; --seed 2026 \\

&#x20; --preview-count 48

```



Example using only 200 polygons for testing:



```bash

python scripts/01\_build\_rer2023\_single\_layer\_field\_database.py \\

&#x20; --shp /path/to/RER2023.shp \\

&#x20; --output outputs/rer2023\_single\_layer\_field\_database\_test \\

&#x20; --total-samples 200 \\

&#x20; --resolution 256 \\

&#x20; --padding 12 \\

&#x20; --seed 2026 \\

&#x20; --preview-count 20

```



If the shapefile contains a meaningful ID column or class column, they can be specified:



```bash

python scripts/01\_build\_rer2023\_single\_layer\_field\_database.py \\

&#x20; --shp /path/to/RER2023.shp \\

&#x20; --output outputs/rer2023\_single\_layer\_field\_database \\

&#x20; --id-column FID \\

&#x20; --class-column landslide\_type \\

&#x20; --total-samples 0 \\

&#x20; --resolution 256 \\

&#x20; --seed 2026

```



Expected output:



```text

outputs/rer2023\_single\_layer\_field\_database/

├── features/

├── metadata/

├── previews/

└── reports/

```



### 4.2 Train a RER2023 single-layer auto-decoder model



This script supports both `autodecoder` and `vae`. The auto-decoder is usually used when posterior latent fitting and inverse reconstruction are important.



```bash

python scripts/02\_train\_rer2023\_single\_layer\_field\_model.py train \\

&#x20; --feature-root data/rer2023/single\_layer\_field\_database \\

&#x20; --output-root outputs/rer2023\_single\_layer\_autodecoder\_d64 \\

&#x20; --model-type autodecoder \\

&#x20; --latent-dim 64 \\

&#x20; --epochs 5000 \\

&#x20; --batch-size 16 \\

&#x20; --points-per-sample 8192 \\

&#x20; --validate-every 1 \\

&#x20; --patience 50 \\

&#x20; --posterior-steps 300 \\

&#x20; --posterior-lr 2e-2 \\

&#x20; --val-posterior-steps 100 \\

&#x20; --val-posterior-lr 2e-2 \\

&#x20; --device cuda \\

&#x20; --final-val-visuals 64 \\

&#x20; --final-test-visuals 64

```



Short CPU test:



```bash

python scripts/02\_train\_rer2023\_single\_layer\_field\_model.py train \\

&#x20; --feature-root data/rer2023/single\_layer\_field\_database \\

&#x20; --output-root outputs/rer2023\_single\_layer\_autodecoder\_d64\_cpu\_test \\

&#x20; --model-type autodecoder \\

&#x20; --latent-dim 64 \\

&#x20; --epochs 5 \\

&#x20; --batch-size 4 \\

&#x20; --points-per-sample 2048 \\

&#x20; --val-posterior-steps 10 \\

&#x20; --posterior-steps 20 \\

&#x20; --device cpu \\

&#x20; --final-max-samples 20 \\

&#x20; --final-val-visuals 4 \\

&#x20; --final-test-visuals 4

```



### 4.3 Train a RER2023 single-layer VAE model



```bash

python scripts/02\_train\_rer2023\_single\_layer\_field\_model.py train \\

&#x20; --feature-root data/rer2023/single\_layer\_field\_database \\

&#x20; --output-root outputs/rer2023\_single\_layer\_vae\_d64 \\

&#x20; --model-type vae \\

&#x20; --latent-dim 64 \\

&#x20; --epochs 5000 \\

&#x20; --batch-size 16 \\

&#x20; --points-per-sample 8192 \\

&#x20; --validate-every 1 \\

&#x20; --patience 50 \\

&#x20; --device cuda

```



### 4.4 Resume RER2023 model training



```bash

python scripts/02\_train\_rer2023\_single\_layer\_field\_model.py train \\

&#x20; --feature-root data/rer2023/single\_layer\_field\_database \\

&#x20; --output-root outputs/rer2023\_single\_layer\_autodecoder\_d64 \\

&#x20; --model-type autodecoder \\

&#x20; --latent-dim 64 \\

&#x20; --resume outputs/rer2023\_single\_layer\_autodecoder\_d64/checkpoints/latest.pt \\

&#x20; --epochs 6000 \\

&#x20; --device cuda

```



If restarting the monitoring logic is needed after resuming, add:



```bash

\--reset-monitor-on-resume

```



### 4.5 Evaluate an existing RER2023 checkpoint



Evaluate the test split:



```bash

python scripts/02\_train\_rer2023\_single\_layer\_field\_model.py evaluate \\

&#x20; --checkpoint results/rer2023\_single\_layer\_latent\_model/checkpoints/best.pt \\

&#x20; --feature-root data/rer2023/single\_layer\_field\_database \\

&#x20; --output-root outputs/rer2023\_single\_layer\_eval \\

&#x20; --split test \\

&#x20; --posterior-steps 300 \\

&#x20; --posterior-lr 2e-2 \\

&#x20; --save-visuals 64 \\

&#x20; --device cuda

```



Quick CPU evaluation:



```bash

python scripts/02\_train\_rer2023\_single\_layer\_field\_model.py evaluate \\

&#x20; --checkpoint results/rer2023\_single\_layer\_latent\_model/checkpoints/best.pt \\

&#x20; --feature-root data/rer2023/single\_layer\_field\_database \\

&#x20; --output-root outputs/rer2023\_single\_layer\_eval\_quick \\

&#x20; --split test \\

&#x20; --max-samples 50 \\

&#x20; --posterior-steps 20 \\

&#x20; --save-visuals 8 \\

&#x20; --device cpu

```



### 4.6 Run full reconstruction evaluation and outlier-filtered visualization



This script performs full evaluation over train, validation, and test splits. It can reuse existing full `sample\_metrics.csv` files if they are already available.



Full example:



```bash

python scripts/03\_evaluate\_rer2023\_reconstruction\_visuals.py \\

&#x20; --train-script scripts/02\_train\_rer2023\_single\_layer\_field\_model.py \\

&#x20; --feature-root data/rer2023/single\_layer\_field\_database \\

&#x20; --checkpoint results/rer2023\_single\_layer\_latent\_model/checkpoints/best.pt \\

&#x20; --output-root outputs/rer2023\_full\_reconstruction\_evaluation \\

&#x20; --splits train val test \\

&#x20; --class-column abcd\_class \\

&#x20; --device cuda \\

&#x20; --posterior-steps 200 \\

&#x20; --posterior-lr 2e-2 \\

&#x20; --outlier-group-mode split\_class \\

&#x20; --outlier-iqr-k 1.5 \\

&#x20; --visuals-per-class 3

```



Force recomputation instead of reusing existing metric tables:



```bash

python scripts/03\_evaluate\_rer2023\_reconstruction\_visuals.py \\

&#x20; --train-script scripts/02\_train\_rer2023\_single\_layer\_field\_model.py \\

&#x20; --feature-root data/rer2023/single\_layer\_field\_database \\

&#x20; --checkpoint results/rer2023\_single\_layer\_latent\_model/checkpoints/best.pt \\

&#x20; --output-root outputs/rer2023\_full\_reconstruction\_evaluation\_recompute \\

&#x20; --splits train val test \\

&#x20; --device cuda \\

&#x20; --posterior-steps 200 \\

&#x20; --force-recompute \\

&#x20; --visuals-per-class 3

```



Quick CPU test:



```bash

python scripts/03\_evaluate\_rer2023\_reconstruction\_visuals.py \\

&#x20; --train-script scripts/02\_train\_rer2023\_single\_layer\_field\_model.py \\

&#x20; --feature-root data/rer2023/single\_layer\_field\_database \\

&#x20; --checkpoint results/rer2023\_single\_layer\_latent\_model/checkpoints/best.pt \\

&#x20; --output-root outputs/rer2023\_full\_reconstruction\_evaluation\_quick \\

&#x20; --splits test \\

&#x20; --device cpu \\

&#x20; --posterior-steps 20 \\

&#x20; --max-samples-per-split 50 \\

&#x20; --visuals-per-class 2

```



### 4.7 Export RER2023 latent feature tables



This script is GUI-based. It does not currently expose a full command-line parser. Run it directly:



```bash

python scripts/05\_export\_rer2023\_latent\_features.py

```



The script will ask the user to select or input:



1\. the dataset root, usually `data/rer2023/single\_layer\_field\_database/`;

2\. the trained checkpoint, for example `results/rer2023\_single\_layer\_latent\_model/checkpoints/best.pt`;

3\. the output folder, for example `outputs/rer2023\_latent\_feature\_tables/`;

4\. the device, usually `cuda` or `cpu`;

5\. posterior fitting steps;

6\. posterior fitting learning rate;

7\. batch size;

8\. IoU threshold, for example `0.85`;

9\. maximum number of samples, where `0` means all samples.



The script fixes the trained decoder and obtains latent vectors for all samples by posterior fitting. It then exports full latent feature tables and IoU-filtered tables for downstream classification or tabular modelling.



A typical interactive run should use:



```text

dataset root: data/rer2023/single\_layer\_field\_database/

checkpoint: results/rer2023\_single\_layer\_latent\_model/checkpoints/best.pt

output folder: outputs/rer2023\_latent\_feature\_tables/

device: cuda

posterior fitting steps: 300

posterior fitting learning rate: 0.02

batch size: 4

IoU threshold: 0.85

max samples: 0

```



The precomputed exported tables are available at:



```text

data/rer2023/latent\_feature\_tables/

```



## 5. Existing precomputed outputs



The repository already contains precomputed outputs for manuscript inspection:



```text

data/synthetic\_stageA\_feature\_database/

data/synthetic\_no\_learning\_representation\_database/

results/synthetic\_stageB\_latent\_dimension\_sweep/

data/rer2023/single\_layer\_field\_database/

results/rer2023\_single\_layer\_latent\_model/

results/rer2023\_reconstruction\_evaluation\_visuals/

data/rer2023/latent\_feature\_tables/

```



Users do not need to rerun all scripts to inspect the manuscript-related outputs. Full training and posterior fitting can be computationally expensive.



## 6. Suggested quick verification order



For a lightweight sanity check after cloning the repository:



```bash

git lfs pull

python scripts/stageA\_sdf\_feature\_database.py \\

&#x20; --input\_dir data/synthetic\_profiles/layer\_polygon\_database \\

&#x20; --output\_dir outputs/test\_stageA\_small

```



For a short synthetic Stage B CPU test:



```bash

python scripts/stageB\_joint\_implicit\_train\_eval\_predict\_timing\_resume\_realtime.py train \\

&#x20; --model\_type vae \\

&#x20; --feature\_root data/synthetic\_stageA\_feature\_database \\

&#x20; --output\_root outputs/test\_stageB\_vae\_cpu \\

&#x20; --latent\_dim 8 \\

&#x20; --epochs 2 \\

&#x20; --batch\_size 2 \\

&#x20; --points\_per\_sample 512 \\

&#x20; --device cpu

```



For a short RER2023 evaluation test:



```bash

python scripts/02\_train\_rer2023\_single\_layer\_field\_model.py evaluate \\

&#x20; --checkpoint results/rer2023\_single\_layer\_latent\_model/checkpoints/best.pt \\

&#x20; --feature-root data/rer2023/single\_layer\_field\_database \\

&#x20; --output-root outputs/test\_rer2023\_eval\_cpu \\

&#x20; --split test \\

&#x20; --max-samples 20 \\

&#x20; --posterior-steps 10 \\

&#x20; --save-visuals 4 \\

&#x20; --device cpu

```



## 7. Notes and cautions



1\. The original RER2023 shapefile is not redistributed in this repository. The script `01\_build\_rer2023\_single\_layer\_field\_database.py` requires users to obtain the original shapefile separately if they want to rebuild the database from raw polygons.



2\. The script `05\_export\_rer2023\_latent\_features.py` is currently GUI-driven. It is suitable for Windows desktop use. For headless servers, users may need to adapt it by adding command-line arguments.



3\. Auto-decoder validation and test evaluation require posterior latent fitting for unseen samples. This can be slow when the number of samples, posterior steps, or points per sample is large.



4\. The `outputs/` folder is recommended for newly generated local files. It avoids overwriting precomputed manuscript results.



5\. During latent-dimension selection, avoid repeated test-set evaluation. Prefer validation-set evaluation first, then run test evaluation only for the final selected configuration.



6\. If GPU memory is limited, reduce `--batch\_size`, `--points\_per\_sample`, `--eval\_grid\_size`, `--posterior-steps`, or `--val-posterior-steps`.



7\. Some scripts can open graphical file-selection dialogs when command-line paths are omitted. For reproducible batch execution, explicit command-line paths are recommended whenever supported.



