# DeepResponse: Large Scale Prediction of Cancer Cell Line Drug Response with Deep Learning Based Pharmacogenomic Modelling


[![Codacy Badge](https://app.codacy.com/project/badge/Grade/2c363a3a149c48fa9b6f75af1307e1b2)](https://www.codacy.com/gh/burakcan-izmirli/DeepResponse/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=burakcan-izmirli/DeepResponse&amp;utm_campaign=Badge_Grade)   ![Platform](https://img.shields.io/static/v1?label=platform&message=macos%20%7C%20linux&color=informational)
![License](https://img.shields.io/static/v1?label=license&message=CC-BY-NC-ND-4.0&color=green)

## Abstract

  Assessing the best treatment option for each patient is the main goal of precision medicine. Patients with the same diagnosis may display varying sensitivity to the applied treatment due to genetic heterogeneity, especially in cancers. 
  
  Here, we propose DeepResponse, a machine learning-based system that predicts drug responses (sensitivity) of cancer cells. DeepResponse employs multi-omics profiles of different cancer cell-lines obtained from large-scale screening projects, together with drugs’ molecular features at the input level, and processes them via a hybrid convolutional (cell encoder) and transformer-based (drug encoder) neural network to learn the relationship between tumour multi-omics features and sensitivity to the administered drug.
  
  Both the performance results and in vitro validation experiments indicated DeepResponse successfully predicts drug sensitivity of cancer cells, and especially the multi-omics aspect benefited the learning process and yielded better performance compared to the state-of-the-art. DeepResponse can be used for early-stage discovery of new drug candidates and for repurposing the existing ones against resistant tumours.


## Architecture

This repository implements a hybrid architecture consisting of a **SELFormer-based drug encoder**, an **enhanced CNN cell-line encoder**, and an **MLP prediction head**:

<img width="1500" alt="Architecture of DeepResponse" src="https://github.com/burakcan-izmirli/DeepResponse/assets/65293991/97a7fd5c-28ae-43b8-977d-9503f2627abd">

**Figure 1.** Hybrid deep convolutional and graph neural network (HDCGNN) architecture of DeepResponse. Multi-omic features of cell lines are processed via deep convolutional neural networks, whereas graph represented drug molecules are proessed by message passing networks containing transformer encoder layers.

Implementation references: `src/dataset/base_dataset_strategy.py`, `src/model/build/base_model_build_strategy.py`, `src/model/build/architecture/selformer_architecture.py`.


## Results

**Table 1.** Evaluation results (in terms of RMSE) of DeepResponse and other methods on the GDSC dataset (10-fold cross-validation). DeepResponse is the **state-of-art** compared to existing models on all split strategies.

<img width="1150" alt="Results of DeepResponse" src="https://github.com/burakcan-izmirli/DeepResponse/assets/65293991/1c8ed994-8771-4c11-a7e6-0979e9ea6c3f">



## Installation

1.  [Miniforge](https://github.com/conda-forge/miniforge) is recommended for compatibility with Apple Silicon devices.
For the other devices, you can install ```Anaconda``` or ```Miniconda```.

*  Please check the prefix field in environment files and change it based on your installation type and directory.

2.  Datasets are stored under `dataset/<source>/`:
    - Raw inputs: `dataset/<source>/raw/`
    - Processed artifacts used for training: `dataset/<source>/processed/`

    Download link (Google Drive): [Google Drive folder](https://drive.google.com/drive/folders/1xfcCyPMVGzhtBxrfv3VtTyCqsOG9oRQk?usp=sharing)

    You can either download ready-to-use processed datasets or place the raw files under `dataset/<source>/raw/` and generate the processed artifacts locally (see “Dataset Creation”).

3.  Execute the following commands with the appropriate environment file for your operating system. 

4.  You need to create a conda environment, all the related packages will be installed.

```
conda env create -f [apple_silicon_env.yml/linux_env.yml]
```
## Execution

5.  It will create an environment as "deep-response", and you need to activate it.
```
conda activate deep-response
```
6.  You can run the model via the terminal:
```
python3 -m deep_response [--use_comet --data_source --evaluation_source --data_type --split_type --random_state --batch_size --epoch --learning_rate]
```
You can check the arguments and their default values:
```
python3 -m deep_response --help
```
An example of a running statement with all parameters:
```
python3 -m deep_response --use_comet True --data_source 'gdsc' --evaluation_source 'ccle' --data_type 'normal' --split_type 'random' --random_state 28 --batch_size 64 --epoch 50 --learning_rate 0.01
```

## Dataset Creation

If you have placed the required raw files under `dataset/<source>/raw/`, you can regenerate the processed artifacts via:

```
python3 -m dataset.depmap.create_depmap_dataset
python3 -m dataset.ccle.create_ccle_dataset
python3 -m dataset.gdsc.create_gdsc_dataset
```

Each script writes outputs under `dataset/<source>/processed/` (including `dataset_records.csv` and `cell_features_lookup.npz`).

## Supported Configurations

DeepResponse supports the following high-level CLI configuration rules:

- `--split_type`: `random`, `cell_stratified`, `drug_stratified`, `drug_cell_stratified`, `cross_domain`
- `--data_type`: currently `normal`
- `--data_source`: any dataset source that has processed artifacts available under `dataset/<source>/processed/`
- `--evaluation_source`:
  - Required when `--split_type cross_domain`
  - Must be omitted (`None`) for all other split types

In `cross_domain`, the model is trained on `--data_source` and evaluated on `--evaluation_source`, and both sources must have their processed datasets generated/downloaded.

### Usage with Comet

You can run DeepResponse with [Comet](https://www.comet.com) support.

In order to do that, you need to pass ```True``` as Comet variable.

```
python3 -m deep_response --use_comet True
```

You need to specify ```api_key```, ```project_name``` and ```workspace```. Recommended way is to create ```dev.env``` at the same level as ```.yml``` files and store these variables in there. 

## License

<a rel="license" href="http://creativecommons.org/licenses/by-nc-nd/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-nd/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-nd/4.0/">Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License</a>.
