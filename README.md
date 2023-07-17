# DeepResponse: Large Scale Prediction of Cancer Cell Line Drug Response with Deep Learning Based Pharmacogenomic Modelling


[![Codacy Badge](https://app.codacy.com/project/badge/Grade/2c363a3a149c48fa9b6f75af1307e1b2)](https://www.codacy.com/gh/burakcan-izmirli/DeepResponse/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=burakcan-izmirli/DeepResponse&amp;utm_campaign=Badge_Grade)   ![Platform](https://img.shields.io/static/v1?label=platform&message=macos%20%7C%20linux&color=informational)
![License](https://img.shields.io/static/v1?label=license&message=CC-BY-NC-ND-4.0&color=green)

## Abstract

  Assessing the best treatment option for each patient is the main goal of precision medicine. Patients with the same diagnosis may display varying sensitivity to the applied treatment due to genetic heterogeneity, especially in cancers. 
  
  Here, we propose DeepResponse, a machine learning-based system that predicts drug responses (sensitivity) of cancer cells. DeepResponse employs multi-omics profiles of different cancer cell-lines obtained from large-scale screening projects, together with drugs’ molecular features at the input level, and processing them via hybrid convolutional and graph-transformer deep neural networks to learn the relationship between multi-omics features of the tumour and its sensitivity to the administered drug. 
  
  Both the performance results and in vitro validation experiments indicated DeepResponse successfully predicts drug sensitivity of cancer cells, and especially the multi-omics aspect benefited the learning process and yielded better performance compared to the state-of-the-art. DeepResponse can be used for early-stage discovery of new drug candidates and for repurposing the existing ones against resistant tumours.


## Architecture

<img width="1500" alt="Architecture of DeepResponse" src="https://github.com/burakcan-izmirli/DeepResponse/assets/65293991/97a7fd5c-28ae-43b8-977d-9503f2627abd">

**Figure 1.** Hybrid deep convolutional and graph neural network (HDCGNN) architecture of DeepResponse. Multi-omic features of cell lines are processed via deep convolutional neural networks, whereas graph represented drug molecules are proessed by message passing networks containing transformer encoder layers.


## Results

**Table 1.** Evaluation results (in terms of RMSE) of DeepResponse and other methods on the GDSC dataset (10-fold cross-validation). DeepResponse is the **state-of-art** compared to existing models on all split strategies.

<img width="1720" alt="Results of DeepResponse" src="https://github.com/burakcan-izmirli/DeepResponse/assets/65293991/e4f75524-057d-403a-b083-59eb264a91b4">



## Installation

1.  [Miniforge](https://github.com/conda-forge/miniforge) is recommended for compatibility with Apple Silicon devices.
For the other devices, you can install ```Anaconda``` or ```Miniconda```.

*  Please check the prefix field in environment files and change it based on your installation type and directory.

2.  You need to download datasets [here](https://drive.google.com/drive/folders/1xfcCyPMVGzhtBxrfv3VtTyCqsOG9oRQk?usp=sharing).

*  Do not change naming and folder structure. You should put the ```data``` folder at the same level as the ```src``` folder.

*  If you'd like to get ready-to-use datasets, you can only download ```data/processed``` folder. However, if you'd like to get raw data and create the dataset from scratch you need to download ```data/raw```.

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
python3 -m deep_response --use_comet True --data_source 'gdsc' --evaluation_source 'ccle' --data_type 'l1000' --split_type 'random' --random_state 28 --batch_size 64 --epoch 50 --learning_rate 0.01
```

## Parameter Compliance Matrix

| Data Source | Evaluation Source | Data Type                                              | Split Type                                                           |
|-------------|-------------------|--------------------------------------------------------|----------------------------------------------------------------------|
| `gdsc`      | `None`            | `[normal, l1000, pathway, pathway_reduced, digestive]` | `[random, cell_stratified,  drug_stratified,  drug_cell_stratified]` |
| `gdsc`      | `ccle`            | `[normal, l1000]`                                      | `[cross_domain]`                                                     |
| `gdsc`      | `nci_60`          | `[normal, l1000]`                                      | `[cross_domain]`                                                     |
| `gdsc`      | `ccle`            | `[normal, l1000]`                                      | `[cross_domain]`                                                     |
| `ccle`      | `None`            | `[normal, l1000]`                                      | `[random, cell_stratified,  drug_stratified,  drug_cell_stratified]` |
| `ccle`      | `nci_60`          | `[normal, l1000]`                                      | `[cross_domain]`                                                     |
| `ccle`      | `gdsc`            | `[normal, l1000]`                                      | `[cross_domain]`                                                     |


### Usage with Comet

You can run DeepResponse with [Comet](https://www.comet.com) support.

In order to do that, you need to pass ```True``` as Comet variable.

```
python3 -m deep_response --use_comet True
```

You need to specify ```api_key```, ```project_name``` and ```workspace```. Recommended way is to create ```dev.env``` at the same level as ```.yml``` files and store these variables in there. 

## License

<a rel="license" href="http://creativecommons.org/licenses/by-nc-nd/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-nd/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-nd/4.0/">Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License</a>.

