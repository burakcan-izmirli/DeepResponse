# Deep Response

[![Codacy Badge](https://api.codacy.com/project/badge/Grade/09c5b2bbb6c24d83ba363c7a5c059f7f)](https://app.codacy.com/gh/burakcan-izmirli/DeepResponse?utm_source=github.com&utm_medium=referral&utm_content=burakcan-izmirli/DeepResponse&utm_campaign=Badge_Grade_Settings)

## Installation

1. [Miniforge](https://github.com/conda-forge/miniforge) is recommended for compatibility with Apple Silicon devices.
For the other devices, you can install ```Anaconda``` or ```Miniconda```.

* Please check the prefix field in environment files and change it based on your installation type and directory.

2. You need to download datasets [here](https://drive.google.com/drive/folders/1xfcCyPMVGzhtBxrfv3VtTyCqsOG9oRQk?usp=sharing).

* Do not change naming and folder structure. You should put the ```data``` folder at the same level as the ```src``` folder.

* If you'd like to get ready-to-use datasets, you can only download ```data/processed``` folder. However, if you'd like to get raw data and create the dataset from scratch you need to download ```data/raw```.

3. Execute the following commands with the appropriate environment file for your operating system. 

4. You need to create a conda environment, all the related packages will be installed.

```
conda env create -f [apple_silicon_env.yml/linux_env.yml]
```
## Execution

5. It will create an environment as "deep-response", and you need to activate it.
```
conda activate deep-response
```
6. You can run the model via the terminal:
```
python3 -m src.model.deep_response [-seed -batch_size -epoch -learning_rate -data_type -comet]
```
You can check the arguments and their default values:
```
python3 -m src.model.deep_response --help
```
An example of a running statement with all variables:
```
python3 -m src.model.deep_response -s 12 -b 64 -e 50 -l 0.01 -d 'pathway' -c False
```
### Usage with Comet

You can run DeepResponse with [Comet](https://www.comet.com) support.

In order to do that, you need to pass ```True``` as Comet variable.

```
python3 -m src.model.deep_response -comet True
```

You need to specify ```api_key```, ```project_name``` and ```workspace```. Recommended way is to create ```dev.env``` at the same level as ```.yml``` files and store these variables in there. 
