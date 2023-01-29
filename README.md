# Deep Response

[Miniforge](https://github.com/conda-forge/miniforge) is recommended for compatibility.

1. You need to download datasets [here](https://drive.google.com/drive/folders/1xfcCyPMVGzhtBxrfv3VtTyCqsOG9oRQk?usp=sharing).

* Do not change naming and folder structure. You should put the ```data``` folder at the same level as the ```src``` folder.

* If you'd like to get ready-to-use datasets, you can only download ```data/processed``` folder. However, if you'd like to get raw data and create the dataset from scratch you need to download ```data/raw```.

2. Execute the following commands with the appropriate environment file for your operating system. 

3. You need to create a conda environment, all the related packages will be installed.

```
conda env create -f deep_response_apple_silicon.yml/deep_response_ubuntu.yml
```

4. It will create an environment as "deep_response", and you need to activate it.
```
conda activate deep_response
```
5. You can run the model via the terminal:
```
python3 -m src.python.deep_response [-seed -batch_size -epoch -learning_rate -d data_type]
```
You can check the arguments and their default values:
```
python3 -m src.python.deep_response --help
```
An example of a running statement with all variables:
```
python3 -m src.python.deep_response -s 12 -b 64 -e 50 -l 0.01 -d 'pathway'
```
