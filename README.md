# Deep Response

[Miniforge](https://github.com/conda-forge/miniforge) is recommended for compabilitiy.

1. You need to create a conda environment, all the related packages will be installed.

```
conda env create -f enviroment.yml
```

2. It will create an environment as "deep_response", and you need to activate it.
```
conda activate deep_response
```
3. You can run the model via terminal:
```
python3 deep_response.py [-seed -batch_size -epoch -learning_rate -d data_type]
```
You can check the arguments and their default values:
```
python3 deep_response.py --help
```
An example of running statement with all variables:
```
python3 deep_response.py -s 12 -b 64 -e 50 -l 0.01 -d 'pathway'
```
