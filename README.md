# Nuclei Segmentation
## Setup conda environment

- Verify that you don't have already the `nuclei_seg_env` environment
```
conda env list
```


- Create conda environment
```
conda create --name nuclei_seg_env  python=3.10
```

- Activate environment 

```
conda activate nuclei_seg_env
```

- Verify you are using the correct `python` and `pip` 

```
which python
```

```
which pip
```

## Install dependencies 
```
pip install tensorflow
```

```
pip install stardist
```

# Run sample script

```
python src/2D_Stardist_prediction.py -d '<image_path>/'
```

```
python src/2D_Stardist_prediction.py -d 'data/input'
```



