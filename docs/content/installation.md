# Installation

`maxent_disaggregation` is an open source Python software package. It's available via [`pip`](https://pypi.org/project/pip/) and [`anaconda`](https://anaconda.org). 


## Installation via `pip` 
```console
pip install maxent_disaggregation
```

## Installation via `Conda`

### Option 1: Create a new `Conda environment` for you project
1. Create a new Conda environment (in this example named `maxent_disaggregation`):

```console
conda create -n maxent_disaggregation -c jakobs maxent_disaggregation
```

2. Activate the environment:

```console
conda activate maxent_disaggregation
```

3. If you want to run the example notebooks with the correlation plots, then run:
```console
pip install corner
```




### Option 2: Install in a existing `Conda environment`:
1. Activate `your_conda_environment`
```console
conda activate your_conda_environment
```

2. Install maxent_disaggregation
```console
conda install -c jakobs maxent_disaggregation
```