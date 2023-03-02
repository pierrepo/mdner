# Molecular dynamics named entity recognition

## Setup your environment

Clone the repository:

```bash
git clone https://github.com/pierrepo/mdner.git
```

Install [miniconda](https://docs.conda.io/en/latest/miniconda.html).

Install [mamba](https://github.com/mamba-org/mamba):

```bash
conda install mamba -n base -c conda-forge
```

Create the `mdner` conda environment:
```
mamba env create -f env/environment.yml
```

Load the `mdner` conda environment:
```
conda activate mdde
```

Note: you can also update the conda environment with:

```bash
mamba env update -f env/environment.yml
```

To deactivate an active environment, use

```
conda deactivate
```
