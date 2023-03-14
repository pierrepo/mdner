# Molecular dynamics named entity recognition

A Named Entity Recognition model for molecular dynamics data.

[![Python 3.10.9](https://img.shields.io/badge/python-%E2%89%A5_3.10.9-blue.svg)](https://www.python.org/downloads/release/python-397/)
[![Conda 22.11.1](https://img.shields.io/badge/conda-%E2%89%A5_22.11.1-green.svg)](https://docs.conda.io/en/latest/miniconda.html)
[![GitHub last commit](https://img.shields.io/github/last-commit/pierrepo/mdner.svg)](https://github.com/pierrepo/mdner)
![GitHub stars](https://img.shields.io/github/stars/pierrepo/mdner.svg?style=social)

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
mamba env create -f binder/environment.yml
```

Load the `mdner` conda environment:

```
conda activate mdner
```

Note: you can also update the conda environment with:

```bash
mamba env update -f env/environment.yml
```

To deactivate an active environment, use

```
conda deactivate
```

## Generate annotations data

Generate json files for spaCy NER and text files containing titles and descriptions of our MD datasets available [here](https://sandbox.zenodo.org/record/1171298).

Launch the generation of text files and json files :

```
python3 scripts/generate_annotation.py
```

#### Parameters

```
usage: generate_annotation.py [-h] [-c] [threshold] [n]

Generate text files containing the title and description of the dataset in the annotation folder.

positional arguments:
  threshold    The threshold for the description length.
  n            The number of least similar descriptions to be selected.

options:
  -h, --help   show this help message and exit
  -c, --clear  Clear the annotation.
```

Annotating json files requires manual annotation and must be in the `annotations` folder. Use the `JSON Corrector` to annotate and edit json files by typing the following command :

```
streamlit run scripts/JSON_Corrector.py
```

Here is a site that allows it: [https://tecoholic.github.io/ner-annotator/](https://tecoholic.github.io/ner-annotator/)

## Create a MDNER

Create the `mdner`:

```
python3 scripts/mdner.py -c
```

## Use the MDNER

Lauch the `mdner` :

```
python3 scripts/mdner.py -p predictions.txt
```

The text should be in the `results/outputs/` folder
