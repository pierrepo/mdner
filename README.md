# Molecular dynamics named entity recognition

A Named Entity Recognition model for molecular dynamics data.

[![Python 3.10.9](https://img.shields.io/badge/python-%E2%89%A5_3.10.9-blue.svg)](https://www.python.org/downloads/release/python-397/)
[![Conda 22.11.1](https://img.shields.io/badge/conda-%E2%89%A5_22.11.1-green.svg)](https://docs.conda.io/en/latest/miniconda.html)
[![GitHub last commit](https://img.shields.io/github/last-commit/pierrepo/mdner.svg)](https://github.com/pierrepo/mdner)
![GitHub stars](https://img.shields.io/github/stars/pierrepo/mdner.svg?style=social)

## Prerequisites

### Hardware

For the GPU code, it is essential to have a relatively new Nvidia GPU that has a minimum memory capacity of 8.0 GiB. No specific requirements are needed for the CPU code.

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

### Parameters

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

Annotating json files requires manual annotation and must be in the `annotations` folder. Use the `Entity Annotator` to annotate and edit json files by typing the following command :

```
streamlit run scripts/Entity_Annotator.py
```

There are various other tools for annotating such as [Prodigy](https://prodi.gy/) or a site that allows it: [https://tecoholic.github.io/ner-annotator/](https://tecoholic.github.io/ner-annotator/).

## Create a MDNER

To create the `mdner`, the `-c` and `-t` options must be used. The `-c` option tells the script to create a model. The `-t` option is the hyperparameters to be used to train the model.

### Parameters

```
usage: mdner.py [-h] [-p | -c] [-t d f p r] [-g]

Create or call a model for the molecular dynamics data.

options:
  -h, --help            show this help message and exit
  -p, --predict         Call an existing model and extracts the MD information
                        which can be viewed via HTML file.
  -c, --create          Create a dedicated Named Entity Recognition model for
                        our molecular dynamics data.
  -t d f p r, --train d f p r
                        Hyperparameters for the training process where d is
                        the percentage of dropout. The f, p and r scores
                        define what SpaCy believes to be the best model after
                        the training process.
  -g, --gpu             Use GPU for training.

```

### Example

```
cd scripts
python3 mdner.py -c -t 0.4 0.0 0.9 0.1 -g
```

Here, we define a model where the dropout will be 0.4 (40% of the nodes will be hidden). The three other values correspond to the metrics. They allow us to consider what is the best model. Here, for example, we prefer the precision score rather than the recall score.

## Evaluate the MDNER

Evaluate the `mdner` :

```
cd scripts
python3 mdner.py -p
```

After creating the model, a html file will be create in the `results/outputs/` folder.
