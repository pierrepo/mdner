# Molecular dynamics named entity recognition

A Named Entity Recognition model for molecular dynamics data.

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

## Generate training data

Generate json files for spaCy NER and text files containing titles and descriptions of our MD datasets available [here](https://sandbox.zenodo.org/record/1171298).

### Generate text files

Launch the generation of text files :

```
python3 scripts/generate_annotation.py
```

#### Parameters

```
usage: generate_annotation.py [-h] [-c] [threshold] [n]

Generate text files containing the title and description of the dataset in the annotation folder.

positional arguments:
  threshold    The threshold for the description length.
  n            The number of data to be selected for the corpus similarity.

options:
  -h, --help   show this help message and exit
  -c, --clear  Clear the annotation.
```

### Generate json files

Annotating json files requires manual annotation and must be in the `annotations` folder. Here is a site that allows it: [https://tecoholic.github.io/ner-annotator/](https://tecoholic.github.io/ner-annotator/)

Edit the json files :

```
streamlit run scripts/JSON_Corrector.py
```
