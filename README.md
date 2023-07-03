# Molecular dynamics named entity recognition

**A Named Entity Recognition model for molecular dynamics data.**

MDNER is a NER model developed specifically to extract information from MD simulations. It is used to identify the names of the molecules simulated, the simulation time, the force field or molecular model used, the simulation temperature and the name of the software used to run the simulations. This is the minimum information required to describe a MD simulation. 



[![Python 3.10.9](https://img.shields.io/badge/python-%E2%89%A5_3.10.9-blue.svg)](https://www.python.org/downloads/release/python-397/)
[![Conda 22.11.1](https://img.shields.io/badge/conda-%E2%89%A5_22.11.1-green.svg)](https://docs.conda.io/en/latest/miniconda.html)
[![GitHub last commit](https://img.shields.io/github/last-commit/pierrepo/mdner.svg)](https://github.com/pierrepo/mdner)
![Black](https://img.shields.io/badge/code%20style-black-black)
![GitHub stars](https://img.shields.io/github/stars/pierrepo/mdner.svg?style=social)

## 🔧 Prerequisites

### Hardware

For the GPU code, it's essential to have a relatively new Nvidia GPU that has a minimum memory capacity of 8.0 GiB. No specific requirements are needed for the CPU code. To use spaCy, see the [spaCy documentation](https://spacy.io/usage).

## 📦 Setup your environment

Clone the repository :

```bash
git clone https://github.com/pierrepo/mdner.git
```

Install [mamba](https://github.com/mamba-org/mamba) :

```bash
conda install mamba -n base -c conda-forge
```

Create the `mdner` conda environment :

```
mamba env create -f binder/environment.yml
```

Load the `mdner` conda environment :

```
conda activate mdner
```

Note: you can also update the conda environment with :

```bash
mamba env update -f binder/environment.yml
```

To deactivate an active environment, use :

```
conda deactivate
```

## ✍ Generate annotations data

Generate json files for spaCy NER and text files containing titles and descriptions of our MD datasets available [here](https://sandbox.zenodo.org/record/1171298).

Launch the generation of text files and json files :

```
python3 scripts/generate_annotation.py
```

### Parameters

```
usage: generate_annotation.py [-h] [-c] [-p {mbart,bart-paraphrase,pegasus}] [-s SEED] [threshold] [cutoff]

Generate text and json files in the annotation folder to be used as training sets.

positional arguments:
  threshold             The threshold for the length of the descriptive texts.
  cutoff                Select the descriptive texts where the cosine similarity is below the threshold.

options:
  -h, --help            show this help message and exit
  -c, --clear           Clear the annotation.
  -p {mbart,bart-paraphrase,pegasus}, --paraphrase {mbart,bart-paraphrase,pegasus}
                        Paraphrase the annotation.
  -s SEED, --seed SEED  Set the seed for reproducibility in paraphrase.
```

Annotating json files requires manual annotation and must be in the `annotations` folder. Use the `Entity Annotator` to annotate and edit json files by typing the following command :

```
streamlit run scripts/Entity_Annotator.py
```

There are various other tools for annotating such as [Prodigy](https://prodi.gy/) or a site that allows it: [https://tecoholic.github.io/ner-annotator/](https://tecoholic.github.io/ner-annotator/).

If you think you don't have enough data, you can duplicate the annotated texts with the following command:

```
python3 scripts/generate_annotation.py -p mbart
```

Duplication consists of paraphrasing, i.e. keeping the context of the original text and reformulating it in another way. Here you will use the mBART model for paraphrasing.

## 🛠 Create a MDNER

The `mdner.py` script is used to create the model according to the defined parameters.

### Parameters

```
usage: mdner.py [-h] [-c] [-t d f p r] [-n NAME] [-g] [-p] [-m] [-s SEED]

Create a model for the molecular dynamics data.

options:
  -h, --help            show this help message and exit
  -c, --create          Create a dedicated Named Entity Recognition model for our molecular dynamics data.
  -t d f p r, --train d f p r
                        Hyperparameters for the training process where d is the percentage of dropout. The f, p and r scores define what SpaCy believes to be the best model after the training process.
  -n NAME, --name NAME  Name of the model.
  -g, --gpu             Use GPU for training.
  -p, --paraphrase      Add paraphrase in the training dataset.
  -m, --mol             Use only MOL entities.
  -s SEED, --seed SEED  Set the seed for reproducibility.
```

To create the `mdner`, the `-c`, `-t` and `-n` options must be used. The `-c` option tells the script to create a model. The `-t` option is the hyperparameters to be used to train the model. The `-n` option corresponds to the name model that will be created.

You can introduce paraphrases only in the learning set (training + test) with the `-p` option.

### Example

```
python3 scripts/mdner.py -c -t 0.4 0.0 0.9 0.1 -n my_model -g
```

Here, we define a model where the dropout will be 0.4 (40% of the nodes will be deactivate). The three other values correspond to the metrics. They allow us to consider what is the best model. Here we prefer the precision score rather than the recall score. We have also chosen to create a model based on Transformers by using the `-g` option. If the `-g` option is not chosen, the model generated will be based on the cpu.

## 📋 Try MDNER

In order to run an exemple, you can launch a streamlit site to apply the MDNER model to a text and evaluate it.  Simply enter the name of the model as an argument, as in the following command :

```
streamlit run scripts/MDner.py -- --model my_model
```
## 📈 Results
From the original and paraphrased texts obtained with the mBART model, we have trained two NER model based on the Transformers "*BioMed-RoBERTa-base*" and we evaluated the models on the validation set as shown in Table 1.

<figure class="table" align="center">
<figcaption> Table 1 - Mean precision scores with standard deviation for each entity of the Transformers model based on "<i>BioMed-RoBERTa-base</i>" without and with paraphrase. Each model was generated over 3 replicates. The best precision scores per entity are shown in bold.</figcaption>
<style type="text/css">
table {
  text-align:center !important;
}
td {
  text-align:center !important;
}
th {
  text-align:center !important;
}
</style>
<table align="center">
<thead>
  <tr>
    <th>Entities<br></th>
    <th colspan="2">Precision score (%)<br></th>
  </tr>
</thead>
<tbody>
  <tr>
    <td></td>
    <td>Transformers</td>
    <td>Transformers + Paraphrase</td>
  </tr>
  <tr style="font-weight: bold;">
    <td>MOL (molecule)</td>
    <td>75 ± 1.3</td>
    <td>84 ± 1.4</td>
  </tr>
  <tr align="center">
    <td>FFM (force field &amp; model)</td>
    <td>86 ± 2.1</td>
    <td>90 ± 1.6</td>
  </tr>
  <tr>
    <td>TEMP (temperature)</td>
    <td>90 ± 2.1</td>
    <td>91 ± 1.9</td>
  </tr>
  <tr>
    <td>STIME (simulation time)</td>
    <td>71 ± 6.3</td>
    <td>73 ± 8.2</td>
  </tr>
  <tr>
    <td>SOFT (software)</td>
    <td>77 ± 1.5</td>
    <td>66 ± 5.3</td>
  </tr>
  <tr>
    <td>Total</td>
    <td>75 ± 0.7</td>
    <td>82 ± 1.4</td>
  </tr>
</tbody>
</table>
</figure>

We note an increase in the accuracy score, particularly for our key entity, the MOL entity, which rises from 75% to 84%. Performance for the other entities is improved slightly, except for the SOFT entity.
The NER models were able to identify molecule names not present in the learning dataset, perfectly underlining the ability of the NER model to generalize and identify the desired entities, and demonstrating the relevance of fine-tuning on Transformer models.

